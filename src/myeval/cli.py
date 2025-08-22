"""Command line interface for myeval.

Reads a YAML configuration file and runs the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .backends.lighteval_runner import build_pipeline
from .datasets import base as dataset_base
from .tasks.registry import get_task
from .utils import set_seed
from .schemas import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run myeval")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.parse_obj(raw)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(42)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(cfg.task.name)
    pipeline = build_pipeline(cfg)

    # Load data
    if cfg.data.source.type == "jsonl":
        samples = dataset_base.load_jsonl(
            cfg.data.source.path,
            input_col=cfg.data.source.input_col,
            target_col=cfg.data.source.target_col,
        )
    elif cfg.data.source.type == "csv":
        samples = dataset_base.load_csv(
            cfg.data.source.path,
            input_col=cfg.data.source.input_col,
            target_col=cfg.data.source.target_col,
        )
    elif cfg.data.source.type == "hf":
        samples = dataset_base.load_hf_dataset(
            cfg.data.source.name,
            split=cfg.data.source.split,
            input_col=cfg.data.source.input_col,
            target_col=cfg.data.source.target_col,
        )
    else:
        raise ValueError(f"Unknown data source type: {cfg.data.source.type}")

    results: List[Dict[str, Any]] = []
    metrics_sum: Dict[str, float] = {}
    n_samples = 0

    for sample in samples:
        if cfg.max_samples and n_samples >= cfg.max_samples:
            break
        item = task.to_lighteval_item(sample)
        pred = pipeline.generate([item["prompt"]])[0]
        post_pred = task.postprocess(pred)
        metric_vals = task.metric_fn(post_pred, item["reference"])
        record = {
            "prompt": item["prompt"],
            "reference": item["reference"],
            "prediction": post_pred,
            **metric_vals,
        }
        results.append(record)
        for k, v in metric_vals.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        n_samples += 1

    summary = {k: v / max(n_samples, 1) for k, v in metrics_sum.items()}
    summary["num_samples"] = n_samples
    summary["model"] = cfg.model.name
    summary["task"] = cfg.task.name

    with open(output_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {n_samples} results to {output_dir}")


if __name__ == "__main__":
    main()
