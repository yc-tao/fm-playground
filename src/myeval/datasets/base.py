"""Dataset helpers and sample definition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import csv
import json


@dataclass
class Sample:
    """Simple data container for an evaluation sample."""

    input: str
    target: str
    metadata: Optional[Dict[str, str]] = None


def load_jsonl(path: str, input_col: str = "input", target_col: str = "target") -> Iterator[Sample]:
    """Yield samples from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield Sample(input=obj[input_col], target=obj[target_col], metadata=obj)


def load_csv(
    path: str, input_col: str = "input", target_col: str = "target"
) -> Iterator[Sample]:
    """Yield samples from a CSV file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield Sample(input=row[input_col], target=row[target_col], metadata=row)


def load_hf_dataset(
    name: str,
    split: str,
    input_col: str = "input",
    target_col: str = "target",
    **load_kwargs,
) -> Iterator[Sample]:
    """Yield samples from a HuggingFace dataset."""
    from datasets import load_dataset  # local import to avoid dependency during import time

    ds = load_dataset(name, split=split, **load_kwargs)
    for item in ds:
        yield Sample(input=item[input_col], target=item[target_col], metadata=dict(item))
