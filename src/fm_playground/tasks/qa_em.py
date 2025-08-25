"""Simple QA task with exact-match metric."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..datasets.base import Sample
from ..metrics.exact_match import exact_match


@dataclass
class TaskSpec:
    name: str
    fewshot_k: int = 0
    allow_truncate: bool = False
    metric_fn: Callable[[str, str], Dict[str, float]] = exact_match

    def build_prompt(self, sample: Sample) -> str:
        return f"Question: {sample.input}\nAnswer:"

    def postprocess(self, prediction: str) -> str:
        return prediction.strip()

    def to_lighteval_item(self, sample: Sample) -> Dict[str, str]:
        prompt = self.build_prompt(sample)
        return {"prompt": prompt, "reference": sample.target, "metadata": sample.metadata or {}}


def create_task(**kwargs) -> TaskSpec:
    return TaskSpec(name="qa_em", **kwargs)
