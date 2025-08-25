"""Task registry."""
from __future__ import annotations

from typing import Dict

from .qa_em import create_task, TaskSpec

_TASKS: Dict[str, TaskSpec] = {
    "qa_em": create_task(),
}


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError:
        raise KeyError(f"Task '{name}' is not registered")
