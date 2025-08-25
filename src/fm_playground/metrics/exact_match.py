"""Exact match metric."""
from __future__ import annotations

from typing import Dict


def exact_match(prediction: str, reference: str) -> Dict[str, float]:
    """Return 1.0 if strings match exactly, else 0.0."""
    return {"em": float(prediction.strip() == reference.strip())}
