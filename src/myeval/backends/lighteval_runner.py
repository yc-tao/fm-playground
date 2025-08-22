"""Build generation pipeline using different backends.

This is a very small wrapper that chooses between `accelerate` and `vllm` to
provide a `generate(prompts: List[str]) -> List[str]` function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dep
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

from ..schemas import Config


@dataclass
class AcceleratePipeline:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device

    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=32)
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts


@dataclass
class VLLMPipeline:
    llm: "LLM"

    def generate(self, prompts: List[str]) -> List[str]:  # pragma: no cover - requires GPU
        params = SamplingParams(max_tokens=32)
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]


def build_pipeline(cfg: Config):
    """Create a generation pipeline based on configuration."""
    backend = cfg.backend
    model_name = cfg.model.name
    if backend == "accelerate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        return AcceleratePipeline(model=model, tokenizer=tokenizer, device=device)
    elif backend == "vllm":  # pragma: no cover - requires GPU
        if LLM is None:
            raise RuntimeError("vllm is not installed")
        llm = LLM(model=model_name)
        return VLLMPipeline(llm=llm)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
