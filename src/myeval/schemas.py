"""Pydantic schemas for configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class ModelConfig(BaseModel):
    name: str
    dtype: Optional[str] = None
    use_chat_template: bool = False
    tensor_parallel_size: Optional[int] = None
    data_parallel_size: Optional[int] = None
    generation_parameters: dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    name: str
    fewshot_k: int = 0
    allow_truncate: bool = False


class DataSourceConfig(BaseModel):
    type: str
    path: Optional[str] = None
    name: Optional[str] = None
    split: Optional[str] = None
    input_col: str = "input"
    target_col: str = "target"


class DataConfig(BaseModel):
    source: DataSourceConfig


class Config(BaseModel):
    backend: str = "accelerate"
    output_dir: str = "evals_out"
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None
    override_batch_size: Optional[int] = None
    model: ModelConfig
    task: TaskConfig
    data: DataConfig
