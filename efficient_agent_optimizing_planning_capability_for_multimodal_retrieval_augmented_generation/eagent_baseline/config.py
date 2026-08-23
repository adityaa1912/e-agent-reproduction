from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel


class RuntimeMode(str, Enum):
    DEVELOPMENT = "development"
    RESEARCH = "research"


class ModelSpec(BaseModel):
    provider: str
    model_name: str


class ToolSpec(BaseModel):
    provider: str


class ToolsConfig(BaseModel):
    image_search: ToolSpec
    text_search: ToolSpec


class DatasetConfig(BaseModel):
    benchmark_size: int
    planner_train_samples: int


class PlannerTrainingScaffoldConfig(BaseModel):
    base_model: str
    train_samples: int
    objective: Optional[str] = None
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    scheduler: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    precision: Optional[str] = None
    seed: Optional[int] = None
    hardware: Optional[str] = None
    data_split: Optional[str] = None


class BaselineConfig(BaseModel):
    mode: RuntimeMode
    planner: ModelSpec
    executor: ModelSpec
    tools: ToolsConfig
    dataset: DatasetConfig
    planner_training: PlannerTrainingScaffoldConfig

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BaselineConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw)


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
