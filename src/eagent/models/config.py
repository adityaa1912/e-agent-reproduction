"""Typed configuration models for the E-Agent model layer.

Configuration is expressed with three types:

* :class:`RuntimeMode`      -> ``development`` or ``research``.
* :class:`ModelSpec`        -> ``provider`` + ``model_name`` for one model role.
* :class:`EAgentModelConfig`-> top-level config holding ``planner`` + ``executor``.

The research configuration merely *records* the models named in the paper
(InternVL2-8B for the planner, Qwen2-VL-72B for the executor). It does not
imply that a ``research`` provider is implemented.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel


class RuntimeMode(str, Enum):
    """Supported runtime modes."""

    DEVELOPMENT = "development"
    RESEARCH = "research"


class ModelSpec(BaseModel):
    """Specification of a single model role."""

    provider: str
    model_name: str


class EAgentModelConfig(BaseModel):
    """Top-level E-Agent model configuration.

    Holds the two logical model roles separately:

    * ``planner``  -> the mRAG planner model (InternVL2-8B in research).
    * ``executor`` -> the MLLM tool backbone (Qwen2-VL-72B in research).
    """

    mode: RuntimeMode
    planner: ModelSpec
    executor: ModelSpec

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EAgentModelConfig":
        """Load and validate an :class:`EAgentModelConfig` from a YAML file."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw)
