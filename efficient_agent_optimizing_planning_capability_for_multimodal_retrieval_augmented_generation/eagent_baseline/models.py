from __future__ import annotations

from typing import List

import eagent_baseline._bootstrap

from eagent.models.config import EAgentModelConfig
from eagent.models.factory import create_executor, create_planner
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel

from eagent_baseline.config import BaselineConfig


def to_eagent_model_config(config: BaselineConfig) -> EAgentModelConfig:
    return EAgentModelConfig.model_validate(
        {
            "mode": config.mode.value,
            "planner": config.planner.model_dump(),
            "executor": config.executor.model_dump(),
        }
    )


def build_planner_model(config: BaselineConfig) -> VisionLanguageModel:
    return create_planner(to_eagent_model_config(config))


def build_executor_model(config: BaselineConfig) -> VisionLanguageModel:
    return create_executor(to_eagent_model_config(config))


class ScriptedVisionLanguageModel(VisionLanguageModel):
    """Development double that replays preset responses in order.

    Not a paper artifact. Intended only to drive deterministic local runs and
    tests of the planner and executor without a real vision-language model.
    """

    def __init__(self, model_name: str, scripted_responses: List[str]) -> None:
        self._model_name = model_name
        self._scripted_responses = list(scripted_responses)
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        return self._call_count

    def generate(self, request: ModelRequest) -> ModelResponse:
        index = min(self._call_count, len(self._scripted_responses) - 1)
        text = self._scripted_responses[index]
        self._call_count += 1
        return ModelResponse(text=text, model_name=self._model_name, usage=None, raw=None)
