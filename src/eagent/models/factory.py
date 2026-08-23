"""Factory for constructing :class:`VisionLanguageModel` instances.

Application code asks the factory for a *planner* or *executor* model built
from an :class:`EAgentModelConfig`. Provider-specific construction details are
kept behind this factory so future code depends only on the abstract
interface.

Two executable providers are supported:
- ``stub`` – deterministic development double.
- ``real_transformers`` – CPU-only HuggingFace multimodal model.

The ``research`` provider (real InternVL2-8B / Qwen2-VL-72B) is intentionally
NOT implemented here; requesting it fails clearly.
"""

from __future__ import annotations

from eagent.models.config import EAgentModelConfig, ModelSpec
from eagent.models.protocols import VisionLanguageModel
from eagent.models.providers.stub import StubVisionLanguageModel
from eagent.models.providers.transformers import RealTransformersVisionLanguageModel

STUB_PROVIDER = "stub"
REAL_TRANSFORMERS_PROVIDER = "real_transformers"


class UnsupportedProviderError(ValueError):
    """Raised when a model is requested for a provider that is not supported."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"Unsupported model provider: {provider!r}. "
            f"Only {STUB_PROVIDER!r} and {REAL_TRANSFORMERS_PROVIDER!r} are executable."
        )


def create_model(spec: ModelSpec) -> VisionLanguageModel:
    """Construct a :class:`VisionLanguageModel` from a :class:`ModelSpec`."""
    if spec.provider == STUB_PROVIDER:
        return StubVisionLanguageModel(model_name=spec.model_name)
    if spec.provider == REAL_TRANSFORMERS_PROVIDER:
        model_id = spec.model_name
        return RealTransformersVisionLanguageModel(model_id=model_id)
    raise UnsupportedProviderError(spec.provider)


def create_planner(config: EAgentModelConfig) -> VisionLanguageModel:
    """Construct the planner model from the E-Agent configuration."""
    return create_model(config.planner)


def create_executor(config: EAgentModelConfig) -> VisionLanguageModel:
    """Construct the executor model from the E-Agent configuration."""
    return create_model(config.executor)
