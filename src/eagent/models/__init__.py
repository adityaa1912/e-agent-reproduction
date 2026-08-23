"""Provider-independent multimodal model abstraction for E-Agent.

Public surface for the model layer: the provider-neutral interface and
request/response types, the typed configuration models, and the factory used
to construct planner and executor models.
"""

from eagent.models.config import (
    EAgentModelConfig,
    ModelSpec,
    RuntimeMode,
)
from eagent.models.factory import (
    UnsupportedProviderError,
    create_executor,
    create_model,
    create_planner,
)
from eagent.models.protocols import (
    ModelRequest,
    ModelResponse,
    VisionLanguageModel,
)

__all__ = [
    "EAgentModelConfig",
    "ModelSpec",
    "RuntimeMode",
    "UnsupportedProviderError",
    "create_executor",
    "create_model",
    "create_planner",
    "ModelRequest",
    "ModelResponse",
    "VisionLanguageModel",
]
