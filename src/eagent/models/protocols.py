"""Provider-neutral interface for multimodal (vision-language) models.

This module defines the contract that every model provider must implement.
It deliberately contains NO provider-specific logic: there are no references
to Hugging Face, Transformers, vLLM, remote API clients, CUDA, or any concrete
inference backend. Application code (future planner / executor) depends only on
these abstractions, never on a concrete provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eagent.common.types import Image


@dataclass
class ModelRequest:
    """A provider-neutral request to a vision-language model.

    Attributes:
        prompt: The text prompt / instruction for the model.
        images: Zero or more images accompanying the prompt.
        temperature: Sampling temperature. ``0.0`` requests deterministic output.
        max_tokens: Optional cap on the number of generated tokens.
        metadata: Optional free-form metadata carried alongside the request.
    """

    prompt: str
    images: List[Image] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        """Return ``True`` if the request carries one or more images."""
        return len(self.images) > 0


@dataclass
class ModelResponse:
    """A provider-neutral response from a vision-language model.

    Attributes:
        text: The generated text.
        model_name: The name of the model that produced the response.
        usage: Optional usage information (e.g. token counts).
        raw: Optional raw, provider-specific response object for debugging.
    """

    text: str
    model_name: str
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Any] = None


class VisionLanguageModel(ABC):
    """Abstract interface for a multimodal vision-language model.

    Concrete providers (e.g. the development stub, or future research
    backends for InternVL2-8B / Qwen2-VL-72B) implement this interface.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model name."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for the given multimodal request."""
        raise NotImplementedError
