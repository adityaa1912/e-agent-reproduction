"""Minimal, provider-independent common types for the E-Agent model layer.

These types intentionally stay small and free of any provider-specific
concepts (no Hugging Face, Transformers, vLLM, API-client, or CUDA details).
They only model the two primitives the model abstraction needs at this step:

* ``Image``    -> a provider-neutral reference to an image.
* ``Question`` -> a natural-language question, optionally with images.

Do NOT add planner-specific or executor-specific types here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Image:
    """A provider-neutral reference to an image.

    An image may be described by a URL, a local file path, or raw bytes.
    All fields are optional so the type can represent whichever form is
    available, without committing to a particular loading mechanism.
    """

    url: Optional[str] = None
    path: Optional[str] = None
    data: Optional[bytes] = None
    mime_type: Optional[str] = None

    def has_content(self) -> bool:
        """Return ``True`` if the image references any concrete content."""
        return any(value is not None for value in (self.url, self.path, self.data))


@dataclass
class Question:
    """A natural-language question, optionally accompanied by images."""

    text: str
    images: List[Image] = field(default_factory=list)

    @property
    def is_multimodal(self) -> bool:
        """Return ``True`` if the question carries one or more images."""
        return len(self.images) > 0
