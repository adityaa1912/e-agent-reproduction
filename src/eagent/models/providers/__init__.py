"""Provider implementations of the :class:`VisionLanguageModel` interface.

Development/test providers (``stub``) and the Transformers-based
real-model provider (``real_transformers``) are available.
"""

from eagent.models.providers.stub import StubVisionLanguageModel
from eagent.models.providers.transformers import RealTransformersVisionLanguageModel

__all__ = [
    "StubVisionLanguageModel",
    "RealTransformersVisionLanguageModel",
]
