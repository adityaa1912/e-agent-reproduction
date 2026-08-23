from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import eagent_baseline._bootstrap

from eagent.common.types import Image

from eagent_baseline.tools.base import Tool, ToolResult


class ImageSearchProvider(ABC):
    @abstractmethod
    def search(self, image: Image) -> List[dict]:
        raise NotImplementedError


class StubImageSearchProvider(ImageSearchProvider):
    """Deterministic, offline image-search provider for local development.

    Not connected to any external service. Returns a fixed, empty-by-default
    payload so the executor can run without network access or credentials.
    """

    def __init__(self, canned_results: Optional[List[dict]] = None) -> None:
        self._canned_results = list(canned_results) if canned_results is not None else []

    def search(self, image: Image) -> List[dict]:
        return list(self._canned_results)


class ImageSearchTool(Tool):
    def __init__(self, provider: Optional[ImageSearchProvider] = None) -> None:
        self._provider = provider if provider is not None else StubImageSearchProvider()

    @property
    def name(self) -> str:
        return "image_search"

    def run(self, **kwargs: Any) -> ToolResult:
        image = kwargs.get("image")
        if not isinstance(image, Image):
            raise TypeError("image_search requires an 'image' of type eagent.common.types.Image")
        results = self._provider.search(image)
        return ToolResult(tool=self.name, content=results, metadata={"result_count": len(results)})
