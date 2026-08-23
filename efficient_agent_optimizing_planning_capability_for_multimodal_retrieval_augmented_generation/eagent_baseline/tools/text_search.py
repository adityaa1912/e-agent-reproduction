from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from eagent_baseline.tools.base import Tool, ToolResult


class TextSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str) -> List[dict]:
        raise NotImplementedError


class StubTextSearchProvider(TextSearchProvider):
    """Deterministic, offline text-search provider for local development.

    Not connected to any external service. Returns a fixed, empty-by-default
    payload so the executor can run without network access or credentials.
    """

    def __init__(self, canned_results: Optional[List[dict]] = None) -> None:
        self._canned_results = list(canned_results) if canned_results is not None else []

    def search(self, query: str) -> List[dict]:
        return list(self._canned_results)


class TextSearchTool(Tool):
    def __init__(self, provider: Optional[TextSearchProvider] = None) -> None:
        self._provider = provider if provider is not None else StubTextSearchProvider()

    @property
    def name(self) -> str:
        return "text_search"

    def run(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query:
            raise TypeError("text_search requires a non-empty string 'query'")
        results = self._provider.search(query)
        return ToolResult(
            tool=self.name,
            content=results,
            metadata={"query": query, "result_count": len(results)},
        )
