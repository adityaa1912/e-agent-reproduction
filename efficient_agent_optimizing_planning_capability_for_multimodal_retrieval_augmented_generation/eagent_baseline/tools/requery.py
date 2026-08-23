from __future__ import annotations

from typing import Any, List, Optional

import eagent_baseline._bootstrap

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest, VisionLanguageModel

from eagent_baseline.tools.base import Tool, ToolResult

ASSUMED_REQUERY_PROMPT_TEMPLATE = (
    "You generate a concise web search query. "
    "Given the image, the user question, and any image-search results, "
    "produce a short keyword-style query that captures the information needed. "
    "Question: {question}\n"
    "Image search results: {image_search_results}"
)


class RequeryTool(Tool):
    def __init__(self, model: VisionLanguageModel) -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "requery"

    def run(self, **kwargs: Any) -> ToolResult:
        question = kwargs.get("question")
        if not isinstance(question, str) or not question:
            raise TypeError("requery requires a non-empty string 'question'")
        image: Optional[Image] = kwargs.get("image")
        image_search_results = kwargs.get("image_search_results", [])
        images: List[Image] = [image] if isinstance(image, Image) else []
        prompt = ASSUMED_REQUERY_PROMPT_TEMPLATE.format(
            question=question,
            image_search_results=image_search_results,
        )
        request = ModelRequest(prompt=prompt, images=images, temperature=0.0)
        response = self._model.generate(request)
        return ToolResult(
            tool=self.name,
            content=response.text,
            metadata={"model_name": response.model_name},
        )
