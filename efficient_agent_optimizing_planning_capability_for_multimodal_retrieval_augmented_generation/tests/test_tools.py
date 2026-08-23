import unittest

import tests._paths

from eagent.common.types import Image

from eagent_baseline.models import build_executor_model
from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.tools import (
    ImageSearchTool,
    RequeryTool,
    ResponseTool,
    StubImageSearchProvider,
    StubTextSearchProvider,
    TextSearchTool,
)


def _config() -> BaselineConfig:
    return BaselineConfig.from_yaml(default_config_path())


class SearchToolTests(unittest.TestCase):
    def test_image_search_offline_default_empty(self) -> None:
        tool = ImageSearchTool()
        result = tool.run(image=Image(url="http://example.invalid/a.png"))
        self.assertEqual(result.tool, "image_search")
        self.assertEqual(result.content, [])

    def test_image_search_returns_canned_results(self) -> None:
        tool = ImageSearchTool(StubImageSearchProvider([{"url": "x"}]))
        result = tool.run(image=Image(url="http://example.invalid/a.png"))
        self.assertEqual(result.metadata["result_count"], 1)

    def test_image_search_requires_image(self) -> None:
        with self.assertRaises(TypeError):
            ImageSearchTool().run(image="not-an-image")

    def test_text_search_offline(self) -> None:
        tool = TextSearchTool(StubTextSearchProvider([{"url": "y"}]))
        result = tool.run(query="usa basketball olympics gold")
        self.assertEqual(result.tool, "text_search")
        self.assertEqual(result.metadata["result_count"], 1)

    def test_text_search_requires_query(self) -> None:
        with self.assertRaises(TypeError):
            TextSearchTool().run(query="")


class MLLMToolTests(unittest.TestCase):
    def test_requery_uses_model(self) -> None:
        model = build_executor_model(_config())
        tool = RequeryTool(model)
        result = tool.run(question="Who is this?", image=Image(url="x"), image_search_results=[])
        self.assertEqual(result.tool, "requery")
        self.assertIsInstance(result.content, str)
        self.assertEqual(result.metadata["model_name"], "stub-executor")

    def test_response_uses_model(self) -> None:
        model = build_executor_model(_config())
        tool = ResponseTool(model)
        result = tool.run(
            question="Who is this?",
            image=Image(url="x"),
            image_search_results=[],
            text_search_results=[],
        )
        self.assertEqual(result.tool, "response")
        self.assertIsInstance(result.content, str)


if __name__ == "__main__":
    unittest.main()
