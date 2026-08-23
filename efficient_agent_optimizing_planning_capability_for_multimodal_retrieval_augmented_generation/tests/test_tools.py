import unittest

import tests._paths

from eagent.common.types import Image

from eagent_baseline.models import build_executor_model
from eagent_baseline.tools import (
    ImageSearchTool,
    RequeryTool,
    ResponseTool,
    StubImageSearchProvider,
    StubTextSearchProvider,
    TextSearchTool,
)

from tests._fixtures import _config


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

    def test_requery_requires_non_empty_question(self) -> None:
        model = build_executor_model(_config())
        tool = RequeryTool(model)
        with self.assertRaises(TypeError):
            tool.run(question="", image=Image(url="x"), image_search_results=[])

    def test_requery_input_contract_question_type(self) -> None:
        model = build_executor_model(_config())
        tool = RequeryTool(model)
        with self.assertRaises(TypeError):
            tool.run(question=123, image=Image(url="x"), image_search_results=[])

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

    def test_response_requires_non_empty_question(self) -> None:
        model = build_executor_model(_config())
        tool = ResponseTool(model)
        with self.assertRaises(TypeError):
            tool.run(question="", image=Image(url="x"), image_search_results=[], text_search_results=[])

    def test_response_input_contract_all_parameters_passed(self) -> None:
        model = build_executor_model(_config())
        tool = ResponseTool(model)
        result = tool.run(
            question="q",
            image=Image(url="x"),
            image_search_results=[{"url": "img1"}],
            text_search_results=[{"url": "txt1"}],
        )
        self.assertEqual(result.tool, "response")
        self.assertTrue(isinstance(result.content, str))

    def test_image_search_tool_name(self) -> None:
        tool = ImageSearchTool()
        self.assertEqual(tool.name, "image_search")

    def test_text_search_tool_name(self) -> None:
        tool = TextSearchTool()
        self.assertEqual(tool.name, "text_search")

    def test_requery_tool_name(self) -> None:
        model = build_executor_model(_config())
        tool = RequeryTool(model)
        self.assertEqual(tool.name, "requery")

    def test_response_tool_name(self) -> None:
        model = build_executor_model(_config())
        tool = ResponseTool(model)
        self.assertEqual(tool.name, "response")


if __name__ == "__main__":
    unittest.main()
