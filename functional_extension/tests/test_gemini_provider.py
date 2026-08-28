"""Offline tests for the functional Gemini provider.

These tests patch the google-genai client. They make no network call and are
not part of the paper-skeleton suites. The gated end-to-end test is skipped
unless both EAGENT_FUNCTIONAL_GEMINI=1 and the API key are present.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_BASELINE = (
    _REPO_ROOT
    / "efficient_agent_optimizing_planning_capability_for_multimodal_retrieval_augmented_generation"
)
_EXT = _REPO_ROOT / "functional_extension"
for _extra in (str(_SRC), str(_BASELINE), str(_EXT)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from google.genai import types

from eagent.common.types import Image, Question
from eagent.models.protocols import ModelRequest

from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.models import build_executor_model
from eagent_baseline.plan import MRAGPlan

from eagent_functional.gemini_provider import (
    MRAG_PLAN_RESPONSE_SCHEMA,
    GeminiAPIError,
    MissingGeminiAPIKeyError,
    RealGeminiVisionLanguageModel,
)
from eagent_functional.planner_integration import run_functional_pipeline

_PLAN_JSON = '{"steps": [{"tool": "response", "arguments": {}}]}'
_IMAGE_BYTES = b"\x89PNG-fake-bytes"

def _fake_response(
    text: str | None = _PLAN_JSON,
    finish_reason: str = "STOP",
    thought_text: str | None = None,
) -> MagicMock:
    response = MagicMock()
    response.text = text
    usage = MagicMock()
    usage.prompt_token_count = 11
    usage.candidates_token_count = 22
    usage.total_token_count = 33
    response.usage_metadata = usage
    candidate = MagicMock()
    candidate.finish_reason = finish_reason
    parts = []
    if thought_text is not None:
        thought = MagicMock()
        thought.text = thought_text
        thought.thought = True
        parts.append(thought)
    candidate.content.parts = parts
    response.candidates = [candidate]
    return response


class _SDKHarness:
    def __init__(self, test: unittest.TestCase, response: MagicMock | None = None):
        self.client_cls = patch("google.genai.Client").start()
        test.addCleanup(patch.stopall)
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        env.start()
        test.addCleanup(env.stop)
        self.generate = self.client_cls.return_value.models.generate_content
        self.generate.return_value = (
            _fake_response() if response is None else response
        )

    @property
    def kwargs(self) -> dict:
        return self.generate.call_args.kwargs

    @property
    def contents(self) -> list:
        return self.kwargs["contents"]

    @property
    def config(self):
        return self.kwargs["config"]


class GeminiProviderConfigTests(unittest.TestCase):
    def test_default_model_id_is_gemini_flash(self) -> None:
        self.assertEqual(
            RealGeminiVisionLanguageModel().model_name, "gemini-3.6-flash"
        )

    def test_model_name_returns_specified_id(self) -> None:
        model = RealGeminiVisionLanguageModel(model_id="custom/model-id")
        self.assertEqual(model.model_name, "custom/model-id")

    def test_construction_does_not_build_client(self) -> None:
        harness = _SDKHarness(self)
        RealGeminiVisionLanguageModel()
        harness.client_cls.assert_not_called()

    def test_missing_api_key_raises_without_sdk_call(self) -> None:
        harness = _SDKHarness(self)
        model = RealGeminiVisionLanguageModel(api_key_env_var="EAGENT_ABSENT_KEY")
        os.environ.pop("EAGENT_ABSENT_KEY", None)
        with self.assertRaises(MissingGeminiAPIKeyError):
            model.generate(ModelRequest(prompt="q"))
        harness.client_cls.assert_not_called()
        harness.generate.assert_not_called()

    def test_api_key_is_passed_from_environment(self) -> None:
        harness = _SDKHarness(self)
        RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))
        harness.client_cls.assert_called_once_with(api_key="test-key")

    def test_client_is_built_once_and_reused(self) -> None:
        harness = _SDKHarness(self)
        model = RealGeminiVisionLanguageModel()
        model.generate(ModelRequest(prompt="q"))
        model.generate(ModelRequest(prompt="q"))
        harness.client_cls.assert_called_once()
        self.assertEqual(harness.generate.call_count, 2)


class GeminiProviderRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _SDKHarness(self)
        self.model = RealGeminiVisionLanguageModel()

    def test_model_id_is_sent_unchanged(self) -> None:
        self.model.generate(ModelRequest(prompt="q"))
        self.assertEqual(self.harness.kwargs["model"], "gemini-3.6-flash")

    def test_contents_carry_image_part_then_prompt(self) -> None:
        self.model.generate(
            ModelRequest(
                prompt="who is this player?",
                images=[Image(data=_IMAGE_BYTES, mime_type="image/png")],
                temperature=0.0,
                max_tokens=200,
            )
        )
        contents = self.harness.contents
        self.assertIsInstance(contents[0], types.Part)
        self.assertEqual(contents[0].inline_data.data, _IMAGE_BYTES)
        self.assertEqual(contents[0].inline_data.mime_type, "image/png")
        self.assertEqual(contents[1], "who is this player?")

    def test_config_is_deterministic_structured_json(self) -> None:
        self.model.generate(ModelRequest(prompt="q", max_tokens=200))
        config = self.harness.config
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_output_tokens, 200)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(
            config.response_schema,
            types.GenerateContentConfig(
                response_schema=MRAG_PLAN_RESPONSE_SCHEMA
            ).response_schema,
        )

    def test_response_schema_matches_mrag_plan_shape(self) -> None:
        step = MRAG_PLAN_RESPONSE_SCHEMA["properties"]["steps"]["items"]
        self.assertEqual(
            step["properties"]["tool"]["enum"],
            ["image_search", "text_search", "requery", "response"],
        )
        self.assertEqual(MRAG_PLAN_RESPONSE_SCHEMA["required"], ["steps"])
        self.assertEqual(step["required"], ["tool"])
        self.assertIn("query", step["properties"]["arguments"]["properties"])

    def test_max_output_tokens_override_wins(self) -> None:
        RealGeminiVisionLanguageModel(max_output_tokens=1024).generate(
            ModelRequest(prompt="q", max_tokens=200)
        )
        self.assertEqual(self.harness.config.max_output_tokens, 1024)


class GeminiProviderImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _SDKHarness(self)
        self.model = RealGeminiVisionLanguageModel()

    def test_image_path_is_read_from_disk(self) -> None:
        import tempfile

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        fixture = Path(temp_dir.name) / "frame.png"
        fixture.write_bytes(_IMAGE_BYTES)
        self.model.generate(
            ModelRequest(prompt="q", images=[Image(path=str(fixture))])
        )
        inline = self.harness.contents[0].inline_data
        self.assertEqual(inline.data, _IMAGE_BYTES)
        self.assertEqual(inline.mime_type, "image/png")

    def test_image_url_is_fetched(self) -> None:
        fetched = MagicMock()
        fetched.__enter__ = lambda self_: self_
        fetched.__exit__ = lambda *_a: False
        fetched.read.return_value = _IMAGE_BYTES
        with patch(
            "eagent_functional.gemini_provider.urlopen", return_value=fetched
        ) as opener:
            self.model.generate(
                ModelRequest(
                    prompt="q", images=[Image(url="http://example.invalid/a.png")]
                )
            )
        self.assertEqual(opener.call_args.args[0], "http://example.invalid/a.png")
        self.assertEqual(self.harness.contents[0].inline_data.data, _IMAGE_BYTES)

    def test_image_without_content_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.generate(ModelRequest(prompt="q", images=[Image()]))


class GeminiProviderResponseTests(unittest.TestCase):
    def test_plan_text_and_usage_are_returned(self) -> None:
        harness = _SDKHarness(self)
        response = RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))
        self.assertEqual(response.text, _PLAN_JSON)
        self.assertEqual(response.model_name, "gemini-3.6-flash")
        self.assertEqual(response.usage["prompt_tokens"], 11)
        self.assertEqual(response.usage["completion_tokens"], 22)
        self.assertEqual(response.usage["total_tokens"], 33)
        self.assertEqual(harness.generate.call_count, 1)
        MRAGPlan.from_json(response.text).validate()

    def test_sdk_text_excludes_thought_parts(self) -> None:
        _SDKHarness(self, _fake_response(thought_text="planning..."))
        response = RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))
        self.assertEqual(response.text, _PLAN_JSON)
        self.assertNotIn("planning", response.text)

    def test_empty_text_raises_with_finish_reason(self) -> None:
        _SDKHarness(self, _fake_response(text=None, finish_reason="MAX_TOKENS"))
        with self.assertRaises(GeminiAPIError) as caught:
            RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))
        self.assertIn("MAX_TOKENS", str(caught.exception))

    def test_blank_text_raises(self) -> None:
        _SDKHarness(self, _fake_response(text="   "))
        with self.assertRaises(GeminiAPIError):
            RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))

    def test_sdk_error_propagates_once_without_retry(self) -> None:
        harness = _SDKHarness(self)
        harness.generate.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        with self.assertRaises(RuntimeError) as caught:
            RealGeminiVisionLanguageModel().generate(ModelRequest(prompt="q"))
        self.assertIn("RESOURCE_EXHAUSTED", str(caught.exception))
        self.assertEqual(harness.generate.call_count, 1)


class GeminiPipelineMockedTests(unittest.TestCase):
    def test_pipeline_runs_single_pass_with_gemini_planner(self) -> None:
        harness = _SDKHarness(self)
        question = Question(
            text="who is this player?",
            images=[Image(data=_IMAGE_BYTES, mime_type="image/png")],
        )
        result = run_functional_pipeline(
            RealGeminiVisionLanguageModel(),
            build_executor_model(BaselineConfig.from_yaml(default_config_path())),
            question,
        )
        self.assertTrue(result.parse_ok, msg=result.parse_error)
        self.assertEqual(result.classification, "valid_json_plan")
        self.assertEqual(result.plan_call_count, 1)
        self.assertEqual(result.generate_call_count, 1)
        self.assertEqual(harness.generate.call_count, 1)
        self.assertEqual(result.model_name, "gemini-3.6-flash")
        self.assertEqual(result.tool_sequence, ["response"])
        self.assertFalse(result.stub_fallback_occurred)
        self.assertTrue(result.terminal_response)


@unittest.skipUnless(
    os.environ.get("EAGENT_FUNCTIONAL_GEMINI") == "1"
    and bool(os.environ.get("GEMINI_API_KEY")),
    "real Gemini disabled; set EAGENT_FUNCTIONAL_GEMINI=1 and GEMINI_API_KEY",
)
class RealGeminiEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PIL import Image as PILImage

        fixture_dir = _EXT / ".venv" / "hf_cache"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture = fixture_dir / "gemini_planner_fixture.png"
        canvas = PILImage.new("RGB", (128, 128), (30, 60, 160))
        for x in range(32, 96):
            for y in range(32, 96):
                canvas.putpixel((x, y), (200, 40, 40))
        canvas.save(fixture, format="PNG")

        cls.question = Question(
            text="What is shown in this image?",
            images=[Image(path=str(fixture))],
        )
        cls.result = run_functional_pipeline(
            RealGeminiVisionLanguageModel(),
            build_executor_model(BaselineConfig.from_yaml(default_config_path())),
            cls.question,
        )

    def test_real_planner_call_mechanics(self) -> None:
        result = self.result
        self.assertTrue(self.question.is_multimodal)
        self.assertEqual(result.generate_call_count, 1)
        self.assertEqual(result.plan_call_count, 1)
        self.assertEqual(result.model_name, "gemini-3.6-flash")
        self.assertTrue(
            result.raw_output and result.raw_output.strip(),
            msg=f"empty raw output: {result.raw_output!r}",
        )
        self.assertFalse(result.stub_fallback_occurred)

    def test_real_planner_produces_executable_plan(self) -> None:
        result = self.result
        self.assertTrue(
            result.parse_ok,
            msg=(
                "Gemini did not produce a parseable MRAGPlan; "
                f"classification={result.classification} "
                f"parse_error={result.parse_error} raw={result.raw_output!r}"
            ),
        )
        self.assertIsInstance(result.plan, MRAGPlan)
        result.plan.validate()
        self.assertIsNone(result.executor_error)
        self.assertTrue(result.terminal_response and result.terminal_response.strip())


if __name__ == "__main__":
    unittest.main()






