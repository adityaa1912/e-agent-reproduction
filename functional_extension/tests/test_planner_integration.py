from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

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

from eagent.common.types import Image, Question
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel

from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.models import build_executor_model
from eagent_baseline.plan import MRAGPlan

from eagent_functional.planner_integration import (
    FUNCTIONAL_VALIDATION_PLANNER_PROMPT,
    FunctionalValidationPlanner,
    classify_raw_output,
    detect_stub_fallback,
    run_functional_pipeline,
)

_VALID_RESPONSE_ONLY = '{"steps": [{"tool": "response", "arguments": {}}]}'
_VALID_FULL = (
    '{"steps": ['
    '{"tool": "image_search", "arguments": {}},'
    '{"tool": "requery", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}},'
    '{"tool": "response", "arguments": {}}]}'
)
_FENCED = "```json\n" + _VALID_RESPONSE_ONLY + "\n```"
_NATURAL = "The image shows a red square on a blue background."
_WRONG_SHAPE = '{"foo": 1}'


class _FakePlanModel(VisionLanguageModel):
    def __init__(self, text: str, model_name: str = "fake-planner") -> None:
        self._text = text
        self._model_name = model_name
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            text=self._text, model_name=self._model_name, usage=None, raw=None
        )


def _question() -> Question:
    return Question(
        text="who is this player?",
        images=[Image(url="http://example.invalid/frame.png")],
    )


def _executor_model() -> VisionLanguageModel:
    return build_executor_model(BaselineConfig.from_yaml(default_config_path()))


class FunctionalPlannerPromptTests(unittest.TestCase):
    def test_build_request_uses_functional_prompt_and_inputs(self) -> None:
        planner = FunctionalValidationPlanner(_FakePlanModel(_VALID_RESPONSE_ONLY))
        question = _question()
        request = planner.build_request(question)
        self.assertIn("who is this player?", request.prompt)
        self.assertIn("image_search", request.prompt)
        self.assertIn("text_search", request.prompt)
        self.assertIn("requery", request.prompt)
        self.assertIn("response", request.prompt)
        self.assertNotIn("```", request.prompt)
        self.assertEqual(request.temperature, 0.0)
        self.assertIsNotNone(request.max_tokens)
        self.assertEqual(len(request.images), 1)

    def test_prompt_is_not_the_core_assumed_template(self) -> None:
        from eagent_baseline.planner import ASSUMED_PLANNER_PROMPT_TEMPLATE

        self.assertNotEqual(
            FUNCTIONAL_VALIDATION_PLANNER_PROMPT, ASSUMED_PLANNER_PROMPT_TEMPLATE
        )


class FunctionalPipelineMockedTests(unittest.TestCase):
    def test_valid_response_only_plan_runs_end_to_end(self) -> None:
        model = _FakePlanModel(_VALID_RESPONSE_ONLY)
        result = run_functional_pipeline(model, _executor_model(), _question())
        self.assertTrue(result.parse_ok)
        self.assertEqual(result.classification, "valid_json_plan")
        self.assertEqual(result.plan_call_count, 1)
        self.assertEqual(result.generate_call_count, 1)
        self.assertEqual(model.calls, 1)
        self.assertIsInstance(result.plan, MRAGPlan)
        self.assertEqual(result.tool_sequence, ["response"])
        self.assertIsNone(result.executor_error)
        self.assertEqual(result.executor_tool_trace, ["response"])
        self.assertTrue(result.terminal_response)
        self.assertFalse(result.stub_fallback_occurred)
        self.assertEqual(result.model_name, "fake-planner")

    def test_valid_full_plan_executes(self) -> None:
        result = run_functional_pipeline(
            _FakePlanModel(_VALID_FULL), _executor_model(), _question()
        )
        self.assertTrue(result.parse_ok)
        self.assertEqual(
            result.tool_sequence,
            ["image_search", "requery", "text_search", "response"],
        )
        self.assertEqual(
            result.executor_tool_trace,
            ["image_search", "requery", "text_search", "response"],
        )
        self.assertTrue(result.terminal_response)
        self.assertEqual(result.plan_call_count, 1)

    def test_single_planner_call_only(self) -> None:
        model = _FakePlanModel(_VALID_FULL)
        result = run_functional_pipeline(model, _executor_model(), _question())
        self.assertEqual(model.calls, 1)
        self.assertEqual(result.generate_call_count, 1)
        self.assertEqual(result.plan_call_count, 1)

    def test_fenced_output_raises_and_preserves_raw(self) -> None:
        result = run_functional_pipeline(
            _FakePlanModel(_FENCED), _executor_model(), _question()
        )
        self.assertFalse(result.parse_ok)
        self.assertEqual(result.classification, "markdown_fenced")
        self.assertIsNotNone(result.parse_error)
        self.assertEqual(result.raw_output, _FENCED)
        self.assertIn("```", result.raw_output)
        self.assertIsNone(result.plan)
        self.assertIsNone(result.terminal_response)
        self.assertEqual(result.plan_call_count, 1)

    def test_natural_language_output_raises(self) -> None:
        result = run_functional_pipeline(
            _FakePlanModel(_NATURAL), _executor_model(), _question()
        )
        self.assertFalse(result.parse_ok)
        self.assertEqual(result.classification, "non_json_natural_language_or_invalid")
        self.assertEqual(result.raw_output, _NATURAL)
        self.assertIsNone(result.plan)
        self.assertIsNone(result.terminal_response)

    def test_wrong_shape_json_raises(self) -> None:
        result = run_functional_pipeline(
            _FakePlanModel(_WRONG_SHAPE), _executor_model(), _question()
        )
        self.assertFalse(result.parse_ok)
        self.assertEqual(result.classification, "valid_json_wrong_shape")
        self.assertIsNone(result.plan)

    def test_classify_helper(self) -> None:
        self.assertEqual(classify_raw_output(None), "no_output")
        self.assertEqual(classify_raw_output("   "), "empty")
        self.assertEqual(classify_raw_output("```json\n{}\n```"), "markdown_fenced")
        self.assertEqual(classify_raw_output(_VALID_RESPONSE_ONLY), "valid_json_plan")
        self.assertEqual(
            classify_raw_output("hello there"),
            "non_json_natural_language_or_invalid",
        )
        self.assertEqual(classify_raw_output('{"a": 1}'), "valid_json_wrong_shape")

    def test_stub_fallback_detector(self) -> None:
        self.assertFalse(detect_stub_fallback(_VALID_RESPONSE_ONLY, "fake-planner"))
        self.assertTrue(
            detect_stub_fallback("[stub:dev] prompt=...", "HuggingFaceTB/SmolVLM-256M-Instruct")
        )
        self.assertTrue(detect_stub_fallback("{}", "stub-executor"))


@unittest.skipUnless(
    os.environ.get("EAGENT_FUNCTIONAL_REAL") == "1",
    "real SmolVLM disabled; set EAGENT_FUNCTIONAL_REAL=1 to enable",
)
class RealSmolVlmEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        hf_cache = _EXT / ".venv" / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_cache)
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        from PIL import Image as PILImage

        from eagent_functional.transformers_provider import (
            RealTransformersVisionLanguageModel,
        )

        fixture = hf_cache / "planner_integration_fixture.png"
        canvas = PILImage.new("RGB", (128, 128), (30, 60, 160))
        for x in range(32, 96):
            for y in range(32, 96):
                canvas.putpixel((x, y), (200, 40, 40))
        canvas.save(fixture, format="PNG")

        cls.question = Question(
            text="What is shown in this image?",
            images=[Image(path=str(fixture))],
        )
        planner_model = RealTransformersVisionLanguageModel()
        executor_model = build_executor_model(
            BaselineConfig.from_yaml(default_config_path())
        )
        cls.result = run_functional_pipeline(
            planner_model, executor_model, cls.question
        )

    def test_real_planner_call_mechanics(self) -> None:
        result = self.result
        self.assertTrue(self.question.is_multimodal)
        self.assertEqual(result.generate_call_count, 1)
        self.assertEqual(result.plan_call_count, 1)
        self.assertEqual(result.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")
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
                "SmolVLM-256M did not produce a parseable MRAGPlan; "
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
