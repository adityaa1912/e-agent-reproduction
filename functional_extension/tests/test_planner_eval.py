"""Offline tests for the functional planner evaluation harness.

These tests use mocked planner responses. They make no network call and are not
part of the paper-skeleton suites.
"""

from __future__ import annotations

import json
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

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel

from eagent_baseline.data import QuestionType

from eagent_functional.planner_eval import (
    EvalCase,
    evaluate_case,
    evaluate_cases,
    report_to_json,
    summarize,
)

_RESPONSE_ONLY = '{"steps": [{"tool": "response", "arguments": {}}]}'
_FULL_PLAN = (
    '{"steps": ['
    '{"tool": "image_search", "arguments": {}},'
    '{"tool": "requery", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}},'
    '{"tool": "response", "arguments": {}}]}'
)
_RESPONSE_FIRST = (
    '{"steps": ['
    '{"tool": "response", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}}]}'
)
_NATURAL = "The image shows a red square on a blue background."

class _ScriptedModel(VisionLanguageModel):
    def __init__(self, texts, model_name: str = "mock-planner") -> None:
        self._texts = list(texts)
        self._model_name = model_name
        self.calls = 0
        self.seen_images = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        self.seen_images.append(len(request.images))
        return ModelResponse(text=text, model_name=self._model_name)


def _case(
    case_id: str = "c1",
    question_type: str = "visual_recognition",
    text: str = "who is this player?",
    with_image: bool = True,
) -> EvalCase:
    images = [Image(url="http://example.invalid/frame.png")] if with_image else []
    return EvalCase.build(case_id, question_type, text, images)


class EvalCaseTests(unittest.TestCase):
    def test_build_accepts_question_type_enum(self) -> None:
        case = EvalCase.build("c9", QuestionType.FUNDAMENTAL, "what colour?", [])
        self.assertEqual(case.question_type, "fundamental")
        self.assertEqual(case.question.text, "what colour?")

    def test_build_carries_images_into_question(self) -> None:
        case = _case()
        self.assertEqual(len(case.question.images), 1)
        self.assertTrue(case.question.is_multimodal)


class EvaluateCaseTests(unittest.TestCase):
    def test_valid_response_only_plan_is_recorded(self) -> None:
        model = _ScriptedModel([_RESPONSE_ONLY])
        record = evaluate_case(_case(), model)
        self.assertTrue(record.plan_valid)
        self.assertEqual(record.case_id, "c1")
        self.assertEqual(record.question_type, "visual_recognition")
        self.assertEqual(record.model_name, "mock-planner")
        self.assertEqual(record.raw_output, _RESPONSE_ONLY)
        self.assertEqual(record.tool_sequence, ["response"])
        self.assertEqual(record.plan_length, 1)
        self.assertIsNone(record.failure_reason)
        self.assertEqual(record.classification, "valid_json_plan")
        self.assertFalse(record.stub_fallback_occurred)
        self.assertEqual(record.image_count, 1)

    def test_valid_multi_step_plan_is_recorded(self) -> None:
        record = evaluate_case(_case(), _ScriptedModel([_FULL_PLAN]))
        self.assertTrue(record.plan_valid)
        self.assertEqual(
            record.tool_sequence,
            ["image_search", "requery", "text_search", "response"],
        )
        self.assertEqual(record.plan_length, 4)

    def test_single_planner_call_per_case(self) -> None:
        model = _ScriptedModel([_RESPONSE_ONLY])
        record = evaluate_case(_case(), model)
        self.assertEqual(model.calls, 1)
        self.assertEqual(record.plan_call_count, 1)

    def test_image_reaches_the_planner_request(self) -> None:
        model = _ScriptedModel([_RESPONSE_ONLY])
        evaluate_case(_case(), model)
        self.assertEqual(model.seen_images, [1])

    def test_latency_is_recorded(self) -> None:
        record = evaluate_case(_case(), _ScriptedModel([_RESPONSE_ONLY]))
        self.assertGreater(record.planner_latency_seconds, 0.0)
        self.assertLess(record.planner_latency_seconds, 60.0)


class InvalidPlanTests(unittest.TestCase):
    def test_non_terminal_response_is_invalid_with_reason(self) -> None:
        record = evaluate_case(_case(), _ScriptedModel([_RESPONSE_FIRST]))
        self.assertFalse(record.plan_valid)
        self.assertIsNone(record.plan)
        self.assertIsNone(record.tool_sequence)
        self.assertIsNone(record.plan_length)
        self.assertIn("PlanValidationError", record.failure_reason)
        self.assertIn("terminate", record.failure_reason)
        self.assertEqual(record.raw_output, _RESPONSE_FIRST)
        self.assertEqual(record.plan_call_count, 1)

    def test_natural_language_output_is_invalid(self) -> None:
        record = evaluate_case(_case(), _ScriptedModel([_NATURAL]))
        self.assertFalse(record.plan_valid)
        self.assertEqual(
            record.classification, "non_json_natural_language_or_invalid"
        )
        self.assertIn("PlanValidationError", record.failure_reason)

    def test_stub_planner_is_flagged(self) -> None:
        record = evaluate_case(
            _case(), _ScriptedModel([_RESPONSE_ONLY], model_name="stub-planner")
        )
        self.assertTrue(record.stub_fallback_occurred)


class ModelAgnosticTests(unittest.TestCase):
    def test_same_harness_records_two_different_planners(self) -> None:
        cases = [_case("c1"), _case("c2")]
        first = evaluate_cases(cases, _ScriptedModel([_RESPONSE_ONLY], "planner-a"))
        second = evaluate_cases(cases, _ScriptedModel([_FULL_PLAN], "planner-b"))
        self.assertEqual({r.model_name for r in first}, {"planner-a"})
        self.assertEqual({r.model_name for r in second}, {"planner-b"})
        self.assertEqual(summarize(first).model_name, "planner-a")
        self.assertEqual(summarize(second).model_name, "planner-b")
        self.assertEqual(summarize(first).plan_lengths, [1, 1])
        self.assertEqual(summarize(second).plan_lengths, [4, 4])

    def test_evaluate_cases_returns_one_record_per_case(self) -> None:
        model = _ScriptedModel([_RESPONSE_ONLY, _NATURAL, _FULL_PLAN])
        records = evaluate_cases([_case("c1"), _case("c2"), _case("c3")], model)
        self.assertEqual([r.case_id for r in records], ["c1", "c2", "c3"])
        self.assertEqual([r.plan_valid for r in records], [True, False, True])
        self.assertEqual(model.calls, 3)


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = evaluate_cases(
            [
                _case("c1", "visual_recognition"),
                _case("c2", "visual_recognition"),
                _case("c3", "information_seeking"),
            ],
            _ScriptedModel([_RESPONSE_ONLY, _RESPONSE_FIRST, _FULL_PLAN]),
        )
        self.summary = summarize(self.records)

    def test_counts_and_validity_rate(self) -> None:
        self.assertEqual(self.summary.total, 3)
        self.assertEqual(self.summary.valid, 2)
        self.assertEqual(self.summary.invalid, 1)
        self.assertAlmostEqual(self.summary.validity_rate, 2 / 3)

    def test_tool_sequence_counts_cover_valid_records_only(self) -> None:
        self.assertEqual(
            self.summary.tool_sequence_counts,
            {
                "response": 1,
                "image_search -> requery -> text_search -> response": 1,
            },
        )

    def test_failure_reasons_are_aggregated(self) -> None:
        self.assertEqual(sum(self.summary.failure_reasons.values()), 1)
        self.assertTrue(
            all("PlanValidationError" in key for key in self.summary.failure_reasons)
        )

    def test_validity_by_question_type(self) -> None:
        self.assertEqual(
            self.summary.validity_by_question_type,
            {"visual_recognition": "1/2", "information_seeking": "1/1"},
        )

    def test_plan_lengths_and_mean_latency(self) -> None:
        self.assertEqual(self.summary.plan_lengths, [1, 4])
        self.assertGreater(self.summary.mean_latency_seconds, 0.0)

    def test_empty_records_summarize_without_error(self) -> None:
        empty = summarize([])
        self.assertEqual(empty.total, 0)
        self.assertEqual(empty.validity_rate, 0.0)
        self.assertEqual(empty.mean_latency_seconds, 0.0)
        self.assertEqual(empty.model_name, "mixed")


class ReportTests(unittest.TestCase):
    def test_report_is_valid_json_with_summary_and_records(self) -> None:
        records = evaluate_cases(
            [_case("c1"), _case("c2")], _ScriptedModel([_RESPONSE_ONLY, _NATURAL])
        )
        payload = json.loads(report_to_json(records))
        self.assertEqual(payload["summary"]["total"], 2)
        self.assertEqual([r["case_id"] for r in payload["records"]], ["c1", "c2"])
        self.assertEqual(
            payload["records"][0]["plan"], {"steps": [{"tool": "response", "arguments": {}}]}
        )
        self.assertIsNone(payload["records"][1]["plan"])
        self.assertEqual(payload["records"][1]["raw_output"], _NATURAL)


class GeminiPlannerHarnessTests(unittest.TestCase):
    def test_gemini_planner_runs_through_the_harness_without_network(self) -> None:
        from eagent_functional.gemini_provider import RealGeminiVisionLanguageModel

        client_cls = patch("google.genai.Client").start()
        self.addCleanup(patch.stopall)
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        response = MagicMock()
        response.text = _RESPONSE_ONLY
        response.candidates = []
        response.usage_metadata = None
        generate = client_cls.return_value.models.generate_content
        generate.return_value = response

        record = evaluate_case(
            EvalCase.build(
                "g1",
                "visual_recognition",
                "who is this player?",
                [Image(data=b"\x89PNG-fake-bytes", mime_type="image/png")],
            ),
            RealGeminiVisionLanguageModel(),
        )

        self.assertTrue(record.plan_valid)
        self.assertEqual(record.model_name, "gemini-3.6-flash")
        self.assertEqual(record.tool_sequence, ["response"])
        self.assertEqual(record.plan_call_count, 1)
        self.assertEqual(generate.call_count, 1)
        self.assertFalse(record.stub_fallback_occurred)


if __name__ == "__main__":
    unittest.main()




