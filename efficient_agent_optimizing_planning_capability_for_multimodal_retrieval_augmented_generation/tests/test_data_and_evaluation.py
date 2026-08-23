import unittest

import tests._paths

from eagent_baseline.data import DataValidationError, QuestionType, RemPlanInstance
from eagent_baseline.evaluation import (
    AnswerJudge,
    StubAnswerJudge,
    answer_score,
    exact_tool_sequence_match,
    image_search_precision_recall,
    param_accuracy,
    param_semantic_similarity,
    plan_accuracy,
    text_search_precision_recall,
)
from eagent_baseline.plan import MRAGPlan, PlanStep, ToolName


def _plan(*tools: ToolName) -> MRAGPlan:
    return MRAGPlan(steps=[PlanStep(tool) for tool in tools])


class DataSchemaTests(unittest.TestCase):
    def test_valid_instance(self) -> None:
        instance = RemPlanInstance(
            image_ref="img://1",
            question="What is this?",
            question_type=QuestionType.VISUAL_RECOGNITION,
            gold_plan=_plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE),
            gold_answer="a cat",
        )
        instance.validate()

    def test_missing_question_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            RemPlanInstance(image_ref="img://1", question="").validate()

    def test_invalid_gold_plan_rejected(self) -> None:
        with self.assertRaises(Exception):
            RemPlanInstance(
                image_ref="img://1",
                question="q",
                gold_plan=_plan(ToolName.IMAGE_SEARCH),
            ).validate()

    def test_all_question_types_accepted(self) -> None:
        for qtype in (QuestionType.FUNDAMENTAL, QuestionType.VISUAL_RECOGNITION,
                      QuestionType.INFORMATION_SEEKING, QuestionType.MULTI_FACETED):
            instance = RemPlanInstance(
                image_ref="img://1", question="q", question_type=qtype,
                gold_plan=_plan(ToolName.RESPONSE),
            )
            instance.validate()

    def test_invalid_question_type_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            instance = RemPlanInstance(
                image_ref="img://1", question="q", question_type="not-a-type",
                gold_plan=_plan(ToolName.RESPONSE),
            )
            instance.validate()

    def test_instance_without_question_type_is_valid(self) -> None:
        instance = RemPlanInstance(
            image_ref="img://1", question="q", gold_plan=_plan(ToolName.RESPONSE),
        )
        instance.validate()

    def test_instance_rejects_empty_image_ref(self) -> None:
        with self.assertRaises(DataValidationError):
            RemPlanInstance(image_ref="", question="q").validate()


class MetricInterfaceTests(unittest.TestCase):
    def test_tool_precision_recall(self) -> None:
        predicted = _plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)
        gold = _plan(ToolName.IMAGE_SEARCH, ToolName.TEXT_SEARCH, ToolName.RESPONSE)
        image_pr = image_search_precision_recall(predicted, gold)
        self.assertEqual((image_pr.precision, image_pr.recall), (1.0, 1.0))
        text_pr = text_search_precision_recall(predicted, gold)
        self.assertEqual((text_pr.precision, text_pr.recall), (0.0, 0.0))

    def test_precision_empty_predicted_returns_zero(self) -> None:
        predicted = _plan(ToolName.RESPONSE)
        gold = _plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)
        pr = image_search_precision_recall(predicted, gold)
        self.assertEqual(pr.precision, 0.0)
        self.assertEqual(pr.recall, 0.0)

    def test_recall_empty_gold_returns_zero(self) -> None:
        predicted = _plan(ToolName.IMAGE_SEARCH, ToolName.TEXT_SEARCH, ToolName.RESPONSE)
        gold = _plan(ToolName.RESPONSE)
        pr = image_search_precision_recall(predicted, gold)
        self.assertEqual(pr.recall, 0.0)

    def test_plan_accuracy_exact_match(self) -> None:
        p = [_plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)]
        g = [_plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)]
        self.assertEqual(plan_accuracy(p, g, exact_tool_sequence_match), 1.0)

    def test_plan_accuracy_order_matters(self) -> None:
        p = [_plan(ToolName.TEXT_SEARCH, ToolName.IMAGE_SEARCH, ToolName.RESPONSE)]
        g = [_plan(ToolName.IMAGE_SEARCH, ToolName.TEXT_SEARCH, ToolName.RESPONSE)]
        self.assertEqual(plan_accuracy(p, g, exact_tool_sequence_match), 0.0)

    def test_plan_accuracy_mismatched_lengths_raises(self) -> None:
        with self.assertRaises(ValueError):
            plan_accuracy([_plan(ToolName.RESPONSE)], [_plan(ToolName.RESPONSE), _plan(ToolName.RESPONSE)])

    def test_plan_accuracy_empty_lists_returns_zero(self) -> None:
        self.assertEqual(plan_accuracy([], [], exact_tool_sequence_match), 0.0)

    def test_param_accuracy_requires_validator(self) -> None:
        with self.assertRaises(NotImplementedError):
            param_accuracy([_plan(ToolName.RESPONSE)], [_plan(ToolName.RESPONSE)])

    def test_param_semantic_similarity_requires_model(self) -> None:
        with self.assertRaises(NotImplementedError):
            param_semantic_similarity(["a"], ["b"])

    def test_param_semantic_similarity_with_injected_fn(self) -> None:
        value = param_semantic_similarity(["a", "b"], ["a", "c"], lambda x, y: 1.0 if x == y else 0.0)
        self.assertEqual(value, 0.5)

    def test_answer_judge_scale(self) -> None:
        judge = StubAnswerJudge()
        self.assertEqual(answer_score(judge, "img", "q", "gold", "the gold answer"), 2)
        self.assertEqual(answer_score(judge, "img", "q", "gold", "unrelated"), 0)

    def test_answer_score_validates_range(self) -> None:
        class BadJudge(AnswerJudge):
            def score(self, image_ref: str, question: str, gold_answer: str, response: str) -> int:
                return 3
        with self.assertRaises(ValueError):
            answer_score(BadJudge(), "img", "q", "gold", "ok")

    def test_answer_score_returns_in_range(self) -> None:
        judge = StubAnswerJudge()
        self.assertIn(answer_score(judge, "img", "q", "gold", "the gold answer"), {0, 1, 2})
        self.assertIn(answer_score(judge, "img", "q", "gold", "unrelated"), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
