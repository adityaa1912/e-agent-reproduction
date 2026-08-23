import unittest

import tests._paths

from eagent_baseline.data import DataValidationError, QuestionType, RemPlanInstance
from eagent_baseline.evaluation import (
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


class MetricInterfaceTests(unittest.TestCase):
    def test_tool_precision_recall(self) -> None:
        predicted = _plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)
        gold = _plan(ToolName.IMAGE_SEARCH, ToolName.TEXT_SEARCH, ToolName.RESPONSE)
        image_pr = image_search_precision_recall(predicted, gold)
        self.assertEqual((image_pr.precision, image_pr.recall), (1.0, 1.0))
        text_pr = text_search_precision_recall(predicted, gold)
        self.assertEqual((text_pr.precision, text_pr.recall), (0.0, 0.0))

    def test_plan_accuracy_exact_match(self) -> None:
        p = [_plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)]
        g = [_plan(ToolName.IMAGE_SEARCH, ToolName.RESPONSE)]
        self.assertEqual(plan_accuracy(p, g, exact_tool_sequence_match), 1.0)

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


if __name__ == "__main__":
    unittest.main()
