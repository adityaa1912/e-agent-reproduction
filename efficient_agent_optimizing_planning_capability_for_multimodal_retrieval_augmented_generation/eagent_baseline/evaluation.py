from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

from eagent_baseline.plan import MRAGPlan, ToolName


@dataclass
class PrecisionRecall:
    precision: float
    recall: float


def _precision_recall(predicted: Set[ToolName], gold: Set[ToolName]) -> PrecisionRecall:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return PrecisionRecall(precision=precision, recall=recall)


def tool_precision_recall(
    predicted_tools: Set[ToolName],
    gold_tools: Set[ToolName],
    target: ToolName,
) -> PrecisionRecall:
    predicted = {target} & predicted_tools
    gold = {target} & gold_tools
    return _precision_recall(predicted, gold)


def image_search_precision_recall(
    predicted: MRAGPlan, gold: MRAGPlan
) -> PrecisionRecall:
    return tool_precision_recall(
        set(predicted.tool_sequence()), set(gold.tool_sequence()), ToolName.IMAGE_SEARCH
    )


def text_search_precision_recall(predicted: MRAGPlan, gold: MRAGPlan) -> PrecisionRecall:
    return tool_precision_recall(
        set(predicted.tool_sequence()), set(gold.tool_sequence()), ToolName.TEXT_SEARCH
    )


PlanComparator = Callable[[MRAGPlan, MRAGPlan], bool]


def exact_tool_sequence_match(predicted: MRAGPlan, gold: MRAGPlan) -> bool:
    return predicted.tool_sequence() == gold.tool_sequence()


def plan_accuracy(
    predicted_plans: List[MRAGPlan],
    gold_plans: List[MRAGPlan],
    comparator: PlanComparator = exact_tool_sequence_match,
) -> float:
    if len(predicted_plans) != len(gold_plans):
        raise ValueError("predicted_plans and gold_plans must have equal length.")
    if not gold_plans:
        return 0.0
    correct = sum(1 for p, g in zip(predicted_plans, gold_plans) if comparator(p, g))
    return correct / len(gold_plans)


ParamValidator = Callable[[MRAGPlan, MRAGPlan], bool]


def param_accuracy(
    predicted_plans: List[MRAGPlan],
    gold_plans: List[MRAGPlan],
    validator: Optional[ParamValidator] = None,
) -> float:
    if validator is None:
        raise NotImplementedError(
            "Param-acc requires a parameter-validity criterion. The paper does not "
            "define one; inject a validator to compute this metric."
        )
    if len(predicted_plans) != len(gold_plans):
        raise ValueError("predicted_plans and gold_plans must have equal length.")
    if not gold_plans:
        return 0.0
    valid = sum(1 for p, g in zip(predicted_plans, gold_plans) if validator(p, g))
    return valid / len(gold_plans)


SimilarityFn = Callable[[str, str], float]


def param_semantic_similarity(
    predicted_params: List[str],
    gold_params: List[str],
    similarity_fn: Optional[SimilarityFn] = None,
) -> float:
    if similarity_fn is None:
        raise NotImplementedError(
            "Param-sim requires a semantic similarity model. The paper does not "
            "specify one; inject a similarity_fn to compute this metric."
        )
    if len(predicted_params) != len(gold_params):
        raise ValueError("predicted_params and gold_params must have equal length.")
    if not gold_params:
        return 0.0
    total = sum(similarity_fn(p, g) for p, g in zip(predicted_params, gold_params))
    return total / len(gold_params)


class AnswerJudge(ABC):
    @abstractmethod
    def score(self, image_ref: str, question: str, gold_answer: str, response: str) -> int:
        raise NotImplementedError


class StubAnswerJudge(AnswerJudge):
    """Deterministic development judge returning an exact-match-based 0/2 score.

    Not the paper's GPT-4o judge and not its judging prompt. Provided only so
    the evaluation interface is runnable offline in tests.
    """

    def score(self, image_ref: str, question: str, gold_answer: str, response: str) -> int:
        return 2 if gold_answer.strip() and gold_answer.strip() in response else 0


def answer_score(
    judge: AnswerJudge,
    image_ref: str,
    question: str,
    gold_answer: str,
    response: str,
) -> int:
    value = judge.score(image_ref, question, gold_answer, response)
    if value not in (0, 1, 2):
        raise ValueError("Answer score must be in {0, 1, 2}.")
    return value
