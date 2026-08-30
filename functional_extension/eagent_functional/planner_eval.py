"""Planner evaluation harness used only for functional validation.

This module is not part of the paper skeleton. It lives outside ``src/eagent``
and ``eagent_baseline``. It measures planner behaviour only: any
``VisionLanguageModel`` can be evaluated, no executor or retrieval provider is
involved, and no metric from the paper is computed. Plan validity is decided by
the unmodified ``MRAGPlan`` parser and validator, with no retry and no repair.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from eagent.common.types import Question
from eagent.models.protocols import VisionLanguageModel

from eagent_baseline.plan import MRAGPlan, PlanValidationError

from eagent_functional.planner_integration import (
    FunctionalValidationPlanner,
    RawCapturingModel,
    classify_raw_output,
    detect_stub_fallback,
)


def _question_type_value(question_type: Any) -> str:
    return str(getattr(question_type, "value", question_type))


@dataclass
class EvalCase:
    """One evaluation input: an identifier, a taxonomy label, and a question."""

    case_id: str
    question_type: str
    question: Question

    @classmethod
    def build(
        cls,
        case_id: str,
        question_type: Any,
        text: str,
        images: Optional[Sequence[Any]] = None,
    ) -> "EvalCase":
        return cls(
            case_id=case_id,
            question_type=_question_type_value(question_type),
            question=Question(text=text, images=list(images or [])),
        )


@dataclass
class EvalRecord:
    """The recorded outcome of evaluating one case against one planner."""

    case_id: str
    question_type: str
    model_name: str
    raw_output: Optional[str]
    plan_valid: bool
    plan: Optional[MRAGPlan]
    tool_sequence: Optional[List[str]]
    plan_length: Optional[int]
    planner_latency_seconds: float
    failure_reason: Optional[str]
    classification: str
    stub_fallback_occurred: bool
    plan_call_count: int
    image_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question_type": self.question_type,
            "model_name": self.model_name,
            "raw_output": self.raw_output,
            "plan_valid": self.plan_valid,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "tool_sequence": self.tool_sequence,
            "plan_length": self.plan_length,
            "planner_latency_seconds": round(self.planner_latency_seconds, 4),
            "failure_reason": self.failure_reason,
            "classification": self.classification,
            "stub_fallback_occurred": self.stub_fallback_occurred,
            "plan_call_count": self.plan_call_count,
            "image_count": self.image_count,
        }


@dataclass
class EvalSummary:
    """Model-agnostic aggregate over records, for comparing planners."""

    model_name: str
    total: int
    valid: int
    invalid: int
    validity_rate: float
    mean_latency_seconds: float
    plan_lengths: List[int] = field(default_factory=list)
    tool_sequence_counts: Dict[str, int] = field(default_factory=dict)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    validity_by_question_type: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "validity_rate": self.validity_rate,
            "mean_latency_seconds": round(self.mean_latency_seconds, 4),
            "plan_lengths": self.plan_lengths,
            "tool_sequence_counts": self.tool_sequence_counts,
            "failure_reasons": self.failure_reasons,
            "validity_by_question_type": self.validity_by_question_type,
        }


def evaluate_case(case: EvalCase, planner_model: VisionLanguageModel) -> EvalRecord:
    capturing = RawCapturingModel(planner_model)
    planner = FunctionalValidationPlanner(capturing)

    plan: Optional[MRAGPlan] = None
    failure_reason: Optional[str] = None
    start = time.perf_counter()
    try:
        plan = planner.plan(case.question)
    except PlanValidationError as error:
        failure_reason = f"{type(error).__name__}: {error}"
    latency = time.perf_counter() - start

    raw = capturing.last_response.text if capturing.last_response is not None else None
    model_name = (
        capturing.last_response.model_name
        if capturing.last_response is not None
        else planner_model.model_name
    )

    return EvalRecord(
        case_id=case.case_id,
        question_type=case.question_type,
        model_name=model_name,
        raw_output=raw,
        plan_valid=plan is not None,
        plan=plan,
        tool_sequence=(
            [tool.value for tool in plan.tool_sequence()] if plan is not None else None
        ),
        plan_length=len(plan.steps) if plan is not None else None,
        planner_latency_seconds=latency,
        failure_reason=failure_reason,
        classification=classify_raw_output(raw),
        stub_fallback_occurred=detect_stub_fallback(raw, model_name),
        plan_call_count=planner.plan_call_count,
        image_count=len(case.question.images),
    )


def evaluate_cases(
    cases: Iterable[EvalCase], planner_model: VisionLanguageModel
) -> List[EvalRecord]:
    return [evaluate_case(case, planner_model) for case in cases]


def summarize(records: Sequence[EvalRecord]) -> EvalSummary:
    total = len(records)
    valid_records = [record for record in records if record.plan_valid]
    valid = len(valid_records)

    tool_sequence_counts: Dict[str, int] = {}
    for record in valid_records:
        key = " -> ".join(record.tool_sequence or [])
        tool_sequence_counts[key] = tool_sequence_counts.get(key, 0) + 1

    failure_reasons: Dict[str, int] = {}
    for record in records:
        if record.failure_reason is not None:
            failure_reasons[record.failure_reason] = (
                failure_reasons.get(record.failure_reason, 0) + 1
            )

    by_type: Dict[str, List[EvalRecord]] = {}
    for record in records:
        by_type.setdefault(record.question_type, []).append(record)

    model_names = {record.model_name for record in records}

    return EvalSummary(
        model_name=model_names.pop() if len(model_names) == 1 else "mixed",
        total=total,
        valid=valid,
        invalid=total - valid,
        validity_rate=(valid / total) if total else 0.0,
        mean_latency_seconds=(
            mean(record.planner_latency_seconds for record in records) if total else 0.0
        ),
        plan_lengths=[
            record.plan_length for record in valid_records if record.plan_length
        ],
        tool_sequence_counts=tool_sequence_counts,
        failure_reasons=failure_reasons,
        validity_by_question_type={
            question_type: f"{sum(1 for r in items if r.plan_valid)}/{len(items)}"
            for question_type, items in by_type.items()
        },
    )


def report_to_json(records: Sequence[EvalRecord], indent: int = 2) -> str:
    return json.dumps(
        {
            "summary": summarize(records).to_dict(),
            "records": [record.to_dict() for record in records],
        },
        indent=indent,
    )
