from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from eagent_baseline.plan import MRAGPlan


class QuestionType(str, Enum):
    FUNDAMENTAL = "fundamental"
    VISUAL_RECOGNITION = "visual_recognition"
    INFORMATION_SEEKING = "information_seeking"
    MULTI_FACETED = "multi_faceted"


class DataValidationError(ValueError):
    pass


@dataclass
class RemPlanInstance:
    image_ref: str
    question: str
    question_type: Optional[QuestionType] = None
    gold_plan: Optional[MRAGPlan] = None
    gold_answer: Optional[str] = None

    def validate(self) -> None:
        if not self.image_ref:
            raise DataValidationError("RemPlanInstance requires a non-empty 'image_ref'.")
        if not self.question:
            raise DataValidationError("RemPlanInstance requires a non-empty 'question'.")
        if self.question_type is not None and not isinstance(self.question_type, QuestionType):
            raise DataValidationError("'question_type' must be a QuestionType or None.")
        if self.gold_plan is not None:
            self.gold_plan.validate()
