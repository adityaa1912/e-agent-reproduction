from __future__ import annotations

import eagent_baseline._bootstrap

from eagent.common.types import Question
from eagent.models.protocols import ModelRequest, VisionLanguageModel

from eagent_baseline.plan import MRAGPlan

ASSUMED_PLANNER_PROMPT_TEMPLATE = (
    "You are the mRAG planner. Analyze the question and image and produce a "
    "single JSON plan with a 'steps' list. Each step has a 'tool' "
    "(image_search, text_search, requery, or response) and an 'arguments' "
    "object. The plan must end with a response step.\n"
    "Question: {question}"
)


class MRAGPlanner:
    def __init__(self, model: VisionLanguageModel) -> None:
        self._model = model
        self._plan_call_count = 0

    @property
    def plan_call_count(self) -> int:
        return self._plan_call_count

    def build_request(self, question: Question) -> ModelRequest:
        prompt = ASSUMED_PLANNER_PROMPT_TEMPLATE.format(question=question.text)
        return ModelRequest(prompt=prompt, images=list(question.images), temperature=0.0)

    def parse_plan(self, text: str) -> MRAGPlan:
        plan = MRAGPlan.from_json(text)
        plan.validate()
        return plan

    def plan(self, question: Question) -> MRAGPlan:
        request = self.build_request(question)
        response = self._model.generate(request)
        self._plan_call_count += 1
        return self.parse_plan(response.text)
