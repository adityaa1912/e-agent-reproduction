from __future__ import annotations

import eagent_baseline._bootstrap

from eagent.common.types import Question

from eagent_baseline.executor import ExecutionState, TaskExecutor
from eagent_baseline.plan import MRAGPlan
from eagent_baseline.planner import MRAGPlanner


class EAgent:
    def __init__(self, planner: MRAGPlanner, executor: TaskExecutor) -> None:
        self._planner = planner
        self._executor = executor

    def plan(self, question: Question) -> MRAGPlan:
        return self._planner.plan(question)

    def run(self, question: Question) -> ExecutionState:
        plan = self._planner.plan(question)
        return self._executor.execute(plan, question)

    def answer(self, question: Question) -> str:
        return self.run(question).final_response
