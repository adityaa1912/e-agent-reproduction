from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import eagent_baseline._bootstrap

from eagent.common.types import Image, Question

from eagent_baseline.plan import MRAGPlan, PlanStep, ToolName
from eagent_baseline.tools.base import Tool, ToolResult


class ExecutorError(RuntimeError):
    pass


@dataclass
class ExecutionState:
    image: Optional[Image] = None
    image_search_results: List[dict] = field(default_factory=list)
    text_search_results: List[dict] = field(default_factory=list)
    last_query: Optional[str] = None
    final_response: Optional[str] = None
    trace: List[ToolResult] = field(default_factory=list)


class TaskExecutor:
    def __init__(self, tools: Dict[ToolName, Tool]) -> None:
        self._tools = dict(tools)

    def _tool_for(self, tool_name: ToolName) -> Tool:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ExecutorError(f"No tool registered for {tool_name.value!r}")
        return tool

    def _run_step(self, step: PlanStep, question: Question, state: ExecutionState) -> ToolResult:
        tool = self._tool_for(step.tool)
        if step.tool is ToolName.IMAGE_SEARCH:
            result = tool.run(image=state.image, query=step.arguments.get("query"))
            state.image_search_results = list(result.content or [])
            return result
        if step.tool is ToolName.REQUERY:
            result = tool.run(
                question=question.text,
                image=state.image,
                image_search_results=state.image_search_results,
            )
            state.last_query = str(result.content)
            return result
        if step.tool is ToolName.TEXT_SEARCH:
            query = step.arguments.get("query") or state.last_query
            if not query:
                raise ExecutorError("text_search step has no query and no prior requery output")
            result = tool.run(query=query)
            state.text_search_results = list(result.content or [])
            return result
        if step.tool is ToolName.RESPONSE:
            result = tool.run(
                question=question.text,
                image=state.image,
                image_search_results=state.image_search_results,
                text_search_results=state.text_search_results,
            )
            state.final_response = str(result.content)
            return result
        raise ExecutorError(f"Unhandled tool {step.tool!r}")

    def execute(self, plan: MRAGPlan, question: Question) -> ExecutionState:
        plan.validate()
        state = ExecutionState(image=question.images[0] if question.images else None)
        for step in plan.steps:
            state.trace.append(self._run_step(step, question, state))
        if state.final_response is None:
            raise ExecutorError("Execution finished without a terminal response.")
        return state

    def run(self, plan: MRAGPlan, question: Question) -> str:
        return self.execute(plan, question).final_response
