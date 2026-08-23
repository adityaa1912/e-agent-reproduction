import unittest

import tests._paths

from eagent.common.types import Image, Question

from eagent_baseline.agent import EAgent
from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.executor import ExecutorError, TaskExecutor
from eagent_baseline.models import ScriptedVisionLanguageModel, build_executor_model
from eagent_baseline.plan import MRAGPlan, PlanStep, PlanValidationError, ToolName
from eagent_baseline.planner import MRAGPlanner
from eagent_baseline.tools import (
    ImageSearchTool,
    RequeryTool,
    ResponseTool,
    TextSearchTool,
)

FULL_PLAN_JSON = (
    '{"steps": ['
    '{"tool": "image_search", "arguments": {}},'
    '{"tool": "requery", "arguments": {}},'
    '{"tool": "text_search", "arguments": {}},'
    '{"tool": "response", "arguments": {}}]}'
)


def _config() -> BaselineConfig:
    return BaselineConfig.from_yaml(default_config_path())


def _question() -> Question:
    return Question(text="Which club is this person playing for?", images=[Image(url="x")])


def _executor() -> TaskExecutor:
    model = build_executor_model(_config())
    return TaskExecutor(
        {
            ToolName.IMAGE_SEARCH: ImageSearchTool(),
            ToolName.TEXT_SEARCH: TextSearchTool(),
            ToolName.REQUERY: RequeryTool(model),
            ToolName.RESPONSE: ResponseTool(model),
        }
    )


class PlannerTests(unittest.TestCase):
    def test_planner_single_pass_and_parse(self) -> None:
        model = ScriptedVisionLanguageModel("dev-planner", [FULL_PLAN_JSON])
        planner = MRAGPlanner(model)
        plan = planner.plan(_question())
        self.assertEqual(planner.plan_call_count, 1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(plan.tool_sequence()[-1], ToolName.RESPONSE)

    def test_planner_rejects_invalid_plan_text(self) -> None:
        model = ScriptedVisionLanguageModel("dev-planner", ['{"steps": []}'])
        with self.assertRaises(PlanValidationError):
            MRAGPlanner(model).plan(_question())


class ExecutorTests(unittest.TestCase):
    def test_execute_full_plan_reaches_terminal_response(self) -> None:
        plan = MRAGPlan.from_json(FULL_PLAN_JSON)
        state = _executor().execute(plan, _question())
        self.assertIsNotNone(state.final_response)
        self.assertEqual(state.trace[-1].tool, "response")
        self.assertEqual(len(state.trace), 4)

    def test_missing_tool_raises(self) -> None:
        plan = MRAGPlan(steps=[PlanStep(ToolName.IMAGE_SEARCH), PlanStep(ToolName.RESPONSE)])
        executor = TaskExecutor({ToolName.RESPONSE: ResponseTool(build_executor_model(_config()))})
        with self.assertRaises(ExecutorError):
            executor.execute(plan, _question())

    def test_invalid_plan_rejected_by_executor(self) -> None:
        plan = MRAGPlan(steps=[PlanStep(ToolName.IMAGE_SEARCH)])
        with self.assertRaises(PlanValidationError):
            _executor().execute(plan, _question())


class AgentTests(unittest.TestCase):
    def test_plan_then_execute_end_to_end(self) -> None:
        planner = MRAGPlanner(ScriptedVisionLanguageModel("dev-planner", [FULL_PLAN_JSON]))
        agent = EAgent(planner, _executor())
        answer = agent.answer(_question())
        self.assertIsInstance(answer, str)
        self.assertTrue(answer)


if __name__ == "__main__":
    unittest.main()
