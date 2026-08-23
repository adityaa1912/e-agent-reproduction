import unittest

import tests._paths

from eagent.common.types import Image, Question

from eagent_baseline.agent import EAgent
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

from tests._fixtures import FULL_PLAN_JSON, _config


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

    def test_planner_build_request_includes_question_text(self) -> None:
        model = ScriptedVisionLanguageModel("dev", [])
        planner = MRAGPlanner(model)
        question = Question(text="what is this?", images=[])
        request = planner.build_request(question)
        self.assertIn("what is this?", request.prompt)

    def test_planner_build_request_includes_all_images(self) -> None:
        model = ScriptedVisionLanguageModel("dev", [])
        planner = MRAGPlanner(model)
        question = Question(
            text="q",
            images=[Image(url="http://a.com/1.png"), Image(url="http://a.com/2.png")],
        )
        request = planner.build_request(question)
        self.assertEqual(len(request.images), 2)
        self.assertEqual(request.images[0].url, "http://a.com/1.png")
        self.assertEqual(request.images[1].url, "http://a.com/2.png")

    def test_planner_build_request_temperature_is_zero(self) -> None:
        model = ScriptedVisionLanguageModel("dev", [])
        planner = MRAGPlanner(model)
        request = planner.build_request(Question(text="q", images=[]))
        self.assertEqual(request.temperature, 0.0)

    def test_planner_increments_call_count_per_plan(self) -> None:
        model = ScriptedVisionLanguageModel("dev", [FULL_PLAN_JSON, FULL_PLAN_JSON])
        planner = MRAGPlanner(model)
        planner.plan(_question())
        planner.plan(_question())
        self.assertEqual(planner.plan_call_count, 2)
        self.assertEqual(model.call_count, 2)


class ExecutorDispatchTests(unittest.TestCase):
    def test_execute_dispatches_each_step_to_registered_tool(self) -> None:
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.TEXT_SEARCH, {"query": "keyword"}),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, _question())
        self.assertEqual([result.tool for result in state.trace], ["image_search", "text_search", "requery", "response"])

    def test_executor_sequential_execution_matches_plan_order(self) -> None:
        plan = MRAGPlan.from_json(FULL_PLAN_JSON)
        executor = _executor()
        state = executor.execute(plan, _question())
        self.assertEqual(len(state.trace), 4)
        self.assertEqual(state.trace[0].tool, "image_search")
        self.assertEqual(state.trace[1].tool, "requery")
        self.assertEqual(state.trace[2].tool, "text_search")
        self.assertEqual(state.trace[3].tool, "response")

    def test_image_search_receives_image_from_question(self) -> None:
        question = Question(text="q", images=[Image(url="http://example.com/img.png")])
        plan = MRAGPlan(steps=[PlanStep(ToolName.IMAGE_SEARCH), PlanStep(ToolName.RESPONSE)])
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertEqual(state.image_search_results, [])

    def test_text_search_uses_step_arguments_when_provided(self) -> None:
        question = Question(text="q", images=[])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.TEXT_SEARCH, {"query": "my-query"}),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertEqual(state.text_search_results, [])

    def test_text_search_uses_last_query_when_step_arguments_missing(self) -> None:
        question = Question(text="q", images=[Image(url="http://x.com/i.png")])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.TEXT_SEARCH),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertEqual(state.text_search_results, [])
        self.assertIsNotNone(state.last_query)

    def test_text_search_missing_query_raises(self) -> None:
        question = Question(text="q", images=[])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.TEXT_SEARCH),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        with self.assertRaises(ExecutorError):
            executor.execute(plan, question)

    def test_executor_handles_question_with_no_images(self) -> None:
        question = Question(text="q", images=[])
        plan = MRAGPlan(steps=[PlanStep(ToolName.RESPONSE)])
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertIsNotNone(state.final_response)

    def test_result_propagation_image_search_to_requery(self) -> None:
        question = Question(text="q", images=[Image(url="http://x.com/i.png")])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertEqual(len(state.trace), 3)
        self.assertEqual(state.trace[0].tool, "image_search")
        self.assertEqual(state.trace[1].tool, "requery")

    def test_result_propagation_requery_to_text_search(self) -> None:
        question = Question(text="q", images=[Image(url="http://x.com/i.png")])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.TEXT_SEARCH),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertIsNotNone(state.last_query)
        self.assertEqual(state.trace[-1].tool, "response")

    def test_result_propagation_all_results_to_response(self) -> None:
        question = Question(text="q", images=[Image(url="http://x.com/i.png")])
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.TEXT_SEARCH),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        executor = _executor()
        state = executor.execute(plan, question)
        self.assertIsNotNone(state.final_response)
        self.assertIsInstance(state.final_response, str)
        self.assertTrue(state.final_response)


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
