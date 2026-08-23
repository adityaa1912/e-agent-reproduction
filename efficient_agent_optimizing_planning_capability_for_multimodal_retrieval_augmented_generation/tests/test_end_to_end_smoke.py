import unittest

import tests._paths

from eagent.common.types import Image, Question

from eagent_baseline.agent import EAgent
from eagent_baseline.config import BaselineConfig, default_config_path
from eagent_baseline.executor import ExecutionState, TaskExecutor
from eagent_baseline.models import ScriptedVisionLanguageModel, build_executor_model
from eagent_baseline.plan import ToolName
from eagent_baseline.planner import MRAGPlanner
from eagent_baseline.tools import (
    ImageSearchTool,
    RequeryTool,
    ResponseTool,
    TextSearchTool,
)

from tests._fixtures import _config, EXPECTED_TOOL_SEQUENCE, FULL_PLAN_JSON


def _synthetic_question() -> Question:
    return Question(
        text="Which club is this person playing for?",
        images=[Image(url="http://example.invalid/person.png")],
    )


class EndToEndSmokeTests(unittest.TestCase):
    def test_multimodal_input_reaches_terminal_response(self) -> None:
        config = BaselineConfig.from_yaml(default_config_path())
        executor_model = build_executor_model(config)

        planner = MRAGPlanner(ScriptedVisionLanguageModel("dev-planner", [FULL_PLAN_JSON]))
        executor = TaskExecutor(
            {
                ToolName.IMAGE_SEARCH: ImageSearchTool(),
                ToolName.TEXT_SEARCH: TextSearchTool(),
                ToolName.REQUERY: RequeryTool(executor_model),
                ToolName.RESPONSE: ResponseTool(executor_model),
            }
        )
        agent = EAgent(planner, executor)

        question = _synthetic_question()
        self.assertTrue(question.is_multimodal)

        state = agent.run(question)

        self.assertIsInstance(state, ExecutionState)
        self.assertEqual(planner.plan_call_count, 1)
        self.assertEqual([result.tool for result in state.trace], EXPECTED_TOOL_SEQUENCE)
        self.assertEqual(state.trace[-1].tool, "response")
        self.assertIsInstance(state.final_response, str)
        self.assertTrue(state.final_response)


if __name__ == "__main__":
    unittest.main()
