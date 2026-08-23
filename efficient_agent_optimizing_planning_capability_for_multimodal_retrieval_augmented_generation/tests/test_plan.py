import unittest

import tests._paths

from eagent_baseline.plan import MRAGPlan, PlanStep, PlanValidationError, ToolName


class PlanConstructionTests(unittest.TestCase):
    def test_construction_and_tool_sequence(self) -> None:
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH),
                PlanStep(ToolName.REQUERY),
                PlanStep(ToolName.TEXT_SEARCH),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        self.assertEqual(
            plan.tool_sequence(),
            [ToolName.IMAGE_SEARCH, ToolName.REQUERY, ToolName.TEXT_SEARCH, ToolName.RESPONSE],
        )

    def test_valid_plan_passes_validation(self) -> None:
        plan = MRAGPlan(steps=[PlanStep(ToolName.IMAGE_SEARCH), PlanStep(ToolName.RESPONSE)])
        plan.validate()

    def test_empty_plan_is_invalid(self) -> None:
        with self.assertRaises(PlanValidationError):
            MRAGPlan(steps=[]).validate()

    def test_plan_without_terminal_response_is_invalid(self) -> None:
        with self.assertRaises(PlanValidationError):
            MRAGPlan(steps=[PlanStep(ToolName.IMAGE_SEARCH)]).validate()

    def test_response_only_allowed_as_terminal(self) -> None:
        with self.assertRaises(PlanValidationError):
            MRAGPlan(
                steps=[PlanStep(ToolName.RESPONSE), PlanStep(ToolName.RESPONSE)]
            ).validate()


class PlanSerializationTests(unittest.TestCase):
    def test_round_trip_json(self) -> None:
        plan = MRAGPlan(
            steps=[
                PlanStep(ToolName.IMAGE_SEARCH, {"query": "q"}),
                PlanStep(ToolName.RESPONSE),
            ]
        )
        restored = MRAGPlan.from_json(plan.to_json())
        self.assertEqual(restored.to_dict(), plan.to_dict())

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(PlanValidationError):
            MRAGPlan.from_json("{not json")

    def test_unknown_tool_raises(self) -> None:
        with self.assertRaises(PlanValidationError):
            MRAGPlan.from_dict({"steps": [{"tool": "search_the_web"}]})


if __name__ == "__main__":
    unittest.main()
