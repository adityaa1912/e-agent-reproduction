import unittest
from pathlib import Path

import tests._paths

from eagent_baseline.config import BaselineConfig, RuntimeMode, default_config_path
from eagent_baseline.data import QuestionType, RemPlanInstance
from eagent_baseline.models import build_executor_model, build_planner_model
from eagent.models.factory import UnsupportedProviderError
from eagent_baseline.plan import MRAGPlan, PlanStep, ToolName
from eagent_baseline.training import PlannerTrainingScaffold

from tests._fixtures import _config


class ConfigTests(unittest.TestCase):
    def test_development_config_loads(self) -> None:
        config = _config()
        self.assertEqual(config.mode, RuntimeMode.DEVELOPMENT)
        self.assertEqual(config.dataset.benchmark_size, 200)
        self.assertEqual(config.dataset.planner_train_samples, 10000)

    def test_models_build_from_config(self) -> None:
        config = _config()
        self.assertEqual(build_planner_model(config).model_name, "stub-planner")
        self.assertEqual(build_executor_model(config).model_name, "stub-executor")

    def test_research_config_loads(self) -> None:
        research_path = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
        config = BaselineConfig.from_yaml(research_path)
        self.assertEqual(config.mode, RuntimeMode.RESEARCH)

    def test_research_config_records_paper_models(self) -> None:
        research_path = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
        config = BaselineConfig.from_yaml(research_path)
        self.assertEqual(config.planner.model_name, "InternVL2-8B")
        self.assertEqual(config.executor.model_name, "Qwen2-VL-72B")

    def test_research_config_records_search_services(self) -> None:
        research_path = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
        config = BaselineConfig.from_yaml(research_path)
        self.assertEqual(config.tools.image_search.provider, "baidu")
        self.assertEqual(config.tools.text_search.provider, "tavily")

    def test_research_planner_model_raises_unsupported_provider(self) -> None:
        research_path = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
        config = BaselineConfig.from_yaml(research_path)
        with self.assertRaises(UnsupportedProviderError):
            build_planner_model(config)

    def test_research_executor_model_raises_unsupported_provider(self) -> None:
        research_path = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"
        config = BaselineConfig.from_yaml(research_path)
        with self.assertRaises(UnsupportedProviderError):
            build_executor_model(config)


class TrainingScaffoldTests(unittest.TestCase):
    def test_scaffold_reports_paper_values(self) -> None:
        scaffold = PlannerTrainingScaffold(base_model="InternVL2-8B", train_samples=10000)
        described = scaffold.describe()
        self.assertEqual(described["base_model"], "InternVL2-8B")
        self.assertEqual(described["train_samples"], 10000)
        self.assertFalse(described["recipe_available"])

    def test_prepare_validates_instances(self) -> None:
        scaffold = PlannerTrainingScaffold(base_model="InternVL2-8B", train_samples=10000)
        instances = [
            RemPlanInstance(
                image_ref="img://1",
                question="q",
                question_type=QuestionType.FUNDAMENTAL,
                gold_plan=MRAGPlan(steps=[PlanStep(ToolName.RESPONSE)]),
            )
        ]
        self.assertEqual(scaffold.prepare(instances), 1)

    def test_train_is_not_implemented(self) -> None:
        scaffold = PlannerTrainingScaffold(base_model="InternVL2-8B", train_samples=10000)
        with self.assertRaises(NotImplementedError):
            scaffold.train()


if __name__ == "__main__":
    unittest.main()
