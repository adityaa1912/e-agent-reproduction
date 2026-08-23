"""Tests for the E-Agent model/runtime abstraction layer (Step 5).

All tests run without a GPU, network access, external API keys, or downloaded
model checkpoints. They exercise only the provider-neutral interface, the
typed configuration, and the deterministic development stub provider.

Written with the standard-library ``unittest`` framework so the suite runs
with no third-party test tooling. It is also collectable by ``pytest`` if that
is available.
"""

import sys
import unittest
from pathlib import Path

# Make the ``src`` layout importable without an installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eagent.common.types import Image, Question  # noqa: E402
from eagent.models.config import (  # noqa: E402
    EAgentModelConfig,
    ModelSpec,
    RuntimeMode,
)
from eagent.models.factory import (  # noqa: E402
    UnsupportedProviderError,
    create_executor,
    create_model,
    create_planner,
)
from eagent.models.providers.transformers import (  # noqa: E402
    RealTransformersVisionLanguageModel,
)
from eagent.models.providers.transformers import RealTransformersVisionLanguageModel  # noqa: E402
from eagent.models.protocols import (  # noqa: E402
    ModelRequest,
    ModelResponse,
    VisionLanguageModel,
)

DEV_CONFIG_PATH = REPO_ROOT / "configs" / "development.yaml"


def _load_development_config() -> EAgentModelConfig:
    return EAgentModelConfig.from_yaml(DEV_CONFIG_PATH)


class DevelopmentConstructionTests(unittest.TestCase):
    def test_development_planner_construction(self) -> None:
        config = _load_development_config()
        planner = create_planner(config)
        self.assertIsInstance(planner, VisionLanguageModel)

    def test_development_executor_construction(self) -> None:
        config = _load_development_config()
        executor = create_executor(config)
        self.assertIsInstance(executor, VisionLanguageModel)

    def test_configured_model_names(self) -> None:
        config = _load_development_config()
        self.assertEqual(config.mode, RuntimeMode.DEVELOPMENT)
        self.assertEqual(create_planner(config).model_name, "stub-planner")
        self.assertEqual(create_executor(config).model_name, "stub-executor")


class MultimodalRequestTests(unittest.TestCase):
    def test_multimodal_request_acceptance(self) -> None:
        planner = create_planner(_load_development_config())
        question = Question(
            text="What is shown in these images?",
            images=[Image(url="http://example.invalid/a.png")],
        )
        request = ModelRequest(prompt=question.text, images=question.images)
        self.assertTrue(request.is_multimodal)

        response = planner.generate(request)
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.model_name, "stub-planner")
        self.assertIn("images=1", response.text)
        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage["images"], 1)

    def test_deterministic_stub_output(self) -> None:
        planner = create_planner(_load_development_config())
        request = ModelRequest(prompt="hello", images=[Image(url="x")])
        first = planner.generate(request)
        second = planner.generate(request)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.usage, second.usage)


class ProviderRejectionTests(unittest.TestCase):
    def test_unsupported_provider_rejection(self) -> None:
        spec = ModelSpec(provider="research", model_name="InternVL2-8B")
        with self.assertRaises(UnsupportedProviderError):
            create_model(spec)


class InstanceIndependenceTests(unittest.TestCase):
    def test_independent_planner_and_executor_instances(self) -> None:
        config = _load_development_config()
        planner = create_planner(config)
        executor = create_executor(config)
        # Distinct objects with distinct configured names.
        self.assertIsNot(planner, executor)
        self.assertNotEqual(planner.model_name, executor.model_name)


class UnsupportedProviderTests(unittest.TestCase):
    def test_research_provider_raises(self) -> None:
        spec = ModelSpec(provider="research", model_name="InternVL2-8B")
        with self.assertRaises(UnsupportedProviderError):
            create_model(spec)

    def test_unknown_provider_raises(self) -> None:
        spec = ModelSpec(provider="nonexistent_provider", model_name="dummy")
        with self.assertRaises(UnsupportedProviderError):
            create_model(spec)

    def test_research_error_message_mentions_executable_providers(self) -> None:
        spec = ModelSpec(provider="research", model_name="InternVL2-8B")
        with self.assertRaises(UnsupportedProviderError) as ctx:
            create_model(spec)
        msg = str(ctx.exception).lower()
        self.assertIn("research", msg)
        self.assertIn("stub", msg)
        self.assertIn("real_transformers", msg)

    def test_research_provider_does_not_create_planner(self) -> None:
        config = EAgentModelConfig(
            mode=RuntimeMode.RESEARCH,
            planner=ModelSpec(provider="research", model_name="InternVL2-8B"),
            executor=ModelSpec(provider="research", model_name="Qwen2-VL-72B"),
        )
        with self.assertRaises(UnsupportedProviderError):
            create_planner(config)

    def test_research_provider_does_not_create_executor(self) -> None:
        config = EAgentModelConfig(
            mode=RuntimeMode.RESEARCH,
            planner=ModelSpec(provider="research", model_name="InternVL2-8B"),
            executor=ModelSpec(provider="research", model_name="Qwen2-VL-72B"),
        )
        with self.assertRaises(UnsupportedProviderError):
            create_executor(config)


class TransformersProviderTests(unittest.TestCase):
    """Tests for the ``real_transformers`` VisionLanguageModel provider.

    These tests construct provider instances and assert the factory wiring
    without downloading weights or running inference (``transformers`` is
    imported lazily inside ``generate``).
    """

    def test_real_transformers_provider_exists(self) -> None:
        spec = ModelSpec(
            provider="real_transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        model = create_model(spec)
        self.assertIsInstance(model, RealTransformersVisionLanguageModel)
        self.assertEqual(model.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")

    def test_default_model_id_is_smlvlm(self) -> None:
        model = RealTransformersVisionLanguageModel()
        self.assertEqual(model.model_name, RealTransformersVisionLanguageModel.DEFAULT_MODEL_ID)
        self.assertEqual(model.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")

    def test_factory_creates_real_transformers_planner(self) -> None:
        config = EAgentModelConfig(
            mode=RuntimeMode.DEVELOPMENT,
            planner=ModelSpec(
                provider="real_transformers",
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            ),
            executor=ModelSpec(
                provider="real_transformers",
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            ),
        )
        planner = create_planner(config)
        self.assertIsInstance(planner, RealTransformersVisionLanguageModel)
        self.assertEqual(planner.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")

    def test_factory_creates_real_transformers_executor(self) -> None:
        config = EAgentModelConfig(
            mode=RuntimeMode.DEVELOPMENT,
            planner=ModelSpec(
                provider="real_transformers",
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            ),
            executor=ModelSpec(
                provider="real_transformers",
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
            ),
        )
        executor = create_executor(config)
        self.assertIsInstance(executor, RealTransformersVisionLanguageModel)

    def test_real_transformers_not_research(self) -> None:
        spec = ModelSpec(
            provider="real_transformers",
            model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        self.assertIsInstance(create_model(spec), RealTransformersVisionLanguageModel)
        with self.assertRaises(UnsupportedProviderError):
            create_model(ModelSpec(provider="research", model_name="InternVL2-8B"))

    def test_real_transformers_no_weights_on_import(self) -> None:
        import sys

        before = set(sys.modules)
        RealTransformersVisionLanguageModel()
        after = set(sys.modules)
        downloaded = after - before
        hf_downloaded = {
            m for m in downloaded
            if "huggingface" in m.lower() or "transformers" in m.lower()
        }
        self.assertEqual(hf_downloaded, set())

    def test_model_name_returns_specified_id(self) -> None:
        custom_id = "custom/model-id"
        model = RealTransformersVisionLanguageModel(model_id=custom_id)
        self.assertEqual(model.model_name, custom_id)


if __name__ == "__main__":
    unittest.main()
