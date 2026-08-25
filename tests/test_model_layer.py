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


if __name__ == "__main__":
    unittest.main()
