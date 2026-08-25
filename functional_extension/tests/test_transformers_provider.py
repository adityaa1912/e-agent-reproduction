"""Offline tests for the functional HuggingFace CPU provider.

These tests mock Transformers, Torch, and PIL. They do not download weights
and they are not part of the paper-skeleton suites.
"""

from __future__ import annotations

import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
EXT = REPO_ROOT / "functional_extension"
for extra in (str(SRC), str(EXT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest
from eagent_functional.transformers_provider import RealTransformersVisionLanguageModel


class _Ids:
    shape = (1, 4)

    def to(self, _device):
        return self

    def __getitem__(self, _key):
        return self


class _Output:
    def __getitem__(self, _key):
        return _Ids()


class _Model:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def generate(self, **_kwargs):
        return _Output()


class TransformersProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "torch",
                "transformers",
                "transformers.image_utils",
                "PIL",
                "PIL.Image",
            )
        }
        self.opened: list = []
        self.load_image_calls: list = []
        self.from_pretrained_kwargs: dict = {}
        self.chat_messages = None
        self.add_generation_prompt = None
        self.processor_text = None
        self.processor_images = None
        self.model = _Model()
        self.processor = None

    def tearDown(self) -> None:
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _install_fakes(self) -> None:
        torch_mod = types.ModuleType("torch")
        torch_mod.float32 = "float32"

        processor = MagicMock()
        processor.apply_chat_template.side_effect = self._capture_chat
        processor.side_effect = self._capture_processor
        processor.decode.return_value = "ok"

        auto_processor = MagicMock()
        auto_processor.from_pretrained.return_value = processor

        auto_model = MagicMock()

        def _from_pretrained(model_id, **kwargs):
            self.from_pretrained_kwargs = dict(kwargs)
            self.from_pretrained_id = model_id
            return self.model

        auto_model.from_pretrained.side_effect = _from_pretrained

        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoProcessor = auto_processor
        transformers_mod.AutoModelForVision2Seq = auto_model

        image_utils = types.ModuleType("transformers.image_utils")

        def _load_image(url):
            self.load_image_calls.append(url)
            return f"url:{url}"

        image_utils.load_image = _load_image
        transformers_mod.image_utils = image_utils

        pil_image = types.ModuleType("PIL.Image")

        def _open(src):
            self.opened.append(src)
            return f"pil:{src}"

        pil_image.open = _open
        pil_mod = types.ModuleType("PIL")
        pil_mod.Image = pil_image

        sys.modules["torch"] = torch_mod
        sys.modules["transformers"] = transformers_mod
        sys.modules["transformers.image_utils"] = image_utils
        sys.modules["PIL"] = pil_mod
        sys.modules["PIL.Image"] = pil_image
        self.processor = processor

    def _capture_chat(self, messages, add_generation_prompt=True):
        self.chat_messages = messages
        self.add_generation_prompt = add_generation_prompt
        return "<chat>"

    def _capture_processor(self, text, images, return_tensors):
        self.processor_text = text
        self.processor_images = images
        return {"input_ids": _Ids()}

    def test_default_model_id_is_smolvlm(self) -> None:
        model = RealTransformersVisionLanguageModel()
        self.assertEqual(model.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")

    def test_model_name_returns_specified_id(self) -> None:
        model = RealTransformersVisionLanguageModel(model_id="custom/model-id")
        self.assertEqual(model.model_name, "custom/model-id")

    def test_construction_does_not_import_transformers(self) -> None:
        before = set(sys.modules)
        RealTransformersVisionLanguageModel()
        after = set(sys.modules)
        self.assertEqual(after, before)

    def test_from_pretrained_does_not_use_device_map(self) -> None:
        self._install_fakes()
        model = RealTransformersVisionLanguageModel()
        model.generate(ModelRequest(prompt="q"))
        self.assertNotIn("device_map", self.from_pretrained_kwargs)
        self.assertEqual(self.model.moved_to, "cpu")

    def test_generate_decodes_image_bytes(self) -> None:
        self._install_fakes()
        payload = b"\x89PNG-bytes"
        model = RealTransformersVisionLanguageModel()
        model.generate(
            ModelRequest(prompt="what", images=[Image(data=payload)])
        )
        self.assertEqual(len(self.load_image_calls), 0)
        self.assertEqual(len(self.opened), 1)
        src = self.opened[0]
        self.assertIsInstance(src, BytesIO)
        self.assertEqual(src.getvalue(), payload)

    def test_generate_opens_path_not_empty_url(self) -> None:
        self._install_fakes()
        model = RealTransformersVisionLanguageModel()
        model.generate(
            ModelRequest(prompt="what", images=[Image(path="local.png")])
        )
        self.assertEqual(self.opened, ["local.png"])
        self.assertEqual(self.load_image_calls, [])

    def test_generate_loads_url(self) -> None:
        self._install_fakes()
        model = RealTransformersVisionLanguageModel()
        model.generate(
            ModelRequest(
                prompt="what",
                images=[Image(url="http://example.invalid/a.png")],
            )
        )
        self.assertEqual(self.load_image_calls, ["http://example.invalid/a.png"])

    def test_generate_applies_chat_template_with_image_placeholder(self) -> None:
        self._install_fakes()
        model = RealTransformersVisionLanguageModel()
        model.generate(
            ModelRequest(prompt="describe", images=[Image(data=b"img")])
        )
        self.assertTrue(self.add_generation_prompt)
        content = self.chat_messages[0]["content"]
        self.assertEqual(content[0], {"type": "image"})
        self.assertEqual(content[1], {"type": "text", "text": "describe"})
        self.assertEqual(self.processor_text, "<chat>")
        self.processor.apply_chat_template.assert_called_once()


if __name__ == "__main__":
    unittest.main()
