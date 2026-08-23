"""Transformers-based :class:`VisionLanguageModel` provider.

Runs a HuggingFace multimodal model on CPU with no quantisation.
Model weights are downloaded from Hugging Face only when
:class:`RealTransformersVisionLanguageModel` is instantiated with a
``model_id`` and ``generate`` is called; importing this module does not
download anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel


@dataclass(frozen=True)
class _HfDevice:
    cpu: str = "cpu"
    cuda: str = "cuda"
    mps: str = "mps"


@dataclass(frozen=True)
class _HfDtype:
    float32: str = "float32"
    bfloat16: str = "bfloat16"
    float16: str = "float16"


class RealTransformersVisionLanguageModel(VisionLanguageModel):
    """A CPU-only Transformers provider for multimodal generation.

    Loads a HuggingFace ``AutoModelForVision2Seq`` model and its
    ``AutoProcessor`` on construction.  Image inputs are accepted via
    :class:`eagent.common.types.Image` (``url`` or ``path`` are supported).
    """

    DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._torch_dtype = torch_dtype
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_id

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=getattr(__import__("torch"), self._torch_dtype),
                device_map=self._device,
            )
        return self._model, self._processor

    def generate(self, request: ModelRequest) -> ModelResponse:
        model, processor = self._ensure_loaded()

        images: List[Any] = []
        for img in request.images:
            from PIL import Image as _PILImage
            from transformers.image_utils import load_image

            if img.path is not None:
                pil_img = _PILImage.open(img.path)
            else:
                pil_img = load_image(img.url if img.url is not None else "")
            images.append(pil_img)

        inputs = processor(
            text=request.prompt,
            images=images if images else None,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": request.max_tokens if request.max_tokens is not None else 512,
        }
        if request.temperature > 0.0:
            generate_kwargs["temperature"] = request.temperature

        output_ids = model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        text = processor.decode(generated_ids[0], skip_special_tokens=True)

        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(text.split())

        return ModelResponse(
            text=text,
            model_name=self._model_id,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "images": len(request.images),
            },
            raw=output_ids,
        )
