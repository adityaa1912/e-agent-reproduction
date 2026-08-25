"""CPU HuggingFace VisionLanguageModel used only for functional validation.

This module is not part of the paper skeleton. It lives outside ``src/eagent``
and ``eagent_baseline``. Callers inject the returned model into planner or
executor constructors; the reproduction factory stays stub-only.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel


class RealTransformersVisionLanguageModel(VisionLanguageModel):
    """CPU-only Transformers provider for a public substitute VLM.

    Loads ``AutoModelForVision2Seq`` without ``device_map`` (no ``accelerate``).
    Image bytes, paths, and URLs are all accepted. Prompts are chat-templated
    with image placeholders so Idefics3 / SmolVLM processors receive tokens.
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
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_id,
                torch_dtype=getattr(torch, self._torch_dtype),
            )
            self._model = self._model.to(self._device)
        return self._model, self._processor

    def _pil_image(self, img: Image) -> Any:
        from PIL import Image as PILImage
        from transformers.image_utils import load_image

        if img.path is not None:
            return PILImage.open(img.path)
        if img.data is not None:
            return PILImage.open(BytesIO(img.data))
        if img.url is not None:
            return load_image(img.url)
        raise ValueError("Image has no path, data, or url")

    def _chat_messages(self, request: ModelRequest) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "image"} for _ in request.images]
        content.append({"type": "text", "text": request.prompt})
        return [{"role": "user", "content": content}]

    def generate(self, request: ModelRequest) -> ModelResponse:
        model, processor = self._ensure_loaded()
        images = [self._pil_image(img) for img in request.images]
        templated = processor.apply_chat_template(
            self._chat_messages(request),
            add_generation_prompt=True,
        )
        inputs = processor(
            text=templated,
            images=images if images else None,
            return_tensors="pt",
        )
        if callable(getattr(inputs, "to", None)):
            inputs = inputs.to(self._device)
        else:
            inputs = {
                k: v.to(self._device) if callable(getattr(v, "to", None)) else v
                for k, v in inputs.items()
            }

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
