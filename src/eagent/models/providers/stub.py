"""Development/test stub provider for :class:`VisionLanguageModel`.

.. warning::

    This is a **development and test double only**. It performs no real
    inference: it returns deterministic, canned output. It must **never** be
    used for paper-reproduction claims or to represent the behaviour of the
    real InternVL2-8B or Qwen2-VL-72B models.

The stub is intentionally free of any GPU, network, external API, or model
download requirements so the model layer can be exercised in local
development and in CI.
"""

from __future__ import annotations

from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel


class StubVisionLanguageModel(VisionLanguageModel):
    """A deterministic, dependency-free vision-language model double.

    The stub echoes a compact, deterministic summary of the incoming request
    so tests can assert on stable output. It reports its configured model name
    and returns dummy usage information.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return deterministic output derived only from the request.

        The output depends solely on the model name, the prompt, and the number
        of images, so identical requests always produce identical responses.
        """
        image_count = len(request.images)
        text = (
            f"[stub:{self._model_name}] prompt={request.prompt!r} "
            f"images={image_count}"
        )

        # Deterministic dummy usage information (no real tokenization).
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(text.split())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "images": image_count,
        }

        return ModelResponse(
            text=text,
            model_name=self._model_name,
            usage=usage,
            raw=None,
        )
