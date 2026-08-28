"""Gemini VisionLanguageModel used only for functional validation.

This module is not part of the paper skeleton. It lives outside ``src/eagent``
and ``eagent_baseline``. Callers inject the returned model into planner
constructors; the reproduction factory stays stub-only. Transport and
authentication are delegated to the official ``google-genai`` SDK.
"""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from eagent.common.types import Image
from eagent.models.protocols import ModelRequest, ModelResponse, VisionLanguageModel


MRAG_PLAN_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["image_search", "text_search", "requery", "response"],
                    },
                    "arguments": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
                "required": ["tool"],
            },
        }
    },
    "required": ["steps"],
}


class MissingGeminiAPIKeyError(RuntimeError):
    """Raised when the configured API-key environment variable is absent."""


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini SDK call returns no usable text."""


class RealGeminiVisionLanguageModel(VisionLanguageModel):
    """Gemini provider that returns MRAGPlan-shaped structured JSON.

    Credentials come from an environment variable only and are handed to the
    ``google-genai`` client. There is no retry, no output repair, and no stub
    fallback: SDK errors propagate, and empty or truncated generations raise.
    """

    DEFAULT_MODEL_ID = "gemini-3.6-flash"
    API_KEY_ENV_VAR = "GEMINI_API_KEY"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        api_key_env_var: str = API_KEY_ENV_VAR,
        response_schema: Optional[Dict[str, Any]] = None,
        max_output_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> None:
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._response_schema = response_schema or MRAG_PLAN_RESPONSE_SCHEMA
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._client: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_id

    def _api_key(self) -> str:
        key = os.environ.get(self._api_key_env_var)
        if not key:
            raise MissingGeminiAPIKeyError(
                f"Environment variable {self._api_key_env_var} is not set; "
                "Gemini credentials must come from the environment."
            )
        return key

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key())
        return self._client

    def _image_bytes(self, img: Image) -> tuple[bytes, str]:
        if img.data is not None:
            source = None
            data = img.data
        elif img.path is not None:
            source = img.path
            with open(img.path, "rb") as handle:
                data = handle.read()
        elif img.url is not None:
            source = img.url
            with urlopen(img.url, timeout=self._timeout) as raw:
                data = raw.read()
        else:
            raise ValueError("Image has no path, data, or url")

        mime_type = img.mime_type
        if mime_type is None and source is not None:
            mime_type = mimetypes.guess_type(source)[0]
        return data, mime_type or "image/png"

    def _contents(self, request: ModelRequest) -> List[Any]:
        from google.genai import types

        parts: List[Any] = []
        for img in request.images:
            data, mime_type = self._image_bytes(img)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(request.prompt)
        return parts

    def _config(self, request: ModelRequest) -> Any:
        from google.genai import types

        max_output_tokens = (
            self._max_output_tokens
            if self._max_output_tokens is not None
            else request.max_tokens
        )
        return types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=self._response_schema,
        )

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if text is not None and text.strip():
            return text
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        raise GeminiAPIError(
            "Gemini returned no output text "
            f"(candidates={len(candidates)}, finish_reason={finish_reason!r}); "
            "raise max_output_tokens if the generation was truncated."
        )

    def _usage(self, response: Any, request: ModelRequest) -> Dict[str, Any]:
        usage_metadata = getattr(response, "usage_metadata", None)
        return {
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
            "completion_tokens": getattr(
                usage_metadata, "candidates_token_count", None
            ),
            "total_tokens": getattr(usage_metadata, "total_token_count", None),
            "images": len(request.images),
        }

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._ensure_client()
        response = client.models.generate_content(
            model=self._model_id,
            contents=self._contents(request),
            config=self._config(request),
        )
        return ModelResponse(
            text=self._extract_text(response),
            model_name=self._model_id,
            usage=self._usage(response, request),
            raw=response,
        )
