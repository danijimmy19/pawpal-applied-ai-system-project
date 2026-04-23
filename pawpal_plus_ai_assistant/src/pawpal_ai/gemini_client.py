from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Iterable, List, Optional


class GeminiConnectionError(RuntimeError):
    """Raised when Gemini is unavailable or returns an unusable response."""


@dataclass
class GeminiModelOption:
    name: str
    label: str


DEFAULT_GEMINI_MODELS = [
    GeminiModelOption(name="gemini-2.5-flash", label="gemini-2.5-flash"),
    GeminiModelOption(name="gemini-2.5-pro", label="gemini-2.5-pro"),
]


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiConnectionError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise GeminiConnectionError(
                "google-genai is not installed. Run `pip install google-genai`."
            ) from exc

        self._genai = genai
        self._types = types
        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:  # pragma: no cover - library specific
            raise GeminiConnectionError(f"Failed to initialize Gemini client: {exc}") from exc

    @staticmethod
    def is_configured() -> bool:
        return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    def embed_texts(
        self,
        *,
        texts: List[str],
        task_type: str,
        titles: Optional[List[str]] = None,
        output_dimensionality: int = 768,
        model: str = "gemini-embedding-001",
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for index, text in enumerate(texts):
            config_kwargs: dict[str, Any] = {
                "task_type": task_type,
                "output_dimensionality": output_dimensionality,
            }
            if titles and task_type == "RETRIEVAL_DOCUMENT":
                config_kwargs["title"] = titles[index]
            try:
                result = self._client.models.embed_content(
                    model=model,
                    contents=text,
                    config=self._types.EmbedContentConfig(**config_kwargs),
                )
            except Exception as exc:
                raise GeminiConnectionError(f"Gemini embeddings request failed: {exc}") from exc

            result_embeddings = getattr(result, "embeddings", None)
            if not result_embeddings:
                raise GeminiConnectionError("Gemini returned no embeddings.")
            embedding_obj = result_embeddings[0]
            values = list(getattr(embedding_obj, "values", []) or [])
            if not values:
                raise GeminiConnectionError("Gemini returned an empty embedding vector.")
            embeddings.append([float(v) for v in values])
        return embeddings

    def generate_json(self, *, model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                },
            )
        except Exception as exc:
            raise GeminiConnectionError(f"Gemini generation request failed: {exc}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiConnectionError("Gemini returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiConnectionError("Gemini did not return valid structured JSON.") from exc
