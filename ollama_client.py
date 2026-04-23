from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional
from urllib import error, request


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server cannot be reached or returns invalid data."""


@dataclass
class OllamaModel:
    name: str
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""

    def label(self) -> str:
        details = " · ".join(part for part in [self.family, self.parameter_size, self.quantization_level] if part)
        return f"{self.name} ({details})" if details else self.name


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/api"

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        url = f"{self.api_base}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at {self.api_base}. Make sure `ollama serve` is running."
            ) from exc
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.code} for {path}: {message or exc.reason}"
            ) from exc

        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise OllamaConnectionError(f"Ollama returned invalid JSON for {path}.") from exc

    def list_models(self) -> List[OllamaModel]:
        payload = self._request_json("/tags", method="GET", timeout=10)
        models: List[OllamaModel] = []
        for item in payload.get("models", []):
            details = item.get("details") or {}
            name = item.get("model") or item.get("name") or ""
            if not name:
                continue
            models.append(
                OllamaModel(
                    name=name,
                    family=details.get("family", ""),
                    parameter_size=details.get("parameter_size", ""),
                    quantization_level=details.get("quantization_level", ""),
                )
            )
        models.sort(key=lambda model: model.name.lower())
        return models

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except OllamaConnectionError:
            return False

    def pull_model(self, model: str, *, insecure: bool = False) -> str:
        payload = self._request_json(
            "/pull",
            method="POST",
            payload={"model": model, "insecure": insecure, "stream": False},
            timeout=600,
        )
        status = payload.get("status") or payload.get("error") or "unknown"
        if isinstance(status, str) and status.lower() in {"success", "downloaded", "already exists"}:
            return status
        return str(status)

    def chat_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        keep_alive: str = "5m",
        think: Any = False,
    ) -> Dict[str, Any]:
        payload = self._request_json(
            "/chat",
            method="POST",
            payload={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "keep_alive": keep_alive,
                "think": think,
                "options": options or {"temperature": 0},
            },
            timeout=120,
        )
        content = ((payload.get("message") or {}).get("content") or "").strip()
        if not content:
            raise OllamaConnectionError("Ollama returned an empty response.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaConnectionError("Ollama did not return valid structured JSON.") from exc
