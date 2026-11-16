"""LLM clients. Topic extraction depends on ``echo.interfaces.LLMClient`` only, so the
provider can be swapped; ``OllamaClient`` talks to a local Ollama server."""

from __future__ import annotations

from typing import Any

import httpx

from echo.config import LLMConfig
from echo.errors import LLMError
from echo.interfaces import LLMClient


class OllamaClient:
    """Implements ``echo.interfaces.LLMClient`` over Ollama's ``/api/chat`` endpoint.

    ``transport`` is for tests (e.g. ``httpx.MockTransport``); by default a real
    HTTP connection to ``config.host`` is used.
    """

    def __init__(self, config: LLMConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        self.model_name = f"ollama/{config.model}"
        self._http = httpx.Client(
            base_url=config.host, timeout=config.timeout_s, transport=transport
        )

    def complete(
        self, system: str, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        if json_schema is not None:
            body["format"] = json_schema  # Ollama structured outputs

        try:
            response = self._http.post("/api/chat", json=body)
        except httpx.ConnectError as e:
            raise LLMError(
                f"cannot reach Ollama at {self.config.host}; is it running? (start it with "
                f"`ollama serve`)"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"Ollama request timed out after {self.config.timeout_s:g}s "
                f"(model {self.config.model!r}); raise llm.timeout_s or use a smaller model"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

        if response.status_code == 404 and "not found" in _error_text(response).lower():
            raise LLMError(
                f"Ollama model {self.config.model!r} is not available; "
                f"pull it with `ollama pull {self.config.model}`"
            )
        if response.is_error:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {_error_text(response)}")

        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError(f"unexpected Ollama response: {response.text[:200]!r}") from e
        if not isinstance(content, str):
            raise LLMError(f"unexpected Ollama response: {response.text[:200]!r}")
        return content

    def close(self) -> None:
        self._http.close()


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text


def make_llm_client(config: LLMConfig) -> LLMClient:
    """Build the client for ``config.provider``."""
    if config.provider == "ollama":
        return OllamaClient(config)
    raise LLMError(f"unknown LLM provider {config.provider!r} (supported: 'ollama')")
