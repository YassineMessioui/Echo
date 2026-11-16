"""Hosted LLMs through the OpenAI-compatible chat completions API.

Groq, OpenRouter, Google Gemini, Mistral and OpenAI all expose this
API, so one client covers them; ``PROVIDERS`` holds their base URLs, the environment
variable each reads its API key from, and a default model. Any other compatible
server (LM Studio, vLLM, a proxy…) works with ``provider = "openai-compatible"`` and
``host`` set to its base URL.

API keys are never stored in the config: they're read from ``ECHO_LLM_API_KEY`` or
the provider's own variable (e.g. ``GROQ_API_KEY``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from echo.config import LLMConfig
from echo.errors import LLMError

API_KEY_ENV = "ECHO_LLM_API_KEY"
"""Overrides the provider-specific API key variable."""

GENERIC_PROVIDER = "openai-compatible"

_OLLAMA_DEFAULTS = LLMConfig()


@dataclass(frozen=True)
class HostedProvider:
    base_url: str
    key_env: str
    default_model: str
    signup: str


PROVIDERS: dict[str, HostedProvider] = {
    "groq": HostedProvider(
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-120b",
        "https://console.groq.com/keys",
    ),
    "openrouter": HostedProvider(
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
        "https://openrouter.ai/settings/keys",
    ),
    "gemini": HostedProvider(
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
        "https://aistudio.google.com/apikey",
    ),
    "mistral": HostedProvider(
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
        "mistral-small-latest",
        "https://console.mistral.ai/api-keys",
    ),
    "openai": HostedProvider(
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "gpt-4.1-mini",
        "https://platform.openai.com/api-keys",
    ),
}


class OpenAICompatibleClient:
    """Implements ``echo.interfaces.LLMClient`` over ``POST {base_url}/chat/completions``.

    When ``json_schema`` is given, JSON mode (``response_format: json_object``) is
    requested; the caller still validates the answer. If the server rejects JSON mode,
    the request is repeated without it once and JSON mode stays off for this client.
    """

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
        key_env: str = API_KEY_ENV,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.model_name = f"{provider}/{model}"
        self.temperature = temperature
        self._key_env = key_env
        self._json_mode = True
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    @classmethod
    def from_config(
        cls,
        config: LLMConfig,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> OpenAICompatibleClient:
        env = os.environ if env is None else env
        if config.provider == GENERIC_PROVIDER:
            preset = None
            base_url, key_env, model = config.host, API_KEY_ENV, config.model
        elif config.provider in PROVIDERS:
            preset = PROVIDERS[config.provider]
            # The LLMConfig defaults are Ollama's; a hosted preset replaces them
            # unless they were changed explicitly.
            base_url = preset.base_url if config.host == _OLLAMA_DEFAULTS.host else config.host
            model = preset.default_model if config.model == _OLLAMA_DEFAULTS.model else config.model
            key_env = preset.key_env
        else:
            raise LLMError(f"unknown hosted LLM provider {config.provider!r}")

        api_key = env.get(API_KEY_ENV) or env.get(key_env)
        if preset and not api_key:
            raise LLMError(
                f"no API key for {config.provider}: set {key_env} (or {API_KEY_ENV}); "
                f"get one at {preset.signup}"
            )
        return cls(
            provider=config.provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=config.temperature,
            timeout_s=config.timeout_s,
            key_env=key_env,
            transport=transport,
        )

    def complete(
        self, system: str, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if json_schema is not None and self._json_mode:
            response = self._post({**body, "response_format": {"type": "json_object"}})
            if response.status_code == 400 and _mentions_json_mode(response):
                self._json_mode = False
                response = self._post(body)
        else:
            response = self._post(body)
        self._raise_for_status(response)

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected {self.provider} response: {response.text[:200]!r}") from e
        if not isinstance(content, str):
            raise LLMError(f"unexpected {self.provider} response: {response.text[:200]!r}")
        return content

    def close(self) -> None:
        self._http.close()

    # --- helpers ---------------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._http.post("chat/completions", json=body)
        except httpx.ConnectError as e:
            raise LLMError(f"cannot reach {self.provider} at {self._http.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"{self.provider} request timed out after {self._http.timeout.read:g}s "
                f"(model {self.model!r}); raise llm.timeout_s"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.provider} request failed: {e}") from e

    def _raise_for_status(self, response: httpx.Response) -> None:
        if not response.is_error:
            return
        detail = _error_message(response)
        status = response.status_code
        if status in (401, 403) or (status == 400 and "api key" in detail.lower()):
            raise LLMError(
                f"{self.provider} rejected the API key (HTTP {status}): {detail}. "
                f"Check {self._key_env} (or {API_KEY_ENV})"
            )
        if status == 404:
            raise LLMError(
                f"{self.provider} has no model {self.model!r} (HTTP 404): {detail}. "
                f"Set llm.model to a model this provider serves"
            )
        if status == 429:
            wait = response.headers.get("retry-after")
            hint = f"; retry after {wait}s" if wait else ""
            raise LLMError(f"{self.provider} rate limit reached (HTTP 429){hint}: {detail}")
        raise LLMError(f"{self.provider} returned HTTP {status}: {detail}")


def _error_message(response: httpx.Response) -> str:
    """The error text from OpenAI-style (``{"error": {"message"}}``) or plain bodies."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(data, list) and data:  # Gemini sometimes wraps errors in a list
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        error = error.get("message") or error.get("detail") or error
    return str(error).strip().rstrip(".")[:300]


def _mentions_json_mode(response: httpx.Response) -> bool:
    text = _error_message(response).lower()
    return "response_format" in text or "json" in text
