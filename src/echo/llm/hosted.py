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

import base64
import logging
import math
import os
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from echo.config import LLMConfig
from echo.errors import LLMError

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 502, 503, 504})
"""Rate limited, or the provider is overloaded ("model is experiencing high demand")."""
MAX_RETRIES = 5
"""Retries per request for ``RETRY_STATUSES`` (#56)."""
TIMEOUT_RETRIES = 1
"""Retries for a timeout or dropped connection: each one already cost up to ``timeout_s``."""
MAX_RETRY_WAIT_S = 60.0
"""A longer ``retry-after`` (e.g. a daily quota) is not waited out: the error is raised."""
MAX_TOTAL_WAIT_S = 120.0
"""Most time one request spends sleeping between retries."""


def _jitter(seconds: float) -> float:
    """Spread backoffs so parallel workers limited together don't retry together."""
    return seconds * random.uniform(0.5, 1.5)


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
        "gemini-flash-lite-latest",
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

    Rate limits and overload (HTTP 429/502/503/504) are retried up to ``MAX_RETRIES``
    times, waiting for the server's ``retry-after`` or about 1, 2, 4, … s (jittered);
    a timeout or dropped connection is retried once. Parallel window extraction (#56)
    makes both more likely on free tiers. The client is safe to share between threads
    (``httpx.Client`` is).

    ``close()`` may be called from another thread to stop every request (Ctrl+C, #83):
    retry waits end at once, the HTTP client is closed so requests in flight fail fast,
    and requests raise LLMError from then on.
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
        sleep: Callable[[float], object] | None = None,
        jitter: Callable[[float], float] = _jitter,
    ) -> None:
        self.provider = provider
        self._closed = threading.Event()
        # Retry waits wait on the event that close() sets, so closing ends them.
        self._sleep = self._closed.wait if sleep is None else sleep
        self._jitter = jitter
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
        self,
        system: str,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        images: Sequence[bytes] = (),
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _user_content(prompt, images)},
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
        """Stop this client; safe to call from another thread (see the class docstring)."""
        self._closed.set()
        self._http.close()

    # --- helpers ---------------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        waited = 0.0
        timeouts = 0
        for attempt in range(1, MAX_RETRIES + 2):  # attempt n is retry n - 1
            self._check_open()
            try:
                response = self._http.post("chat/completions", json=body)
            except (httpx.HTTPError, RuntimeError) as e:  # RuntimeError: closed meanwhile
                self._check_open()
                if not isinstance(e, httpx.HTTPError):
                    raise
                if not isinstance(e, httpx.TimeoutException | httpx.RemoteProtocolError):
                    raise self._transport_error(e, retries=attempt - 1) from e
                wait = self._jitter(2.0 ** (attempt - 1))
                if timeouts >= TIMEOUT_RETRIES or waited + wait > MAX_TOTAL_WAIT_S:
                    raise self._transport_error(e, retries=attempt - 1) from e
                timeouts += 1
                reason = "request timed out" if isinstance(e, httpx.TimeoutException) else str(e)
            else:
                if response.status_code not in RETRY_STATUSES:
                    return response
                server_wait = _retry_after(response)
                wait = self._jitter(2.0 ** (attempt - 1)) if server_wait is None else server_wait
                if (
                    attempt > MAX_RETRIES
                    or wait > MAX_RETRY_WAIT_S  # a quota, not a burst: report it now
                    or waited + wait > MAX_TOTAL_WAIT_S
                ):
                    self._raise_for_status(response, retries=attempt - 1)
                reason = f"HTTP {response.status_code}"
            logger.warning(
                "%s: %s; retry %d/%d in %.1f s", self.provider, reason, attempt, MAX_RETRIES, wait
            )
            self._sleep(wait)
            waited += wait
        raise AssertionError("unreachable")

    def _check_open(self) -> None:
        if self._closed.is_set():
            raise LLMError(f"{self.provider} request cancelled: the client was closed")

    def _transport_error(self, e: httpx.HTTPError, *, retries: int) -> LLMError:
        gave_up = f" (gave up after {retries} retries)" if retries else ""
        if isinstance(e, httpx.ConnectError):
            return LLMError(f"cannot reach {self.provider} at {self._http.base_url}: {e}")
        if isinstance(e, httpx.TimeoutException):
            return LLMError(
                f"{self.provider} request timed out after {self._http.timeout.read:g}s "
                f"(model {self.model!r}){gave_up}; raise llm.timeout_s"
            )
        return LLMError(f"{self.provider} request failed: {e}{gave_up}")

    def _raise_for_status(self, response: httpx.Response, *, retries: int = 0) -> None:
        if not response.is_error:
            return
        detail = _error_message(response)
        if retries:
            detail += f" (gave up after {retries} retries)"
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
            wait = _retry_after(response)
            hint = f"; retry after {math.ceil(wait)}s" if wait is not None else ""
            raise LLMError(f"{self.provider} rate limit reached (HTTP 429){hint}: {detail}")
        raise LLMError(f"{self.provider} returned HTTP {status}: {detail}")


def _retry_after(response: httpx.Response) -> float | None:
    """``retry-after`` in seconds: either form, delay-seconds or an HTTP date."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


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


def _user_content(prompt: str, images: Sequence[bytes]) -> str | list[dict[str, Any]]:
    """The user message: plain text, or (with ``images``) OpenAI-style content parts, the
    text first and each JPEG as a data URL."""
    if not images:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        data = base64.b64encode(image).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return parts
