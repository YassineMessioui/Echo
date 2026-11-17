import json

import httpx
import pytest

from echo.config import LLMConfig
from echo.errors import LLMError
from echo.interfaces import LLMClient
from echo.llm.client import make_llm_client
from echo.llm.hosted import API_KEY_ENV, PROVIDERS, OpenAICompatibleClient

SCHEMA = {"type": "object"}
SLEPT: list[float] = []


def _ok(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}
    )


def _client(handler, config: LLMConfig | None = None, env: dict | None = None):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = OpenAICompatibleClient.from_config(
        config or LLMConfig(provider="groq"),
        env={"GROQ_API_KEY": "gsk-test"} if env is None else env,
        transport=httpx.MockTransport(record),
    )
    client._sleep = SLEPT.append  # no real waiting on retries
    client._jitter = lambda seconds: seconds  # exact backoffs in tests
    return client, seen


# --- configuration ------------------------------------------------------------------------


def test_preset_defaults_replace_ollama_defaults():
    client, seen = _client(lambda r: _ok("hi"))
    assert isinstance(client, LLMClient)
    assert client.model_name == "groq/openai/gpt-oss-120b"
    client.complete("s", "p")
    assert str(seen[0].url) == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer gsk-test"


def test_explicit_model_and_host_are_kept():
    config = LLMConfig(provider="groq", model="qwen/qwen3-32b", host="https://proxy.test/v1")
    client, seen = _client(lambda r: _ok("hi"), config)
    client.complete("s", "p")
    assert client.model_name == "groq/qwen/qwen3-32b"
    assert str(seen[0].url) == "https://proxy.test/v1/chat/completions"
    assert json.loads(seen[0].content)["model"] == "qwen/qwen3-32b"


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_preset_builds(name: str):
    preset = PROVIDERS[name]
    client = make_llm_client(LLMConfig(provider=name), env={preset.key_env: "k"})
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model_name == f"{name}/{preset.default_model}"


def test_generic_key_overrides_provider_key():
    client, seen = _client(lambda r: _ok(""), env={"GROQ_API_KEY": "a", API_KEY_ENV: "b"})
    client.complete("s", "p")
    assert seen[0].headers["authorization"] == "Bearer b"


def test_missing_api_key_explains_where_to_get_one():
    with pytest.raises(LLMError, match=r"set GROQ_API_KEY.*console\.groq\.com"):
        _client(lambda r: _ok(""), env={})


def test_openai_compatible_server_without_key():
    config = LLMConfig(provider="openai-compatible", host="http://localhost:1234/v1", model="m")
    client, seen = _client(lambda r: _ok("hi"), config, env={})
    assert client.complete("s", "p") == "hi"
    assert str(seen[0].url) == "http://localhost:1234/v1/chat/completions"
    assert "authorization" not in seen[0].headers


def test_make_llm_client_still_defaults_to_ollama():
    from echo.llm.client import OllamaClient

    assert isinstance(make_llm_client(LLMConfig(), env={}), OllamaClient)
    with pytest.raises(LLMError, match="supported: 'ollama', 'groq'"):
        make_llm_client(LLMConfig(provider="nope"), env={})


# --- requests -----------------------------------------------------------------------------


def test_request_body_and_json_mode():
    client, seen = _client(lambda r: _ok('{"topics": []}'))
    assert client.complete("be brief", "hello", json_schema=SCHEMA) == '{"topics": []}'
    body = json.loads(seen[0].content)
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}


def test_no_json_mode_without_schema():
    client, seen = _client(lambda r: _ok("plain"))
    client.complete("s", "p")
    assert "response_format" not in json.loads(seen[0].content)


def test_falls_back_when_json_mode_is_unsupported():
    def handler(request: httpx.Request) -> httpx.Response:
        if "response_format" in json.loads(request.content):
            return httpx.Response(
                400, json={"error": {"message": "response_format is not supported by this model"}}
            )
        return _ok("{}")

    client, seen = _client(handler)
    assert client.complete("s", "p", json_schema=SCHEMA) == "{}"
    assert client.complete("s", "p", json_schema=SCHEMA) == "{}"
    assert len(seen) == 3, "JSON mode is only tried once"


# --- errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx.Response(401, json={"error": {"message": "Invalid API Key"}}),
            "rejected the API key.*Invalid API Key.*GROQ_API_KEY",
        ),
        (
            httpx.Response(404, json={"error": {"message": "model does not exist"}}),
            "no model 'openai/gpt-oss-120b'.*Set llm.model",
        ),
        (
            # Longer than MAX_RATE_LIMIT_WAIT_S (a quota): raised without waiting.
            httpx.Response(
                429, headers={"retry-after": "120"}, json={"error": {"message": "slow"}}
            ),
            r"rate limit reached \(HTTP 429\); retry after 120s: slow",
        ),
        (
            httpx.Response(400, json=[{"error": {"message": "Please pass a valid API key."}}]),
            "rejected the API key",
        ),
        (
            httpx.Response(401, json={"detail": "Invalid API Key"}),
            r"\(HTTP 401\): Invalid API Key\. Check",
        ),
        (httpx.Response(500, text="boom"), "HTTP 500: boom"),
        (
            httpx.Response(400, json=[{"error": {"code": 400, "message": "bad request"}}]),
            "HTTP 400: bad request",
        ),
        (httpx.Response(200, json={"choices": []}), "unexpected groq response"),
        (httpx.Response(200, text="<html>"), "unexpected groq response"),
    ],
)
def test_error_responses(response: httpx.Response, message: str):
    client, _ = _client(lambda r: response)
    with pytest.raises(LLMError, match=message):
        client.complete("s", "p")


def test_connection_and_timeout_errors():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="cannot reach groq"):
        _client(refuse)[0].complete("s", "p")
    with pytest.raises(LLMError, match="timed out after 120s"):
        _client(slow)[0].complete("s", "p")


# --- rate limits (#56) --------------------------------------------------------------------


def _limited(times: int, retry_after: str | None):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= times:
            headers = {"retry-after": retry_after} if retry_after else {}
            return httpx.Response(429, headers=headers, json={"error": {"message": "slow"}})
        return _ok("{}")

    return handler


def test_429_waits_for_retry_after_then_succeeds():
    SLEPT.clear()
    client, seen = _client(_limited(2, "1.5"))
    assert client.complete("s", "p") == "{}"
    assert len(seen) == 3
    assert SLEPT == [1.5, 1.5]


def test_429_without_retry_after_backs_off_exponentially():
    SLEPT.clear()
    client, _ = _client(_limited(3, None))
    assert client.complete("s", "p") == "{}"
    assert SLEPT == [1.0, 2.0, 4.0]


def test_429_gives_up_after_the_retry_budget():
    from echo.llm.hosted import MAX_RETRIES

    SLEPT.clear()
    client, seen = _client(_limited(100, "2"))
    with pytest.raises(LLMError, match=r"retry after 2s: slow \(gave up after 5 retries\)"):
        client.complete("s", "p")
    assert len(seen) == MAX_RETRIES + 1
    assert SLEPT == [2.0] * MAX_RETRIES


@pytest.mark.parametrize("status", [502, 503, 504])
def test_overloaded_provider_is_retried(status: int):
    """Gemini's free tier: 503 "model is currently experiencing high demand"."""
    SLEPT.clear()
    responses = [httpx.Response(status, json={"error": {"message": "high demand"}}), _ok("{}")]
    client, seen = _client(lambda r: responses.pop(0))
    assert client.complete("s", "p") == "{}"
    assert len(seen) == 2 and SLEPT == [1.0]


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("server disconnected")]
)
def test_timeout_or_dropped_connection_is_retried_once(error: httpx.HTTPError):
    SLEPT.clear()
    outcomes: list = [error, _ok("{}")]

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client, _ = _client(handler)
    assert client.complete("s", "p") == "{}"
    assert SLEPT == [1.0]


def test_second_timeout_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client, seen = _client(handler)
    with pytest.raises(LLMError, match=r"timed out.*gave up after 1 retries.*llm.timeout_s"):
        client.complete("s", "p")
    assert len(seen) == 2


def test_retry_after_as_http_date():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    SLEPT.clear()
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    client, _ = _client(_limited(1, when))
    assert client.complete("s", "p") == "{}"
    assert 28 <= SLEPT[0] <= 30


def test_total_wait_is_capped():
    from echo.llm.hosted import MAX_TOTAL_WAIT_S

    SLEPT.clear()
    client, seen = _client(_limited(100, "50"))  # each wait allowed, but not all of them
    with pytest.raises(LLMError, match="gave up after 2 retries"):
        client.complete("s", "p")
    assert SLEPT == [50.0, 50.0] and sum(SLEPT) <= MAX_TOTAL_WAIT_S
    assert len(seen) == 3


def test_429_on_json_mode_is_retried_before_the_json_fallback():
    SLEPT.clear()
    client, seen = _client(_limited(1, "0"))
    assert client.complete("s", "p", json_schema=SCHEMA) == "{}"
    assert [("response_format" in json.loads(r.content)) for r in seen] == [True, True]
