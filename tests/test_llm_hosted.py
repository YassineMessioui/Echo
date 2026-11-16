import json

import httpx
import pytest

from echo.config import LLMConfig
from echo.errors import LLMError
from echo.interfaces import LLMClient
from echo.llm.client import make_llm_client
from echo.llm.hosted import API_KEY_ENV, PROVIDERS, OpenAICompatibleClient

SCHEMA = {"type": "object"}


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
    return client, seen


# --- configuration ------------------------------------------------------------------------


def test_preset_defaults_replace_ollama_defaults():
    client, seen = _client(lambda r: _ok("hi"))
    assert isinstance(client, LLMClient)
    assert client.model_name == "groq/llama-3.3-70b-versatile"
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
            "no model 'llama-3.3-70b-versatile'.*Set llm.model",
        ),
        (
            httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow"}}),
            r"rate limit reached \(HTTP 429\); retry after 7s: slow",
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
