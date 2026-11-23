import json

import httpx
import pytest

from echo.config import LLMConfig
from echo.errors import LLMError
from echo.interfaces import LLMClient
from echo.llm.client import OllamaClient, make_llm_client

CONFIG = LLMConfig(model="llama3.2:3b", host="http://ollama.test:11434", temperature=0.3)


def _client(handler) -> tuple[OllamaClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return OllamaClient(CONFIG, transport=httpx.MockTransport(record)), seen


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


def test_is_an_llm_client():
    client, _ = _client(lambda r: _ok(""))
    assert isinstance(client, LLMClient)
    assert client.model_name == "ollama/llama3.2:3b"


def test_complete_sends_chat_request():
    client, seen = _client(lambda r: _ok('{"topics": []}'))
    schema = {"type": "object"}
    assert client.complete("be brief", "hello", json_schema=schema) == '{"topics": []}'

    (request,) = seen
    assert request.url == "http://ollama.test:11434/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "llama3.2:3b"
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert body["stream"] is False
    assert body["format"] == schema
    assert body["options"] == {"temperature": 0.3}


def test_complete_without_schema_omits_format():
    client, seen = _client(lambda r: _ok("plain"))
    assert client.complete("s", "p") == "plain"
    assert "format" not in json.loads(seen[0].content)


def test_connection_refused():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    client, _ = _client(refuse)
    with pytest.raises(LLMError, match="cannot reach Ollama.*ollama serve"):
        client.complete("s", "p")


def test_model_not_found():
    client, _ = _client(
        lambda r: httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})
    )
    with pytest.raises(LLMError, match=r"ollama pull llama3\.2:3b"):
        client.complete("s", "p")


def test_timeout():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, _ = _client(slow)
    with pytest.raises(LLMError, match="timed out after 120s"):
        client.complete("s", "p")


def test_server_error():
    client, _ = _client(lambda r: httpx.Response(500, json={"error": "out of memory"}))
    with pytest.raises(LLMError, match="HTTP 500: out of memory"):
        client.complete("s", "p")


@pytest.mark.parametrize(
    "payload", [b"not json", b'{"done": true}', b'{"message": {"content": 3}}']
)
def test_malformed_response(payload: bytes):
    client, _ = _client(lambda r: httpx.Response(200, content=payload))
    with pytest.raises(LLMError, match="unexpected Ollama response"):
        client.complete("s", "p")


def test_make_llm_client():
    assert isinstance(make_llm_client(LLMConfig()), OllamaClient)
    with pytest.raises(LLMError, match="unknown LLM provider"):
        make_llm_client(LLMConfig(provider="nope"))


def test_images_go_in_the_user_message():
    """#129: Ollama takes base64 images on the message (vision models like llava)."""
    client, seen = _client(lambda r: _ok("ok"))
    assert client.complete("sys", "describe", images=[b"\xff\xd8one"]) == "ok"
    user = json.loads(seen[0].content)["messages"][1]
    assert user == {"role": "user", "content": "describe", "images": ["/9hvbmU="]}


def test_fake_client_records_images():
    from echo.testing import FakeLLMClient

    fake = FakeLLMClient(["a", "b"])
    fake.complete("s", "p", images=[b"x"])
    fake.complete("s", "p")
    assert [c.images for c in fake.calls] == [(b"x",), ()]
