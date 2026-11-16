"""Live smoke test against a real LLM provider. Skipped unless ``ECHO_LIVE_LLM`` is set.

    ECHO_LIVE_LLM=ollama uv run pytest tests/test_llm_live.py
    ECHO_LIVE_LLM=groq GROQ_API_KEY=... uv run pytest tests/test_llm_live.py

``ECHO_LIVE_LLM_MODEL`` overrides the provider's default model. Keys (and these
variables) can also come from a local ``.env``. It is *read*, never loaded into
``os.environ``, because pytest imports this module while collecting every test, so
loading it here would leak real keys into the whole session.
"""

import json
import os

import pytest
from dotenv import dotenv_values, find_dotenv

from echo.config import LLMConfig
from echo.llm.client import make_llm_client

# Shell variables win over .env, as in the CLI.
ENV: dict[str, str] = {
    **{k: v for k, v in dotenv_values(find_dotenv(usecwd=True)).items() if v is not None},
    **os.environ,
}
PROVIDER = ENV.get("ECHO_LIVE_LLM")

pytestmark = pytest.mark.skipif(not PROVIDER, reason="set ECHO_LIVE_LLM=<provider> to run")


def test_live_provider_returns_json():
    overrides = {}
    if model := ENV.get("ECHO_LIVE_LLM_MODEL"):
        overrides["model"] = model
    client = make_llm_client(LLMConfig(provider=PROVIDER or "ollama", **overrides), env=ENV)
    schema = {
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
    }
    raw = client.complete(
        "You answer with JSON only.",
        'What is the capital of France? Answer as {"capital": "<city>"}.',
        json_schema=schema,
    )
    assert "paris" in json.loads(raw)["capital"].lower()
