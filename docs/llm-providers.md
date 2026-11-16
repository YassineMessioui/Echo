# LLM providers

Topic extraction talks to an `LLMClient`, so the model can run **locally** (Ollama)
or on a **hosted API**. Pick one with `llm.provider` in `PipelineConfig`
(`echo.config.LLMConfig`):

| `llm.provider` | Where it runs | API key variable | Default model |
|---|---|---|---|
| `ollama` (default) | your machine, `llm.host` (default `http://localhost:11434`) | – | `llama3.2:3b` |
| `groq` | [Groq](https://console.groq.com/keys) | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `openrouter` | [OpenRouter](https://openrouter.ai/settings/keys) | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` |
| `gemini` | [Google AI Studio](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| `mistral` | [Mistral](https://console.mistral.ai/api-keys) | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `openai` | [OpenAI](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| `openai-compatible` | any OpenAI-compatible server at `llm.host` (LM Studio, vLLM, …) | optional | set `llm.model` |

- `ECHO_LLM_API_KEY` overrides the provider-specific variable.
- Keys are read from the environment. For convenience, `echo-clips` also loads a
  `.env` file (git-ignored; start from `cp .env.example .env`). Variables set in your
  shell win over `.env`. Keys never go in config files, so they can't end up in git.
- Set `llm.model` to use a different model from the same provider; set `llm.host`
  to go through a proxy.
- Groq, OpenRouter, Gemini and Mistral have free tiers with rate limits. Free tiers
  change often, so check the provider's page. Some (e.g. Gemini's free tier) may
  use your prompts to improve their models: transcripts are sent to the provider.

## Local: Ollama

```bash
# install from https://ollama.com (Windows: winget install Ollama.Ollama)
ollama pull llama3.2:3b
```

## Check a provider works

```bash
ECHO_LIVE_LLM=ollama uv run pytest tests/test_llm_live.py
ECHO_LIVE_LLM=groq GROQ_API_KEY=... uv run pytest tests/test_llm_live.py
```

`ECHO_LIVE_LLM_MODEL` overrides the default model for that run.
