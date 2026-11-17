"""Split a transcript into overlapping windows that fit the LLM's context budget.

Each segment is rendered as one numbered line::

    [12] 00:03:41–00:03:47 text...

The LLM refers to segments by these numbers, so a window never splits a segment.

``ChunkConfig.max_tokens = "auto"`` (the default) is resolved per provider and model
by ``resolve_max_tokens``: see ``LONG_CONTEXT_MODELS``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from echo.config import AUTO_MAX_TOKENS, ChunkConfig, LLMConfig
from echo.llm.hosted import GENERIC_PROVIDER, PROVIDERS
from echo.models import Segment, Transcript, TranscriptWindow

TokenCounter = Callable[[str], int]

LONG_CONTEXT_MAX_TOKENS = 12_000
"""``"auto"`` budget for hosted models with a context of 128k tokens or more.

Not as big as the context allows: on a 68-min talk with ``gemini-flash-lite-latest``
(#63), one 24k window missed a whole 25-min stretch and misplaced spans, while 12k
windows (4x fewer calls than 3k) still covered the whole talk with correct spans."""

LONG_CONTEXT_MODELS: tuple[str, ...] = (
    "openai/gpt-oss-",  # Groq, OpenRouter: 131k
    "gpt-oss-",
    "gemini-",  # Gemini 2.x/flash/flash-lite/pro, *-latest aliases: 1M
    "google/gemini-",  # through OpenRouter
    "meta-llama/llama-3.3-70b",  # OpenRouter: 128k
    "llama-3.3-70b",  # Groq: 128k
    "mistral-small",  # mistral-small-latest (3.x): 128k
    "mistral-medium",  # 128k
    "mistral-large",  # 128k
    "gpt-4.1",  # gpt-4.1, -mini, -nano: 1M
    "gpt-4o",  # 128k
)
"""Model-name prefixes (lower case) that get ``LONG_CONTEXT_MAX_TOKENS`` under
``"auto"`` on a hosted provider. Anything else gets ``AUTO_MAX_TOKENS``."""

_LLM_DEFAULTS = LLMConfig()


def resolve_max_tokens(chunk: ChunkConfig, llm: LLMConfig | None = None) -> int:
    """The window budget for ``chunk`` with the model of ``llm``.

    An explicit ``chunk.max_tokens`` wins. ``"auto"`` gives ``LONG_CONTEXT_MAX_TOKENS``
    for a hosted provider (one of ``echo.llm.hosted.PROVIDERS`` or
    ``openai-compatible``) whose model starts with one of ``LONG_CONTEXT_MODELS``, and
    ``AUTO_MAX_TOKENS`` otherwise: Ollama and other local models (small models, and
    Ollama's default context is a few thousand tokens), unknown models and providers,
    or no ``llm`` at all. As in ``OpenAICompatibleClient.from_config``, a hosted
    provider left on the Ollama default model uses the provider's default model.
    """
    if chunk.max_tokens != "auto":
        return chunk.max_tokens
    if llm is None or llm.provider == "ollama":
        return AUTO_MAX_TOKENS
    model = llm.model
    if llm.provider in PROVIDERS and model == _LLM_DEFAULTS.model:
        model = PROVIDERS[llm.provider].default_model
    elif llm.provider not in PROVIDERS and llm.provider != GENERIC_PROVIDER:
        return AUTO_MAX_TOKENS
    if model.lower().startswith(LONG_CONTEXT_MODELS):
        return LONG_CONTEXT_MAX_TOKENS
    return AUTO_MAX_TOKENS


def approx_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English)."""
    return math.ceil(len(text) / 4)


def format_timestamp(seconds: float) -> str:
    """``HH:MM:SS``, truncating fractions of a second."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_segment(segment: Segment) -> str:
    start, end = format_timestamp(segment.start), format_timestamp(segment.end)
    return f"[{segment.id}] {start}–{end} {segment.text.strip()}"


class WindowChunker:
    """Implements ``echo.interfaces.Chunker``.

    Windows are filled greedily up to the budget (``resolve_max_tokens(config, llm)``,
    so ``"auto"`` follows the model of ``llm``); each window after the
    first starts ``config.overlap_segments`` segments before the previous one ended.
    A segment that alone exceeds the budget gets a window of its own. The overlap is
    smaller than configured only when the previous window has too few segments to
    overlap by that much and still make progress.
    """

    def __init__(
        self, count_tokens: TokenCounter = approx_tokens, llm: LLMConfig | None = None
    ) -> None:
        self.count_tokens = count_tokens
        self.llm = llm

    def __call__(self, transcript: Transcript, config: ChunkConfig) -> list[TranscriptWindow]:
        budget = resolve_max_tokens(config, self.llm)
        lines = [format_segment(seg) for seg in transcript.segments]
        # Count each line with its trailing newline: for a sub-additive counter
        # (like chars/4 rounded up) the sum never underestimates the joined text.
        costs = [self.count_tokens(line + "\n") for line in lines]

        windows: list[TranscriptWindow] = []
        start = 0
        while start < len(lines):
            end, used = start, costs[start]
            while end + 1 < len(lines) and used + costs[end + 1] <= budget:
                end += 1
                used += costs[end]
            windows.append(
                TranscriptWindow(
                    start_segment=start, end_segment=end, text="\n".join(lines[start : end + 1])
                )
            )
            if end == len(lines) - 1:
                break
            start = max(end + 1 - config.overlap_segments, start + 1)
        return windows


chunk_transcript = WindowChunker()
"""Default chunker, using the chars/4 token estimate."""
