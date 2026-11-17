import pytest

from echo.config import AUTO_MAX_TOKENS, ChunkConfig, LLMConfig
from echo.job import Job
from echo.llm.chunking import (
    LONG_CONTEXT_MAX_TOKENS,
    WindowChunker,
    approx_tokens,
    chunk_transcript,
    format_segment,
    format_timestamp,
    resolve_max_tokens,
)
from echo.models import Segment, Transcript, TranscriptWindow


def _transcript(texts: list[str], seg_len: float = 5.0) -> Transcript:
    segments = [
        Segment(id=i, start=i * seg_len, end=(i + 1) * seg_len, text=text)
        for i, text in enumerate(texts)
    ]
    return Transcript(language="en", duration=len(texts) * seg_len, segments=segments)


def _ids(window: TranscriptWindow) -> set[int]:
    return set(range(window.start_segment, window.end_segment + 1))


def _check_invariants(
    transcript: Transcript, windows: list[TranscriptWindow], config: ChunkConfig
) -> None:
    covered = set().union(*(_ids(w) for w in windows))
    assert covered == set(range(len(transcript.segments))), "every segment is in a window"
    for prev, cur in zip(windows, windows[1:], strict=False):
        assert prev.start_segment < cur.start_segment, "windows are ordered and progress"
        assert cur.start_segment <= prev.end_segment + 1, "no gaps between windows"
    for w in windows:
        lines = w.text.split("\n")
        assert len(lines) == w.end_segment - w.start_segment + 1, "whole segments only"
        assert lines[0].startswith(f"[{w.start_segment}] ")
        assert lines[-1].startswith(f"[{w.end_segment}] ")
        if len(lines) > 1:
            assert approx_tokens(w.text) <= config.max_tokens


def test_format_segment():
    seg = Segment(id=12, start=221.4, end=227.9, text="  hello there ")
    assert format_segment(seg) == "[12] 00:03:41–00:03:47 hello there"
    assert format_timestamp(3725.99) == "01:02:05"


def test_short_transcript_is_one_window(golden_job: Job):
    transcript = golden_job.read_transcript()
    windows = chunk_transcript(transcript, ChunkConfig())
    assert len(windows) == 1
    assert (windows[0].start_segment, windows[0].end_segment) == (0, 7)
    assert windows[0].text.splitlines()[1].startswith("[1] 00:00:08–00:00:17 Let's start")


def test_empty_transcript_has_no_windows():
    assert chunk_transcript(Transcript(language="en", duration=0, segments=[]), ChunkConfig()) == []


@pytest.mark.parametrize("overlap", [0, 1, 2, 3])
def test_consecutive_windows_overlap_by_configured_amount(overlap: int):
    # Each line "[i] 00:00:..–00:00:.. xxxxxxxx…" costs 10 tokens; a budget of 50 fits 5.
    transcript = _transcript(["x" * 11] * 20)
    chunker = WindowChunker(count_tokens=lambda _: 10)
    config = ChunkConfig(max_tokens=50, overlap_segments=overlap)
    windows = chunker(transcript, config)

    assert all(w.end_segment - w.start_segment + 1 <= 5 for w in windows)
    for prev, cur in zip(windows, windows[1:], strict=False):
        assert prev.end_segment - cur.start_segment + 1 == overlap
    assert windows[-1].end_segment == 19


def test_no_window_exceeds_budget():
    texts = [("word " * n).strip() for n in [3, 40, 7, 12, 90, 1, 25, 60, 5, 33] * 5]
    transcript = _transcript(texts)
    config = ChunkConfig(max_tokens=200, overlap_segments=2)
    windows = chunk_transcript(transcript, config)
    assert len(windows) > 1
    _check_invariants(transcript, windows, config)


def test_oversized_segment_gets_its_own_window():
    transcript = _transcript(["short", "y" * 2000, "short", "short"])
    config = ChunkConfig(max_tokens=100, overlap_segments=1)
    windows = chunk_transcript(transcript, config)
    assert [(w.start_segment, w.end_segment) for w in windows] == [(0, 0), (1, 1), (2, 3)]
    _check_invariants(transcript, windows, config)


def test_overlap_larger_than_window_still_progresses():
    transcript = _transcript(["z" * 40] * 10)
    config = ChunkConfig(max_tokens=40, overlap_segments=10)
    windows = chunk_transcript(transcript, config)
    _check_invariants(transcript, windows, config)
    assert len(windows) < 20


def test_pluggable_token_counter():
    transcript = _transcript(["a", "b", "c", "d"])
    chunker = WindowChunker(count_tokens=lambda _: 1)
    windows = chunker(transcript, ChunkConfig(max_tokens=2, overlap_segments=0))
    assert [(w.start_segment, w.end_segment) for w in windows] == [(0, 1), (2, 3)]


# --- "auto" window budget (#63) ---------------------------------------------------

LONG, SHORT = LONG_CONTEXT_MAX_TOKENS, AUTO_MAX_TOKENS


@pytest.mark.parametrize(
    ("llm", "expected"),
    [
        ({}, SHORT),  # Ollama defaults
        ({"provider": "ollama", "model": "llama3.3:70b"}, SHORT),  # local: always small
        ({"provider": "ollama", "model": "gpt-oss:120b"}, SHORT),
        # Hosted providers on their default model
        ({"provider": "groq"}, LONG),
        ({"provider": "openrouter"}, LONG),
        ({"provider": "gemini"}, LONG),
        ({"provider": "mistral"}, LONG),
        ({"provider": "openai"}, LONG),
        # Hosted, explicit models
        ({"provider": "groq", "model": "openai/gpt-oss-120b"}, LONG),
        ({"provider": "groq", "model": "llama-3.1-8b-instant"}, SHORT),
        ({"provider": "gemini", "model": "gemini-flash-lite-latest"}, LONG),
        ({"provider": "gemini", "model": "gemini-2.5-flash-lite"}, LONG),
        ({"provider": "gemini", "model": "gemini-flash-latest"}, LONG),
        ({"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free"}, LONG),
        ({"provider": "openrouter", "model": "some/unknown-model"}, SHORT),
        ({"provider": "mistral", "model": "mistral-small-latest"}, LONG),
        ({"provider": "mistral", "model": "open-mistral-7b"}, SHORT),
        ({"provider": "openai", "model": "gpt-4.1-mini"}, LONG),
        ({"provider": "openai", "model": "GPT-4.1-mini"}, LONG),  # case-insensitive
        # Generic server: small unless the model is known
        ({"provider": "openai-compatible", "model": "llama3.2:3b"}, SHORT),
        ({"provider": "openai-compatible", "model": "local-model"}, SHORT),
        ({"provider": "openai-compatible", "model": "openai/gpt-oss-120b"}, LONG),
        # Unknown provider
        ({"provider": "nope", "model": "gpt-4.1-mini"}, SHORT),
    ],
)
def test_auto_resolves_per_provider_and_model(llm: dict, expected: int):
    assert resolve_max_tokens(ChunkConfig(), LLMConfig(**llm)) == expected


def test_auto_without_llm_config_is_the_small_budget():
    assert resolve_max_tokens(ChunkConfig()) == AUTO_MAX_TOKENS == 3000
    assert LONG_CONTEXT_MAX_TOKENS == 12_000


@pytest.mark.parametrize("llm", [{}, {"provider": "groq"}, {"provider": "gemini"}])
def test_explicit_max_tokens_wins(llm: dict):
    assert resolve_max_tokens(ChunkConfig(max_tokens=1234), LLMConfig(**llm)) == 1234


def test_chunker_uses_the_resolved_budget():
    # 2000 segments of ~85 tokens each (~170k tokens): 64 windows at 3k, 15 at 12k.
    transcript = _transcript([("word " * 60).strip()] * 2000)
    small = WindowChunker(llm=LLMConfig())(transcript, ChunkConfig())
    big = WindowChunker(llm=LLMConfig(provider="gemini"))(transcript, ChunkConfig())
    explicit = WindowChunker(llm=LLMConfig(provider="gemini"))(
        transcript, ChunkConfig(max_tokens=3000)
    )

    assert [(w.start_segment, w.end_segment) for w in small] == [
        (w.start_segment, w.end_segment) for w in explicit
    ]
    assert len(small) == 64
    assert len(big) == 15
    for windows, budget in [(small, SHORT), (big, LONG)]:
        _check_invariants(transcript, windows, ChunkConfig(max_tokens=budget))
        for prev, cur in zip(windows, windows[1:], strict=False):
            assert prev.end_segment - cur.start_segment + 1 == 5, "overlap is kept"
