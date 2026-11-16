import pytest

from echo.config import ChunkConfig
from echo.job import Job
from echo.llm.chunking import (
    WindowChunker,
    approx_tokens,
    chunk_transcript,
    format_segment,
    format_timestamp,
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
