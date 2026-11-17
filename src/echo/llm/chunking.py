"""Split a transcript into overlapping windows that fit the LLM's context budget.

Each segment is rendered as one numbered line::

    [12] 00:03:41–00:03:47 text...

The LLM refers to segments by these numbers, so a window never splits a segment.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from echo.config import AUTO_MAX_TOKENS, ChunkConfig
from echo.models import Segment, Transcript, TranscriptWindow

TokenCounter = Callable[[str], int]


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

    Windows are filled greedily up to ``config.max_tokens``; each window after the
    first starts ``config.overlap_segments`` segments before the previous one ended.
    A segment that alone exceeds the budget gets a window of its own. The overlap is
    smaller than configured only when the previous window has too few segments to
    overlap by that much and still make progress.
    """

    def __init__(self, count_tokens: TokenCounter = approx_tokens) -> None:
        self.count_tokens = count_tokens

    def __call__(self, transcript: Transcript, config: ChunkConfig) -> list[TranscriptWindow]:
        budget = AUTO_MAX_TOKENS if config.max_tokens == "auto" else config.max_tokens
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
