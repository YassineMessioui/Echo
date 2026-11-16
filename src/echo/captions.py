"""Burned-in captions: word-timed subtitle tracks for each clip (#17).

Short-form captions show a few words at a time and highlight the word being spoken.
``build_ass`` turns the transcript's word timestamps inside one clip into an ASS
subtitle track (times relative to the clip start); ``subtitles_filter`` gives the
ffmpeg filter that burns it in, so the clip stage can chain it after the 9:16
reframe filter in the same encode::

    vf = chain_filters(reframe.video_filter(w, h, cfg), subtitles_filter(ass_path))

Words are grouped into short lines (``max_words``), broken early at sentence
punctuation or pauses, and each line is shown as one event per word, with that word
in ``highlight_colour``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from echo.config import CaptionConfig
from echo.models import ClipSpec, Transcript, Word

PLAY_RES = (1080, 1920)
"""Caption coordinates are laid out for the 9:16 output; libass scales them otherwise."""


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial"
    font_size: int = 84
    colour: str = "#FFFFFF"
    highlight_colour: str = "#FFD400"
    outline_colour: str = "#000000"
    outline: int = 6
    margin_bottom: int = 420
    """Distance from the bottom edge, in 1920-high pixels (keeps clear of app UI)."""
    max_words: int = 3
    """Words on screen at once."""
    max_gap: float = 0.6
    """A pause longer than this (seconds) starts a new line."""
    uppercase: bool = True

    @classmethod
    def from_config(cls, config: CaptionConfig) -> CaptionStyle:
        """The style for ``PipelineConfig.caption`` (``enabled`` is the caller's concern)."""
        return cls(**config.model_dump(exclude={"enabled"}))


@dataclass(frozen=True)
class _TimedWord:
    start: float
    end: float
    text: str


# --- grouping ---------------------------------------------------------------------------


def clip_words(transcript: Transcript, clip: ClipSpec) -> list[_TimedWord]:
    """Words whose midpoint lies inside ``clip``, with times relative to the clip start and
    clamped to it. (The clip's padding often grazes the previous sentence's last word; a
    word mostly outside the clip isn't shown.)"""
    words: list[_TimedWord] = []
    for segment in transcript.segments:
        if segment.end <= clip.start or segment.start >= clip.end:
            continue
        for word in segment.words:
            middle = (word.start + word.end) / 2
            if not clip.start <= middle < clip.end or not word.text.strip():
                continue
            words.append(_shift(word, clip))
    return words


def _shift(word: Word, clip: ClipSpec) -> _TimedWord:
    start = max(word.start, clip.start) - clip.start
    end = min(word.end, clip.end) - clip.start
    return _TimedWord(round(start, 3), round(max(end, start), 3), word.text.strip())


def group_lines(words: list[_TimedWord], style: CaptionStyle) -> list[list[_TimedWord]]:
    """Split words into caption lines of at most ``max_words``, breaking after sentence
    punctuation or before a pause longer than ``max_gap``."""
    lines: list[list[_TimedWord]] = []
    line: list[_TimedWord] = []
    for word in words:
        if line and (
            len(line) >= style.max_words
            or word.start - line[-1].end > style.max_gap
            or line[-1].text[-1] in ".?!"
        ):
            lines.append(line)
            line = []
        line.append(word)
    if line:
        lines.append(line)
    return lines


# --- ASS output -------------------------------------------------------------------------


def _ass_time(seconds: float) -> str:
    """``H:MM:SS.cc`` (centiseconds), the ASS timestamp format."""
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_colour(hex_rgb: str) -> str:
    """``#RRGGBB`` → ASS ``&H00BBGGRR``."""
    value = hex_rgb.lstrip("#")
    try:
        int(value, 16)
    except ValueError:
        value = ""
    if len(value) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_rgb!r}")
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ass_text(text: str, style: CaptionStyle) -> str:
    # Braces start override blocks and backslashes start escapes in ASS text.
    cleaned = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    return cleaned.upper() if style.uppercase else cleaned


def _header(style: CaptionStyle) -> str:
    width, height = PLAY_RES
    colour = _ass_colour(style.colour)
    outline = _ass_colour(style.outline_colour)
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{style.font},{style.font_size},{colour},{colour},{outline},"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline},0,2,60,60,"
        f"{style.margin_bottom},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _events(lines: Iterable[list[_TimedWord]], style: CaptionStyle) -> list[str]:
    highlight = _ass_colour(style.highlight_colour)
    events: list[str] = []
    for line in lines:
        for i, word in enumerate(line):
            # Each word's event lasts until the next word starts (no flicker in between);
            # the last one until the word ends.
            end = line[i + 1].start if i + 1 < len(line) else word.end
            if end <= word.start:
                continue
            parts = [
                f"{{\\c{highlight}}}{_ass_text(w.text, style)}{{\\r}}"
                if j == i
                else _ass_text(w.text, style)
                for j, w in enumerate(line)
            ]
            events.append(
                f"Dialogue: 0,{_ass_time(word.start)},{_ass_time(end)},Caption,,0,0,0,,"
                + " ".join(parts)
            )
    return events


def build_ass(transcript: Transcript, clip: ClipSpec, style: CaptionStyle | None = None) -> str:
    """The ASS subtitle track for ``clip``, timed relative to the clip's start."""
    style = style or CaptionStyle()
    lines = group_lines(clip_words(transcript, clip), style)
    return _header(style) + "".join(e + "\n" for e in _events(lines, style))


def write_captions(
    transcript: Transcript,
    clips: list[ClipSpec],
    outputs: list[Path],
    style: CaptionStyle | None = None,
) -> None:
    """Write one ``.ass`` file per clip (UTF-8 with BOM, which libass reads everywhere)."""
    if len(clips) != len(outputs):
        raise ValueError(f"got {len(clips)} clips but {len(outputs)} output paths")
    for clip, out in zip(clips, outputs, strict=True):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_ass(transcript, clip, style), encoding="utf-8-sig")


# --- ffmpeg -----------------------------------------------------------------------------


def _escape_option(value: str) -> str:
    """First escaping level (filter option value): ``\\``, ``'`` and ``:``."""
    for char in "\\':":
        value = value.replace(char, "\\" + char)
    return value


def _escape_graph(value: str) -> str:
    """Second escaping level (filtergraph description): ``\\ ' [ ] , ;``."""
    for char in "\\'[],;":
        value = value.replace(char, "\\" + char)
    return value


def subtitles_filter(ass_path: Path) -> str:
    """``subtitles`` filter for ``-vf`` burning ``ass_path`` in; safe for Windows paths
    (``C:\\...``) and names with spaces, quotes, commas or brackets."""
    return "subtitles=filename=" + _escape_graph(_escape_option(ass_path.as_posix()))


def chain_filters(*filters: str | None) -> str | None:
    """Join video filters into one ``-vf`` chain, skipping ``None``; None if all are."""
    present = [f for f in filters if f]
    return ",".join(present) if present else None
