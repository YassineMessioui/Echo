"""Thin wrappers around the ffmpeg / ffprobe binaries, shared by the media layers.

Every failure surfaces as ``ExternalToolError`` (tool missing, non-zero exit) with the
tail of stderr, or ``InvalidInputError`` when the input file itself is the problem.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from echo.errors import ExternalToolError, InvalidInputError

STDERR_TAIL_LINES = 15


def require(tool: str) -> str:
    """Absolute path of ``tool`` (``ffmpeg`` or ``ffprobe``), or raise ExternalToolError."""
    path = shutil.which(tool)
    if path is None:
        raise ExternalToolError(f"{tool} not found on PATH; install ffmpeg (https://ffmpeg.org)")
    return path


def _tail(stderr: str) -> str:
    return "\n".join(stderr.strip().splitlines()[-STDERR_TAIL_LINES:])


def run(
    tool: str, args: list[str], *, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``tool`` with ``args``; raise ExternalToolError with the stderr tail on failure.

    Output is decoded as UTF-8 (what ffmpeg writes) rather than the locale encoding,
    which is cp1252 on Windows and would fail on non-ASCII paths or metadata.
    """
    cmd = [require(tool), *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ExternalToolError(f"{tool} timed out after {timeout} s") from e
    except OSError as e:
        raise ExternalToolError(f"could not run {tool}: {e}") from e
    if result.returncode != 0:
        raise ExternalToolError(
            f"{tool} exited with code {result.returncode}:\n{_tail(result.stderr)}"
        )
    return result


def ffmpeg(args: list[str], *, timeout: float | None = None) -> None:
    """Run ffmpeg quietly, overwriting outputs: ``ffmpeg -hide_banner -loglevel error -y ...``."""
    run("ffmpeg", ["-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], timeout=timeout)


@functools.cache
def _filters() -> frozenset[str]:
    """Names of the filters this ffmpeg build has (``ffmpeg -filters``), read once."""
    names = set()
    for line in run("ffmpeg", ["-hide_banner", "-filters"]).stdout.splitlines():
        parts = line.split()
        # " T.. subtitles  V->V  Render text subtitles ..." (flags, name, in->out, help)
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    """Whether this ffmpeg build has the ``name`` filter (e.g. ``subtitles`` needs libass)."""
    return name in _filters()


@functools.cache
def http_options() -> frozenset[str]:
    """Options of this ffmpeg's HTTP protocol (``ffmpeg -h protocol=http``), read once.

    They vary by version: e.g. ``request_size`` is missing from ffmpeg 8.0.
    """
    names = set()
    for line in run("ffmpeg", ["-hide_banner", "-h", "protocol=http"]).stdout.splitlines():
        # "  -request_size      <int64>      .D......... size (in bytes) of requests ..."
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("-") and parts[1].startswith("<"):
            names.add(parts[0][1:])
    return frozenset(names)


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    rotation: int = 0
    """Display rotation in degrees (0, 90, 180, 270); phones store portrait video this way."""
    sample_aspect_ratio: float = 1.0
    """Pixel shape (SAR); not 1 for anamorphic sources."""

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """(width, height) of the frames a ``-vf`` filter receives.

        ffmpeg auto-rotates before user filters, so ±90° rotations swap the stored size.
        """
        if self.width is None or self.height is None:
            return None
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height

    @property
    def frame_sar(self) -> float:
        """Sample aspect ratio of the frames a ``-vf`` filter receives (inverted by ±90°)."""
        return (
            1 / self.sample_aspect_ratio if self.rotation in (90, 270) else self.sample_aspect_ratio
        )


def _rotation(stream: dict) -> int:
    """Rotation from the display matrix side data, falling back to the legacy ``rotate`` tag."""
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return round(float(side["rotation"])) % 360
    return round(float(stream.get("tags", {}).get("rotate", 0))) % 360


def _sar(stream: dict) -> float:
    num, _, den = str(stream.get("sample_aspect_ratio", "1:1")).partition(":")
    try:
        sar = int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        return 1.0
    return sar if sar > 0 else 1.0  # "0:1" means unknown


def probe(path: Path) -> MediaInfo:
    """Inspect a media file with ffprobe. Raises InvalidInputError for missing/unreadable files."""
    if not path.is_file():
        raise InvalidInputError(f"input not found: {path}")
    try:
        result = run(
            "ffprobe",
            ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        )
    except ExternalToolError as e:
        if shutil.which("ffprobe") is None:
            raise
        raise InvalidInputError(f"not a readable media file: {path}\n{e}") from e
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    # Embedded cover art (e.g. in an .m4a) is a "video" stream with attached_pic set.
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return MediaInfo(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        has_audio=audio is not None,
        has_video=video is not None,
        width=int(video["width"]) if video else None,
        height=int(video["height"]) if video else None,
        sample_rate=int(audio["sample_rate"]) if audio else None,
        channels=int(audio["channels"]) if audio else None,
        rotation=_rotation(video) if video else 0,
        sample_aspect_ratio=_sar(video) if video else 1.0,
    )
