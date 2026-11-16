"""Thin wrappers around the ffmpeg / ffprobe binaries, shared by the media layers.

Every failure surfaces as ``ExternalToolError`` (tool missing, non-zero exit) with the
tail of stderr, or ``InvalidInputError`` when the input file itself is the problem.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None


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
    )
