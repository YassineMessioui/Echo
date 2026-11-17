"""Ingest stage: ``input.mp4`` → ``audio.wav`` (16 kHz mono 16-bit PCM by default).

The input is probed first so a missing file or a video without sound fails with a
clear ``InvalidInputError`` instead of an ffmpeg error. Only the first audio stream is
decoded; video, subtitle and data streams are dropped.

The WAV is encoded to a temp file next to ``out`` and moved into place with
``os.replace``, so a failed or interrupted run never leaves a partial ``audio.wav``
that a later run would mistake for a finished stage.

**Focus range (#59).** With ``IngestConfig.start`` / ``end`` set, only that span is
extracted (``-ss`` / ``-to``); percentages are of the input's full duration. The stage
records the resolved span in ``job.json`` as ``meta.models["ingest_range"] =
"<start>:<end>"`` (seconds of the original video), and the transcribe stage uses its
start as ``Transcript.offset``. A run without a range removes the key.
``IngestStage.settings`` fingerprints the range, so changing it re-runs ingest and
every later stage.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo.config import IngestConfig, PipelineConfig
from echo.errors import InvalidInputError
from echo.ffmpeg import ffmpeg, probe
from echo.interfaces import AudioExtractor
from echo.job import Job
from echo.models import JobMeta, StageName

RANGE_KEY = "ingest_range"
"""``JobMeta.models`` key holding the resolved focus range, ``"<start>:<end>"`` in seconds.
``echo.transcribe`` reads it (by the same name) for the transcript offset."""


def _point(value: str, duration: float) -> float:
    if value.endswith("%"):
        return duration * float(value[:-1]) / 100
    return float(value)


def resolve_range(config: IngestConfig, duration: float) -> tuple[float, float] | None:
    """The focus range as ``(start, end)`` seconds of a ``duration``-second input, or None
    when no range is set (the whole input).

    Raises InvalidInputError for an empty or inverted range, or one past the end.
    """
    if config.start is None and config.end is None:
        return None
    start = 0.0 if config.start is None else _point(config.start, duration)
    end = duration if config.end is None else _point(config.end, duration)
    shown = f"{config.start or ''}:{config.end or ''}"
    if start >= duration:
        raise InvalidInputError(
            f"focus range {shown} starts at {start:.1f} s, at or past the end of the input "
            f"({duration:.1f} s)"
        )
    if end > duration + 0.001:
        raise InvalidInputError(
            f"focus range {shown} ends at {end:.1f} s, past the end of the input ({duration:.1f} s)"
        )
    if end <= start:
        raise InvalidInputError(
            f"focus range {shown} is empty or inverted ({start:.1f} s to {end:.1f} s)"
        )
    return round(start, 3), round(min(end, duration), 3)


def extract_audio(video: Path, out: Path, config: IngestConfig) -> None:
    """Implements ``echo.interfaces.AudioExtractor``."""
    info = probe(video)
    if not info.has_audio:
        raise InvalidInputError(f"input has no audio stream: {video}")
    span = resolve_range(config, info.duration)
    # As input options: a fast seek, then exact trimming of the decoded audio.
    seek = [] if span is None else ["-ss", f"{span[0]:.3f}", "-to", f"{span[1]:.3f}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    os.close(fd)  # ffmpeg reopens it by name; Windows can't share an open handle
    tmp = Path(tmp_name)
    try:
        ffmpeg(
            [
                *seek,
                "-i",
                str(video),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                str(config.channels),
                "-ar",
                str(config.sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",  # the .tmp suffix gives ffmpeg no format hint
                str(tmp),
            ]
        )
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


_: AudioExtractor = extract_audio


@dataclass
class IngestStage:
    """Stage adapter: ``input.mp4`` (or a link job's ``input-audio.m4a``) → ``audio.wav``."""

    extractor: AudioExtractor = extract_audio

    name: ClassVar[StageName] = StageName.INGEST

    def outputs(self, job: Job) -> list[Path]:
        return [job.audio_path]

    def settings(self, config: PipelineConfig) -> str:
        """The focus range as given (``"range=15%:85%"``; ``"range=:"`` is the whole input)."""
        return f"range={config.ingest.start or ''}:{config.ingest.end or ''}"

    def run(self, job: Job, config: PipelineConfig) -> None:
        source = job.input_video
        if not (source.is_file() or source.is_symlink()) and job.input_audio.is_file():
            source = job.input_audio  # link job: only the audio was downloaded
        span = None
        if config.ingest.start is not None or config.ingest.end is not None:
            span = resolve_range(config.ingest, probe(source).duration)
        self.extractor(source, job.audio_path, config.ingest)
        meta = job.read_meta()
        _record_range(meta, span)
        job.write_meta(meta)


def _record_range(meta: JobMeta, span: tuple[float, float] | None) -> None:
    if span is None:
        meta.models.pop(RANGE_KEY, None)
    else:
        meta.models[RANGE_KEY] = f"{span[0]:.3f}:{span[1]:.3f}"
