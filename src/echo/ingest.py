"""Ingest stage: ``input.mp4`` → ``audio.wav`` (16 kHz mono 16-bit PCM by default).

The input is probed first so a missing file or a video without sound fails with a
clear ``InvalidInputError`` instead of an ffmpeg error. Only the first audio stream is
decoded; video, subtitle and data streams are dropped.

The WAV is encoded to a temp file next to ``out`` and moved into place with
``os.replace``, so a failed or interrupted run never leaves a partial ``audio.wav``
that a later run would mistake for a finished stage.
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
from echo.models import StageName


def extract_audio(video: Path, out: Path, config: IngestConfig) -> None:
    """Implements ``echo.interfaces.AudioExtractor``."""
    info = probe(video)
    if not info.has_audio:
        raise InvalidInputError(f"input has no audio stream: {video}")

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    os.close(fd)  # ffmpeg reopens it by name; Windows can't share an open handle
    tmp = Path(tmp_name)
    try:
        ffmpeg(
            [
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

    def run(self, job: Job, config: PipelineConfig) -> None:
        source = job.input_video
        if not (source.is_file() or source.is_symlink()) and job.input_audio.is_file():
            source = job.input_audio  # link job: only the audio was downloaded
        self.extractor(source, job.audio_path, config.ingest)
