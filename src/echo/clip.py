"""Clipping stage: ``input.mp4`` + ``clips.json`` → ``clips/<nn>-<slug>.mp4``.

Each clip is one re-encode (stream copy snaps to keyframes, so boundaries would be
off by up to a GOP). ``-ss`` goes before ``-i`` with ``-accurate_seek``: ffmpeg seeks
quickly to the previous keyframe, then decodes and drops frames up to the exact start.
``-t`` is an output option, so the duration is measured from that exact start.

Every clip is written to a temp file next to its final path and moved into place with
``os.replace``, so an interrupted run never leaves a half-written ``.mp4`` that a later
run would mistake for a finished clip.

The 9:16 reframing (#26) plugs in through ``build_ffmpeg_args(..., video_filter=...)``;
``aspect`` / ``background`` / ``crop_x`` in ``ClipConfig`` are not applied here yet.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo.config import ClipConfig, PipelineConfig
from echo.errors import ArtifactError, InvalidInputError
from echo.ffmpeg import ffmpeg
from echo.interfaces import Clipper
from echo.job import Job
from echo.models import ClipSpec, StageName


def build_ffmpeg_args(
    video: Path,
    clip: ClipSpec,
    out: Path,
    config: ClipConfig,
    *,
    video_filter: str | None = None,
) -> list[str]:
    """ffmpeg arguments (after the global flags) that cut ``clip`` from ``video`` into ``out``.

    ``video_filter`` is passed as ``-vf`` in the same encode (e.g. the 9:16 crop/scale
    graph from ``echo.reframe``); ``None`` keeps the source framing.
    """
    args = ["-accurate_seek", "-ss", f"{clip.start:.3f}", "-i", str(video)]
    args += ["-t", f"{clip.duration:.3f}", "-map", "0:v:0", "-map", "0:a:0?"]
    if video_filter is not None:
        args += ["-vf", video_filter]
    args += ["-c:v", config.video_codec, "-crf", str(config.crf), "-preset", config.preset]
    args += ["-pix_fmt", "yuv420p", "-c:a", config.audio_codec]
    # The temp file has no .mp4 extension, so the muxer is named explicitly.
    args += ["-movflags", "+faststart", "-f", "mp4", str(out)]
    return args


def _cut_one(
    video: Path, clip: ClipSpec, out: Path, config: ClipConfig, video_filter: str | None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    os.close(fd)  # ffmpeg reopens it; Windows can't share an open handle
    tmp = Path(tmp_name)
    try:
        ffmpeg(build_ffmpeg_args(video, clip, tmp, config, video_filter=video_filter))
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def cut_clips(
    video: Path,
    clips: list[ClipSpec],
    outputs: list[Path],
    config: ClipConfig,
    *,
    video_filter: str | None = None,
) -> None:
    """Implements ``echo.interfaces.Clipper``.

    Raises ValueError if ``clips`` and ``outputs`` differ in length, InvalidInputError if
    ``video`` is missing and ExternalToolError if ffmpeg fails. Clips cut before a
    failure are kept.
    """
    if len(clips) != len(outputs):
        raise ValueError(f"got {len(clips)} clips but {len(outputs)} output paths")
    if not video.is_file():
        raise InvalidInputError(f"input video not found: {video}")
    for clip, out in zip(clips, outputs, strict=True):
        _cut_one(video, clip, out, config, video_filter)


_: Clipper = cut_clips


@dataclass
class ClipStage:
    """Stage adapter: ``input.mp4`` + ``clips.json`` → ``clips/<nn>-<slug>.mp4``.

    Clips whose file already exists are skipped, so an interrupted run resumes where it
    stopped; ``force`` re-cuts all of them.
    """

    clipper: Clipper = cut_clips
    force: bool = False

    name: ClassVar[StageName] = StageName.CLIP

    def outputs(self, job: Job) -> list[Path]:
        if not job.clips_path.is_file():
            # Not known until refine has run. A missing path keeps the stage from being skipped.
            return [job.clips_path]
        return _output_paths(job, job.read_clips())

    def run(self, job: Job, config: PipelineConfig) -> None:
        if not job.input_video.is_file():
            raise ArtifactError(f"missing artifact: {job.input_video}")
        clips = job.read_clips()
        paths = _output_paths(job, clips)
        todo = [(c, p) for c, p in zip(clips, paths, strict=True) if self.force or not p.exists()]
        job.clips_dir.mkdir(parents=True, exist_ok=True)
        if todo:
            self.clipper(job.input_video, [c for c, _ in todo], [p for _, p in todo], config.clip)


def _output_paths(job: Job, clips: list[ClipSpec]) -> list[Path]:
    """One path per clip. Topics whose titles slug the same (``"Q&A"`` and ``"Q & A"``)
    still get distinct files because the index prefix differs; this is checked so a
    future naming change can't silently make two clips overwrite each other."""
    paths = [job.clip_video_path(i, c.topic_title) for i, c in enumerate(clips, start=1)]
    if len(set(paths)) != len(paths):
        raise ArtifactError(f"clip file names collide: {[p.name for p in paths]}")
    return paths
