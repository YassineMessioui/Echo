"""Clipping stage: ``input.mp4`` + ``clips.json`` → ``clips/<nn>-<slug>.mp4``.

Each clip is one re-encode (stream copy snaps to keyframes, so boundaries would be
off by up to a GOP). ``-ss`` goes before ``-i`` with ``-accurate_seek``: ffmpeg seeks
quickly to the previous keyframe, then decodes and drops frames up to the exact start.
``-t`` is an output option, so the duration is measured from that exact start.

Every clip is written to a temp file next to its final path and moved into place with
``os.replace``, so an interrupted run never leaves a half-written ``.mp4`` that a later
run would mistake for a finished clip.

``ClipStage`` keeps ``clips/manifest.json`` mapping each file it wrote to a fingerprint
of its ``ClipSpec`` and ``ClipConfig``. A clip is only reused if its fingerprint still
matches, so re-running refine with new boundaries re-cuts the clips that moved, and
files from a previous ``clips.json`` that no longer belong to it are deleted.

The 9:16 reframing (#26) plugs in through ``build_ffmpeg_args(..., video_filter=...)``;
``aspect`` / ``background`` / ``crop_x`` in ``ClipConfig`` are not applied here yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo.config import ClipConfig, PipelineConfig
from echo.errors import ArtifactError, InvalidInputError
from echo.ffmpeg import ffmpeg
from echo.interfaces import Clipper
from echo.job import Job, _atomic_write
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
    # -crf / -preset are x264/x265 options; hardware encoders (e.g. h264_nvenc) need others.
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


MANIFEST = "manifest.json"
"""``clips/manifest.json``: ``{file name: {"clip": ..., "config": ...}}`` fingerprints
for the clips this stage wrote."""


def _hash(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _clip_hash(clip: ClipSpec) -> str:
    return _hash(clip.model_dump(mode="json"))


def _entry(clip: ClipSpec, config: ClipConfig) -> dict[str, str]:
    return {"clip": _clip_hash(clip), "config": _hash(config.model_dump(mode="json"))}


@dataclass
class ClipStage:
    """Stage adapter: ``input.mp4`` + ``clips.json`` → ``clips/<nn>-<slug>.mp4``.

    A clip is reused only if its file exists and the manifest says it was cut with the
    same boundaries and ``ClipConfig``; ``force`` re-cuts all of them.

    ``outputs()`` gets no config (``Stage`` protocol), so it only detects changed
    boundaries; changed encode settings are picked up whenever the stage runs.
    """

    clipper: Clipper = cut_clips
    force: bool = False

    name: ClassVar[StageName] = StageName.CLIP

    def outputs(self, job: Job) -> list[Path]:
        if not job.clips_path.is_file():
            # Not known until refine has run. A missing path keeps the stage from being skipped.
            return [job.clips_path]
        clips = job.read_clips()
        paths = clip_paths(job, clips)
        manifest = _read_manifest(job)
        moved = any(
            manifest.get(p.name, {}).get("clip") != _clip_hash(c)
            for c, p in zip(clips, paths, strict=True)
        )
        stale = moved or set(manifest) - {p.name for p in paths}
        # A path that never exists makes the orchestrator run the stage.
        return [*paths, job.clips_dir / ".stale"] if stale else paths

    def run(self, job: Job, config: PipelineConfig) -> None:
        if not job.input_video.is_file():
            raise ArtifactError(f"missing artifact: {job.input_video}")
        clips = job.read_clips()
        paths = clip_paths(job, clips)
        job.clips_dir.mkdir(parents=True, exist_ok=True)
        manifest = {k: v for k, v in _read_manifest(job).items() if (job.clips_dir / k).exists()}

        for clip, path in zip(clips, paths, strict=True):
            entry = _entry(clip, config.clip)
            if not self.force and path.exists() and manifest.get(path.name) == entry:
                continue
            manifest.pop(path.name, None)
            self.clipper(job.input_video, [clip], [path], config.clip)
            manifest[path.name] = entry
            _write_manifest(job, manifest)  # after every clip, so a crash loses at most one

        # Delete clips written for a previous clips.json that are no longer wanted.
        for name in set(manifest) - {p.name for p in paths}:
            (job.clips_dir / name).unlink(missing_ok=True)
            del manifest[name]
        _write_manifest(job, manifest)


def _read_manifest(job: Job) -> dict[str, dict[str, str]]:
    try:
        data = json.loads((job.clips_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def _write_manifest(job: Job, manifest: dict[str, dict[str, str]]) -> None:
    _atomic_write(job.clips_dir / MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode())


def clip_paths(job: Job, clips: list[ClipSpec]) -> list[Path]:
    """One path per clip. Topics whose titles slug the same (``"Q&A"`` and ``"Q & A"``)
    still get distinct files because the index prefix differs; this is checked so a
    future naming change can't silently make two clips overwrite each other."""
    paths = [job.clip_video_path(i, c.topic_title) for i, c in enumerate(clips, start=1)]
    if len(set(paths)) != len(paths):
        raise ArtifactError(f"clip file names collide: {[p.name for p in paths]}")
    return paths
