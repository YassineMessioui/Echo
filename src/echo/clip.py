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

Reframing (#26) happens in the same encode: ``cut_clips`` asks ``echo.reframe`` for the
``-vf`` filter matching ``ClipConfig.aspect`` / ``background`` / ``crop_x`` (1080x1920 by
default, none for ``aspect="source"``).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo.captions import CaptionStyle, build_ass, chain_filters, subtitles_filter
from echo.config import ClipConfig, PipelineConfig
from echo.download import video_source
from echo.errors import ArtifactError, ExternalToolError, InvalidInputError
from echo.ffmpeg import ffmpeg, has_filter, http_options
from echo.interfaces import Clipper, RemoteVideo
from echo.job import Job, _atomic_write
from echo.models import ClipSpec, StageName
from echo.reframe import video_filter as reframe_filter
from echo.reframe import video_filter_for

REMOTE_REQUEST_SIZE = 2**20
"""Bytes per HTTP request when reading a ``RemoteVideo``."""

_REMOTE_OPTIONS = {
    # Ranged 1 MiB requests: YouTube throttles one long-running request to ~0.2 MB/s
    # (a 60 s 1080p piece took 25 s; with this, 2.7 s). ffmpeg >= 9 only.
    "request_size": str(REMOTE_REQUEST_SIZE),
    "reconnect": "1",
    "reconnect_on_network_error": "1",
    "reconnect_delay_max": "5",
}


def _remote_input_options() -> list[str]:
    """The ``_REMOTE_OPTIONS`` this ffmpeg supports (older builds lack some; without
    ``request_size`` remote cuts still work, just slower)."""
    supported = http_options()
    return [arg for k, v in _REMOTE_OPTIONS.items() if k in supported for arg in (f"-{k}", v)]


def build_ffmpeg_args(
    video: Path | RemoteVideo,
    clip: ClipSpec,
    out: Path,
    config: ClipConfig,
    *,
    video_filter: str | None = None,
    threads: int | None = None,
) -> list[str]:
    """ffmpeg arguments (after the global flags) that cut ``clip`` from ``video`` into ``out``.

    ``threads`` caps the encoder's threads, so parallel encodes share the cores (#57).

    ``video_filter`` is passed as ``-vf`` in the same encode (e.g. the 9:16 crop/scale
    graph from ``echo.reframe``); ``None`` keeps the source framing.

    A ``RemoteVideo`` is read over HTTP: ffmpeg seeks with range requests, so only this
    clip's bytes are fetched, and the audio comes from the local file. Both inputs are
    seeked to the same start, so they stay in sync.
    """
    start = f"{clip.start:.3f}"
    if isinstance(video, RemoteVideo):
        args = ["-headers", "".join(f"{k}: {v}\r\n" for k, v in video.headers.items())]
        args += _remote_input_options()
        args += ["-accurate_seek", "-ss", start, "-i", video.url]
        args += ["-accurate_seek", "-ss", start, "-i", str(video.audio)]
        audio_map = "1:a:0"
    else:
        args = ["-accurate_seek", "-ss", start, "-i", str(video)]
        audio_map = "0:a:0?"
    args += ["-t", f"{clip.duration:.3f}", "-map", "0:v:0", "-map", audio_map]
    if video_filter is not None:
        args += ["-vf", video_filter]
    # -crf / -preset are x264/x265 options; hardware encoders (e.g. h264_nvenc) need others.
    args += ["-c:v", config.video_codec, "-crf", str(config.crf), "-preset", config.preset]
    args += ["-pix_fmt", "yuv420p", "-c:a", config.audio_codec]
    if threads is not None:
        args += ["-threads", str(threads)]
    # The temp file has no .mp4 extension, so the muxer is named explicitly.
    args += ["-movflags", "+faststart", "-f", "mp4", str(out)]
    return args


def _cut_one(
    video: Path | RemoteVideo,
    clip: ClipSpec,
    out: Path,
    config: ClipConfig,
    video_filter: str | None,
    threads: int | None = None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    os.close(fd)  # ffmpeg reopens it; Windows can't share an open handle
    tmp = Path(tmp_name)
    try:
        ffmpeg(
            build_ffmpeg_args(video, clip, tmp, config, video_filter=video_filter, threads=threads)
        )
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def cut_clips(
    video: Path | RemoteVideo,
    clips: list[ClipSpec],
    outputs: list[Path],
    config: ClipConfig,
    *,
    video_filter: str | None = None,
    extra_filter: str | None = None,
) -> None:
    """Implements ``echo.interfaces.Clipper``.

    The frame is shaped by ``config.aspect`` (see ``echo.reframe``); pass
    ``video_filter`` to use a different ``-vf`` filter instead. ``extra_filter`` (e.g.
    burned-in captions) is chained after it, in the same encode.

    Raises ValueError if ``clips`` and ``outputs`` differ in length, InvalidInputError if
    ``video`` is missing or has no video stream and ExternalToolError if ffmpeg fails.
    Clips cut before a failure are kept.
    """
    if len(clips) != len(outputs):
        raise ValueError(f"got {len(clips)} clips but {len(outputs)} output paths")
    if isinstance(video, RemoteVideo):
        if not video.audio.is_file():
            raise InvalidInputError(f"input audio not found: {video.audio}")
    elif not video.is_file():
        raise InvalidInputError(f"input video not found: {video}")
    if extra_filter and "subtitles=" in extra_filter and not has_filter("subtitles"):
        raise ExternalToolError(
            "this ffmpeg has no 'subtitles' filter (it was built without libass), so captions "
            "can't be burned in. Install an ffmpeg build with libass, or turn captions off "
            "(--no-captions, or caption.enabled = false)."
        )
    if video_filter is None and clips:
        if isinstance(video, RemoteVideo):
            video_filter = reframe_filter(video.width, video.height, config)
        else:
            video_filter = video_filter_for(video, config)
    video_filter = chain_filters(video_filter, extra_filter)
    threads = encoder_threads(config)
    for clip, out in zip(clips, outputs, strict=True):
        _cut_one(video, clip, out, config, video_filter, threads)


_: Clipper = cut_clips


def clip_workers(config: ClipConfig) -> int:
    """Clips encoded at once: ``config.max_workers``, else ``min(4, cpu_count // 2)``."""
    if config.max_workers is not None:
        return config.max_workers
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def encoder_threads(config: ClipConfig) -> int | None:
    """Each parallel ffmpeg's share of the cores; None (ffmpeg decides) when sequential."""
    workers = clip_workers(config)
    return None if workers == 1 else max(1, (os.cpu_count() or workers) // workers)


MANIFEST = "manifest.json"
"""``clips/manifest.json``: ``{file name: {"clip": ..., "config": ..., "captions": ...}}``
fingerprints for the clips this stage wrote (``captions`` only when burned in)."""


def _hash(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _clip_hash(clip: ClipSpec) -> str:
    return _hash(clip.model_dump(mode="json"))


def _entry(clip: ClipSpec, config: ClipConfig, captions: str | None) -> dict[str, str]:
    # max_workers doesn't change the output (and leaving it out keeps older hashes valid).
    settings = config.model_dump(mode="json", exclude={"max_workers"})
    entry = {"clip": _clip_hash(clip), "config": _hash(settings)}
    if captions is not None:
        # The caption track's content: re-transcribing or restyling re-cuts the clip.
        entry["captions"] = _hash(captions)
    return entry


def captions_path(clip_video: Path) -> Path:
    """``clips/<nn>-<slug>.ass``, the caption track burned into ``clip_video``."""
    return clip_video.with_suffix(".ass")


@dataclass
class ClipStage:
    """Stage adapter: ``input.mp4`` + ``clips.json`` → ``clips/<nn>-<slug>.mp4``.

    A clip is reused only if its file exists and the manifest says it was cut with the
    same boundaries, ``ClipConfig`` and caption track; ``force`` re-cuts all of them.

    With ``PipelineConfig.caption.enabled`` (the default), each clip gets a word-timed
    caption track (``clips/<nn>-<slug>.ass``, kept for inspection) burned in after the
    9:16 framing, in the same encode (#17).

    ``outputs()`` gets no config (``Stage`` protocol), so it only detects changed
    boundaries; changed encode settings are picked up whenever the stage runs.
    """

    clipper: Clipper = cut_clips
    force: bool = False
    source: Callable[[Job], Path | RemoteVideo] = video_source
    """Local ``input.mp4``, or a link job's remote video (resolved only if a clip is cut)."""

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
        if not (job.input_video.is_file() or job.input_audio.is_file()):
            raise ArtifactError(f"missing artifact: {job.input_video}")
        clips = job.read_clips()
        resolved: list[Path | RemoteVideo] = []
        lock = threading.Lock()  # guards ``resolved`` and ``manifest`` across workers

        def video() -> Path | RemoteVideo:
            # Resolved once, on the first clip that needs cutting: a link job's stream
            # URL is only fetched when there's work to do (and it expires anyway).
            with lock:
                if not resolved:
                    resolved.append(self.source(job))
                return resolved[0]

        paths = clip_paths(job, clips)
        job.clips_dir.mkdir(parents=True, exist_ok=True)
        manifest = {k: v for k, v in _read_manifest(job).items() if (job.clips_dir / k).exists()}
        burn = config.caption.enabled and bool(clips)
        if burn and not job.transcript_path.is_file():
            raise ArtifactError(
                f"captions need {job.transcript_path}; run the transcribe stage first, or "
                "turn captions off (--no-captions, or caption.enabled = false)"
            )
        transcript = job.read_transcript() if burn else None
        style = CaptionStyle.from_config(config.caption)

        todo: list[_Cut] = []
        for clip, path in zip(clips, paths, strict=True):
            ass = build_ass(transcript, clip, style) if transcript is not None else None
            entry = _entry(clip, config.clip, ass)
            if not self.force and path.exists() and manifest.get(path.name) == entry:
                continue
            manifest.pop(path.name, None)
            todo.append(_Cut(clip, path, entry, ass))

        def cut(item: _Cut) -> None:
            ass_path = captions_path(item.path)
            if item.ass is None:
                ass_path.unlink(missing_ok=True)
                self.clipper(video(), [item.clip], [item.path], config.clip)
            else:
                # Burn from a temp file and move it into place only once the clip is cut,
                # so a failed encode never leaves a .ass that doesn't match its .mp4.
                tmp = ass_path.with_name(f".{ass_path.name}.tmp")
                tmp.write_bytes(item.ass.encode("utf-8-sig"))  # BOM: libass reads it anywhere
                try:
                    self.clipper(
                        video(),
                        [item.clip],
                        [item.path],
                        config.clip,
                        extra_filter=subtitles_filter(tmp),
                    )
                    os.replace(tmp, ass_path)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
            with lock:
                manifest[item.path.name] = item.entry
                # After every clip, so a crash only loses the clips still encoding.
                _write_manifest(job, manifest)

        workers = min(clip_workers(config.clip), len(todo))
        if workers <= 1:
            for item in todo:
                cut(item)
        else:
            _run_parallel(cut, todo, workers)

        # Delete clips written for a previous clips.json that are no longer wanted.
        for name in set(manifest) - {p.name for p in paths}:
            (job.clips_dir / name).unlink(missing_ok=True)
            captions_path(job.clips_dir / name).unlink(missing_ok=True)
            del manifest[name]
        _write_manifest(job, manifest)


@dataclass(frozen=True)
class _Cut:
    """A clip ``ClipStage.run`` has to (re-)encode."""

    clip: ClipSpec
    path: Path
    entry: dict[str, str]
    ass: str | None


def _run_parallel(cut: Callable[[_Cut], None], todo: list[_Cut], workers: int) -> None:
    """Run ``cut`` for every clip on ``workers`` threads (each one waits on an ffmpeg).

    The first failure stops new clips from starting; clips already encoding finish and
    keep their manifest entries, then the error of the earliest failed clip is raised.
    """
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clip")
    try:
        futures = [pool.submit(cut, item) for item in todo]
        wait(futures, return_when=FIRST_EXCEPTION)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for future in futures:
        error = None if future.cancelled() else future.exception()
        if error is not None:
            raise error


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
