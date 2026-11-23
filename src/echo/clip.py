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
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo import progress
from echo.captions import CaptionStyle, build_ass, chain_filters, subtitles_filter
from echo.config import ClipConfig, PipelineConfig
from echo.download import video_source
from echo.errors import ArtifactError, ExternalToolError, InvalidInputError
from echo.faces import FaceDetector, default_detector
from echo.ffmpeg import ffmpeg, has_filter, http_options
from echo.framing import frame_crop
from echo.interfaces import Clipper, RemoteVideo
from echo.job import Job, _atomic_write
from echo.models import ClipSpec, Speakers, StageName
from echo.reframe import crop_window, frame_geometry, video_filter_for
from echo.reframe import video_filter as reframe_filter
from echo.webcam import ExcludingDetector, detect_webcam, overlay_filter

log = logging.getLogger(__name__)

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


def video_input_args(video: Path | RemoteVideo, start: float, *, exact: bool = True) -> list[str]:
    """ffmpeg arguments that open ``video`` as input 0, seeked exactly to ``start`` (or,
    with ``exact=False``, to the keyframe before it: much cheaper, for sampling)."""
    seek = ["-accurate_seek" if exact else "-noaccurate_seek", "-ss", f"{start:.3f}"]
    if not isinstance(video, RemoteVideo):
        return [*seek, "-i", str(video)]
    # -headers puts them in ffmpeg's argv (visible in the process list). That's fine:
    # echo.download never resolves a stream whose headers carry credentials
    # (CREDENTIAL_HEADER), and echo.ffmpeg.run never logs argv (errors show stderr).
    args = []
    if video.headers:
        args += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in video.headers.items())]
    return [*args, *_remote_input_options(), *seek, "-i", video.url]


def build_ffmpeg_args(
    video: Path | RemoteVideo,
    clip: ClipSpec,
    out: Path,
    config: ClipConfig,
    *,
    video_filter: str | None = None,
) -> list[str]:
    """ffmpeg arguments (after the global flags) that cut ``clip`` from ``video`` into ``out``.

    ``video_filter`` is passed as ``-vf`` in the same encode (e.g. the 9:16 crop/scale
    graph from ``echo.reframe``); ``None`` keeps the source framing.

    A ``RemoteVideo`` is read over HTTP: ffmpeg seeks with range requests, so only this
    clip's bytes are fetched, and the audio comes from the local file. Both inputs are
    seeked to the same start, so they stay in sync.
    """
    args = video_input_args(video, clip.start)
    if isinstance(video, RemoteVideo):
        args += ["-accurate_seek", "-ss", f"{clip.start:.3f}", "-i", str(video.audio)]
        audio_map = "1:a:0"
    else:
        audio_map = "0:a:0?"
    args += ["-t", f"{clip.duration:.3f}", "-map", "0:v:0", "-map", audio_map]
    if video_filter is not None:
        args += ["-vf", video_filter]
    # -crf / -preset are x264/x265 options; hardware encoders (e.g. h264_nvenc) need others.
    args += ["-c:v", config.video_codec, "-crf", str(config.crf), "-preset", config.preset]
    args += ["-pix_fmt", "yuv420p", "-c:a", config.audio_codec]
    # The temp file has no .mp4 extension, so the muxer is named explicitly.
    args += ["-movflags", "+faststart", "-f", "mp4", str(out)]
    return args


def _cut_one(
    video: Path | RemoteVideo,
    clip: ClipSpec,
    out: Path,
    config: ClipConfig,
    video_filter: str | None,
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
    video: Path | RemoteVideo,
    clips: list[ClipSpec],
    outputs: list[Path],
    config: ClipConfig,
    *,
    video_filter: str | None = None,
    extra_filter: str | None = None,
    speakers: Speakers | None = None,
    detector: FaceDetector | None = None,
) -> None:
    """Implements ``echo.interfaces.Clipper``.

    The frame is shaped by ``config.aspect`` (see ``echo.reframe``); pass
    ``video_filter`` to use a different ``-vf`` filter instead. ``extra_filter`` (e.g.
    burned-in captions) is chained after it, in the same encode.

    With ``config.reframe`` ``"face"`` (or ``"speaker"``, which frames like ``"face"``
    until #109), each clip is sampled first and the window follows the faces shot by
    shot (``echo.framing``), using ``detector`` (YuNet by default). ``speakers`` is for
    ``"speaker"`` (#109).

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
    # Face framing (#16) replaces the fixed crop_x, on sources the 9:16 filter crops; a
    # streamer's webcam overlay (#126) is kept in a corner of them.
    framed = video_filter is None and config.aspect != "source"
    face = framed and config.reframe != "centre"
    webcam = framed and config.webcam == "auto"
    if video_filter is None and clips:
        if isinstance(video, RemoteVideo):
            video_filter = reframe_filter(video.width, video.height, config)
        else:
            video_filter = video_filter_for(video, config)
    geometry, window = None, None
    if (face or webcam) and clips:
        if isinstance(video, RemoteVideo):
            geometry = (video.width, video.height, 1.0)
        else:
            geometry = frame_geometry(video)
        window = crop_window(geometry[0], geometry[1], sar=geometry[2])
    known_webcam = None
    for clip, out in zip(clips, outputs, strict=True):
        clip_filter = video_filter
        if window is not None and geometry is not None:
            width, height, sar = geometry
            found = detector or default_detector()
            box = None
            if webcam:
                started = time.monotonic()
                box = detect_webcam(
                    lambda t: video_input_args(video, t, exact=False),
                    clip.start,
                    clip.duration,
                    *window[:2],
                    found,
                    known=known_webcam,
                )
                known_webcam = box or known_webcam  # a stream's webcam stays put
                log.info("webcam search took %.1f s", time.monotonic() - started)
            if face:
                started = time.monotonic()
                x = frame_crop(
                    video_input_args(video, clip.start),
                    clip.duration,
                    *window,
                    config.crop_x,
                    found if box is None else ExcludingDetector(found, box),
                )
                log.info("face framing took %.1f s: crop x = %s", time.monotonic() - started, x)
                clip_filter = reframe_filter(width, height, config, sar=sar, crop_x_expr=x)
            if box is not None and clip_filter is not None:
                clip_filter = overlay_filter(clip_filter, box, width, height, config)
        _cut_one(video, clip, out, config, chain_filters(clip_filter, extra_filter))


_: Clipper = cut_clips


MANIFEST = "manifest.json"
"""``clips/manifest.json``: ``{file name: {"clip": ..., "config": ..., "captions": ...}}``
fingerprints for the clips this stage wrote (``captions`` only when burned in)."""


def _hash(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _clip_hash(clip: ClipSpec) -> str:
    return _hash(clip.model_dump(mode="json"))


def _config_data(config: ClipConfig) -> dict[str, object]:
    """``config`` as fingerprinted. Settings added later are left out while they have the
    value that reproduces the earlier output, so upgrading Echo doesn't re-cut old jobs."""
    data = config.model_dump(mode="json")
    for key, earlier in _ADDED_SETTINGS.items():
        if data.get(key) == earlier:
            del data[key]
    return data


_ADDED_SETTINGS: dict[str, object] = {
    "reframe": "centre",  # #16: what every clip cut before it used
    # #126: "auto" finds no webcam in non-stream clips, so they come out as before;
    # stream clips cut earlier aren't re-cut just because of the upgrade (use --force).
    "webcam": "auto",
    "webcam_corner": "top-left",
    "webcam_size": 0.4,
}
"""Settings added after release, with the value whose fingerprint matches clips cut before
they existed."""


def _entry(clip: ClipSpec, config: ClipConfig, captions: str | None) -> dict[str, str]:
    entry = {"clip": _clip_hash(clip), "config": _hash(_config_data(config))}
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

    With diarization on (``--speakers``), each speaker's captions get their own colour
    (#68).

    ``outputs()`` gets no config (``Stage`` protocol), so it only detects changed
    boundaries; ``settings()`` covers the rest (encode, framing, captions, speakers), so
    the orchestrator runs the stage when they change, and the manifest then re-cuts
    exactly the clips whose output differs.
    """

    clipper: Clipper = cut_clips
    force: bool = False
    source: Callable[[Job], Path | RemoteVideo] = video_source
    """Local ``input.mp4``, or a link job's remote video (resolved only if a clip is cut)."""

    name: ClassVar[StageName] = StageName.CLIP

    def settings(self, config: PipelineConfig) -> str:
        """Fingerprint of everything besides the clip boundaries that changes the clips."""
        return _hash(
            {
                "clip": _config_data(config.clip),
                "caption": config.caption.model_dump(mode="json"),
                "speakers": config.diarize.enabled,
            }
        )

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

        def video() -> Path | RemoteVideo:
            # Resolved once, on the first clip that needs cutting: a link job's stream
            # URL is only fetched when there's work to do (and it expires anyway).
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
        colour_speakers = burn and config.diarize.enabled and job.speakers_path.is_file()
        speakers = job.read_speakers() if colour_speakers else None

        def clip_done(done: int) -> None:  # skipped (up-to-date) clips count as done
            progress.report(done / len(clips), f"clip {done}/{len(clips)}")

        for done, (clip, path) in enumerate(zip(clips, paths, strict=True), start=1):
            ass = build_ass(transcript, clip, style, speakers) if transcript is not None else None
            entry = _entry(clip, config.clip, ass)
            if not self.force and path.exists() and manifest.get(path.name) == entry:
                clip_done(done)
                continue
            manifest.pop(path.name, None)
            ass_path = captions_path(path)
            if ass is None:
                ass_path.unlink(missing_ok=True)
                self.clipper(video(), [clip], [path], config.clip)
            else:
                # Burn from a temp file and move it into place only once the clip is cut,
                # so a failed encode never leaves a .ass that doesn't match its .mp4.
                tmp = ass_path.with_name(f".{ass_path.name}.tmp")
                tmp.write_bytes(ass.encode("utf-8-sig"))  # BOM: libass reads it anywhere
                try:
                    self.clipper(
                        video(),
                        [clip],
                        [path],
                        config.clip,
                        extra_filter=subtitles_filter(tmp),
                    )
                    os.replace(tmp, ass_path)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
            manifest[path.name] = entry
            _write_manifest(job, manifest)  # after every clip, so a crash loses at most one
            clip_done(done)

        # Delete clips written for a previous clips.json that are no longer wanted.
        for name in set(manifest) - {p.name for p in paths}:
            (job.clips_dir / name).unlink(missing_ok=True)
            captions_path(job.clips_dir / name).unlink(missing_ok=True)
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
