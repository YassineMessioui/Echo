"""Video links as input (#60): download a URL with yt-dlp into the job directory.

``echo-clips run <URL>`` probes the link (no download) and picks a job id from the site
and video id (``youtube-jNQXAC9IVRw``). Only the **audio** is downloaded
(``input-audio.m4a``), which is all ingest and transcription need; the clip stage later
reads just each clip's section of the video over HTTP (``RemoteVideo``, #75). On a
68-min podcast that's ~67 MB instead of ~240 MB before transcription can start. Streams
ffmpeg can't seek into fall back to downloading the full video as ``input.mp4``.
Running the same link again reopens that job and skips the download, whichever URL
form is used (``youtu.be/...``, ``youtube.com/watch?v=...``).

Only clip content you have the rights to use.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from yt_dlp.utils import DownloadError

from echo.errors import ArtifactError, InvalidInputError
from echo.interfaces import RemoteVideo
from echo.job import Job
from echo.models import JobMeta
from echo.progress import report

logger = logging.getLogger(__name__)

FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"
"""Full download (fallback): best video up to 1080p + best audio, merged to MP4."""

AUDIO_FORMAT = "ba[ext=m4a]/ba"

VIDEO_FORMAT = "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]/b[height<=1080]/b"
"""Video read remotely at clip time; H.264 first because it's the cheapest to decode."""

SEEKABLE_PROTOCOLS = frozenset({"http", "https", "m3u8", "m3u8_native"})
"""Streams ffmpeg can seek into over the network (fragmented DASH can't be)."""

CREDENTIAL_HEADER = re.compile(r"cookie|auth|token|secret", re.IGNORECASE)
"""Header names that look like credentials: a stream needing them is never read remotely.

``RemoteVideo.headers`` go on ffmpeg's command line (``-headers``), which other users
can see in the process list. yt-dlp's ``http_headers`` for YouTube are User-Agent,
Accept and the like: yt-dlp drops ``Cookie`` from them (cookies, e.g. from a cookie
file, go in the format's separate ``cookies`` field, which Echo never passes) and
YouTube's ``Authorization`` is only sent to its API, not to the stream. Other sites may
still put auth in there, so such a stream falls back to the full download, where the
headers stay inside yt-dlp. Dropping them instead isn't safe: the remote read would just
fail (403) instead of falling back."""

JS_RUNTIMES = ("deno", "node", "bun")
"""YouTube needs a JavaScript runtime; yt-dlp only enables deno by default."""


class _YoutubeDL(Protocol):
    """The part of ``yt_dlp.YoutubeDL`` we use (so tests can pass a fake)."""

    def __enter__(self) -> _YoutubeDL: ...
    def __exit__(self, *exc: object) -> None: ...
    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]: ...


YDLFactory = Callable[[dict[str, Any]], _YoutubeDL]
ProgressCallback = Callable[[float], None]
"""Called with the download fraction (0..1) of the current stream."""


def _default_factory(params: dict[str, Any]) -> _YoutubeDL:
    import yt_dlp  # the full package is only needed when a URL is actually downloaded

    return yt_dlp.YoutubeDL(params)


def is_url(text: str) -> bool:
    """True for ``http(s)://host/...`` links; local paths (including ``C:\\...``) are not."""
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


@dataclass(frozen=True)
class VideoInfo:
    url: str
    id: str
    extractor: str
    title: str | None = None
    uploader: str | None = None
    duration: float | None = None
    video_protocol: str | None = None
    """Protocol of the ``VIDEO_FORMAT`` stream; decides remote vs full download."""

    @property
    def seekable(self) -> bool:
        return self.video_protocol in SEEKABLE_PROTOCOLS

    @property
    def job_id(self) -> str:
        """Stable per video, so re-running a link resumes its job. Case is kept (ids are)."""
        return re.sub(r"[^A-Za-z0-9_-]+", "-", f"{self.extractor.lower()}-{self.id}").strip("-")

    def source_info(self) -> dict[str, str | float]:
        info = {"id": self.id, "extractor": self.extractor, "title": self.title}
        info |= {"uploader": self.uploader, "duration": self.duration}
        return {k: v for k, v in info.items() if v is not None}


def _base_params() -> dict[str, Any]:
    params: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,  # the CLI prints its own progress via ``on_progress``
        "noplaylist": True,
    }
    runtimes = {name: {} for name in JS_RUNTIMES if shutil.which(name)}
    if runtimes:
        params["js_runtimes"] = runtimes
    return params


def _error(url: str, e: Exception) -> InvalidInputError:
    message = re.sub(r"^ERROR:\s*", "", str(e)).strip()
    return InvalidInputError(f"could not download {url}: {message}")


def probe_url(url: str, *, ydl_factory: YDLFactory = _default_factory) -> VideoInfo:
    """Read the video's metadata without downloading it.

    Raises InvalidInputError for private, removed, geo-blocked or unsupported links, and
    for playlists (pass a single video).
    """
    info = _extract(url, ydl_factory)
    return VideoInfo(
        url=url,
        id=str(info["id"]),
        extractor=str(info.get("extractor_key") or info.get("extractor") or "web"),
        title=info.get("title"),
        uploader=info.get("uploader"),
        duration=info.get("duration"),
        video_protocol=_video_format(info).get("protocol"),
    )


def _extract(url: str, ydl_factory: YDLFactory) -> dict[str, Any]:
    try:
        with ydl_factory(_base_params() | {"format": VIDEO_FORMAT}) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise _error(url, e) from e
    if info.get("_type") in ("playlist", "multi_video"):
        raise InvalidInputError(f"{url} is a playlist; pass a link to a single video")
    return info


def _video_format(info: dict[str, Any]) -> dict[str, Any]:
    """The selected video stream: the video part of a merged pair, or the single format."""
    for f in info.get("requested_formats") or [info]:
        if f.get("vcodec") not in (None, "none"):
            return f
    return info


def resolve_video(
    url: str, audio: Path, *, ydl_factory: YDLFactory = _default_factory
) -> RemoteVideo | None:
    """A fresh ``RemoteVideo`` for ``url`` (stream URLs expire, so call this right before
    cutting), or None if the stream can't be read remotely (see ``video_source``)."""
    remote = _resolve(url, audio, ydl_factory)
    return remote if isinstance(remote, RemoteVideo) else None


def _resolve(url: str, audio: Path, ydl_factory: YDLFactory) -> RemoteVideo | str:
    """The ``RemoteVideo``, or why the stream can't be read remotely."""
    f = _video_format(_extract(url, ydl_factory))
    headers = {str(k): str(v) for k, v in (f.get("http_headers") or {}).items()}
    if (protocol := f.get("protocol")) not in SEEKABLE_PROTOCOLS:
        return f"ffmpeg can't seek into its {protocol or 'unknown'} protocol"
    if not f.get("width") or not f.get("height"):
        return "its frame size is unknown"
    if secret := sorted(k for k in headers if CREDENTIAL_HEADER.search(k)):
        return f"it needs credential headers ({', '.join(secret)}) kept off ffmpeg's argv"
    return RemoteVideo(
        url=f["url"], audio=audio, width=int(f["width"]), height=int(f["height"]), headers=headers
    )


def video_source(job: Job, *, ydl_factory: YDLFactory = _default_factory) -> Path | RemoteVideo:
    """What the clip stage cuts from: the local ``input.mp4``, or, for a link job, the
    remote video (downloading it in full only if the stream can't be read remotely, with
    progress reported as ``"downloading full video"``)."""
    if job.input_video.is_file():
        return job.input_video
    url = job.read_meta().source_url if job.meta_path.is_file() else None
    if url is None or not job.input_audio.is_file():
        raise ArtifactError(f"missing artifact: {job.input_video}")
    remote = _resolve(url, job.input_audio, ydl_factory)
    if isinstance(remote, RemoteVideo):
        return remote
    logger.warning(
        "the video stream can't be read remotely (%s), so the full video is being "
        "downloaded to %s; this can take a few minutes",
        remote,
        job.input_video,
    )
    download_video(
        url,
        job.input_video,
        on_progress=lambda fraction: report(fraction, "downloading full video"),
        ydl_factory=ydl_factory,
    )
    return job.input_video


def download_video(
    url: str,
    out: Path,
    *,
    format_spec: str = FORMAT,
    on_progress: ProgressCallback | None = None,
    ydl_factory: YDLFactory = _default_factory,
) -> None:
    """Download ``url`` (``format_spec``: the full video by default) to ``out`` atomically: a
    failed download leaves nothing at ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=out.parent, prefix=".download-"))

    def hook(status: dict[str, Any]) -> None:
        if on_progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if total:
            on_progress(min(1.0, status.get("downloaded_bytes", 0) / total))

    params = _base_params() | {
        "format": format_spec,
        "merge_output_format": "mp4",
        "outtmpl": str(tmp_dir / "video.%(ext)s"),
        "progress_hooks": [hook],
    }
    if ffmpeg := shutil.which("ffmpeg"):
        params["ffmpeg_location"] = ffmpeg
    try:
        try:
            with ydl_factory(params) as ydl:
                ydl.extract_info(url, download=True)
        except DownloadError as e:
            raise _error(url, e) from e
        # Single-file formats can come out as .webm etc.; ingest reads any container.
        files = [p for p in tmp_dir.iterdir() if p.name.startswith("video.") and p.is_file()]
        if not files:
            raise InvalidInputError(f"could not download {url}: yt-dlp produced no file")
        os.replace(max(files, key=lambda p: p.stat().st_size), out)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def job_from_url(
    jobs_dir: Path,
    url: str,
    job_id: str | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    ydl_factory: YDLFactory = _default_factory,
) -> Job:
    """Create (or reopen) the job for the video at ``url``, downloading it if needed.

    Raises InvalidInputError if the link can't be downloaded, or if ``job_id`` already
    belongs to a different video.
    """
    info = probe_url(url, ydl_factory=ydl_factory)
    job = Job(jobs_dir / (job_id or info.job_id))
    if job.meta_path.exists():
        meta = job.read_meta()
        previous = (meta.source_info.get("extractor"), meta.source_info.get("id"))
        if previous != (info.extractor, info.id):
            raise InvalidInputError(f"job {job.job_id!r} was created for {meta.source}, not {url}")
    job.root.mkdir(parents=True, exist_ok=True)
    if not (job.input_video.exists() or job.input_audio.exists()):
        if info.seekable:
            out, format_spec = job.input_audio, AUDIO_FORMAT
        else:
            out, format_spec = job.input_video, FORMAT
        download_video(
            url, out, format_spec=format_spec, on_progress=on_progress, ydl_factory=ydl_factory
        )
    if not job.meta_path.exists():
        job.write_meta(
            JobMeta(
                job_id=job.job_id,
                source=url,
                created_at=datetime.now(UTC),
                source_url=url,
                source_info=info.source_info(),
            )
        )
    return job
