"""Video links as input (#60): download a URL with yt-dlp into the job directory.

``echo-clips run <URL>`` probes the link (no download), picks a job id from the site
and video id (``youtube-jNQXAC9IVRw``), and downloads the best video up to 1080p plus
audio, merged to MP4, as the job's ``input.mp4``. Running the same link again reopens
that job and skips the download, whichever URL form is used (``youtu.be/...``,
``youtube.com/watch?v=...``).

Only clip content you have the rights to use.
"""

from __future__ import annotations

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

from echo.errors import InvalidInputError
from echo.job import Job
from echo.models import JobMeta

FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"
"""Best video up to 1080p + best audio; MP4/M4A first so merging needs no re-encode."""

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
    try:
        with ydl_factory(_base_params()) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise _error(url, e) from e
    if info.get("_type") in ("playlist", "multi_video"):
        raise InvalidInputError(f"{url} is a playlist; pass a link to a single video")
    return VideoInfo(
        url=url,
        id=str(info["id"]),
        extractor=str(info.get("extractor_key") or info.get("extractor") or "web"),
        title=info.get("title"),
        uploader=info.get("uploader"),
        duration=info.get("duration"),
    )


def download_video(
    url: str,
    out: Path,
    *,
    on_progress: ProgressCallback | None = None,
    ydl_factory: YDLFactory = _default_factory,
) -> None:
    """Download ``url`` to ``out`` atomically (a failed download leaves nothing at ``out``)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=out.parent, prefix=".download-"))

    def hook(status: dict[str, Any]) -> None:
        if on_progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        if total:
            on_progress(min(1.0, status.get("downloaded_bytes", 0) / total))

    params = _base_params() | {
        "format": FORMAT,
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
    if not job.input_video.exists():
        download_video(url, job.input_video, on_progress=on_progress, ydl_factory=ydl_factory)
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
