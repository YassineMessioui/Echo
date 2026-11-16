import os
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import DownloadError

from echo import download
from echo.download import VideoInfo, download_video, is_url, job_from_url, probe_url
from echo.errors import InvalidInputError
from echo.job import Job

URL = "https://youtu.be/abcDEF12345"
INFO = {
    "id": "abcDEF12345",
    "extractor_key": "Youtube",
    "title": "A talk",
    "uploader": "Someone",
    "duration": 615.0,
}


class FakeYDL:
    """Stands in for ``yt_dlp.YoutubeDL``: returns ``info``; on download writes the file
    yt-dlp would, calling the progress hooks."""

    def __init__(self, params, *, info=INFO, error=None, ext="mp4", calls=None):
        self.params, self.info, self.error, self.ext = params, info, error, ext
        self.calls = calls if calls is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self.calls.append(("download" if download else "probe", url))
        if self.error is not None:
            raise self.error
        if download:
            for hook in self.params["progress_hooks"]:
                hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
                hook({"status": "finished"})
            Path(self.params["outtmpl"].replace("%(ext)s", self.ext)).write_bytes(b"video")
        return self.info


def factory(calls=None, **kwargs):
    calls = [] if calls is None else calls
    return lambda params: FakeYDL(params, calls=calls, **kwargs)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("http://vimeo.com/123", True),
        ("episode.mp4", False),
        ("/Users/me/Videos/a b.mp4", False),
        (r"C:\Videos\episode.mp4", False),
        ("ftp://host/file.mp4", False),
        ("https://", False),
    ],
)
def test_is_url(text, expected):
    assert is_url(text) is expected


def test_job_id_is_stable_and_keeps_case():
    assert VideoInfo(URL, "abcDEF12345", "Youtube").job_id == "youtube-abcDEF12345"
    assert VideoInfo(URL, "a/b c", "Generic").job_id == "generic-a-b-c"


def test_probe_reads_metadata():
    info = probe_url(URL, ydl_factory=factory())
    assert (info.id, info.extractor, info.title, info.duration) == (
        "abcDEF12345",
        "Youtube",
        "A talk",
        615.0,
    )


def test_probe_rejects_playlists():
    with pytest.raises(InvalidInputError, match="playlist"):
        probe_url(URL, ydl_factory=factory(info={"_type": "playlist", "id": "PL1"}))


@pytest.mark.parametrize(
    "message",
    ["ERROR: [youtube] abc: Private video", "ERROR: [youtube] abc: Video unavailable"],
)
def test_download_errors_become_invalid_input(message):
    with pytest.raises(InvalidInputError, match="could not download") as e:
        probe_url(URL, ydl_factory=factory(error=DownloadError(message)))
    assert "ERROR:" not in str(e.value)


def test_download_is_atomic_and_reports_progress(tmp_path):
    out = tmp_path / "job" / "input.mp4"
    progress: list[float] = []
    download_video(URL, out, on_progress=progress.append, ydl_factory=factory())
    assert out.read_bytes() == b"video"
    assert progress == [0.5]
    assert [p.name for p in out.parent.iterdir()] == ["input.mp4"]  # no temp dir left


def test_failed_download_leaves_nothing(tmp_path):
    out = tmp_path / "job" / "input.mp4"
    with pytest.raises(InvalidInputError):
        download_video(URL, out, ydl_factory=factory(error=DownloadError("ERROR: geo-blocked")))
    assert list(out.parent.iterdir()) == []


def test_single_file_formats_are_kept_as_input(tmp_path):
    out = tmp_path / "input.mp4"
    download_video(URL, out, ydl_factory=factory(ext="webm"))
    assert out.read_bytes() == b"video"


def test_job_from_url_creates_job_with_source_info(tmp_path):
    job = job_from_url(tmp_path, URL, ydl_factory=factory())
    assert job.root == tmp_path / "youtube-abcDEF12345"
    assert job.input_video.is_file()
    meta = job.read_meta()
    assert meta.source == URL and meta.source_url == URL
    assert meta.source_info == {
        "id": "abcDEF12345",
        "extractor": "Youtube",
        "title": "A talk",
        "uploader": "Someone",
        "duration": 615.0,
    }


def test_same_video_reuses_the_job_without_downloading(tmp_path):
    job_from_url(tmp_path, URL, ydl_factory=factory())
    calls: list[tuple[str, str]] = []
    other_form = "https://www.youtube.com/watch?v=abcDEF12345"
    again = job_from_url(tmp_path, other_form, ydl_factory=factory(calls))
    assert again.root == tmp_path / "youtube-abcDEF12345"
    assert calls == [("probe", other_form)]


def test_job_id_of_another_video_is_rejected(tmp_path):
    job_from_url(tmp_path, URL, job_id="ep1", ydl_factory=factory())
    other = INFO | {"id": "zzz"}
    with pytest.raises(InvalidInputError, match="was created for"):
        job_from_url(
            tmp_path, "https://youtu.be/zzz", job_id="ep1", ydl_factory=factory(info=other)
        )


def test_local_file_job_id_is_rejected_for_a_url(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    Job.create(tmp_path / "jobs", video, job_id="ep1")
    with pytest.raises(InvalidInputError, match="was created for"):
        job_from_url(tmp_path / "jobs", URL, job_id="ep1", ydl_factory=factory())


def test_js_runtime_is_enabled_when_installed(monkeypatch):
    monkeypatch.setattr(
        download.shutil, "which", lambda name: f"/bin/{name}" if name == "node" else None
    )
    assert download._base_params()["js_runtimes"] == {"node": {}}


@pytest.mark.skipif(
    not os.environ.get("ECHO_LIVE_DOWNLOAD"), reason="set ECHO_LIVE_DOWNLOAD=1 to run"
)
def test_live_download(tmp_path):
    """Downloads a short public-domain clip (NASA, Wikimedia Commons). Override with
    ``ECHO_LIVE_DOWNLOAD_URL``."""
    from echo.ffmpeg import probe

    url = os.environ.get(
        "ECHO_LIVE_DOWNLOAD_URL",
        "https://commons.wikimedia.org/wiki/File:Apollo_15_feather_and_hammer_drop.ogv",
    )
    job = job_from_url(tmp_path, url)
    info = probe(job.input_video)
    assert info.has_video and info.duration > 0
