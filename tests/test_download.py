import logging
import os
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import DownloadError

from echo import download
from echo.download import VideoInfo, download_video, is_url, job_from_url, probe_url
from echo.errors import ArtifactError, InvalidInputError
from echo.job import Job
from echo.progress import capture

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


# --- audio first / remote video (#75) ------------------------------------------------

SEEKABLE = INFO | {
    "requested_formats": [
        {
            "format_id": "137",
            "vcodec": "avc1",
            "protocol": "https",
            "url": "https://video.example/137",
            "width": 1920,
            "height": 1080,
            "http_headers": {"User-Agent": "UA"},
        },
        {"format_id": "140", "vcodec": "none", "protocol": "https", "url": "https://a"},
    ]
}
DASH_ONLY = INFO | {"protocol": "http_dash_segments", "vcodec": "avc1"}


def test_seekable_link_downloads_only_the_audio(tmp_path):
    calls: list[tuple[str, str]] = []
    params: list[dict] = []

    def recording(p):
        params.append(p)
        return FakeYDL(p, info=SEEKABLE, calls=calls)

    job = job_from_url(tmp_path, URL, ydl_factory=recording)
    assert job.input_audio.is_file() and not job.input_video.exists()
    assert params[-1]["format"] == download.AUDIO_FORMAT


def test_unseekable_link_downloads_the_full_video(tmp_path):
    job = job_from_url(tmp_path, URL, ydl_factory=factory(info=DASH_ONLY))
    assert job.input_video.is_file() and not job.input_audio.exists()


def test_resolve_video_returns_the_stream(tmp_path):
    remote = download.resolve_video(URL, tmp_path / "a.m4a", ydl_factory=factory(info=SEEKABLE))
    assert remote is not None
    assert (remote.url, remote.width, remote.height) == ("https://video.example/137", 1920, 1080)
    assert remote.headers == {"User-Agent": "UA"} and remote.audio == tmp_path / "a.m4a"
    assert (
        download.resolve_video(URL, tmp_path / "a.m4a", ydl_factory=factory(info=DASH_ONLY)) is None
    )


def test_video_source(tmp_path):
    job = job_from_url(tmp_path, URL, ydl_factory=factory(info=SEEKABLE))
    remote = download.video_source(job, ydl_factory=factory(info=SEEKABLE))
    assert remote.url == "https://video.example/137" and remote.audio == job.input_audio

    # The stream stopped being seekable since: fall back to a full download, once.
    assert download.video_source(job, ydl_factory=factory(info=DASH_ONLY)) == job.input_video
    assert job.input_video.is_file()
    assert download.video_source(job, ydl_factory=factory(error=AssertionError)) == job.input_video


def test_video_source_fallback_logs_and_reports_progress(tmp_path, caplog):
    job = job_from_url(tmp_path, URL, ydl_factory=factory(info=SEEKABLE))
    reports: list[tuple[float, str]] = []
    with (
        caplog.at_level(logging.INFO, logger="echo.download"),
        capture(lambda f, d: reports.append((f, d))),
    ):
        assert download.video_source(job, ydl_factory=factory(info=DASH_ONLY)) == job.input_video
    assert reports == [(0.5, "downloading full video")]  # the fake reports 50/100 bytes
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "can't be read remotely" in record.message and "full video" in record.message
    assert "http_dash_segments" in record.message  # why


@pytest.mark.parametrize("name", ["Cookie", "Authorization", "X-Session-Token"])
def test_streams_needing_credentials_are_not_read_remotely(tmp_path, caplog, name):
    # Remote headers go on ffmpeg's argv (process list), so credentials never do:
    # such a stream is downloaded in full by yt-dlp instead.
    video, audio = SEEKABLE["requested_formats"]
    headers = {"User-Agent": "UA", name: "s3cret"}
    info = SEEKABLE | {"requested_formats": [video | {"http_headers": headers}, audio]}
    assert download.resolve_video(URL, tmp_path / "a.m4a", ydl_factory=factory(info=info)) is None
    job = job_from_url(tmp_path, URL, ydl_factory=factory(info=SEEKABLE))
    with caplog.at_level(logging.INFO, logger="echo.download"):
        assert download.video_source(job, ydl_factory=factory(info=info)) == job.input_video
    assert name in caplog.text and "s3cret" not in caplog.text


def test_video_source_without_any_input(tmp_path):
    job = Job(tmp_path / "j")
    job.root.mkdir()
    with pytest.raises(ArtifactError, match="input.mp4"):
        download.video_source(job, ydl_factory=factory())


def _picks(formats: list[dict]) -> str:
    """The format id yt-dlp's own selector picks with ``VIDEO_FORMAT``."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        selector = ydl.build_format_selector(download.VIDEO_FORMAT)
        ctx = {"formats": formats, "has_merged_format": False, "incomplete_formats": False}
        return next(iter(selector(ctx)))["format_id"]


def _fmt(format_id: str, height: int, vcodec: str) -> dict:
    return {
        "format_id": format_id,
        "height": height,
        "width": height * 16 // 9,
        "vcodec": vcodec,
        "acodec": "none",
        "ext": "mp4" if vcodec.startswith("avc1") else "webm",
        "url": f"https://example.com/{format_id}",
        "protocol": "https",
    }


def test_remote_video_prefers_the_sharpest_stream_up_to_4k():
    """A 9:16 crop of 1080p is stretched 1.8x; from 2160p it's shrunk, so it stays sharp."""
    formats = [
        _fmt("137", 1080, "avc1.640028"),
        _fmt("248", 1080, "vp09.00.40.08"),
        _fmt("271", 1440, "vp09.00.50.08"),
        _fmt("401", 2160, "av01.0.12M.08"),
        _fmt("313", 2160, "vp09.00.51.08"),
        _fmt("571", 4320, "av01.0.16M.08"),
    ]
    assert _picks(formats) == "313"  # 4K VP9: never 8K, and VP9 before AV1
    assert _picks([f for f in formats if f["format_id"] != "313"]) == "401"  # 4K AV1
    assert _picks(formats[:2]) == "137"  # up to 1080p: H.264 first, as before
    hls = _fmt("625", 2160, "vp09.00.50.08") | {"protocol": "m3u8_native"}
    assert _picks([*formats, hls]) == "313"  # HTTPS (byte-range seeks) before HLS
    assert _picks([formats[0], hls]) == "625"  # but HLS 4K still beats 1080p
