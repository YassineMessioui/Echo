import sys

import pytest

from echo import ffmpeg
from echo.errors import ExternalToolError, InvalidInputError


def test_missing_tool_is_reported(monkeypatch):
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _: None)
    with pytest.raises(ExternalToolError, match="not found on PATH"):
        ffmpeg.require("ffmpeg")


def test_probe_missing_file(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        ffmpeg.probe(tmp_path / "nope.mp4")


@pytest.mark.ffmpeg
def test_failure_includes_stderr_tail(tmp_path):
    with pytest.raises(ExternalToolError, match=r"(?s)exited with code.*No such file"):
        ffmpeg.ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "out.wav")])


@pytest.mark.ffmpeg
def test_probe_generated_video(make_video):
    info = ffmpeg.probe(make_video(duration=2, size="320x180"))
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (320, 180)
    assert info.duration == pytest.approx(2, abs=0.1)


@pytest.mark.ffmpeg
def test_probe_video_without_audio(make_video):
    info = ffmpeg.probe(make_video(audio=False))
    assert not info.has_audio and info.sample_rate is None


@pytest.mark.ffmpeg
def test_probe_rejects_non_media(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("not a video")
    with pytest.raises(InvalidInputError, match="not a readable media file"):
        ffmpeg.probe(bogus)


def _fake_tool(monkeypatch):
    """Make ``run("fake", ...)`` execute the current Python interpreter."""
    monkeypatch.setattr(ffmpeg, "require", lambda tool: sys.executable)


def test_non_ascii_output_is_decoded_as_utf8(monkeypatch):
    _fake_tool(monkeypatch)
    script = (
        "import sys; sys.stderr.buffer.write('Łódź 🎬.mp4: invalid data'.encode()); "
        "sys.stdout.buffer.write(b'\\xff'); sys.exit(1)"
    )
    with pytest.raises(ExternalToolError, match="Łódź 🎬.mp4: invalid data"):
        ffmpeg.run("fake", ["-c", script])


def test_timeout_is_reported(monkeypatch):
    _fake_tool(monkeypatch)
    with pytest.raises(ExternalToolError, match="timed out") as e:
        ffmpeg.run("fake", ["-c", "import time; time.sleep(5)  # -headers X"], timeout=0.2)
    # argv (which can hold -headers) is in neither the message nor a chained cause.
    assert "-headers" not in str(e.value) and e.value.__cause__ is None
    assert e.value.__suppress_context__


def test_failure_message_has_stderr_not_argv(monkeypatch):
    _fake_tool(monkeypatch)
    script = "import sys; sys.stderr.write('boom'); sys.exit(1)  # -headers User-Agent: UA"
    with pytest.raises(ExternalToolError, match="boom") as e:
        ffmpeg.run("fake", ["-c", script])
    assert "-headers" not in str(e.value)


@pytest.mark.ffmpeg
def test_probe_ignores_cover_art(tmp_path, make_video):
    cover = tmp_path / "cover.png"
    ffmpeg.ffmpeg(["-f", "lavfi", "-i", "color=red:size=64x64", "-frames:v", "1", str(cover)])
    audio = tmp_path / "song.m4a"
    ffmpeg.ffmpeg(
        ["-f", "lavfi", "-i", "sine=duration=1", "-i", str(cover), "-map", "0:a", "-map", "1:v"]
        + ["-c:a", "aac", "-c:v", "png", "-disposition:v", "attached_pic", str(audio)]
    )
    info = ffmpeg.probe(audio)
    assert info.has_audio and not info.has_video and info.width is None


def test_has_filter_parses_the_filter_list(monkeypatch):
    listing = (
        "Filters:\n  T.. = Timeline support\n  ------\n"
        " TSC aformat           A->A       Convert the input audio to one of the formats.\n"
        " T.. subtitles         V->V       Render text subtitles using libass.\n"
    )

    class Result:
        stdout = listing

    ffmpeg._filters.cache_clear()
    monkeypatch.setattr(ffmpeg, "run", lambda tool, args: Result())
    try:
        assert ffmpeg.has_filter("subtitles")
        assert ffmpeg.has_filter("aformat")
        assert not ffmpeg.has_filter("Timeline")
        assert not ffmpeg.has_filter("drawtext")
    finally:
        ffmpeg._filters.cache_clear()


@pytest.mark.ffmpeg
def test_http_options_are_parsed():
    options = ffmpeg.http_options()
    assert {"headers", "reconnect", "multiple_requests"} <= options
    assert all(not name.startswith("-") for name in options)
