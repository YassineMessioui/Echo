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
    with pytest.raises(ExternalToolError, match="exited with code"):
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
