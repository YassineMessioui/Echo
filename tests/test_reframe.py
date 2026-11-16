import pytest

from echo.config import ClipConfig
from echo.errors import InvalidInputError
from echo.ffmpeg import ffmpeg, probe
from echo.reframe import TARGET, video_filter, video_filter_for

BLUR_1080 = (
    "split=2[bg][fg];"
    "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:2,"
    "scale=1080:1920,setsar=1[back];"
    "[fg]scale=1080:1080,setsar=1[front];"
    "[back][front]overlay=0:420,setsar=1"
)


@pytest.mark.parametrize(
    ("size", "config", "expected"),
    [
        # Wider than 4:3: full-height 9:16 crop placed by crop_x, then scale.
        ((1920, 1080), ClipConfig(), "crop=606:1080:657:0,scale=1080:1920,setsar=1"),
        ((1920, 1080), ClipConfig(crop_x=0), "crop=606:1080:0:0,scale=1080:1920,setsar=1"),
        ((1920, 1080), ClipConfig(crop_x=1), "crop=606:1080:1314:0,scale=1080:1920,setsar=1"),
        ((1280, 720), ClipConfig(), "crop=404:720:438:0,scale=1080:1920,setsar=1"),
        (
            (1280, 720),
            ClipConfig(background="black"),
            "crop=404:720:438:0,scale=1080:1920,setsar=1",
        ),
        ((1440, 1080), ClipConfig(), "crop=606:1080:417:0,scale=1080:1920,setsar=1"),
        # Already 9:16 (or within tolerance): scale only, never cropped.
        ((1080, 1920), ClipConfig(), "scale=1080:1920,setsar=1"),
        ((720, 1280), ClipConfig(), "scale=1080:1920,setsar=1"),
        ((720, 1280), ClipConfig(crop_x=0, background="black"), "scale=1080:1920,setsar=1"),
        ((1080, 1918), ClipConfig(), "scale=1080:1920,setsar=1"),
        # 1:1 and narrower than 9:16: fit inside and pad, never stretched.
        ((1080, 1080), ClipConfig(), BLUR_1080),
        ((1080, 1080), ClipConfig(crop_x=0), BLUR_1080),
        (
            (1080, 1080),
            ClipConfig(background="black"),
            "scale=1080:1080,pad=1080:1920:0:420:black,setsar=1",
        ),
        (
            (720, 1600),
            ClipConfig(background="black"),
            "scale=864:1920,pad=1080:1920:108:0:black,setsar=1",
        ),
        # Odd fitted sizes are rounded down to even.
        (
            (1001, 999),
            ClipConfig(background="black"),
            "scale=1080:1076,pad=1080:1920:0:422:black,setsar=1",
        ),
    ],
)
def test_video_filter(size, config, expected):
    assert video_filter(*size, config) == expected


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (1080, 1080), (720, 1280)])
@pytest.mark.parametrize("background", ["blur", "black"])
def test_source_aspect_keeps_framing(size, background):
    assert video_filter(*size, ClipConfig(aspect="source", background=background)) is None


def test_invalid_size():
    with pytest.raises(ValueError, match="invalid video size"):
        video_filter(0, 1080, ClipConfig())


def test_source_aspect_skips_probe(tmp_path):
    assert video_filter_for(tmp_path / "missing.mp4", ClipConfig(aspect="source")) is None


@pytest.mark.ffmpeg
@pytest.mark.parametrize("size", ["320x180", "180x320", "240x240"])
@pytest.mark.parametrize("background", ["blur", "black"])
def test_reframe_outputs_vertical(make_video, tmp_path, size, background):
    src = make_video(duration=1.0, size=size)
    config = ClipConfig(background=background)
    out = tmp_path / "out.mp4"
    vf = video_filter_for(src, config)
    assert vf is not None
    ffmpeg(["-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", str(out)])
    info, before = probe(out), probe(src)
    assert (info.width, info.height) == TARGET
    assert info.duration == pytest.approx(before.duration, abs=0.1)


@pytest.mark.ffmpeg
def test_video_filter_for_rejects_audio_only(tmp_path):
    wav = tmp_path / "a.wav"
    ffmpeg(["-f", "lavfi", "-i", "sine=duration=0.5", str(wav)])
    with pytest.raises(InvalidInputError, match="no video stream"):
        video_filter_for(wav, ClipConfig())
