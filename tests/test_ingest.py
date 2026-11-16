import wave
from pathlib import Path

import pytest

from echo import ffmpeg, ingest
from echo.config import IngestConfig, PipelineConfig
from echo.errors import ExternalToolError, InvalidInputError
from echo.ingest import IngestStage, extract_audio
from echo.job import Job
from echo.models import StageName


def _wav_info(path: Path) -> tuple[int, int, int, float]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        return rate, wav.getnchannels(), wav.getsampwidth(), wav.getnframes() / rate


@pytest.mark.ffmpeg
def test_extracts_16k_mono_pcm(make_video, tmp_path: Path):
    video = make_video(duration=2.0)
    out = tmp_path / "audio.wav"
    extract_audio(video, out, IngestConfig())

    rate, channels, width, duration = _wav_info(out)
    assert (rate, channels, width) == (16_000, 1, 2)
    assert duration == pytest.approx(ffmpeg.probe(video).duration, abs=0.1)
    assert not ffmpeg.probe(out).has_video
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


@pytest.mark.ffmpeg
def test_honours_config(make_video, tmp_path: Path):
    out = tmp_path / "audio.wav"
    extract_audio(make_video(duration=1.0), out, IngestConfig(sample_rate=8_000, channels=2))
    assert _wav_info(out)[:3] == (8_000, 2, 2)


@pytest.mark.ffmpeg
def test_overwrites_existing_output(make_video, tmp_path: Path):
    out = tmp_path / "audio.wav"
    out.write_bytes(b"stale")
    extract_audio(make_video(duration=1.0), out, IngestConfig())
    assert _wav_info(out)[0] == 16_000


def test_missing_input(tmp_path: Path):
    with pytest.raises(InvalidInputError, match="not found"):
        extract_audio(tmp_path / "nope.mp4", tmp_path / "audio.wav", IngestConfig())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.ffmpeg
def test_non_media_input(tmp_path: Path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("not a video")
    with pytest.raises(InvalidInputError, match="not a readable media file"):
        extract_audio(bogus, tmp_path / "audio.wav", IngestConfig())
    assert list(tmp_path.iterdir()) == [bogus]


@pytest.mark.ffmpeg
def test_video_without_audio(make_video, tmp_path: Path):
    video = make_video(audio=False)
    out = tmp_path / "audio.wav"
    with pytest.raises(InvalidInputError, match="no audio stream"):
        extract_audio(video, out, IngestConfig())
    assert not out.exists()


def test_ffmpeg_missing(monkeypatch, tmp_path: Path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _: None)
    with pytest.raises(ExternalToolError, match="not found on PATH"):
        extract_audio(video, tmp_path / "audio.wav", IngestConfig())


@pytest.mark.ffmpeg
def test_failed_encode_leaves_no_partial_output(make_video, monkeypatch, tmp_path: Path):
    video = make_video(duration=1.0)
    out = tmp_path / "audio.wav"
    out.write_bytes(b"previous run")

    def failing_ffmpeg(args: list[str]) -> None:
        Path(args[-1]).write_bytes(b"partial")  # ffmpeg dies halfway through writing
        raise ExternalToolError("ffmpeg exited with code 1:\nboom")

    monkeypatch.setattr(ingest, "ffmpeg", failing_ffmpeg)
    with pytest.raises(ExternalToolError, match="boom"):
        extract_audio(video, out, IngestConfig())
    assert out.read_bytes() == b"previous run"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.wav", video.name]


@pytest.mark.ffmpeg
def test_stage_writes_job_audio(make_video, tmp_path: Path):
    job = Job.create(tmp_path / "jobs", make_video(duration=1.5))
    stage = IngestStage()

    assert stage.name is StageName.INGEST
    assert stage.outputs(job) == [job.audio_path]
    stage.run(job, PipelineConfig())

    rate, channels, width, duration = _wav_info(job.audio_path)
    assert (rate, channels, width) == (16_000, 1, 2)
    assert duration == pytest.approx(1.5, abs=0.1)


def test_stage_passes_ingest_config(job_copy: Job):
    calls: list[tuple[Path, Path, IngestConfig]] = []
    config = PipelineConfig(ingest=IngestConfig(sample_rate=22_050))

    IngestStage(extractor=lambda *args: calls.append(args)).run(job_copy, config)
    assert calls == [(job_copy.input_video, job_copy.audio_path, config.ingest)]
