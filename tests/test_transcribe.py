import math
import os
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from echo.config import PipelineConfig, TranscribeConfig
from echo.errors import ArtifactError, TranscriptionError
from echo.job import Job
from echo.models import Transcript
from echo.testing import FakeTranscriber
from echo.transcribe import TranscribeStage, WhisperTranscriber, load_wav


def _write_wav(
    path: Path, seconds: float = 0.5, rate: int = 16_000, channels: int = 1, width: int = 2
) -> Path:
    """A 440 Hz tone at half amplitude."""
    n = int(seconds * rate)
    tone = [int(16384 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        if width == 2:
            frames = b"".join(struct.pack("<h", v) * channels for v in tone)
        else:
            frames = bytes(n * channels * width)
        wav.writeframes(frames)
    return path


def _word(start: float, end: float, word: str, probability: float | None = 0.9):
    return SimpleNamespace(start=start, end=end, word=word, probability=probability)


def _segment(start: float, end: float, text: str, words: list | None = None):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


@dataclass
class FakeWhisperModel:
    segments: list[Any]
    language: str = "en"
    duration: float = 30.0
    calls: list[tuple[np.ndarray, dict]] = field(default_factory=list)

    def transcribe(self, audio: np.ndarray, **kwargs: Any):
        self.calls.append((audio, kwargs))
        info = SimpleNamespace(language=self.language, duration=self.duration)
        return iter(self.segments), info


def _transcriber(model: FakeWhisperModel, **config: Any) -> tuple[WhisperTranscriber, list]:
    loaded: list[TranscribeConfig] = []

    def factory(cfg: TranscribeConfig) -> FakeWhisperModel:
        loaded.append(cfg)
        return model

    return WhisperTranscriber(TranscribeConfig(**config), model_factory=factory), loaded


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    return _write_wav(tmp_path / "audio.wav")


def test_transcribes_with_word_timestamps_and_sequential_ids(audio: Path):
    model = FakeWhisperModel(
        [
            _segment(
                0.0, 2.5, " Hello there.", [_word(0.0, 1.0, " Hello"), _word(1.1, 2.4, " there.")]
            ),
            _segment(2.5, 5.0, " General Kenobi.", [_word(2.6, 5.0, " General")]),
        ]
    )
    transcriber, _ = _transcriber(model, language="en")
    transcript = transcriber.transcribe(audio)

    assert isinstance(transcript, Transcript)
    assert transcript.language == "en"
    assert transcript.duration == 30.0
    assert [s.id for s in transcript.segments] == [0, 1]
    assert transcript.segments[0].text == "Hello there."
    assert [w.text for w in transcript.segments[0].words] == ["Hello", "there."]
    assert transcript.segments[0].words[1].start == 1.1

    ((samples, kwargs),) = model.calls
    assert samples.dtype == np.float32 and samples.shape == (8000,)
    assert kwargs == {"language": "en", "word_timestamps": True}


def test_model_config_and_name(audio: Path):
    model = FakeWhisperModel([])
    transcriber, loaded = _transcriber(model, model_size="tiny", device="cpu", compute_type="int8")
    assert transcriber.model_name == "faster-whisper/tiny"
    transcriber.transcribe(audio)
    transcriber.transcribe(audio)
    assert len(loaded) == 1, "model is loaded once and reused"
    assert (loaded[0].model_size, loaded[0].device, loaded[0].compute_type) == (
        "tiny",
        "cpu",
        "int8",
    )
    assert model.calls[0][1]["language"] is None  # auto-detect by default


def test_ids_contiguous_and_timestamps_monotonic(audio: Path):
    model = FakeWhisperModel(
        [
            _segment(0.0, 4.0, "one"),
            _segment(3.5, 6.0, "starts before the previous one"),  # whisper can go backwards
            _segment(6.0, 6.0, "   "),  # empty: dropped
            _segment(7.0, 31.0, "runs past the end", [_word(30.5, 31.2, "end", 1.2)]),
        ],
        duration=30.0,
    )
    transcriber, _ = _transcriber(model)
    transcript = transcriber.transcribe(audio)

    segments = transcript.segments
    assert [s.id for s in segments] == [0, 1, 2]
    assert [s.text for s in segments][2] == "runs past the end"
    for prev, cur in zip(segments, segments[1:], strict=False):
        assert cur.start >= prev.start
    assert segments[1].start == 3.5
    assert segments[-1].end == 30.0
    (word,) = segments[-1].words
    assert (word.start, word.end, word.probability) == (30.0, 30.0, 1.0)


def test_missing_audio(tmp_path: Path):
    transcriber, loaded = _transcriber(FakeWhisperModel([]))
    with pytest.raises(TranscriptionError, match="not found"):
        transcriber.transcribe(tmp_path / "nope.wav")
    assert not loaded


def test_model_failures_become_transcription_errors(audio: Path):
    def broken(config: TranscribeConfig):
        raise RuntimeError("CUDA out of memory")

    transcriber = WhisperTranscriber(TranscribeConfig(), model_factory=broken)
    with pytest.raises(TranscriptionError, match="faster-whisper/small failed.*CUDA out of memory"):
        transcriber.transcribe(audio)


def test_decoding_failures_become_transcription_errors(audio: Path):
    def segments():
        yield _segment(0, 1, "ok")
        raise ValueError("corrupt audio")

    model = FakeWhisperModel([])
    model.transcribe = lambda audio, **kw: (segments(), SimpleNamespace(language="en", duration=5))
    transcriber, _ = _transcriber(model)
    with pytest.raises(TranscriptionError, match="corrupt audio"):
        transcriber.transcribe(audio)


# --- WAV loading ----------------------------------------------------------------------------


def test_load_wav_returns_float_samples(tmp_path: Path):
    samples = load_wav(_write_wav(tmp_path / "a.wav", seconds=1))
    assert samples.dtype == np.float32
    assert samples.shape == (16_000,)
    assert 0.49 < float(np.abs(samples).max()) <= 0.5


def test_load_wav_mixes_stereo_to_mono(tmp_path: Path):
    samples = load_wav(_write_wav(tmp_path / "a.wav", seconds=1, channels=2))
    assert samples.shape == (16_000,)
    assert 0.49 < float(np.abs(samples).max()) <= 0.5


@pytest.mark.parametrize(("rate", "width"), [(44_100, 2), (16_000, 4)])
def test_load_wav_rejects_other_formats(tmp_path: Path, rate: int, width: int):
    path = _write_wav(tmp_path / "a.wav", rate=rate, width=width)
    with pytest.raises(TranscriptionError, match="expected 16 kHz 16-bit PCM"):
        load_wav(path)


def test_load_wav_rejects_non_wav(tmp_path: Path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"not a wav")
    with pytest.raises(TranscriptionError, match="not a PCM WAV"):
        load_wav(path)


# --- stage ---------------------------------------------------------------------------------


def test_stage_writes_transcript_and_records_model(job_copy: Job, golden_job: Job):
    transcript = golden_job.read_transcript()
    job_copy.transcript_path.unlink()
    _write_wav(job_copy.audio_path)
    fake = FakeTranscriber(transcript, model_name="faster-whisper/tiny")
    stage = TranscribeStage(fake)
    assert stage.outputs(job_copy) == [job_copy.transcript_path]

    stage.run(job_copy, PipelineConfig())

    assert job_copy.read_transcript() == transcript
    assert fake.calls == [job_copy.audio_path]
    meta = job_copy.read_meta()
    assert meta.models["whisper"] == "faster-whisper/tiny"
    assert meta.models["llm"] == "ollama/llama3.2:3b", "other entries are kept"


def test_stage_requires_audio(job_copy: Job, golden_job: Job):
    stage = TranscribeStage(FakeTranscriber(golden_job.read_transcript()))
    with pytest.raises(ArtifactError, match="audio.wav"):
        stage.run(job_copy, PipelineConfig())


# --- real model (opt-in: uv run pytest -m model) -------------------------------------------


@pytest.mark.model
def test_real_whisper_model(tmp_path: Path):
    """Runs the real faster-whisper model (downloads it on first run).

    Uses ``ECHO_TEST_AUDIO`` if set (a short speech clip as 16 kHz 16-bit PCM WAV),
    otherwise a generated tone, which only checks that the model loads and runs.
    """
    audio = Path(os.environ["ECHO_TEST_AUDIO"]) if "ECHO_TEST_AUDIO" in os.environ else None
    audio = audio or _write_wav(tmp_path / "tone.wav", seconds=3)
    config = TranscribeConfig(model_size=os.environ.get("ECHO_TEST_WHISPER", "tiny"), device="cpu")
    transcript = WhisperTranscriber(config).transcribe(audio)

    assert transcript.duration > 0
    assert [s.id for s in transcript.segments] == list(range(len(transcript.segments)))
    if "ECHO_TEST_AUDIO" in os.environ:
        assert transcript.segments, "expected speech in ECHO_TEST_AUDIO"
        assert any(s.words for s in transcript.segments)


# --- first run on Windows (#65) ---------------------------------------------------------


def test_quiet_model_libraries_sets_defaults_without_overriding(monkeypatch):
    from echo import transcribe

    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    monkeypatch.setenv("HF_HUB_VERBOSITY", "debug")  # the user's own choice wins
    transcribe._quiet_model_libraries()
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_VERBOSITY"] == "debug"


def test_first_download_is_announced_once(monkeypatch, caplog):
    import faster_whisper.utils

    from echo import transcribe

    def not_cached(size, local_files_only=False):
        raise FileNotFoundError(size)

    monkeypatch.setattr(faster_whisper.utils, "download_model", not_cached)
    with caplog.at_level("WARNING", logger="echo.transcribe"):
        transcribe._announce_download("small")
    assert caplog.messages == ['downloading whisper model "small" (~480 MB, first run only)']

    caplog.clear()
    monkeypatch.setattr(faster_whisper.utils, "download_model", lambda *a, **k: "/cache/small")
    with caplog.at_level("WARNING", logger="echo.transcribe"):
        transcribe._announce_download("small")
    assert caplog.messages == [], "nothing printed once the model is cached"
