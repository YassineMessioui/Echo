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

from echo import progress
from echo.config import PipelineConfig, TranscribeConfig
from echo.errors import ArtifactError, TranscriptionError
from echo.job import Job
from echo.models import Transcript
from echo.pipeline import run_pipeline
from echo.testing import FakeTranscriber
from echo.transcribe import TranscribeStage, WhisperTranscriber, fixed_chunks, load_wav


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


@dataclass
class FakeBatchedPipeline:
    """Stands in for ``BatchedInferencePipeline``: records that the model was wrapped."""

    model: FakeWhisperModel

    def transcribe(self, audio: np.ndarray, **kwargs: Any):
        return self.model.transcribe(audio, batched=True, **kwargs)


def _transcriber(model: FakeWhisperModel, **config: Any) -> tuple[WhisperTranscriber, list]:
    loaded: list[TranscribeConfig] = []

    def factory(cfg: TranscribeConfig) -> FakeWhisperModel:
        loaded.append(cfg)
        return model

    transcriber = WhisperTranscriber(
        TranscribeConfig(**config), model_factory=factory, pipeline_factory=FakeBatchedPipeline
    )
    return transcriber, loaded


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
    assert kwargs == {"language": "en", "word_timestamps": True, "vad_filter": True}


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


# --- settings fingerprint: re-transcribe when the mode changes (#82) -----------------------


def _config(**transcribe: Any) -> PipelineConfig:
    return PipelineConfig(transcribe=transcribe)


def test_settings_fingerprint_is_readable():
    stage = TranscribeStage(FakeTranscriber(Transcript(language="en", duration=0, segments=[])))
    assert stage.settings(_config()) == "small|lang=auto|sequential|vad(min_silence=2000ms)"
    assert (
        stage.settings(_config(model_size="tiny", language="fr", batch_size=8))
        == "tiny|lang=fr|batched|vad(min_silence=160ms)"
    )
    assert (
        stage.settings(_config(vad_filter=False, vad_min_silence_ms=500))
        == "small|lang=auto|sequential|no-vad"
    )


@pytest.fixture
def transcribed(job_copy: Job, golden_job: Job) -> tuple[Job, FakeTranscriber]:
    """A job transcribed once with the default settings, and the transcriber's fake."""
    job_copy.transcript_path.unlink()
    job_copy.audio_path.write_bytes(b"")
    fake = FakeTranscriber(golden_job.read_transcript())
    run_pipeline(job_copy, [TranscribeStage(fake)], PipelineConfig())
    assert len(fake.calls) == 1
    return job_copy, fake


@pytest.mark.parametrize(
    "change",
    [
        {"model_size": "tiny"},
        {"language": "en"},
        {"batch_size": 8},
        {"vad_filter": False},
        {"vad_min_silence_ms": 500},
    ],
)
def test_changed_settings_re_transcribe(transcribed: tuple[Job, FakeTranscriber], change):
    job, fake = transcribed
    config = _config(**change)
    run_pipeline(job, [TranscribeStage(fake)], config)
    assert len(fake.calls) == 2
    stage = TranscribeStage(fake)
    assert job.read_meta().stages[stage.name].settings == stage.settings(config)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"device": "cpu"},
        {"compute_type": "int8"},
        {"vad_min_silence_ms": 2000},  # the sequential default, spelled out
    ],
)
def test_unchanged_transcript_settings_skip(transcribed: tuple[Job, FakeTranscriber], change):
    job, fake = transcribed
    run_pipeline(job, [TranscribeStage(fake)], _config(**change))
    assert len(fake.calls) == 1


def test_old_job_without_settings_loads_and_is_not_re_transcribed(job_copy: Job, golden_job: Job):
    assert job_copy.read_meta().stages[TranscribeStage.name].settings is None  # pre-#82 job
    fake = FakeTranscriber(golden_job.read_transcript())
    config = _config(batch_size=8)
    stage = TranscribeStage(fake)
    run_pipeline(job_copy, [stage], config)
    assert fake.calls == []
    meta = job_copy.read_meta()
    assert meta.stages[stage.name].settings == stage.settings(config), "adopted"
    assert meta.models["whisper"] == "faster-whisper/small"
    assert "whisper_mode" not in meta.models


@pytest.mark.parametrize(
    ("change", "mode"),
    [
        ({}, "sequential, vad"),
        ({"batch_size": 8}, "batched-8, vad"),
        ({"vad_filter": False}, "sequential, no vad"),
    ],
)
def test_stage_records_the_whisper_mode(job_copy: Job, golden_job: Job, change, mode):
    _write_wav(job_copy.audio_path)
    stage = TranscribeStage(FakeTranscriber(golden_job.read_transcript()))
    stage.run(job_copy, _config(**change))
    models = job_copy.read_meta().models
    assert models["whisper_mode"] == mode
    assert models["whisper"] == "fake/whisper"


# --- batched pipeline + VAD (#58) ------------------------------------------------------


def test_batched_mode_wraps_the_model_and_asks_for_segment_timestamps(audio: Path):
    model = FakeWhisperModel([_segment(0.0, 1.0, "hi")])
    transcriber, loaded = _transcriber(model, batch_size=8)
    transcriber.transcribe(audio)
    transcriber.transcribe(audio)
    assert len(loaded) == 1, "model (and pipeline) built once"
    assert model.calls[0][1] == {
        "language": None,
        "word_timestamps": True,
        "vad_filter": True,
        "batch_size": 8,
        "without_timestamps": False,  # sentence-sized segments, not one per 30 s chunk
        "batched": True,
    }


def test_batched_mode_keeps_the_transcript_contract(audio: Path):
    model = FakeWhisperModel(
        [
            _segment(0.0, 29.8, " first chunk", [_word(0.0, 0.5, " first")]),
            _segment(29.5, 31.0, " overlaps it", [_word(29.6, 30.0, " overlaps")]),
            _segment(31.0, 31.0, " "),
            _segment(31.0, 35.0, " third", [_word(31.2, 31.6, " third")]),
        ],
        duration=40.0,
    )
    transcriber, _ = _transcriber(model, batch_size=16)
    transcript = transcriber.transcribe(audio)
    assert model.calls[0][1]["batched"] and model.calls[0][1]["batch_size"] == 16
    assert [s.id for s in transcript.segments] == [0, 1, 2]
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert all(s.words for s in transcript.segments)


def test_batched_without_vad_passes_fixed_30s_chunks(tmp_path: Path):
    audio = _write_wav(tmp_path / "long.wav", seconds=65)
    model = FakeWhisperModel([])
    transcriber, _ = _transcriber(model, batch_size=8, vad_filter=False)
    transcriber.transcribe(audio)
    ((_, kwargs),) = model.calls
    assert kwargs["vad_filter"] is False
    assert kwargs["clip_timestamps"] == [
        {"start": 0.0, "end": pytest.approx(65 / 3)},
        {"start": pytest.approx(65 / 3), "end": pytest.approx(130 / 3)},
        {"start": pytest.approx(130 / 3), "end": 65.0},
    ]


def test_vad_min_silence_is_passed_only_with_vad(audio: Path):
    model = FakeWhisperModel([])
    transcriber, _ = _transcriber(model, vad_min_silence_ms=500)  # sequential default
    transcriber.transcribe(audio)
    assert model.calls[-1][1]["vad_parameters"] == {"min_silence_duration_ms": 500}

    model = FakeWhisperModel([])
    transcriber, _ = _transcriber(model, vad_filter=False, vad_min_silence_ms=500)
    transcriber.transcribe(audio)
    assert "vad_parameters" not in model.calls[-1][1]


def test_fixed_chunks_short_and_exact_lengths():
    assert fixed_chunks(16_000 * 10) == [{"start": 0.0, "end": 10.0}]
    assert fixed_chunks(16_000 * 60) == [{"start": 0.0, "end": 30.0}, {"start": 30.0, "end": 60.0}]
    assert fixed_chunks(0) == []
    tail = fixed_chunks(int(16_000 * 60.05))  # no 50 ms last chunk
    assert len(tail) == 3 and min(c["end"] - c["start"] for c in tail) > 20
    assert tail[-1]["end"] == pytest.approx(60.05)
    assert all(a["end"] == b["start"] for a, b in zip(tail, tail[1:], strict=False))


def test_batched_cpu_fallback_rewraps_the_cpu_model(audio: Path):
    good = FakeWhisperModel([_segment(0.0, 1.0, "ok")])

    class NoCuda:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    wrapped: list[object] = []

    def pipeline(model):
        wrapped.append(model)
        return FakeBatchedPipeline(model) if model is good else model

    def factory(cfg: TranscribeConfig):
        return NoCuda() if cfg.device == "auto" else good

    transcriber = WhisperTranscriber(
        TranscribeConfig(device="auto", batch_size=8),
        model_factory=factory,
        pipeline_factory=pipeline,
    )
    assert [s.text for s in transcriber.transcribe(audio).segments] == ["ok"]
    assert len(wrapped) == 2 and wrapped[1] is good, "the CPU model is wrapped again"
    assert good.calls[0][1]["batched"], "and still runs batched"


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


@pytest.mark.model
@pytest.mark.ffmpeg
def test_real_batched_pipeline_on_fixture_podcast(tmp_path: Path):
    """Batched + VAD on the fixture podcast: real speech, words, contract intact."""
    from echo.config import IngestConfig
    from echo.ingest import extract_audio

    video = Path(__file__).parent / "fixtures" / "podcast" / "podcast.mp4"
    wav = tmp_path / "audio.wav"
    extract_audio(video, wav, IngestConfig())
    config = TranscribeConfig(
        model_size=os.environ.get("ECHO_TEST_WHISPER", "tiny"), device="cpu", batch_size=8
    )
    transcript = WhisperTranscriber(config).transcribe(wav)

    segments = transcript.segments
    assert len(segments) > 5
    assert [s.id for s in segments] == list(range(len(segments)))
    assert [s.start for s in segments] == sorted(s.start for s in segments)
    assert all(s.words for s in segments)
    assert segments[-1].end > 0.8 * transcript.duration, "VAD kept the speech to the end"
    text = " ".join(s.text for s in segments).lower()
    assert "caffeine" in text and "bicycle" in text


# --- first run on Windows (#65) ---------------------------------------------------------


def test_first_download_is_announced_once(monkeypatch, caplog):
    import faster_whisper.utils
    from huggingface_hub.errors import LocalEntryNotFoundError

    from echo import transcribe

    def not_cached(size, local_files_only=False):
        raise LocalEntryNotFoundError(size)

    monkeypatch.setattr(faster_whisper.utils, "download_model", not_cached)
    with caplog.at_level("WARNING", logger="echo.transcribe"):
        transcribe._announce_download("small")
    assert caplog.messages == ['downloading whisper model "small" (~480 MB, first run only)']

    caplog.clear()
    monkeypatch.setattr(faster_whisper.utils, "download_model", lambda *a, **k: "/cache/small")
    with caplog.at_level("WARNING", logger="echo.transcribe"):
        transcribe._announce_download("small")
    assert caplog.messages == [], "nothing printed once the model is cached"


def test_invalid_model_name_is_not_announced(caplog):
    from echo import transcribe

    with caplog.at_level("WARNING", logger="echo.transcribe"):
        transcribe._announce_download("large-v3-typo")
    assert caplog.messages == [], "WhisperModel reports the invalid name itself"


def test_auto_device_falls_back_to_cpu_without_cuda_libraries(audio: Path):
    """An NVIDIA card makes device="auto" pick CUDA; without the CUDA runtime installed,
    CTranslate2 fails on first use. Retry on the CPU instead of failing the stage."""
    good = FakeWhisperModel([_segment(0.0, 1.0, "hello")])

    class NoCuda:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    devices: list[str] = []

    def factory(cfg: TranscribeConfig):
        devices.append(cfg.device)
        return NoCuda() if cfg.device == "auto" else good

    transcriber = WhisperTranscriber(
        TranscribeConfig(device="auto", batch_size=0), model_factory=factory
    )
    transcript = transcriber.transcribe(audio)
    assert [s.text for s in transcript.segments] == ["hello"]
    assert devices == ["auto", "cpu"]
    transcriber.transcribe(audio)
    assert devices == ["auto", "cpu"], "stays on the CPU afterwards"


def test_explicit_cuda_device_does_not_fall_back(audio: Path):
    def factory(cfg: TranscribeConfig):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    transcriber = WhisperTranscriber(TranscribeConfig(device="cuda"), model_factory=factory)
    with pytest.raises(TranscriptionError, match="cublas64_12"):
        transcriber.transcribe(audio)


@pytest.mark.parametrize(
    "message",
    [
        # CTranslate2's real formats for ordinary GPU runtime errors: no fallback.
        "CUDA failed with error out of memory",
        "cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED",
        "cuDNN failed with status CUDNN_STATUS_EXECUTION_FAILED",
    ],
)
def test_real_gpu_errors_do_not_fall_back(audio: Path, message: str):
    devices: list[str] = []

    def factory(cfg: TranscribeConfig):
        devices.append(cfg.device)
        raise RuntimeError(message)

    transcriber = WhisperTranscriber(
        TranscribeConfig(device="auto", batch_size=0), model_factory=factory
    )
    with pytest.raises(TranscriptionError, match=message.split(" with ")[0]):
        transcriber.transcribe(audio)
    assert devices == ["auto"]


@pytest.mark.parametrize(
    "message",
    [
        "Library cublas64_12.dll is not found or cannot be loaded",
        "Could not load library libcudnn_ops.so.9",
        "libcublas.so.12: cannot open shared object file: No such file or directory",
        "CUDA failed with error no CUDA-capable device is detected",
        "CUDA failed with error CUDA driver version is insufficient for CUDA runtime version",
    ],
)
def test_missing_gpu_stack_falls_back(audio: Path, message: str):
    good = FakeWhisperModel([_segment(0.0, 1.0, "ok")])

    def factory(cfg: TranscribeConfig):
        if cfg.device == "auto":
            raise RuntimeError(message)
        return good

    transcriber = WhisperTranscriber(
        TranscribeConfig(device="auto", batch_size=0), model_factory=factory
    )
    assert [s.text for s in transcriber.transcribe(audio).segments] == ["ok"]


def test_cpu_fallback_resets_float16_compute_type(audio: Path):
    seen: list[tuple[str, str]] = []
    good = FakeWhisperModel([_segment(0.0, 1.0, "ok")])

    def factory(cfg: TranscribeConfig):
        seen.append((cfg.device, cfg.compute_type))
        if cfg.device == "auto":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return good

    config = TranscribeConfig(device="auto", compute_type="float16", batch_size=0)
    WhisperTranscriber(config, model_factory=factory).transcribe(audio)
    assert seen == [("auto", "float16"), ("cpu", "default")]


def test_failed_cpu_retry_keeps_the_cuda_error(audio: Path):
    def factory(cfg: TranscribeConfig):
        if cfg.device == "auto":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        raise RuntimeError("disk full")

    transcriber = WhisperTranscriber(
        TranscribeConfig(device="auto", batch_size=0), model_factory=factory
    )
    with pytest.raises(TranscriptionError, match=r"(?s)cublas64_12\.dll.*disk full"):
        transcriber.transcribe(audio)


def test_reports_progress_per_segment(audio: Path):
    model = FakeWhisperModel(
        [_segment(0.0, 750.0, " a"), _segment(750.0, 2000.0, " b"), _segment(2000.0, 3400.0, " c")],
        duration=3490.0,
    )
    transcriber, _ = _transcriber(model)
    reports: list[tuple[float, str]] = []
    with progress.capture(lambda fraction, detail: reports.append((fraction, detail))):
        transcriber.transcribe(audio)
    fractions = [f for f, _ in reports]
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(750 / 3490)
    assert reports[0][1] == "12:30 / 58:10"
    assert reports[-1] == (1.0, "58:10 / 58:10")


def test_progress_detail_shows_hours():
    from echo.transcribe import _clock

    assert [_clock(s) for s in (0, 59.9, 750, 3725)] == ["0:00", "0:59", "12:30", "1:02:05"]


def test_progress_without_a_duration_still_ends_at_one(audio: Path):
    transcriber, _ = _transcriber(FakeWhisperModel([], duration=0.0))
    reports: list[tuple[float, str]] = []
    with progress.capture(lambda fraction, detail: reports.append((fraction, detail))):
        transcriber.transcribe(audio)
    assert reports == [(1.0, "0:00 / 0:00")]


def test_batch_size_alone_does_not_re_transcribe(transcribed: tuple[Job, FakeTranscriber]):
    job, fake = transcribed
    run_pipeline(job, [TranscribeStage(fake)], _config(batch_size=8))
    assert len(fake.calls) == 2  # sequential → batched: re-run
    run_pipeline(job, [TranscribeStage(fake)], _config(batch_size=16))
    assert len(fake.calls) == 2, "8 → 16 only changes speed"
    assert job.read_meta().models["whisper_mode"] == "batched-8, vad"
