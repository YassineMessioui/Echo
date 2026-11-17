"""Transcription stage: ``audio.wav`` → ``transcript.json`` with faster-whisper.

Segment ids are assigned 0..n-1 here; the LLM later refers to segments by these ids,
so they must be stable for a given transcript.
"""

from __future__ import annotations

import logging
import math
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from echo import progress
from echo.config import PipelineConfig, TranscribeConfig
from echo.errors import ArtifactError, TranscriptionError
from echo.interfaces import Transcriber
from echo.job import Job
from echo.models import Segment, StageName, Transcript, Word

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


class _WhisperLike(Protocol):
    """The part of ``faster_whisper.WhisperModel`` we use (so tests can pass a fake)."""

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


ModelFactory = Callable[[TranscribeConfig], _WhisperLike]
PipelineFactory = Callable[[_WhisperLike], _WhisperLike]
"""Wraps a loaded model in the batched pipeline (``TranscribeConfig.batch_size > 0``)."""


def _batched_pipeline(model: _WhisperLike) -> _WhisperLike:
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model)  # type: ignore[arg-type]


CHUNK_SECONDS = 30
"""Whisper's input window; the batched pipeline decodes chunks of at most this length."""


def fixed_chunks(n_samples: int, rate: int = 16_000) -> list[dict[str, float]]:
    """Equal back-to-back windows of at most ``CHUNK_SECONDS`` covering the audio.

    The batched pipeline needs chunk boundaries: from VAD, or these when VAD is off.
    Equal lengths avoid a tiny last chunk (60.05 s → 30 + 30 + 0.05 s), which Whisper
    pads to 30 s and tends to fill with hallucinated text. Empty audio gives none.
    """
    duration = n_samples / rate
    n = math.ceil(duration / CHUNK_SECONDS)
    return [
        {"start": duration * i / n, "end": duration if i == n - 1 else duration * (i + 1) / n}
        for i in range(n)
    ]


# Approximate download sizes, for the one-line first-run notice.
_MODEL_SIZES = {"tiny": "75 MB", "base": "145 MB", "small": "480 MB", "medium": "1.5 GB"}


def _announce_download(model_size: str) -> None:
    """Log one clear line when the model isn't in the local cache yet (#65).

    Only for the real "not cached" case: an invalid model name or a local directory
    says nothing here and is handled by ``WhisperModel`` itself.
    """
    from faster_whisper.utils import download_model
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        download_model(model_size, local_files_only=True)
    except LocalEntryNotFoundError:
        size = _MODEL_SIZES.get(model_size)
        detail = f" (~{size}, first run only)" if size else " (first run only)"
        logger.warning('downloading whisper model "%s"%s', model_size, detail)
    except Exception:
        pass


def _load_whisper(config: TranscribeConfig) -> WhisperModel:
    from faster_whisper import WhisperModel  # heavy import, only when actually transcribing

    _announce_download(config.model_size)
    return WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)


logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000


def load_wav(path: Path) -> np.ndarray:
    """Read the ingest stage's 16 kHz 16-bit PCM WAV as float32 samples in [-1, 1].

    Decoding the WAV here, instead of letting faster-whisper open the file, avoids its
    PyAV dependency for the common case (PyAV >= 15 broke faster-whisper's file decoding).
    Multi-channel audio is averaged to mono.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise TranscriptionError(f"not a PCM WAV file: {path} ({e})") from e
    if rate != WHISPER_SAMPLE_RATE or width != 2:
        raise TranscriptionError(
            f"{path}: expected 16 kHz 16-bit PCM, got {rate} Hz {8 * width}-bit "
            f"(re-run the ingest stage)"
        )
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


# Only "the GPU stack can't be used at all" errors. Ordinary runtime GPU errors share
# the "cuda"/"cublas"/"cudnn" words (``CUDA failed with error out of memory``,
# ``cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED``) and must NOT fall back:
# silently switching a large model to the CPU can turn minutes into hours.
_GPU_UNAVAILABLE_HINTS = (
    "is not found or cannot be loaded",  # Library cublas64_12.dll is not found or ...
    "could not load library",
    "cannot open shared object file",
    "no cuda-capable device",
    "cuda driver version is insufficient",
)


def _is_missing_gpu_runtime(error: Exception) -> bool:
    """CTranslate2's error when the GPU was chosen but its libraries/driver can't be
    loaded, e.g. ``Library cublas64_12.dll is not found or cannot be loaded``."""
    text = str(error).lower()
    return any(hint in text for hint in _GPU_UNAVAILABLE_HINTS)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _clock(seconds: float) -> str:
    """``750`` → ``"12:30"``, ``3725`` → ``"1:02:05"`` (for progress details)."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class WhisperTranscriber:
    """Implements ``echo.interfaces.Transcriber`` with faster-whisper.

    The model is loaded on first use and reused. With ``batch_size > 0`` it runs through
    faster-whisper's ``BatchedInferencePipeline``, which decodes several ~30 s chunks at
    once (#58). ``model_factory`` and ``pipeline_factory`` exist for tests.
    """

    def __init__(
        self,
        config: TranscribeConfig,
        *,
        model_factory: ModelFactory = _load_whisper,
        pipeline_factory: PipelineFactory = _batched_pipeline,
    ) -> None:
        self.config = config
        self.model_name = f"faster-whisper/{config.model_size}"
        self._model_factory = model_factory
        self._pipeline_factory = pipeline_factory
        self._model: _WhisperLike | None = None

    def transcribe(self, audio: Path) -> Transcript:
        if not audio.is_file():
            raise TranscriptionError(f"audio file not found: {audio}")
        samples = load_wav(audio)
        try:
            return self._run(samples)
        except Exception as e:
            if not (self.config.device == "auto" and _is_missing_gpu_runtime(e)):
                raise TranscriptionError(
                    f"transcription with {self.model_name} failed: {type(e).__name__}: {e}"
                ) from e
            gpu_error = f"{type(e).__name__}: {e}"
        # device="auto" picked the GPU (an NVIDIA card is present) but the CUDA runtime
        # (cuBLAS/cuDNN) isn't installed: fall back to the CPU instead of failing.
        logger.warning("CUDA libraries not found; transcribing on the CPU instead")
        update: dict[str, str] = {"device": "cpu"}
        if "16" in self.config.compute_type:  # float16 / int8_float16 / bfloat16: GPU-only
            update["compute_type"] = "default"
        self.config = self.config.model_copy(update=update)
        self._model = None
        try:
            return self._run(samples)
        except Exception as e:
            raise TranscriptionError(
                f"transcription with {self.model_name} failed on the CPU after falling back "
                f"from CUDA ({gpu_error}): {type(e).__name__}: {e}"
            ) from e

    def _run(self, samples: np.ndarray) -> Transcript:
        config = self.config
        if self._model is None:
            model = self._model_factory(config)
            self._model = self._pipeline_factory(model) if config.batch_size else model
        kwargs: dict[str, Any] = {
            "language": config.language,
            "word_timestamps": True,
            "vad_filter": config.vad_filter,
        }
        if config.vad_filter and config.vad_min_silence_ms is not None:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": config.vad_min_silence_ms}
        if config.batch_size:
            kwargs["batch_size"] = config.batch_size
            # The batched pipeline defaults to one segment per ~30 s chunk; refine and the
            # LLM need sentence-sized segments like the sequential decoder gives.
            kwargs["without_timestamps"] = False
            if not config.vad_filter:
                # Without VAD the batched pipeline has no chunk boundaries of its own.
                kwargs["clip_timestamps"] = fixed_chunks(len(samples))
        raw_segments, info = self._model.transcribe(samples, **kwargs)
        # Segments are a lazy generator: decoding happens while iterating, so report
        # progress (#61) as each one arrives.
        duration = max(float(info.duration), 0.0)
        segments: list[Any] = []
        for segment in raw_segments:
            segments.append(segment)
            if duration > 0:
                done = min(float(segment.end), duration)
                progress.report(done / duration, f"{_clock(done)} / {_clock(duration)}")
        progress.report(1.0, f"{_clock(duration)} / {_clock(duration)}")
        return self._build(segments, language=info.language, duration=info.duration)

    @staticmethod
    def _build(raw_segments: list[Any], *, language: str, duration: float) -> Transcript:
        """Validated transcript with contiguous ids, non-decreasing starts and all times
        inside ``[0, duration]``. Segments without text are dropped."""
        duration = max(float(duration), 0.0)
        segments: list[Segment] = []
        prev_start = 0.0
        for raw in raw_segments:
            text = raw.text.strip()
            if not text:
                continue
            start = _clamp(float(raw.start), prev_start, duration)
            end = _clamp(float(raw.end), start, duration)
            words = [
                Word(
                    start=(w_start := _clamp(float(w.start), 0.0, duration)),
                    end=_clamp(float(w.end), w_start, duration),
                    text=w.word.strip(),
                    probability=None
                    if w.probability is None
                    else _clamp(float(w.probability), 0.0, 1.0),
                )
                for w in (raw.words or [])
            ]
            segments.append(Segment(id=len(segments), start=start, end=end, text=text, words=words))
            prev_start = start
        return Transcript(language=language or "unknown", duration=duration, segments=segments)


@dataclass
class TranscribeStage:
    """Stage adapter: ``audio.wav`` → ``transcript.json``; records the model in ``job.json``."""

    transcriber: Transcriber

    name: ClassVar[StageName] = StageName.TRANSCRIBE

    def outputs(self, job: Job) -> list[Path]:
        return [job.transcript_path]

    def run(self, job: Job, config: PipelineConfig) -> None:
        if not job.audio_path.is_file():
            raise ArtifactError(f"missing artifact: {job.audio_path}")
        job.write_transcript(self.transcriber.transcribe(job.audio_path))
        meta = job.read_meta()
        meta.models["whisper"] = self.transcriber.model_name
        job.write_meta(meta)
