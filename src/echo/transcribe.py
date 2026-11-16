"""Transcription stage: ``audio.wav`` → ``transcript.json`` with faster-whisper.

Segment ids are assigned 0..n-1 here; the LLM later refers to segments by these ids,
so they must be stable for a given transcript.
"""

from __future__ import annotations

import logging
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

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


def _load_whisper(config: TranscribeConfig) -> WhisperModel:
    from faster_whisper import WhisperModel  # heavy import, only when actually transcribing

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


class WhisperTranscriber:
    """Implements ``echo.interfaces.Transcriber`` with faster-whisper.

    The model is loaded on first use and reused. ``model_factory`` exists for tests.
    """

    def __init__(
        self, config: TranscribeConfig, *, model_factory: ModelFactory = _load_whisper
    ) -> None:
        self.config = config
        self.model_name = f"faster-whisper/{config.model_size}"
        self._model_factory = model_factory
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
        if self._model is None:
            self._model = self._model_factory(self.config)
        raw_segments, info = self._model.transcribe(
            samples, language=self.config.language, word_timestamps=True
        )
        # Segments are a lazy generator: decoding happens while iterating.
        return self._build(list(raw_segments), language=info.language, duration=info.duration)

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
