"""Layer interfaces: what each layer must implement and what it may rely on.

Each layer is a pure-ish component with typed inputs and outputs. The matching
``Stage`` adapter (built by the orchestrator issue) moves data between these
components and the job directory, so a layer never needs another layer's code,
only the models in ``echo.models``.

Every implementation raises a subclass of ``echo.errors.EchoError`` for expected
failures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from echo.config import (
    ChunkConfig,
    ClipConfig,
    IngestConfig,
    PipelineConfig,
    PublishConfig,
    RefineConfig,
)
from echo.job import Job
from echo.models import (
    ClipPost,
    ClipSpec,
    Platform,
    PublishResult,
    Score,
    Speakers,
    SpeakerTurn,
    StageName,
    Topic,
    Transcript,
    TranscriptWindow,
)

# --- stage: ingest ----------------------------------------------------------------


class AudioExtractor(Protocol):
    def __call__(self, video: Path, out: Path, config: IngestConfig) -> None:
        """Write a ``config.sample_rate`` Hz, ``config.channels``-channel PCM WAV to ``out``.

        Raises InvalidInputError (bad/missing input, no audio stream) or
        ExternalToolError (ffmpeg missing or failing).
        """
        ...


# --- stage: transcribe ------------------------------------------------------------


class Transcriber(Protocol):
    model_name: str
    """Recorded in job.json, e.g. 'faster-whisper/small'."""

    def transcribe(self, audio: Path) -> Transcript:
        """Return segments with ids 0..n-1 and word-level timestamps.

        Raises TranscriptionError.
        """
        ...


# --- stage: diarize (optional, #22) --------------------------------------------------


class Diarizer(Protocol):
    model_name: str
    """Recorded in job.json, e.g. 'sherpa-onnx/pyannote-3.0+titanet-small'."""

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        """Who speaks when in ``audio``, in seconds of the audio file, sorted by start.
        Labels are stable within one call (the stage renames them S1, S2, ...).

        Raises TranscriptionError.
        """
        ...


# --- stage: llm -------------------------------------------------------------------


class Chunker(Protocol):
    def __call__(self, transcript: Transcript, config: ChunkConfig) -> list[TranscriptWindow]:
        """Split into overlapping windows. Every segment is in at least one window;
        windows are ordered and never split a segment."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    model_name: str
    """Recorded in job.json, e.g. 'ollama/llama3.2:3b'."""

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        images: Sequence[bytes] = (),
    ) -> str:
        """Return the raw model text. ``json_schema`` asks for structured output if supported.

        ``images`` are JPEG bytes shown to the model with ``prompt`` (#129: a frame of each
        clip, so the text can tell a game's cutscene from the streamer). Only vision models
        accept them; others fail with LLMError.

        Raises LLMError.
        """
        ...


class TopicExtractor(Protocol):
    def extract(self, window: TranscriptWindow) -> list[Topic]:
        """Topics whose segment ids all fall inside ``window``.

        Retries on invalid output, then raises TopicExtractionError.
        """
        ...


class TopicMerger(Protocol):
    def __call__(self, per_window: list[list[Topic]]) -> list[Topic]:
        """Deduplicate topics across overlapping windows; return them sorted by start_segment."""
        ...


# --- stage: score (optional, #21) ----------------------------------------------------


class ClipScorer(Protocol):
    def score(self, clips: list[str]) -> list[Score]:
        """One ``Score`` per clip transcript text, in the same order. The LLM refers to
        clips by their index in ``clips``, never by text or time.

        Retries on invalid output, then raises TopicExtractionError.
        """
        ...


# --- stage: refine ----------------------------------------------------------------


class BoundaryRefiner(Protocol):
    def __call__(
        self, transcript: Transcript, topics: list[Topic], config: RefineConfig
    ) -> list[ClipSpec]:
        """Clips on segment edges (± padding) within [0, transcript.duration], durations in
        [min_duration, max_duration], sorted and non-overlapping."""
        ...


# --- stage: clip ------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteVideo:
    """A video read over HTTP(S) instead of from disk (link jobs, #60).

    ffmpeg seeks into ``url`` with range requests, so cutting a clip only fetches that
    clip's bytes. The audio comes from the local ``audio`` file (``Job.input_audio``).
    ``url`` is a direct stream URL that expires, so resolve it right before cutting.
    """

    url: str
    audio: Path
    width: int
    height: int
    headers: dict[str, str] = field(default_factory=dict)


class Clipper(Protocol):
    def __call__(
        self,
        video: Path | RemoteVideo,
        clips: list[ClipSpec],
        outputs: list[Path],
        config: ClipConfig,
        *,
        extra_filter: str | None = None,
        speakers: Speakers | None = None,
    ) -> None:
        """Cut ``clips[i]`` from ``video`` into ``outputs[i]`` (frame-accurate re-encode).

        ``extra_filter`` is a ``-vf`` filter chained *after* the framing set by ``config``
        (e.g. burned-in captions, #17), in the same encode.

        ``speakers`` (who speaks when, in original-video time) lets
        ``config.reframe == "speaker"`` follow the active speaker (#16); without it, that
        mode frames like ``"face"``. Other modes ignore it.

        Raises ExternalToolError.
        """
        ...


# --- orchestration ----------------------------------------------------------------


class Stage(Protocol):
    """Adapter between a layer and the job directory, run by the orchestrator.

    Optional: ``settings(config) -> str | None``, a fingerprint of the config its outputs
    depend on (e.g. the whisper model and batch size). When it changes, the orchestrator
    re-runs the stage even though its outputs exist. Report progress inside ``run`` with
    ``echo.progress.report(fraction, detail)``.
    """

    name: StageName

    def outputs(self, job: Job) -> list[Path]:
        """Artifacts this stage produces. If all exist, the stage is skipped (unless forced)."""
        ...

    def run(self, job: Job, config: PipelineConfig) -> None:
        """Read the previous stage's artifacts from ``job`` and write this stage's outputs."""
        ...


# --- command: publish (#67) ---------------------------------------------------------


class Publisher(Protocol):
    """Posts clips to one platform. Used by the ``echo-clips publish`` command, never by
    a pipeline run."""

    platform: Platform

    def publish(self, video: Path, post: ClipPost, config: PublishConfig) -> PublishResult:
        """Post ``video`` with ``post.text_for(self.platform)``, at ``config.privacy``.

        The caller only passes approved posts (``post.approval``) not yet published on
        this platform. A failed upload the user can act on (expired sign-in, quota,
        rejected video) returns a ``PublishResult`` with ``error``, so the other clips
        still go out; raises PublishError when nothing can be posted at all (e.g. no
        credentials).
        """
        ...
