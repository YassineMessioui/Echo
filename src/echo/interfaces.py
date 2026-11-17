"""Layer interfaces: what each layer must implement and what it may rely on.

Each layer is a pure-ish component with typed inputs and outputs. The matching
``Stage`` adapter (built by the orchestrator issue) moves data between these
components and the job directory, so a layer never needs another layer's code,
only the models in ``echo.models``.

Every implementation raises a subclass of ``echo.errors.EchoError`` for expected
failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from echo.config import (
    ChunkConfig,
    ClipConfig,
    IngestConfig,
    PipelineConfig,
    RefineConfig,
)
from echo.job import Job
from echo.models import ClipSpec, StageName, Topic, Transcript, TranscriptWindow

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
        self, system: str, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Return the raw model text. ``json_schema`` asks for structured output if supported.

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
    ) -> None:
        """Cut ``clips[i]`` from ``video`` into ``outputs[i]`` (frame-accurate re-encode).

        ``extra_filter`` is a ``-vf`` filter chained *after* the framing set by ``config``
        (e.g. burned-in captions, #17), in the same encode.

        Raises ExternalToolError.
        """
        ...


# --- orchestration ----------------------------------------------------------------


class Stage(Protocol):
    """Adapter between a layer and the job directory, run by the orchestrator."""

    name: StageName

    def outputs(self, job: Job) -> list[Path]:
        """Artifacts this stage produces. If all exist, the stage is skipped (unless forced)."""
        ...

    def run(self, job: Job, config: PipelineConfig) -> None:
        """Read the previous stage's artifacts from ``job`` and write this stage's outputs."""
        ...
