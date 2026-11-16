"""Data contracts: the artifacts every stage reads and writes.

These models are the single source of truth for the JSON files in a job
directory. Change them only in a dedicated PR, since every layer depends on them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Tolerance (seconds) when comparing floats that should be equal.
_EPS = 1e-3


class _Artifact(BaseModel):
    """Base for artifact models: strict and immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- transcript.json (stage: transcribe) ---------------------------------------


class Word(_Artifact):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    probability: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> Word:
        if self.end < self.start:
            raise ValueError(f"word end ({self.end}) < start ({self.start})")
        return self


class Segment(_Artifact):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[Word] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"segment {self.id}: end ({self.end}) < start ({self.start})")
        return self


class Transcript(_Artifact):
    language: str
    duration: float = Field(ge=0, description="Duration of the source audio in seconds.")
    segments: list[Segment]

    @model_validator(mode="after")
    def _consistent(self) -> Transcript:
        for expected, seg in enumerate(self.segments):
            if seg.id != expected:
                raise ValueError(
                    f"segment ids must be contiguous from 0; got {seg.id} at {expected}"
                )
        for prev, cur in zip(self.segments, self.segments[1:], strict=False):
            if cur.start < prev.start:
                raise ValueError(f"segment {cur.id} starts before segment {prev.id}")
        if self.segments and self.segments[-1].end > self.duration + _EPS:
            raise ValueError("last segment ends after transcript duration")
        return self

    def segment(self, segment_id: int) -> Segment:
        """Look up a segment by id, raising KeyError if it does not exist."""
        if not 0 <= segment_id < len(self.segments):
            raise KeyError(segment_id)
        return self.segments[segment_id]


class TranscriptWindow(_Artifact):
    """A chunk of the transcript sent to the LLM in one call (in memory only)."""

    start_segment: int = Field(ge=0)
    end_segment: int = Field(ge=0)
    text: str = Field(description="Numbered lines: '[12] 00:03:41–00:03:47 text...'")

    @model_validator(mode="after")
    def _ordered(self) -> TranscriptWindow:
        if self.end_segment < self.start_segment:
            raise ValueError("window end_segment < start_segment")
        return self

    def contains(self, segment_id: int) -> bool:
        return self.start_segment <= segment_id <= self.end_segment


# --- topics.json (stage: llm) ---------------------------------------------------


class Topic(_Artifact):
    title: str = Field(min_length=1)
    summary: str
    start_segment: int = Field(ge=0)
    end_segment: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Topic:
        if self.end_segment < self.start_segment:
            raise ValueError(f"topic {self.title!r}: end_segment < start_segment")
        return self


# --- clips.json (stage: refine) -------------------------------------------------


class ClipSpec(_Artifact):
    topic_title: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    duration: float = Field(gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> ClipSpec:
        if self.end <= self.start:
            raise ValueError(f"clip {self.topic_title!r}: end <= start")
        if abs((self.end - self.start) - self.duration) > _EPS:
            raise ValueError(f"clip {self.topic_title!r}: duration != end - start")
        return self

    @classmethod
    def between(cls, topic_title: str, start: float, end: float) -> ClipSpec:
        """Build a clip, computing the duration (all values rounded to milliseconds)."""
        start, end = round(start, 3), round(end, 3)
        return cls(topic_title=topic_title, start=start, end=end, duration=round(end - start, 3))


# --- job.json (orchestrator) ----------------------------------------------------


class StageName(StrEnum):
    """Pipeline stages, in execution order."""

    INGEST = "ingest"
    TRANSCRIBE = "transcribe"
    LLM = "llm"
    REFINE = "refine"
    CLIP = "clip"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class JobMeta(BaseModel):
    """Mutable job metadata, owned by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    source: str = Field(description="Original path of the input video.")
    created_at: datetime
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in StageName}
    )
    models: dict[str, str] = Field(
        default_factory=dict,
        description="Component -> model/version used, e.g. {'whisper': 'small', 'llm': '...'}",
    )
