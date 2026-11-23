"""Data contracts: the artifacts every stage reads and writes.

These models are the single source of truth for the JSON files in a job
directory. Change them only in a dedicated PR, since every layer depends on them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

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
    duration: float = Field(
        ge=0, description="End of the transcribed audio, in seconds of the original video."
    )
    segments: list[Segment]
    offset: float = Field(
        default=0.0,
        ge=0,
        description="Where audio.wav starts in the original video (focus range, #59). "
        "Segment and word times are already in original-video time.",
    )

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


# --- speakers.json (stage: diarize, #22) ------------------------------------------


class SpeakerTurn(_Artifact):
    """One stretch of speech by one speaker, in original-video time."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str = Field(
        min_length=1, description="Stable label, e.g. 'S1', in order of first appearance."
    )

    @model_validator(mode="after")
    def _ordered(self) -> SpeakerTurn:
        if self.end <= self.start:
            raise ValueError(f"speaker turn end ({self.end}) <= start ({self.start})")
        return self


class Speakers(_Artifact):
    turns: list[SpeakerTurn]

    @model_validator(mode="after")
    def _sorted(self) -> Speakers:
        for prev, cur in zip(self.turns, self.turns[1:], strict=False):
            if cur.start < prev.start:
                raise ValueError("speaker turns must be sorted by start")
        return self

    @property
    def labels(self) -> list[str]:
        """Speakers in order of first appearance."""
        return list(dict.fromkeys(turn.speaker for turn in self.turns))

    def speaker_at(self, start: float, end: float) -> str | None:
        """The speaker who talks most during ``[start, end]`` (a word or segment), or the
        nearest turn's speaker if none overlaps; None without turns."""
        if not self.turns:
            return None
        best, best_overlap = None, 0.0
        for turn in self.turns:
            overlap = min(end, turn.end) - max(start, turn.start)
            if overlap > best_overlap:
                best, best_overlap = turn.speaker, overlap
        if best is not None:
            return best
        mid = (start + end) / 2
        return min(self.turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


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
    DIARIZE = "diarize"
    """Optional (#22): only part of a run when ``DiarizeConfig.enabled``."""
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
    settings: str | None = None
    """Fingerprint of the settings that produced this stage's outputs (``Stage.settings``);
    the stage re-runs when it changes."""


class JobMeta(BaseModel):
    """Mutable job metadata, owned by the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    source: str = Field(description="Original path (or URL) of the input video.")
    created_at: datetime
    source_url: str | None = Field(default=None, description="Set when the input was a link.")
    source_info: dict[str, str | float] = Field(
        default_factory=dict,
        description="Link metadata from yt-dlp: id, extractor, title, uploader, duration.",
    )
    stages: dict[StageName, StageRecord] = Field(
        default_factory=lambda: {name: StageRecord() for name in StageName}
    )
    models: dict[str, str] = Field(
        default_factory=dict,
        description="Component -> model/version used, e.g. {'whisper': 'small', 'llm': '...'}",
    )

    @model_validator(mode="after")
    def _every_stage(self) -> JobMeta:
        """job.json files written before a stage existed (e.g. ``diarize``) still load."""
        for name in StageName:
            self.stages.setdefault(name, StageRecord())
        return self


# --- clips/publish.json (command: publish, #67) ----------------------------------


class Platform(StrEnum):
    """Where a clip can be posted (#67)."""

    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


class Approval(StrEnum):
    """A human's decision on a clip (#102). Only ``APPROVED`` clips are ever published."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


TITLE_MAX_LEN = 100
"""The longest title any platform accepts (YouTube: 100 characters)."""

Hashtag = Annotated[str, Field(pattern=r"^[^\s#]+$", max_length=100)]
"""A hashtag without its ``#`` (``"podcast"``); publishers add the ``#``."""


class PostText(_Artifact):
    """Per-platform overrides of a ``ClipPost``'s text; ``None`` keeps the clip's own."""

    title: str | None = Field(default=None, min_length=1, max_length=TITLE_MAX_LEN)
    description: str | None = None
    hashtags: list[Hashtag] | None = None


class PublishResult(_Artifact):
    """The outcome of posting one clip to one platform: an ``id`` (and usually a
    ``url``) on success, else an ``error``."""

    id: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    privacy: str | None = None
    """The visibility it was posted with (``PublishConfig.privacy``)."""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.id is not None and self.error is None

    @model_validator(mode="after")
    def _outcome(self) -> PublishResult:
        if (self.id is None) == (self.error is None):
            raise ValueError("a publish result has exactly one of id (posted) or error")
        return self


class ClipPost(_Artifact):
    """One clip as it would be posted: its text, a human's approval and, per platform,
    what happened when it was published. One entry per clip in ``clips/publish.json``.

    The text is generated by #101 and edited in review (#102); ``overrides`` adapt it
    for a platform (e.g. a shorter TikTok caption).
    """

    clip: str = Field(min_length=1, pattern=r"^[^/\\]+\.mp4$")
    """File name of the clip in ``clips/`` (``01-sleep-and-productivity.mp4``)."""
    topic_title: str = Field(min_length=1)
    """``ClipSpec.topic_title`` of that clip, to match it after a re-run of refine."""
    title: str = Field(min_length=1, max_length=TITLE_MAX_LEN)
    description: str = ""
    hashtags: list[Hashtag] = Field(default_factory=list)
    overrides: dict[Platform, PostText] = Field(default_factory=dict)
    approval: Approval = Approval.PENDING
    results: dict[Platform, PublishResult] = Field(default_factory=dict)
    """The latest attempt per platform. A platform with an ``ok`` result is never
    posted again."""

    def text_for(self, platform: Platform) -> PostText:
        """The title, description and hashtags to post on ``platform`` (overrides
        applied), all set."""
        override = self.overrides.get(platform, PostText())
        return PostText(
            title=override.title if override.title is not None else self.title,
            description=(
                override.description if override.description is not None else self.description
            ),
            hashtags=override.hashtags if override.hashtags is not None else self.hashtags,
        )

    def published_on(self, platform: Platform) -> bool:
        result = self.results.get(platform)
        return result is not None and result.ok
