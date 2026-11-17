"""Boundary refinement: turn topic spans (segment ids) into clean clip times.

Steps, all pure functions over the transcript:

1. **Resolve overlaps** between topic spans: merge two topics when they share at least
   half of the shorter one, otherwise trim the later one to start after the earlier.
2. **Enforce durations** on segment edges: trim long clips by dropping segments from
   the end; extend short clips with neighbouring segments (the following one first)
   without entering another clip. A single segment longer than the limit is cut at
   the last word that fits. Clips still under the minimum are dropped.
3. **Pad** each edge by up to ``padding`` seconds, but never past the media bounds and
   never more than half the gap to the neighbouring clip, so clips never overlap.

Padding is reserved inside ``max_duration``, so padded clips never exceed it (60 s is
a hard platform cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from echo.config import PipelineConfig, RefineConfig
from echo.errors import ArtifactError
from echo.interfaces import BoundaryRefiner
from echo.job import Job
from echo.models import ClipSpec, Segment, StageName, Topic, Transcript

MERGE_OVERLAP = 0.5
"""Merge two topics if their shared segments cover at least this share of the shorter."""


@dataclass
class _Span:
    title: str
    start: int  # segment ids, inclusive
    end: int


@dataclass
class _Times:
    title: str
    start: float
    end: float


def _seconds(segments: list[Segment], start: int, end: int) -> float:
    return segments[end].end - segments[start].start


def resolve_overlaps(segments: list[Segment], topics: list[Topic]) -> list[_Span]:
    """Sorted, non-overlapping spans. Heavily overlapping topics merge (keeping the title
    of the longer one); lightly overlapping ones are trimmed to start after the previous."""
    spans: list[_Span] = []
    for topic in sorted(topics, key=lambda t: (t.start_segment, -t.end_segment)):
        start, end = topic.start_segment, topic.end_segment
        if spans and start <= spans[-1].end:
            last = spans[-1]
            shared = min(end, last.end) - start + 1
            shorter = min(end - start, last.end - last.start) + 1
            if shared / shorter >= MERGE_OVERLAP:
                if _seconds(segments, start, end) > _seconds(segments, last.start, last.end):
                    last.title = topic.title
                last.end = max(last.end, end)
                continue
            start = last.end + 1
            if start > end:
                continue
        spans.append(_Span(topic.title, start, end))
    return spans


def _fit_span(
    segments: list[Segment], span: _Span, lo: int, hi: int, min_d: float, max_d: float
) -> tuple[int, int]:
    """Trim/extend ``span`` on segment edges within segment ids ``[lo, hi]``."""
    start, end = span.start, span.end
    while end > start and _seconds(segments, start, end) > max_d:
        end -= 1
    while _seconds(segments, start, end) < min_d:
        if end + 1 <= hi and _seconds(segments, start, end + 1) <= max_d:
            end += 1
        elif start - 1 >= lo and _seconds(segments, start - 1, end) <= max_d:
            start -= 1
        else:
            break
    return start, end


def _cut_long_segment(segment: Segment, max_d: float) -> float:
    """End time for a clip starting at ``segment.start`` that must stop within ``max_d``:
    the last word that fits, or a hard cut if no word does."""
    limit = segment.start + max_d
    fitting = [w.end for w in segment.words if segment.start < w.end <= limit]
    return max(fitting, default=limit)


IDEAL_DURATION = 37.5
"""Seconds; clips closest to this are kept by ``max_clips`` (30–45 s is the sweet spot for
complete, self-contained short-form clips)."""


def _keep_best(clips: list[ClipSpec], limit: int) -> list[ClipSpec]:
    """The ``limit`` clips closest to ``IDEAL_DURATION`` (earlier wins ties), in time order.
    A stand-in until clip scoring (#21) ranks by engagement."""
    ranked = sorted(clips, key=lambda c: (abs(c.duration - IDEAL_DURATION), c.start))
    return sorted(ranked[:limit], key=lambda c: c.start)


def refine_clips(
    transcript: Transcript, topics: list[Topic], config: RefineConfig
) -> list[ClipSpec]:
    """Implements ``echo.interfaces.BoundaryRefiner``.

    With ``config.max_clips``, all clips are refined first, the best ones are chosen
    (``_keep_best``), and then only the chosen topics are refined again: a kept clip
    shouldn't stay trimmed by an overlapping neighbour that was dropped.
    """
    clips = _refine(transcript, topics, config)
    if config.max_clips is None or len(clips) <= config.max_clips:
        return clips
    kept = _keep_best(clips, config.max_clips)
    titles = {c.topic_title for c in kept}
    again = _refine(transcript, [t for t in topics if t.title in titles], config)
    # Re-refining can merge/drop differently in rare cases; then keep the first choice.
    return again if sorted(c.topic_title for c in again) == sorted(titles) else kept


def _refine(transcript: Transcript, topics: list[Topic], config: RefineConfig) -> list[ClipSpec]:
    segments = transcript.segments
    for topic in topics:
        if topic.end_segment >= len(segments):
            raise ArtifactError(
                f"topic {topic.title!r} ends at segment {topic.end_segment}, but the "
                f"transcript has {len(segments)} segments"
            )
    if not segments:
        return []

    # Reserve room for padding on both sides, but never squeeze below min_duration.
    pad = min(config.padding, (config.max_duration - config.min_duration) / 2)
    max_d = config.max_duration - 2 * pad

    spans = resolve_overlaps(segments, topics)
    clips: list[_Times] = []
    prev_end = -1  # last segment id used by a kept clip
    for i, span in enumerate(spans):
        hi = spans[i + 1].start - 1 if i + 1 < len(spans) else len(segments) - 1
        start, end = _fit_span(segments, span, prev_end + 1, hi, config.min_duration, max_d)
        start_t, end_t = segments[start].start, segments[end].end
        if end_t - start_t > max_d:  # one segment alone is too long
            end_t = _cut_long_segment(segments[start], max_d)
        if end_t - start_t < config.min_duration:
            continue
        clips.append(_Times(span.title, start_t, end_t))
        prev_end = end

    # Whisper segments may overlap slightly in time; never let clips do so.
    for prev, cur in zip(clips, clips[1:], strict=False):
        prev.end = min(prev.end, cur.start)
    clips = [c for c in clips if c.end - c.start >= config.min_duration]

    result: list[ClipSpec] = []
    for i, clip in enumerate(clips):
        before = clip.start - clips[i - 1].end if i > 0 else float("inf")
        after = clips[i + 1].start - clip.end if i + 1 < len(clips) else float("inf")
        # The transcribed span is [offset, duration] of the original video (focus range).
        start = max(transcript.offset, clip.start - min(pad, before / 2))
        end = min(transcript.duration, clip.end + min(pad, after / 2))
        start, end = round(start, 3), round(end, 3)
        end = min(end, round(start + config.max_duration, 3))
        result.append(ClipSpec.between(clip.title, start, end))
    return result


_: BoundaryRefiner = refine_clips


@dataclass
class RefineStage:
    """Stage adapter: ``transcript.json`` + ``topics.json`` → ``clips.json``."""

    refiner: BoundaryRefiner = refine_clips

    name: ClassVar[StageName] = StageName.REFINE

    def outputs(self, job: Job) -> list[Path]:
        return [job.clips_path]

    def run(self, job: Job, config: PipelineConfig) -> None:
        clips = self.refiner(job.read_transcript(), job.read_topics(), config.refine)
        job.write_clips(clips)
