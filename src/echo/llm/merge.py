"""Merge topics from overlapping windows into one list, and the LLM stage that runs
chunking → extraction → merging over a whole transcript."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from echo.config import PipelineConfig
from echo.interfaces import Chunker, TopicExtractor, TopicMerger
from echo.job import Job
from echo.models import StageName, Topic

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_CONTAINMENT_THRESHOLD = 0.8
"""A topic cut off by a window edge is a short span mostly inside the full one: its
IoU can be low (3 of 8 segments), so containment is checked too."""


def span_iou(a: Topic, b: Topic) -> float:
    """Intersection over union of two inclusive segment spans."""
    inter = min(a.end_segment, b.end_segment) - max(a.start_segment, b.start_segment) + 1
    if inter <= 0:
        return 0.0
    union = _length(a) + _length(b) - inter
    return inter / union


def span_containment(a: Topic, b: Topic) -> float:
    """Share of the shorter span that lies inside the other one."""
    inter = min(a.end_segment, b.end_segment) - max(a.start_segment, b.start_segment) + 1
    if inter <= 0:
        return 0.0
    return inter / min(_length(a), _length(b))


def _length(topic: Topic) -> int:
    return topic.end_segment - topic.start_segment + 1


@dataclass
class _Cluster:
    topic: Topic
    windows: frozenset[int]


def _union(a: Topic, b: Topic) -> Topic:
    """Union of both spans, keeping the title/summary of the longer topic (``a`` on ties)."""
    keep = b if _length(b) > _length(a) else a
    return Topic(
        title=keep.title,
        summary=keep.summary,
        start_segment=min(a.start_segment, b.start_segment),
        end_segment=max(a.end_segment, b.end_segment),
    )


def merge_topics(
    per_window: list[list[Topic]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    containment_threshold: float = DEFAULT_CONTAINMENT_THRESHOLD,
) -> list[Topic]:
    """Implements ``echo.interfaces.TopicMerger``.

    Topics from *different* windows whose spans have IoU ≥ ``iou_threshold``, or where
    the shorter lies at least ``containment_threshold`` inside the other (a topic cut
    off at a window edge), are the same topic seen twice (windows overlap) and are
    merged; the most overlapping pair
    is merged first, until no pair qualifies. Topics from the same window are never
    merged with each other: the LLM saw them side by side and chose to separate them.
    The result is sorted by ``start_segment``.
    """
    clusters = [
        _Cluster(topic, frozenset({w})) for w, topics in enumerate(per_window) for topic in topics
    ]
    while True:
        best: tuple[float, int, int] | None = None
        for i, a in enumerate(clusters):
            for j in range(i + 1, len(clusters)):
                b = clusters[j]
                if a.windows & b.windows:
                    continue
                iou = span_iou(a.topic, b.topic)
                same = (
                    iou >= iou_threshold
                    or span_containment(a.topic, b.topic) >= containment_threshold
                )
                if same and (best is None or iou > best[0]):
                    best = (iou, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = clusters[i], clusters.pop(j)
        clusters[i] = _Cluster(_union(a.topic, b.topic), a.windows | b.windows)

    return sorted(
        (c.topic for c in clusters), key=lambda t: (t.start_segment, t.end_segment, t.title)
    )


@dataclass
class LLMStage:
    """Stage adapter: ``transcript.json`` → chunk → extract per window → merge → ``topics.json``.

    The components are injected so the stage can be tested with fakes. ``models`` is
    recorded in ``job.json``, e.g. ``{"llm": client.model_name, "llm_prompt": "topics-v1"}``.
    """

    chunker: Chunker
    extractor: TopicExtractor
    merger: TopicMerger = merge_topics
    models: Mapping[str, str] = field(default_factory=dict)

    name: ClassVar[StageName] = StageName.LLM

    def outputs(self, job: Job) -> list[Path]:
        return [job.topics_path]

    def run(self, job: Job, config: PipelineConfig) -> None:
        transcript = job.read_transcript()
        windows = self.chunker(transcript, config.chunk)
        per_window = [self.extractor.extract(window) for window in windows]
        job.write_topics(self.merger(per_window))
        if self.models:
            meta = job.read_meta()
            meta.models.update(self.models)
            job.write_meta(meta)
