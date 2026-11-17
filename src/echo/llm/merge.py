"""Merge topics from overlapping windows into one list, and the LLM stage that runs
chunking → extraction → merging over a whole transcript."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from echo import progress
from echo.config import LLMConfig, PipelineConfig
from echo.interfaces import Chunker, TopicExtractor, TopicMerger
from echo.job import Job
from echo.models import StageName, Topic, TranscriptWindow

logger = logging.getLogger(__name__)

HOSTED_CONCURRENCY = 4
"""``LLMConfig.max_concurrency = None`` for a hosted provider."""


def concurrency(config: LLMConfig) -> int:
    """How many windows go to the LLM at once: the configured value, else 4 for hosted
    providers and 1 for Ollama (a local server queues requests, so parallel calls gain
    little and the queued ones can hit ``timeout_s``)."""
    if config.max_concurrency is not None:
        return config.max_concurrency
    return 1 if config.provider == "ollama" else HOSTED_CONCURRENCY


_POLL_S = 0.05
"""How often the stage thread wakes while waiting for windows, so Ctrl+C is handled
at once (a blocking wait isn't interrupted by ``_thread.interrupt_main``, nor by
Ctrl+C on Windows)."""


def extract_all(
    extractor: TopicExtractor,
    windows: list[TranscriptWindow],
    workers: int,
    *,
    on_done: Callable[[int], None] | None = None,
) -> list[list[Topic]]:
    """``extractor.extract`` for every window, up to ``workers`` at a time (#56).

    Results are in window order, so merging is deterministic. ``on_done(n)`` is called
    from this thread each time another window finishes (``n`` = windows done so far), so
    ``echo.progress`` reports reach the pipeline (#61). As soon as any window fails, the
    windows not started yet are skipped (a dead provider doesn't get every request) and
    the calls in flight finish; the error raised is the one from the lowest failed
    window, so the message is stable.

    On ``KeyboardInterrupt`` (#83) nothing is waited for: windows not started are
    skipped, ``extractor.close()`` is called if it exists (the HTTP client is closed and
    retry waits end, so calls in flight fail fast), and the interrupt is re-raised.
    Errors those calls raise afterwards are ignored. Windows run on daemon threads,
    even with ``workers <= 1``: the interpreter doesn't join them at exit, and this
    thread only waits in short steps, so Ctrl+C is handled promptly.
    """
    report = on_done or (lambda done: None)
    if not windows:
        return []
    results: list[list[Topic]] = [[] for _ in windows]
    errors: dict[int, BaseException] = {}
    finished: queue.SimpleQueue[int] = queue.SimpleQueue()
    pending = iter(range(len(windows)))
    lock = threading.Lock()
    stop = threading.Event()

    def work() -> None:
        while not stop.is_set():
            with lock:
                i = next(pending, None)
            if i is None:
                return
            try:
                results[i] = extractor.extract(windows[i])
            except BaseException as e:  # reported by the stage thread
                errors[i] = e
            finished.put(i)

    threads = [
        threading.Thread(target=work, name=f"llm-{k}", daemon=True)
        for k in range(min(max(workers, 1), len(windows)))
    ]
    try:
        for thread in threads:
            thread.start()
        done = 0
        while done < len(windows):
            try:
                i = finished.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if i in errors:
                break
            done += 1
            report(done)
        stop.set()
        for thread in threads:  # after an error: let the calls in flight finish
            while thread.is_alive():
                thread.join(_POLL_S)
    except KeyboardInterrupt:
        stop.set()
        _close(extractor)
        raise
    finally:
        stop.set()
    if errors:
        raise errors[min(errors)]
    return results


def _close(extractor: TopicExtractor) -> None:
    """Close ``extractor`` if it can be (``LLMTopicExtractor`` closes its client)."""
    close = getattr(extractor, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # never hide the interrupt
            logger.warning("closing the LLM client failed", exc_info=True)


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

        def window_done(done: int) -> None:
            progress.report(done / len(windows), f"window {done}/{len(windows)}")

        per_window = extract_all(
            self.extractor, windows, concurrency(config.llm), on_done=window_done
        )
        job.write_topics(self.merger(per_window))
        if self.models:
            meta = job.read_meta()
            meta.models.update(self.models)
            job.write_meta(meta)
