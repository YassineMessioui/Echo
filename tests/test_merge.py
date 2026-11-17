import _thread
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from echo import progress
from echo.config import ChunkConfig, LLMConfig, PipelineConfig
from echo.errors import ArtifactError, LLMError
from echo.interfaces import Stage
from echo.job import Job
from echo.llm.merge import (
    LLMStage,
    concurrency,
    extract_all,
    merge_topics,
    span_containment,
    span_iou,
)
from echo.llm.topics import LLMTopicExtractor
from echo.models import StageName, Topic, Transcript, TranscriptWindow


def _t(start: int, end: int, title: str | None = None) -> Topic:
    return Topic(
        title=title or f"topic {start}-{end}",
        summary=f"summary of {title or f'{start}-{end}'}",
        start_segment=start,
        end_segment=end,
    )


@pytest.mark.parametrize(
    ("a", "b", "iou"),
    [
        ((0, 9), (0, 9), 1.0),
        ((0, 9), (10, 19), 0.0),
        ((0, 9), (5, 14), 5 / 15),
        ((0, 9), (2, 7), 6 / 10),
        ((3, 3), (3, 3), 1.0),
    ],
)
def test_span_iou(a, b, iou):
    assert span_iou(_t(*a), _t(*b)) == pytest.approx(iou)


def test_topic_cut_off_at_window_edge_is_merged():
    """Seen live with Groq: window 0 ended at segment 15, cutting a topic to 13–15,
    while window 1 saw all of it (13–20). IoU is only 3/8, but 13–15 lies inside."""
    cut, full = _t(13, 15, "Why the 20% rule fails"), _t(13, 20, "Smart savings")
    assert span_iou(cut, full) == pytest.approx(3 / 8)
    assert span_containment(cut, full) == 1.0
    assert merge_topics([[_t(2, 11, "Sleep"), cut], [full, _t(22, 29, "EVs")]]) == [
        _t(2, 11, "Sleep"),
        Topic(title="Smart savings", summary=full.summary, start_segment=13, end_segment=20),
        _t(22, 29, "EVs"),
    ]


def test_mostly_contained_but_below_threshold_stays_separate():
    a, b = _t(0, 9, "A"), _t(6, 20, "B")  # B's 4 of A's 10 segments: containment 0.4
    assert merge_topics([[a], [b]]) == [a, b]


def test_same_topic_in_two_windows_becomes_one():
    short, long = _t(10, 18, "Solar, partial"), _t(10, 20, "Solar power is cheap")
    merged = merge_topics([[_t(0, 5, "Intro"), short], [long, _t(25, 30, "Sleep")]])
    assert merged == [
        _t(0, 5, "Intro"),
        Topic(
            title="Solar power is cheap",
            summary=long.summary,
            start_segment=10,
            end_segment=20,
        ),
        _t(25, 30, "Sleep"),
    ]


def test_merge_takes_union_of_spans_and_longer_title():
    a, b = _t(10, 20, "Longer"), _t(12, 21, "Shorter")  # IoU 9/12
    (merged,) = merge_topics([[a], [b]])
    assert (merged.start_segment, merged.end_segment) == (10, 21)
    assert merged.title == "Longer"


def test_distinct_adjacent_topics_stay_separate():
    a, b = _t(0, 9, "A"), _t(10, 19, "B")
    assert merge_topics([[a], [b]]) == [a, b]
    # Slight overlap (IoU 2/18) is below the threshold.
    c = _t(8, 17, "C")
    assert merge_topics([[a], [c]]) == [a, c]


def test_topic_seen_in_three_windows_merges_into_one():
    windows = [[_t(20, 30, "w0")], [_t(21, 30, "w1")], [_t(20, 29, "w2")]]
    assert merge_topics(windows) == [_t(20, 30, "w0")]


def test_topics_from_the_same_window_are_not_merged():
    a, b = _t(0, 9, "A"), _t(1, 9, "B")
    assert merge_topics([[a, b]]) == [a, b]


def test_single_window_passes_through_sorted():
    topics = [_t(1, 3, "Solar"), _t(4, 7, "Sleep")]
    assert merge_topics([topics]) == topics
    assert merge_topics([list(reversed(topics))]) == topics


def test_empty_input():
    assert merge_topics([]) == []
    assert merge_topics([[], []]) == []


def test_threshold_is_configurable():
    a, b = _t(0, 9), _t(5, 14)  # IoU 1/3
    assert len(merge_topics([[a], [b]])) == 2
    assert merge_topics([[a], [b]], iou_threshold=0.3) == [
        Topic(title=a.title, summary=a.summary, start_segment=0, end_segment=14)
    ]


# --- stage -----------------------------------------------------------------------------


def _fixed_chunker(spans: list[tuple[int, int]]):
    def chunk(transcript: Transcript, config: ChunkConfig) -> list[TranscriptWindow]:
        return [TranscriptWindow(start_segment=s, end_segment=e, text=f"{s}-{e}") for s, e in spans]

    return chunk


@dataclass
class _ScriptedExtractor:
    by_window: dict[int, list[Topic]]
    seen: list[TranscriptWindow] = field(default_factory=list)

    def extract(self, window: TranscriptWindow) -> list[Topic]:
        self.seen.append(window)
        return self.by_window[window.start_segment]


def test_llm_stage_writes_merged_topics(job_copy: Job):
    golden = Job(job_copy.root).read_topics()
    solar, sleep = golden
    job_copy.topics_path.unlink()
    extractor = _ScriptedExtractor(
        {
            0: [solar, _t(4, 5, sleep.title)],  # window 0–5 sees only the start of "sleep"
            3: [sleep],
        }
    )
    stage = LLMStage(
        chunker=_fixed_chunker([(0, 5), (3, 7)]),
        extractor=extractor,
        models={"llm": "fake/llm", "llm_prompt": "topics-v1"},
    )
    _: Stage = stage  # type-checks against the Stage protocol
    assert stage.name is StageName.LLM
    assert stage.outputs(job_copy) == [job_copy.topics_path]

    stage.run(job_copy, PipelineConfig())

    assert job_copy.read_topics() == golden
    assert [w.start_segment for w in extractor.seen] == [0, 3]
    models = job_copy.read_meta().models
    assert models["llm"] == "fake/llm"
    assert models["llm_prompt"] == "topics-v1"
    assert models["whisper"] == "faster-whisper/small", "other entries are kept"


def test_llm_stage_passes_chunk_config(job_copy: Job):
    seen: list[ChunkConfig] = []

    def chunk(transcript: Transcript, config: ChunkConfig) -> list[TranscriptWindow]:
        seen.append(config)
        return []

    LLMStage(chunker=chunk, extractor=_ScriptedExtractor({})).run(
        job_copy, PipelineConfig(chunk=ChunkConfig(max_tokens=42))
    )
    assert seen[0].max_tokens == 42
    assert job_copy.read_topics() == []


def test_llm_stage_requires_transcript(job_copy: Job):
    job_copy.transcript_path.unlink()
    stage = LLMStage(chunker=_fixed_chunker([]), extractor=_ScriptedExtractor({}))
    with pytest.raises(ArtifactError, match="missing"):
        stage.run(job_copy, PipelineConfig())


# --- parallel extraction (#56) ------------------------------------------------------------


@dataclass
class _SlowExtractor:
    """One topic per window after ``delay(window)`` seconds; thread-safe bookkeeping."""

    delay: Callable[[TranscriptWindow], float]
    fail_at: int | None = None
    started: list[int] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: int = 0
    peak: int = 0

    def extract(self, window: TranscriptWindow) -> list[Topic]:
        with self.lock:
            self.started.append(window.start_segment)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(self.delay(window))
            if window.start_segment == self.fail_at:
                raise LLMError(f"provider down at window {window.start_segment}")
            s = window.start_segment
            return [_t(s, s + 3, f"window {s}")]
        finally:
            with self.lock:
                self.in_flight -= 1


def _windows(n: int) -> list[tuple[int, int]]:
    return [(10 * i, 10 * i + 9) for i in range(n)]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (LLMConfig(), 1),  # Ollama
        (LLMConfig(provider="groq"), 4),
        (LLMConfig(provider="groq", max_concurrency=1), 1),
        (LLMConfig(max_concurrency=3), 3),  # explicit value wins, even for Ollama
    ],
)
def test_concurrency_defaults(config: LLMConfig, expected: int):
    assert concurrency(config) == expected


def test_topics_identical_sequential_and_parallel(job_copy: Job):
    """Later windows finish first in parallel; topics.json must not change."""
    results = []
    for workers in (1, 4):
        extractor = _SlowExtractor(delay=lambda w: 0.05 * (8 - w.start_segment // 10))
        LLMStage(chunker=_fixed_chunker(_windows(8)), extractor=extractor).run(
            job_copy, PipelineConfig(llm=LLMConfig(max_concurrency=workers))
        )
        results.append(job_copy.topics_path.read_bytes())
        assert extractor.peak == workers
    assert results[0] == results[1]


def test_parallel_wall_time_is_windows_over_workers():
    extractor = _SlowExtractor(delay=lambda w: 0.3)
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(8)]
    t = time.perf_counter()
    topics = extract_all(extractor, windows, workers=4)
    elapsed = time.perf_counter() - t
    assert [ts[0].start_segment for ts in topics] == [w.start_segment for w in windows]
    assert 0.55 <= elapsed < 1.15, f"8 windows × 0.3 s on 4 workers ≈ 0.6 s, took {elapsed:.2f}"


def test_first_error_cancels_windows_not_started():
    extractor = _SlowExtractor(delay=lambda w: 0.1, fail_at=0)
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(12)]
    with pytest.raises(LLMError, match="window 0"):
        extract_all(extractor, windows, workers=2)
    assert len(extractor.started) < 12, "queued windows were cancelled"
    assert extractor.in_flight == 0, "calls in flight finished before raising"


def test_error_in_a_later_window_stops_new_windows_without_waiting_for_window_0():
    """Review case: window 0 is slow, window 3 fails fast; the rest must not start."""
    extractor = _SlowExtractor(delay=lambda w: 1.0 if w.start_segment == 0 else 0.05, fail_at=30)
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(15)]
    with pytest.raises(LLMError, match="window 30"):
        extract_all(extractor, windows, workers=4)
    assert len(extractor.started) <= 8, f"started {sorted(extractor.started)}"


class _TwoFailures:
    """Window 10 fails slowly, window 20 fails at once: window 10's error is reported."""

    def extract(self, window: TranscriptWindow) -> list[Topic]:
        time.sleep(0.2 if window.start_segment == 10 else 0.0)
        if window.start_segment in (10, 20):
            raise LLMError(f"provider down at window {window.start_segment}")
        return []


def test_lowest_failed_window_is_reported():
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(3)]
    with pytest.raises(LLMError, match="window 10"):
        extract_all(_TwoFailures(), windows, workers=3)


def test_llm_stage_reports_windows_done(job_copy: Job):
    spans = [(0, 2), (2, 4), (4, 6), (6, 7)]
    extractor = _ScriptedExtractor({s: [] for s, _ in spans})
    reports: list[tuple[float, str]] = []
    with progress.capture(lambda fraction, detail: reports.append((fraction, detail))):
        LLMStage(chunker=_fixed_chunker(spans), extractor=extractor).run(job_copy, PipelineConfig())
    assert reports == [
        (0.25, "window 1/4"),
        (0.5, "window 2/4"),
        (0.75, "window 3/4"),
        (1.0, "window 4/4"),
    ]


def test_parallel_extraction_reports_progress_per_finished_window(job_copy: Job):
    """#56 + #61: windows finish out of order on pool threads, but progress is reported
    from the stage thread, once per finished window, and ends at 1.0."""
    from echo.pipeline import run_pipeline

    events = []
    extractor = _SlowExtractor(delay=lambda w: 0.02 * (6 - w.start_segment // 10))
    stage = LLMStage(chunker=_fixed_chunker(_windows(6)), extractor=extractor)
    run_pipeline(
        job_copy,
        [stage],
        PipelineConfig(llm=LLMConfig(max_concurrency=3)),
        force=True,
        on_event=events.append,
    )
    reports = [(e.progress, e.detail) for e in events if e.progress is not None]
    assert [d for _, d in reports] == [f"window {n}/6" for n in range(1, 7)]
    assert reports[-1][0] == 1.0


def test_on_done_counts_only_successes_before_a_failure():
    done: list[int] = []
    extractor = _SlowExtractor(delay=lambda w: 0.05 * (w.start_segment // 10), fail_at=20)
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(6)]
    with pytest.raises(LLMError, match="window 20"):
        extract_all(extractor, windows, workers=2, on_done=done.append)
    assert done == list(range(1, len(done) + 1)) and len(done) < 6


# --- Ctrl+C (#83) ----------------------------------------------------------------------


class _BlockingClient:
    """``LLMClient`` whose calls block for ``seconds`` or until ``close()``."""

    model_name = "fake/blocking"

    def __init__(self, seconds: float = 3.0) -> None:
        self.seconds = seconds
        self.closed = threading.Event()
        self.calls = 0

    def complete(self, system: str, prompt: str, *, json_schema=None) -> str:
        self.calls += 1
        if self.closed.wait(self.seconds):
            raise LLMError("client closed")  # must not replace the KeyboardInterrupt
        return '{"topics": []}'

    def close(self) -> None:
        self.closed.set()


@contextmanager
def _ctrl_c_after(seconds: float):
    """Interrupt the main thread like Ctrl+C (works on Windows too); yields a clock."""
    timer = threading.Timer(seconds, _thread.interrupt_main)
    t0 = time.perf_counter()
    timer.start()
    try:
        yield lambda: time.perf_counter() - t0
    finally:
        timer.cancel()


@pytest.mark.parametrize("workers", [1, 4])
def test_ctrl_c_stops_extraction_without_waiting_for_calls_in_flight(workers: int):
    client = _BlockingClient(seconds=3.0)
    extractor = LLMTopicExtractor(client, LLMConfig(max_retries=0))
    windows = [TranscriptWindow(start_segment=s, end_segment=e, text="") for s, e in _windows(8)]
    with _ctrl_c_after(0.3) as elapsed, pytest.raises(KeyboardInterrupt):
        extract_all(extractor, windows, workers=workers)
    assert elapsed() < 1.0, f"Ctrl+C took {elapsed():.2f} s"
    assert client.closed.is_set(), "the client is closed so calls in flight fail fast"
    time.sleep(0.1)  # the workers see the closed client and stop
    assert client.calls == workers, "no window starts after Ctrl+C"


def test_ctrl_c_writes_no_topics_and_is_recorded_as_interrupted(job_copy: Job):
    from echo.models import StageStatus
    from echo.pipeline import run_pipeline

    job_copy.topics_path.unlink()
    extractor = LLMTopicExtractor(_BlockingClient(), LLMConfig())
    stage = LLMStage(chunker=_fixed_chunker(_windows(6)), extractor=extractor)
    with _ctrl_c_after(0.3) as elapsed, pytest.raises(KeyboardInterrupt):
        run_pipeline(job_copy, [stage], PipelineConfig(llm=LLMConfig(max_concurrency=3)))
    assert elapsed() < 1.0
    assert not job_copy.topics_path.exists()
    record = job_copy.read_meta().stages[StageName.LLM]
    assert record.status == StageStatus.FAILED
    assert record.error == "interrupted"


_EXIT_SCRIPT = """
import _thread, threading, time
from echo.llm.merge import extract_all
from echo.models import TranscriptWindow

class Stuck:  # ignores close(): the worst case for interpreter exit
    def extract(self, window):
        time.sleep(5)
        return []

windows = [TranscriptWindow(start_segment=i, end_segment=i, text="") for i in range(8)]
threading.Timer(0.3, _thread.interrupt_main).start()
try:
    extract_all(Stuck(), windows, workers=4)
except KeyboardInterrupt:
    print("interrupted")
"""


def test_process_exits_promptly_after_ctrl_c():
    """Worker threads stuck in a call must not keep the interpreter alive at exit."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT], capture_output=True, text=True, timeout=30
    )
    elapsed = time.perf_counter() - t0
    assert result.stdout.strip() == "interrupted", result.stderr
    assert elapsed < 4.0, f"exit took {elapsed:.2f} s; a joined worker would take ≥ 5 s"
