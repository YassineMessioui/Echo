from dataclasses import dataclass, field

import pytest

from echo import progress
from echo.config import ChunkConfig, PipelineConfig
from echo.errors import ArtifactError
from echo.interfaces import Stage
from echo.job import Job
from echo.llm.merge import LLMStage, merge_topics, span_containment, span_iou
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
