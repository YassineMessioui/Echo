import random

import pytest

from echo.config import PipelineConfig, RefineConfig
from echo.errors import ArtifactError
from echo.job import Job
from echo.models import ClipSpec, Segment, Topic, Transcript, Word
from echo.refine import RefineStage, refine_clips, resolve_overlaps

NO_PAD = RefineConfig(padding=0)


def _words(start: float, end: float, every: float) -> list[Word]:
    times = [start + k * every for k in range(int((end - start) / every))] + [end]
    return [
        Word(start=round(a, 3), end=round(b, 3), text="w")
        for a, b in zip(times, times[1:], strict=False)
    ]


def _transcript(
    lengths: list[float], gap: float = 0.0, tail: float = 0.0, words_over: float | None = None
) -> Transcript:
    """Back-to-back segments with the given lengths (seconds), ``gap`` apart. Segments
    longer than ``words_over`` seconds get a word every 0.4 s."""
    segments, t = [], 0.0
    for i, length in enumerate(lengths):
        start, end = round(t, 3), round(t + length, 3)
        words = _words(start, end, 0.4) if words_over and length > words_over else []
        segments.append(Segment(id=i, start=start, end=end, text=f"s{i}", words=words))
        t += length + gap
    return Transcript(language="en", duration=round(t - gap + tail, 3), segments=segments)


def _topic(start: int, end: int, title: str | None = None) -> Topic:
    return Topic(title=title or f"t{start}-{end}", summary="", start_segment=start, end_segment=end)


def _spans(clips: list[ClipSpec]) -> list[tuple[float, float]]:
    return [(c.start, c.end) for c in clips]


def check_invariants(transcript: Transcript, clips: list[ClipSpec], config: RefineConfig) -> None:
    edges = {s.start for s in transcript.segments} | {s.end for s in transcript.segments}
    edges |= {w.end for s in transcript.segments for w in s.words}
    tolerance = config.padding + 1e-6
    for clip in clips:
        assert config.min_duration - 1e-6 <= clip.duration <= config.max_duration + 1e-6
        assert 0 <= clip.start and clip.end <= transcript.duration + 1e-6
        assert min(abs(clip.start - e) for e in edges) <= tolerance, "start on a segment edge"
        assert min(abs(clip.end - e) for e in edges) <= tolerance, "end on a segment edge"
    for prev, cur in zip(clips, clips[1:], strict=False):
        assert prev.end <= cur.start, "sorted and non-overlapping"


# --- golden ------------------------------------------------------------------------------


def test_reproduces_golden_clips(golden_job: Job):
    transcript, topics = golden_job.read_transcript(), golden_job.read_topics()
    clips = refine_clips(transcript, topics, RefineConfig())
    assert clips == golden_job.read_clips()
    check_invariants(transcript, clips, RefineConfig())


def test_refine_stage_writes_clips(job_copy: Job, golden_job: Job):
    job_copy.clips_path.unlink()
    stage = RefineStage()
    assert stage.outputs(job_copy) == [job_copy.clips_path]
    stage.run(job_copy, PipelineConfig())
    assert job_copy.read_clips() == golden_job.read_clips()


def test_refine_stage_requires_topics(job_copy: Job):
    job_copy.topics_path.unlink()
    with pytest.raises(ArtifactError, match="missing"):
        RefineStage().run(job_copy, PipelineConfig())


# --- mapping and durations (table-driven, no padding) ------------------------------------


@pytest.mark.parametrize(
    ("lengths", "topics", "expected"),
    [
        pytest.param([10, 10, 10, 10], [(1, 2)], [(10, 30)], id="maps-to-segment-edges"),
        pytest.param([10, 3, 10, 10], [(1, 1)], [(10, 33)], id="short-extends-forward"),
        pytest.param([10, 10, 3], [(2, 2)], [(0, 23)], id="short-extends-backward-at-end"),
        pytest.param([5, 3, 4], [(1, 1)], [], id="short-dropped-when-nothing-to-add"),
        pytest.param([20, 20, 20, 20], [(0, 3)], [(0, 60)], id="long-trimmed-at-segment-edge"),
        pytest.param([25, 25, 25], [(0, 2)], [(0, 50)], id="trim-keeps-start"),
        pytest.param([20, 20, 20], [(0, 2)], [(0, 60)], id="exactly-max-is-kept"),
        pytest.param([15], [(0, 0)], [(0, 15)], id="exactly-min-is-kept"),
        pytest.param([5, 5, 5], [(0, 0), (1, 1)], [(0, 15)], id="short-cannot-enter-next-clip"),
    ],
)
def test_durations(lengths, topics, expected):
    transcript = _transcript(lengths)
    clips = refine_clips(transcript, [_topic(*t) for t in topics], NO_PAD)
    assert _spans(clips) == expected
    check_invariants(transcript, clips, NO_PAD)


def test_extension_does_not_enter_neighbouring_topic():
    transcript = _transcript([10, 5, 5, 20])
    clips = refine_clips(transcript, [_topic(1, 1, "short"), _topic(3, 3, "next")], NO_PAD)
    assert [(c.topic_title, c.start, c.end) for c in clips] == [
        ("short", 0, 20),  # took segment 2, then segment 0; never segment 3
        ("next", 20, 40),
    ]


def test_single_long_segment_is_cut_at_a_word():
    words = [Word(start=i * 7.0, end=i * 7.0 + 6.5, text=f"w{i}") for i in range(13)]
    seg = Segment(id=0, start=0, end=90, text="long", words=words)
    transcript = Transcript(language="en", duration=90, segments=[seg])
    (clip,) = refine_clips(transcript, [_topic(0, 0)], NO_PAD)
    assert (clip.start, clip.end) == (0, 55.5)  # word 7 ends at 55.5, word 8 at 62.5


def test_single_long_segment_without_words_is_hard_cut():
    transcript = _transcript([90])
    (clip,) = refine_clips(transcript, [_topic(0, 0)], NO_PAD)
    assert clip.duration == 60


def test_custom_limits():
    transcript = _transcript([4] * 10)
    config = RefineConfig(min_duration=5, max_duration=10, padding=0)
    clips = refine_clips(transcript, [_topic(0, 0), _topic(3, 9)], config)
    assert _spans(clips) == [(0, 8), (12, 20)]


# --- overlaps -----------------------------------------------------------------------------


def test_heavily_overlapping_topics_merge_keeping_longer_title():
    segments = _transcript([5] * 10).segments
    spans = resolve_overlaps(segments, [_topic(0, 5, "long"), _topic(3, 6, "short")])
    assert [(s.title, s.start, s.end) for s in spans] == [("long", 0, 6)]


def test_lightly_overlapping_topics_are_trimmed():
    segments = _transcript([5] * 10).segments
    spans = resolve_overlaps(segments, [_topic(0, 4, "a"), _topic(4, 9, "b")])
    assert [(s.title, s.start, s.end) for s in spans] == [("a", 0, 4), ("b", 5, 9)]


def test_contained_topic_is_merged():
    segments = _transcript([5] * 10).segments
    spans = resolve_overlaps(segments, [_topic(2, 3, "inner"), _topic(0, 9, "outer")])
    assert [(s.title, s.start, s.end) for s in spans] == [("outer", 0, 9)]


def test_output_never_overlaps():
    transcript = _transcript([6] * 20)
    topics = [_topic(0, 4), _topic(3, 8), _topic(8, 12), _topic(12, 19), _topic(15, 16)]
    clips = refine_clips(transcript, topics, RefineConfig())
    assert clips
    check_invariants(transcript, clips, RefineConfig())


def test_overlapping_segments_in_time_do_not_overlap_clips():
    segments = [
        Segment(id=0, start=0, end=20.5, text="a"),
        Segment(id=1, start=20, end=40, text="b"),
    ]
    transcript = Transcript(language="en", duration=40, segments=segments)
    clips = refine_clips(transcript, [_topic(0, 0), _topic(1, 1)], NO_PAD)
    assert _spans(clips) == [(0, 20), (20, 40)]


# --- padding ------------------------------------------------------------------------------


def test_padding_clamped_to_media_bounds():
    # Segments: [0, 0.1] [0.1, 20.1] [20.1, 20.15]; media lasts 20.2 s.
    transcript = _transcript([0.1, 20, 0.05], tail=0.05)
    (clip,) = refine_clips(transcript, [_topic(1, 1)], RefineConfig(padding=0.2))
    assert (clip.start, clip.end) == (0, 20.2)


def test_padding_splits_the_gap_between_clips():
    transcript = _transcript([20, 20], gap=0.3, tail=1)
    clips = refine_clips(transcript, [_topic(0, 0), _topic(1, 1)], RefineConfig(padding=0.2))
    assert _spans(clips) == [(0, 20.15), (20.15, 40.5)]


def test_padding_never_pushes_a_clip_over_the_max():
    transcript = _transcript([20, 20, 20], tail=5)
    (clip,) = refine_clips(transcript, [_topic(0, 2)], RefineConfig(padding=0.2))
    assert (clip.start, clip.end) == (0, 40.2)  # 60 s + padding would exceed 60 s


def test_padding_is_capped_when_limits_are_tight():
    transcript = _transcript([10, 10, 10], tail=5)
    config = RefineConfig(min_duration=20, max_duration=20.2, padding=1)
    (clip,) = refine_clips(transcript, [_topic(1, 2)], config)
    assert (clip.start, clip.end) == (9.9, 30.1)
    check_invariants(transcript, [clip], config)


def test_no_segments_or_no_topics():
    assert refine_clips(Transcript(language="en", duration=0, segments=[]), [], NO_PAD) == []
    assert refine_clips(_transcript([30]), [], NO_PAD) == []


def test_topic_outside_transcript_is_an_artifact_error():
    with pytest.raises(ArtifactError, match="segment 5"):
        refine_clips(_transcript([30, 30]), [_topic(0, 5)], NO_PAD)


# --- randomised invariants ------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_invariants_hold_on_random_inputs(seed: int):
    rng = random.Random(seed)
    lengths = [round(rng.uniform(0.5, 25 if rng.random() < 0.9 else 90), 3) for _ in range(40)]
    gap = rng.choice([0, 0, 0.1, 0.7])
    transcript = _transcript(lengths, gap=gap, tail=rng.uniform(0, 1), words_over=25)
    topics = []
    for _ in range(rng.randint(0, 12)):
        start = rng.randrange(len(lengths))
        topics.append(_topic(start, min(len(lengths) - 1, start + rng.randint(0, 6))))
    config = RefineConfig(
        min_duration=rng.choice([5, 15, 30]),
        max_duration=rng.choice([30, 60]),
        padding=rng.choice([0, 0.2, 0.5]),
    )
    clips = refine_clips(transcript, topics, config)
    check_invariants(transcript, clips, config)


# --- max_clips (#62) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_clips", "expected"),
    [
        pytest.param(None, ["a", "b", "c", "d"], id="no-limit"),
        pytest.param(9, ["a", "b", "c", "d"], id="limit-above-count"),
        pytest.param(2, ["b", "d"], id="closest-to-ideal-kept-in-time-order"),
        pytest.param(1, ["b"], id="single"),
    ],
)
def test_max_clips(max_clips, expected):
    # Topic lengths: a=16 s, b=36 s, c=58 s, d=40 s (b and d are closest to 37.5 s).
    transcript = _transcript([16, 36, 58, 40])
    topics = [_topic(0, 0, "a"), _topic(1, 1, "b"), _topic(2, 2, "c"), _topic(3, 3, "d")]
    config = RefineConfig(padding=0, max_clips=max_clips)
    clips = refine_clips(transcript, topics, config)
    assert [c.topic_title for c in clips] == expected
    check_invariants(transcript, clips, config)


def test_max_clips_ties_keep_the_earlier_clip():
    transcript = _transcript([30, 30, 30])
    topics = [_topic(i, i, t) for i, t in enumerate("xyz")]
    clips = refine_clips(transcript, topics, RefineConfig(padding=0, max_clips=2))
    assert [c.topic_title for c in clips] == ["x", "y"]


def test_max_clips_must_be_positive():
    with pytest.raises(ValueError):
        RefineConfig(max_clips=0)


def test_kept_clip_is_not_left_trimmed_by_a_dropped_neighbour():
    """Review on #71: A = segs 0-5, B = segs 4-8 (light overlap, B trimmed to start at
    seg 6). With max_clips=1, B wins and must span its whole topic again (40-90)."""
    transcript = _transcript([10] * 9)
    topics = [_topic(0, 5, "A"), _topic(4, 8, "B")]
    all_clips = refine_clips(transcript, topics, RefineConfig(padding=0))
    assert [(c.topic_title, c.start, c.end) for c in all_clips] == [("A", 0, 60), ("B", 60, 90)]
    (only,) = refine_clips(transcript, topics, RefineConfig(padding=0, max_clips=1))
    assert (only.topic_title, only.start, only.end) == ("B", 40, 90)
