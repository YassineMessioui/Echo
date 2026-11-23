"""YouTube's "Most replayed" heatmap in clip scoring (#123)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from echo.config import PipelineConfig
from echo.job import _TOPICS, Job
from echo.llm.scoring import (
    CRITERIA,
    ScoreStage,
    blend,
    clip_span,
    overall,
    replay_score,
    settings_fingerprint,
)
from echo.models import HeatmapPoint, ReplayHeatmap, Score, Segment, StageName, Topic, Transcript
from echo.pipeline import run_pipeline


def heatmap(*values: float, step: float = 10.0, start: float = 0.0) -> ReplayHeatmap:
    """Bars of ``step`` seconds from ``start``, one per value."""
    return ReplayHeatmap(
        points=[
            HeatmapPoint(start=start + i * step, end=start + (i + 1) * step, value=v)
            for i, v in enumerate(values)
        ]
    )


# Median 0.2, peak 1.0.
HEATMAP = heatmap(0.2, 0.2, 0.2, 1.0, 0.6, 0.0, 0.2)


@dataclass
class FixedScorer:
    """A ``ClipScorer`` that gives every clip the same LLM criteria."""

    hook: float = 8
    calls: list[list[str]] = field(default_factory=list)

    def score(self, clips: list[str]) -> list[Score]:
        self.calls.append(clips)
        values = {"hook": self.hook, "payoff": 6, "clarity": 7, "energy": 5}
        return [Score(overall=overall(values), reason="r", **values) for _ in clips]


# --- replay_score -----------------------------------------------------------------------


def test_median_is_5_and_peak_is_10():
    assert replay_score(HEATMAP, 0, 10) == 5.0  # 0.2 = the median
    assert replay_score(HEATMAP, 30, 40) == 10.0  # the peak
    assert replay_score(HEATMAP, 40, 50) == 7.5  # halfway from median to peak
    assert replay_score(HEATMAP, 50, 60) == 0.0
    assert replay_score(HEATMAP, 30, 50) == 8.75  # time-weighted mean 0.8


def test_below_the_median_scales_down_to_0():
    flat_low = heatmap(0.4, 0.4, 0.1, 1.0)  # median 0.4
    assert replay_score(flat_low, 20, 30) == 1.25


def test_partial_overlap_at_the_clip_edges_is_time_weighted():
    # 5 s of the 0.2 bar + 10 s of the peak + 5 s of the 0.6 bar → mean 0.7.
    assert replay_score(HEATMAP, 25, 45) == pytest.approx(5 + 5 * 0.5 / 0.8, abs=0.01)
    # A clip just inside one bar only sees that bar.
    assert replay_score(HEATMAP, 31.5, 38.2) == 10.0


def test_span_outside_the_heatmap_has_no_replay():
    assert replay_score(HEATMAP, 70, 90) is None
    assert replay_score(HEATMAP, 75, 75) is None


def test_flat_heatmap_is_neutral():
    assert replay_score(heatmap(0.5, 0.5, 0.5), 0, 30) == 5.0


# --- blend --------------------------------------------------------------------------------


def test_blend_mixes_the_llm_overall_with_replay():
    llm = FixedScorer().score(["x"])[0]
    assert llm.overall == 6.7  # 0.3*8 + 0.3*6 + 0.25*7 + 0.15*5
    mixed = blend(llm, 10.0, 0.3)
    assert mixed.replay == 10.0
    assert mixed.overall == round(0.7 * 6.7 + 0.3 * 10, 2)
    assert {c: getattr(mixed, c) for c in CRITERIA} == {c: getattr(llm, c) for c in CRITERIA}
    assert blend(llm, 2.0, 0.0).overall == 6.7, "weight 0 shows replay without ranking by it"
    assert blend(llm, 2.0, 0.0).replay == 2.0
    assert blend(llm, None, 0.3) is llm


# --- the score stage ---------------------------------------------------------------------


def test_stage_blends_replay_when_the_job_has_a_heatmap(job_copy: Job):
    # Golden clips: "Solar" 8.5–34 s, "Sleep" 34–80 s; peak replays inside "Solar".
    job_copy.write_heatmap(heatmap(0.1, 1.0, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, step=10))
    config = PipelineConfig(score={"replay_weight": 0.4})
    ScoreStage(FixedScorer()).run(job_copy, config)
    solar, sleep = (t.score for t in job_copy.read_scores())
    assert solar.replay > 8 and sleep.replay == 5.0  # all at the median
    assert solar.overall > 6.7 > sleep.overall, "same LLM score: replay decides"
    assert '"replay":' in job_copy.scores_path.read_text()


def test_without_a_heatmap_scores_json_is_unchanged(job_copy: Job):
    """Byte for byte what the stage wrote before #123: the LLM scores, no "replay"."""
    scorer = FixedScorer()
    ScoreStage(scorer).run(job_copy, PipelineConfig(score={"replay_weight": 0.9}))
    expected = [
        t.model_copy(update={"score": s})
        for t, s in zip(job_copy.read_topics(), scorer.score(["a", "b"]), strict=True)
    ]
    written = job_copy.scores_path.read_bytes()
    assert written == _TOPICS.dump_json(expected, indent=2)
    assert b"replay" not in written


def test_focus_range_job_uses_original_video_time(job_copy: Job):
    """With ``--range 600:``, the transcript starts at 600 s of the video (offset) and
    its times are already original-video time, like the heatmap: no shift."""
    job = job_copy
    segments = [
        Segment(id=0, start=600, end=615, text="quiet"),
        Segment(id=1, start=615, end=630, text="the moment"),
    ]
    job.write_transcript(Transcript(language="en", duration=630, offset=600, segments=segments))
    job.write_topics(
        [
            Topic(title="quiet", summary="", start_segment=0, end_segment=0),
            Topic(title="moment", summary="", start_segment=1, end_segment=1),
        ]
    )
    # Peaks at 615–630 s of the video ("moment") and at 0–15 s, which is where "quiet"
    # would land if its times were wrongly shifted by the offset.
    values = [1.0] + [0.1] * 40 + [1.0] + [0.1] * 20
    job.write_heatmap(heatmap(*values, step=15))
    ScoreStage(FixedScorer()).run(job, PipelineConfig())
    quiet, moment = (t.score for t in job.read_scores())
    assert (quiet.replay, moment.replay) == (5.0, 10.0)  # median, peak


def test_clip_span_matches_clip_text():
    segments = [Segment(id=i, start=i * 10.0, end=i * 10.0 + 10, text=f"s{i}") for i in range(7)]
    topic = Topic(title="t", summary="", start_segment=1, end_segment=6)
    assert clip_span(segments, topic, 60) == (10, 70)
    assert clip_span(segments, topic, 35) == (10, 40)
    assert clip_span(segments, Topic(title="t", summary="", start_segment=2, end_segment=2), 5) == (
        20,
        30,
    )


# --- settings fingerprint ----------------------------------------------------------------


def test_fingerprint_without_a_heatmap_is_unchanged(job_copy: Job):
    """No heatmap: the pre-#123 fingerprint, whatever the weight, so old jobs and local
    files aren't re-scored."""
    base = settings_fingerprint(PipelineConfig())
    assert settings_fingerprint(PipelineConfig(), job_copy) == base
    assert settings_fingerprint(PipelineConfig(score={"replay_weight": 0.9}), job_copy) == base


def test_fingerprint_with_a_heatmap_includes_it_and_the_weight(job_copy: Job):
    base = settings_fingerprint(PipelineConfig(), job_copy)
    job_copy.write_heatmap(HEATMAP)
    with_heatmap = settings_fingerprint(PipelineConfig(), job_copy)
    heavier = settings_fingerprint(PipelineConfig(score={"replay_weight": 0.9}), job_copy)
    assert len({base, with_heatmap, heavier}) == 3


def test_heatmap_arriving_or_weight_change_rescores(job_copy: Job):
    scorer = FixedScorer()
    stages = [ScoreStage(scorer)]
    run_pipeline(job_copy, stages, PipelineConfig(), from_stage=StageName.SCORE)
    assert len(scorer.calls) == 1
    run_pipeline(job_copy, stages, PipelineConfig())
    assert len(scorer.calls) == 1, "same settings: skipped"

    job_copy.write_heatmap(HEATMAP)  # re-probing an old link job saved one
    run_pipeline(job_copy, stages, PipelineConfig())
    assert len(scorer.calls) == 2
    assert all(t.score.replay is not None for t in job_copy.read_scores())

    run_pipeline(job_copy, stages, PipelineConfig(score={"replay_weight": 0.5}))
    assert len(scorer.calls) == 3
    run_pipeline(job_copy, stages, PipelineConfig(score={"replay_weight": 0.5}))
    assert len(scorer.calls) == 3
