"""Clip scoring (#21): parsing, retries, batching and the score stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from echo.config import LLMConfig, PipelineConfig
from echo.errors import ArtifactError, LLMError, TopicExtractionError
from echo.job import Job
from echo.llm import prompts
from echo.llm.scoring import (
    WEIGHTS,
    InvalidScoresError,
    LLMClipScorer,
    ScoreStage,
    batches,
    clip_text,
    overall,
    parse_scores,
)
from echo.models import Score, Segment, StageName, StageStatus, Topic
from echo.pipeline import run_pipeline
from echo.testing import FakeLLMClient

CONFIG = PipelineConfig()


def entry(clip: int, hook=6, payoff=7, clarity=8, energy=5, reason="Good.") -> dict:
    return {
        "clip": clip,
        "hook": hook,
        "payoff": payoff,
        "clarity": clarity,
        "energy": energy,
        "reason": reason,
    }


def answer(*entries: dict) -> str:
    return json.dumps({"scores": list(entries)})


# --- parse_scores ---------------------------------------------------------------------


def test_parses_scores_in_clip_order_and_computes_overall():
    scores = parse_scores(answer(entry(1, hook=2), entry(0, hook=9)), 2)
    assert [s.hook for s in scores] == [9, 2]
    expected = overall({"hook": 9, "payoff": 7, "clarity": 8, "energy": 5})
    assert scores[0].overall == expected == pytest.approx(7.55)
    assert scores[0].reason == "Good."


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + answer(entry(0)) + "\n```",
        json.dumps([entry(0)]),
        json.dumps({"scores": [entry(0, hook=6.5)]}),
    ],
)
def test_accepts_fences_bare_lists_and_fractions(raw):
    (score,) = parse_scores(raw, 1)
    assert 0 <= score.overall <= 10


def test_missing_reason_is_empty():
    item = entry(0)
    del item["reason"]
    assert parse_scores(answer(item), 1)[0].reason == ""


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"topics": []}', "of the form"),
        (answer(entry(0)), "no score for clip(s) 1"),
        (answer(entry(0), entry(0), entry(1)), "more than once"),
        (answer(entry(0), entry(2)), "there is no clip 2"),
        (answer(entry(0), entry(1, hook=11)), '"hook" is 11'),
        (answer(entry(0), entry(1, energy=-1)), '"energy" is -1'),
        (answer(entry(0), entry(1, payoff="high")), '"payoff" must be a number'),
        (answer(entry(0), entry(1, clarity=True)), '"clarity" must be a number'),
        (answer(entry("0"), entry(1)), '"clip" must be an integer'),
        (answer(entry(0), entry(1, reason=3)), '"reason" must be a string'),
        (json.dumps({"scores": [1, 2]}), "not an object"),
    ],
)
def test_rejects_invalid_answers(raw, message):
    with pytest.raises(InvalidScoresError) as e:
        parse_scores(raw, 2)
    assert message in str(e.value)


# --- LLMClipScorer --------------------------------------------------------------------


def test_scorer_sends_numbered_clips_and_asks_for_json():
    client = FakeLLMClient([answer(entry(0), entry(1))])
    scores = LLMClipScorer(client, LLMConfig()).score(["First clip.", "Second clip."])
    assert len(scores) == 2
    (call,) = client.calls
    assert call.system == prompts.SCORE_SYSTEM_PROMPT
    assert "[0] First clip.\n[1] Second clip." in call.prompt
    assert call.json_schema == prompts.SCORES_SCHEMA


def test_scorer_retries_with_the_error_then_succeeds():
    client = FakeLLMClient(["oops", answer(entry(0)), answer(entry(0), entry(1))])
    scores = LLMClipScorer(client, LLMConfig()).score(["a", "b"])
    assert len(scores) == 2 and len(client.calls) == 3
    assert "not valid JSON" in client.calls[1].prompt and "oops" in client.calls[1].prompt
    assert "no score for clip(s) 1" in client.calls[2].prompt
    assert "[0] a" in client.calls[2].prompt  # the original clips are resent


def test_scorer_gives_up_after_max_retries():
    client = FakeLLMClient(["oops"] * 5)
    with pytest.raises(TopicExtractionError, match="still invalid after 2 attempts"):
        LLMClipScorer(client, LLMConfig(max_retries=1)).score(["a"])
    assert len(client.calls) == 2


def test_scorer_does_not_retry_provider_errors():
    client = FakeLLMClient([])
    with pytest.raises(LLMError):
        LLMClipScorer(client, LLMConfig()).score(["a"])
    assert len(client.calls) == 1


def test_scorer_with_no_clips_makes_no_call():
    client = FakeLLMClient([])
    assert LLMClipScorer(client, LLMConfig()).score([]) == []
    assert client.calls == []


# --- clip text and batches ------------------------------------------------------------


def _segments(*lengths: float) -> list[Segment]:
    segments, t = [], 0.0
    for i, length in enumerate(lengths):
        segments.append(Segment(id=i, start=t, end=t + length, text=f" s{i} "))
        t += length
    return segments


def _topic(start: int, end: int, title: str = "t") -> Topic:
    return Topic(title=title, summary="", start_segment=start, end_segment=end)


def test_clip_text_is_what_the_clip_would_say():
    """Refine trims long topics from the end, so the text stops at max_duration."""
    segments = _segments(10, 10, 10, 10, 10, 10, 10)
    assert clip_text(segments, _topic(1, 6), 60) == "s1 s2 s3 s4 s5 s6"
    assert clip_text(segments, _topic(1, 6), 35) == "s1 s2 s3"
    assert clip_text(segments, _topic(2, 2), 5) == "s2", "at least the first segment"


def test_batches_fill_the_budget_in_order():
    texts = ["x" * 400] * 5  # 100 tokens + 40 for the answer = 140 each
    assert batches(texts, 600 + 300) == [range(0, 2), range(2, 4), range(4, 5)]
    assert batches(texts, 12_000) == [range(0, 5)]
    assert batches(["x" * 10_000, "y"], 1_000) == [range(0, 1), range(1, 2)]
    assert batches([], 1_000) == []


# --- ScoreStage -----------------------------------------------------------------------


@dataclass
class RecordingScorer:
    """A ``ClipScorer`` that scores clip ``i`` of each call by its text length."""

    calls: list[list[str]] = field(default_factory=list)

    def score(self, clips: list[str]) -> list[Score]:
        self.calls.append(clips)
        return [
            Score(overall=min(len(c) / 10, 10), hook=5, payoff=5, clarity=5, energy=5)
            for c in clips
        ]


def test_stage_writes_scores_aligned_with_topics(job_copy: Job):
    job_copy.scores_path.unlink()
    client = FakeLLMClient([answer(entry(0, hook=3), entry(1, hook=9))])
    stage = ScoreStage(LLMClipScorer(client, LLMConfig()))
    assert stage.outputs(job_copy) == [job_copy.scores_path]

    stage.run(job_copy, CONFIG)

    topics, scored = job_copy.read_topics(), job_copy.read_scores()
    assert [t.model_copy(update={"score": None}) for t in scored] == topics
    assert [t.score.hook for t in scored] == [3, 9]
    assert len(client.calls) == 1, "one batched call for the whole job"
    assert "solar" in client.calls[0].prompt.lower()
    assert job_copy.read_meta().models["score_prompt"] == prompts.SCORE_PROMPT_VERSION


def test_stage_batches_by_the_window_budget(job_copy: Job):
    scorer = RecordingScorer()
    config = PipelineConfig(chunk={"max_tokens": 650})  # room for ~one clip per call
    ScoreStage(scorer).run(job_copy, config)
    assert [len(c) for c in scorer.calls] == [1, 1]
    assert len(job_copy.read_scores()) == 2


def test_stage_rejects_topics_outside_the_transcript(job_copy: Job):
    job_copy.write_topics([_topic(0, 999)])
    with pytest.raises(ArtifactError, match="999"):
        ScoreStage(RecordingScorer()).run(job_copy, CONFIG)


def test_stage_with_no_topics_writes_an_empty_list(job_copy: Job):
    job_copy.write_topics([])
    scorer = RecordingScorer()
    ScoreStage(scorer).run(job_copy, CONFIG)
    assert job_copy.read_scores() == [] and scorer.calls == []


# --- re-runs (settings fingerprints) ---------------------------------------------------


@dataclass
class TopicsStage:
    """Stands in for the llm stage: writes the golden topics, logging each run."""

    topics: list[Topic]
    log: list[StageName]
    name: StageName = StageName.LLM

    def outputs(self, job: Job) -> list[Path]:
        return [job.topics_path]

    def run(self, job: Job, config: PipelineConfig) -> None:
        self.log.append(self.name)
        job.write_topics(self.topics)


def _rerun(job: Job, config: PipelineConfig, scorer: RecordingScorer, **kwargs) -> int:
    before = len(scorer.calls)
    topics = TopicsStage(job.read_topics(), [])
    run_pipeline(job, [topics, ScoreStage(scorer)], config, **kwargs)
    return len(scorer.calls) - before


def test_score_reruns_only_when_its_settings_change(job_copy: Job):
    job_copy.scores_path.unlink()
    scorer = RecordingScorer()
    assert _rerun(job_copy, CONFIG, scorer) == 1, "first run scores"
    assert _rerun(job_copy, CONFIG, scorer) == 0, "same settings: skipped"
    record = job_copy.read_meta().stages[StageName.SCORE]
    assert record.status is StageStatus.SKIPPED
    assert record.settings.startswith(prompts.SCORE_PROMPT_VERSION)

    gemini = PipelineConfig(llm={"provider": "gemini"})
    assert _rerun(job_copy, gemini, scorer) == 1, "another model re-scores"
    shorter = gemini.model_copy(
        update={"refine": gemini.refine.model_copy(update={"max_duration": 45})}
    )
    assert _rerun(job_copy, shorter, scorer) == 1, "clip text changes with max_duration"
    assert _rerun(job_copy, shorter, scorer) == 0
    # Settings the scores don't depend on don't re-score.
    more = shorter.model_copy(update={"refine": shorter.refine.model_copy(update={"max_clips": 3})})
    assert _rerun(job_copy, more, scorer) == 0


def test_new_topics_are_rescored(job_copy: Job):
    job_copy.scores_path.unlink()
    scorer = RecordingScorer()
    _rerun(job_copy, CONFIG, scorer)
    assert _rerun(job_copy, CONFIG, scorer, from_stage=StageName.LLM) == 1


def test_old_job_without_scores_gets_scored(job_copy: Job):
    """A job from before #21 (golden job.json has no score stage) scores on its next run,
    without re-extracting topics."""
    job_copy.scores_path.unlink()
    scorer, log = RecordingScorer(), []
    topics = TopicsStage(job_copy.read_topics(), log)
    run_pipeline(job_copy, [topics, ScoreStage(scorer)], CONFIG)
    assert log == [] and len(scorer.calls) == 1
    assert job_copy.read_meta().stages[StageName.SCORE].status is StageStatus.DONE
