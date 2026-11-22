from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import echo.pipeline as pipeline
from echo.clip import ClipStage
from echo.config import PipelineConfig
from echo.errors import EchoError, LLMError
from echo.ingest import IngestStage
from echo.job import Job
from echo.llm.merge import LLMStage
from echo.llm.prompts import PROMPT_VERSION, SCORE_PROMPT_VERSION
from echo.llm.scoring import ScoreStage
from echo.models import StageName, StageStatus
from echo.pipeline import PipelineError, StageEvent, build_stages, run_pipeline
from echo.refine import RefineStage
from echo.testing import FakeLLMClient, FakeTranscriber
from echo.transcribe import TranscribeStage

CONFIG = PipelineConfig()


@dataclass
class FakeStage:
    """Writes ``<name>.out`` into the job; optionally raises instead."""

    name: StageName
    log: list[StageName]
    error: BaseException | None = None

    def outputs(self, job: Job) -> list[Path]:
        return [job.root / f"{self.name}.out"]

    def run(self, job: Job, config: PipelineConfig) -> None:
        self.log.append(self.name)
        if self.error is not None:
            raise self.error
        self.outputs(job)[0].write_text("ok")


@dataclass
class ForceableStage(FakeStage):
    force: bool = False
    forced_runs: list[bool] = field(default_factory=list)

    def run(self, job: Job, config: PipelineConfig) -> None:
        self.forced_runs.append(self.force)
        super().run(job, config)


@pytest.fixture
def job(tmp_path: Path) -> Job:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"")
    return Job.create(tmp_path / "jobs", video, job_id="test")


def fake_stages(log: list[StageName]) -> list[FakeStage]:
    return [FakeStage(name, log) for name in StageName]


def statuses(job: Job) -> dict[StageName, StageStatus]:
    return {name: record.status for name, record in job.read_meta().stages.items()}


def test_runs_stages_in_stage_name_order(job: Job) -> None:
    log: list[StageName] = []
    meta = run_pipeline(job, list(reversed(fake_stages(log))), CONFIG)
    assert log == list(StageName)
    assert all(record.status == StageStatus.DONE for record in meta.stages.values())


def test_rerun_skips_completed_stages(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    log.clear()
    meta = run_pipeline(job, fake_stages(log), CONFIG)
    assert log == []
    assert all(record.status == StageStatus.SKIPPED for record in meta.stages.values())


def test_stage_with_a_missing_output_runs(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    (job.root / "refine.out").unlink()
    log.clear()
    run_pipeline(job, fake_stages(log), CONFIG)
    assert log == [StageName.REFINE, StageName.CLIP]  # clip depends on the new clips.json


def test_force_reruns_every_stage(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    log.clear()
    run_pipeline(job, fake_stages(log), CONFIG, force=True)
    assert log == list(StageName)


def test_from_stage_reruns_that_stage_and_later_ones(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    log.clear()
    meta = run_pipeline(job, fake_stages(log), CONFIG, from_stage=StageName.LLM)
    assert log == [StageName.LLM, StageName.SCORE, StageName.REFINE, StageName.CLIP]
    assert meta.stages[StageName.INGEST].status == StageStatus.SKIPPED
    assert meta.stages[StageName.TRANSCRIBE].status == StageStatus.SKIPPED
    assert meta.stages[StageName.LLM].status == StageStatus.DONE


def test_from_stage_still_runs_earlier_stages_that_are_not_done(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG, from_stage=StageName.REFINE)
    assert log == list(StageName)


def test_forced_stage_with_force_field_runs_with_force(job: Job) -> None:
    log: list[StageName] = []
    clip = ForceableStage(StageName.CLIP, log)
    run_pipeline(job, [clip], CONFIG)
    run_pipeline(job, [clip], CONFIG, force=True)
    run_pipeline(job, [clip], CONFIG, from_stage=StageName.REFINE)
    run_pipeline(job, [clip], CONFIG)  # skipped: not run at all
    assert clip.forced_runs == [False, True, True]
    assert clip.force is False  # the caller's stage isn't mutated


def test_job_json_records_status_and_timings(job: Job) -> None:
    log: list[StageName] = []
    seen: list[StageStatus] = []

    @dataclass
    class Inspecting(FakeStage):
        def run(self, job: Job, config: PipelineConfig) -> None:
            seen.append(job.read_meta().stages[self.name].status)
            super().run(job, config)

    stage = Inspecting(StageName.INGEST, log)
    meta = run_pipeline(job, [stage], CONFIG)
    assert seen == [StageStatus.RUNNING]
    record = meta.stages[StageName.INGEST]
    assert record.status == StageStatus.DONE
    assert record.started_at is not None and record.finished_at is not None
    assert record.started_at <= record.finished_at
    assert record.started_at.utcoffset() is not None  # timezone-aware (UTC)
    assert record.error is None
    assert meta.stages[StageName.CLIP].status == StageStatus.PENDING  # not in the run
    assert job.read_meta() == meta

    meta = run_pipeline(job, [stage], CONFIG)
    skipped = meta.stages[StageName.INGEST]
    assert skipped.status == StageStatus.SKIPPED
    assert skipped.finished_at is not None


def test_keeps_what_stages_write_to_job_json(job: Job) -> None:
    @dataclass
    class RecordsModel(FakeStage):
        def run(self, job: Job, config: PipelineConfig) -> None:
            meta = job.read_meta()
            meta.models["whisper"] = "fake/whisper"
            job.write_meta(meta)
            super().run(job, config)

    meta = run_pipeline(job, [RecordsModel(StageName.TRANSCRIBE, [])], CONFIG)
    assert meta.models == {"whisper": "fake/whisper"}


def test_events(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log)[:2], CONFIG)
    events: list[StageEvent] = []
    stages = [*fake_stages(log)[:2], FakeStage(StageName.LLM, log, error=LLMError("down"))]
    with pytest.raises(PipelineError):
        run_pipeline(job, stages, CONFIG, from_stage=StageName.TRANSCRIBE, on_event=events.append)

    assert [(e.stage, e.status) for e in events] == [
        (StageName.INGEST, StageStatus.SKIPPED),
        (StageName.TRANSCRIBE, StageStatus.RUNNING),
        (StageName.TRANSCRIBE, StageStatus.DONE),
        (StageName.LLM, StageStatus.RUNNING),
        (StageName.LLM, StageStatus.FAILED),
    ]
    assert events[0].elapsed is None and events[1].elapsed is None
    assert events[2].elapsed is not None and events[2].elapsed >= 0
    assert events[2].error is None
    assert events[4].elapsed is not None
    assert events[4].error == "down"


def test_echo_error_fails_the_stage_and_stops_the_run(job: Job) -> None:
    log: list[StageName] = []
    stages = fake_stages(log)
    cause = LLMError("ollama is not running")
    stages[list(StageName).index(StageName.LLM)] = FakeStage(StageName.LLM, log, error=cause)

    with pytest.raises(PipelineError) as excinfo:
        run_pipeline(job, stages, CONFIG)

    assert excinfo.value.stage == StageName.LLM
    assert excinfo.value.cause is cause
    assert isinstance(excinfo.value, EchoError)
    assert log == [StageName.INGEST, StageName.TRANSCRIBE, StageName.DIARIZE, StageName.LLM]
    meta = job.read_meta()
    record = meta.stages[StageName.LLM]
    assert record.status == StageStatus.FAILED
    assert record.error == "ollama is not running"
    assert record.started_at is not None and record.finished_at is not None
    assert meta.stages[StageName.REFINE].status == StageStatus.PENDING
    assert meta.stages[StageName.CLIP].status == StageStatus.PENDING


def test_other_exceptions_propagate_but_are_recorded(job: Job) -> None:
    log: list[StageName] = []
    stages = [FakeStage(StageName.INGEST, log, error=ValueError("bug")), *fake_stages(log)[1:]]

    with pytest.raises(ValueError, match="bug"):
        run_pipeline(job, stages, CONFIG)

    assert log == [StageName.INGEST]
    record = job.read_meta().stages[StageName.INGEST]
    assert record.status == StageStatus.FAILED
    assert record.error == "ValueError: bug"
    assert statuses(job)[StageName.TRANSCRIBE] == StageStatus.PENDING


def test_ctrl_c_is_recorded_as_interrupted(job: Job) -> None:
    log: list[StageName] = []
    stages = [FakeStage(StageName.INGEST, log, error=KeyboardInterrupt()), *fake_stages(log)[1:]]

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(job, stages, CONFIG)

    record = job.read_meta().stages[StageName.INGEST]
    assert record.status == StageStatus.FAILED
    assert record.error == "interrupted"


def test_rerun_after_failure_resumes_at_failed_stage(job: Job) -> None:
    log: list[StageName] = []
    stages = fake_stages(log)
    stages[list(StageName).index(StageName.REFINE)] = FakeStage(
        StageName.REFINE, log, error=EchoError("bad clips")
    )
    with pytest.raises(PipelineError):
        run_pipeline(job, stages, CONFIG)

    log.clear()
    meta = run_pipeline(job, fake_stages(log), CONFIG)
    assert log == [StageName.REFINE, StageName.CLIP]
    assert statuses(job) == {
        StageName.INGEST: StageStatus.SKIPPED,
        StageName.TRANSCRIBE: StageStatus.SKIPPED,
        StageName.DIARIZE: StageStatus.SKIPPED,
        StageName.LLM: StageStatus.SKIPPED,
        StageName.SCORE: StageStatus.SKIPPED,
        StageName.REFINE: StageStatus.DONE,
        StageName.CLIP: StageStatus.DONE,
    }
    assert meta.stages[StageName.REFINE].error is None


# --- build_stages -----------------------------------------------------------------


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("heavy resource created while building the stages")


def test_build_stages_is_lazy(monkeypatch: pytest.MonkeyPatch, job: Job) -> None:
    monkeypatch.setattr(pipeline, "WhisperTranscriber", _forbid)
    monkeypatch.setattr(pipeline, "make_llm_client", _forbid)
    monkeypatch.setattr(pipeline, "LLMTopicExtractor", _forbid)

    stages = build_stages(PipelineConfig(llm={"provider": "groq"}))  # no API key needed yet

    # Diarization is optional and off by default, so it isn't built (#22); scoring (#21)
    # is on by default.
    assert [s.name for s in stages] == [n for n in StageName if n is not StageName.DIARIZE]
    assert StageName.SCORE not in [
        s.name for s in build_stages(PipelineConfig(score={"enabled": False}))
    ]
    assert [type(s) for s in stages] == [
        IngestStage,
        TranscribeStage,
        LLMStage,
        ScoreStage,
        RefineStage,
        ClipStage,
    ]
    for stage in stages:
        stage.outputs(job)


def test_build_stages_chunks_for_the_configured_model() -> None:
    """``chunk.max_tokens = "auto"`` follows ``llm`` (#63): the chunker sees the LLM config."""
    config = PipelineConfig(llm={"provider": "gemini"})
    llm = build_stages(config)[2]
    assert isinstance(llm, LLMStage)
    assert llm.chunker.llm == config.llm


def test_build_stages_creates_resources_when_run(
    monkeypatch: pytest.MonkeyPatch, job_copy: Job
) -> None:
    transcript = job_copy.read_transcript()
    client = FakeLLMClient(['{"topics": []}'] * 100)
    monkeypatch.setattr(pipeline, "WhisperTranscriber", lambda config: FakeTranscriber(transcript))
    monkeypatch.setattr(pipeline, "make_llm_client", lambda config: client)
    _, transcribe, llm, score, _, _ = build_stages(CONFIG)

    job_copy.audio_path.write_bytes(b"")
    transcribe.run(job_copy, CONFIG)
    llm.run(job_copy, CONFIG)
    score.run(job_copy, CONFIG)

    assert client.calls
    assert job_copy.read_topics() == []
    assert job_copy.read_scores() == []
    models = job_copy.read_meta().models
    assert models["whisper"] == "fake/whisper"
    assert models["llm"] == "fake/llm"
    assert models["llm_prompt"] == PROMPT_VERSION
    assert models["score_prompt"] == SCORE_PROMPT_VERSION


def test_a_stage_that_reruns_forces_later_stages(job: Job) -> None:
    """Regenerating an earlier artifact must not leave later ones built from the old one."""
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    (job.root / f"{StageName.LLM}.out").unlink()
    log.clear()
    run_pipeline(job, fake_stages(log), CONFIG)
    assert log == [StageName.LLM, StageName.SCORE, StageName.REFINE, StageName.CLIP]
    assert statuses(job)[StageName.TRANSCRIBE] == StageStatus.SKIPPED


def test_skipped_stage_keeps_the_times_it_was_produced(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)
    produced = job.read_meta().stages[StageName.INGEST]
    run_pipeline(job, fake_stages(log), CONFIG)
    run_pipeline(job, fake_stages(log), CONFIG)
    skipped = job.read_meta().stages[StageName.INGEST]
    assert skipped.status == StageStatus.SKIPPED
    assert (skipped.started_at, skipped.finished_at) == (produced.started_at, produced.finished_at)


# --- settings fingerprints + progress (#59, #61, #82) ----------------------------------


@dataclass
class SettingsStage(FakeStage):
    """Re-runs when ``value`` changes (like transcribe when the whisper model changes)."""

    value: str = "a"

    def settings(self, config: PipelineConfig) -> str:
        return self.value


def test_changed_settings_rerun_the_stage_and_later_ones(job: Job) -> None:
    log: list[StageName] = []
    stages = fake_stages(log)
    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="small")
    run_pipeline(job, stages, CONFIG)
    assert job.read_meta().stages[StageName.TRANSCRIBE].settings == "small"

    log.clear()
    run_pipeline(job, stages, CONFIG)
    assert log == [], "same settings: skipped"

    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="medium")
    log.clear()
    run_pipeline(job, stages, CONFIG)
    assert log == list(StageName)[1:]
    assert job.read_meta().stages[StageName.TRANSCRIBE].settings == "medium"


def test_old_job_without_settings_adopts_them_without_rerunning(job: Job) -> None:
    log: list[StageName] = []
    run_pipeline(job, fake_stages(log), CONFIG)  # recorded before the stage had settings
    stages = fake_stages(log)
    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="small")
    log.clear()
    run_pipeline(job, stages, CONFIG)
    assert log == []
    assert job.read_meta().stages[StageName.TRANSCRIBE].settings == "small"


def test_progress_reports_become_non_decreasing_running_events(job: Job) -> None:
    from echo import progress

    @dataclass
    class Reporting(FakeStage):
        def run(self, job: Job, config: PipelineConfig) -> None:
            for fraction, detail in [(0.2, "1/5"), (0.6, "3/5"), (0.4, "late"), (1.5, "5/5")]:
                progress.report(fraction, detail)
            super().run(job, config)

    events: list[StageEvent] = []
    stages = [Reporting(StageName.INGEST, [])]
    run_pipeline(job, stages, CONFIG, on_event=events.append)
    running = [e for e in events if e.status is StageStatus.RUNNING]
    assert running[0].progress is None
    assert [e.progress for e in running[1:]] == [0.2, 0.6, 0.6, 1.0]
    assert [e.detail for e in running[1:]] == ["1/5", "3/5", "late", "5/5"]
    assert events[-1].status is StageStatus.DONE


def test_progress_outside_a_run_is_a_no_op() -> None:
    from echo import progress

    progress.report(0.5, "nobody listening")  # must not raise


def test_failed_rerun_after_a_settings_change_is_retried(job: Job) -> None:
    """Change settings → the re-run fails → the next run must re-run, not adopt."""
    log: list[StageName] = []
    stages = fake_stages(log)
    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="small")
    run_pipeline(job, stages, CONFIG)

    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="medium", error=EchoError("GPU"))
    with pytest.raises(PipelineError):
        run_pipeline(job, stages, CONFIG)
    record = job.read_meta().stages[StageName.TRANSCRIBE]
    assert record.status is StageStatus.FAILED and record.settings == "small"

    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="medium")
    log.clear()
    run_pipeline(job, stages, CONFIG)
    assert log[0] == StageName.TRANSCRIBE, "the old transcript is not adopted"
    assert job.read_meta().stages[StageName.TRANSCRIBE].settings == "medium"


def test_old_job_that_last_failed_does_not_adopt_settings(job: Job) -> None:
    log: list[StageName] = []
    stages = fake_stages(log)
    run_pipeline(job, stages, CONFIG)  # outputs exist, recorded before settings existed
    stages[1] = FakeStage(StageName.TRANSCRIBE, log, error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):  # interrupted mid-transcribe: outputs suspect
        run_pipeline(job, stages, CONFIG, from_stage=StageName.TRANSCRIBE)
    stages[1] = SettingsStage(StageName.TRANSCRIBE, log, value="small")
    log.clear()
    run_pipeline(job, stages, CONFIG)
    assert log[0] == StageName.TRANSCRIBE
