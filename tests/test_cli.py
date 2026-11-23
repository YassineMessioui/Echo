"""``echo-clips`` CLI. ``run_pipeline`` and ``build_stages`` are faked, so no stage runs."""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from echo import cli
from echo.config import PipelineConfig
from echo.errors import InvalidInputError, LLMError
from echo.job import Job
from echo.models import StageName, StageStatus
from echo.pipeline import PipelineError, StageEvent

GOLDEN_CLIPS = Path(__file__).parent / "fixtures" / "golden" / "clips.json"

# Wide, colourless help so Rich doesn't wrap or style option names.
runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


FAKE_STAGES = [SimpleNamespace(name=name) for name in StageName if name is not StageName.DIARIZE]
"""What the fake ``build_stages`` returns: only ``name`` is read by the CLI."""


@dataclass
class FakePipeline:
    """Stands in for ``build_stages`` + ``run_pipeline`` and records what the CLI passed."""

    events: list[StageEvent] = field(default_factory=list)
    error: Exception | None = None
    write_clips: bool = False
    stages_config: PipelineConfig | None = None
    calls: list[dict] = field(default_factory=list)

    def build_stages(self, config):
        self.stages_config = config
        return FAKE_STAGES

    def run_pipeline(self, job, stages, config, *, force=False, from_stage=None, on_event=None):
        self.calls.append(
            {"job": job, "stages": stages, "config": config, "force": force, "from": from_stage}
        )
        for event in self.events:
            on_event(event)
        if self.error is not None:
            raise self.error
        if self.write_clips:
            shutil.copy(GOLDEN_CLIPS, job.clips_path)
        return job.read_meta()

    @property
    def config(self) -> PipelineConfig:
        return self.calls[-1]["config"]


@pytest.fixture(autouse=True)
def _libass_available(monkeypatch):
    """CLI tests don't depend on the local ffmpeg build (e.g. Homebrew without libass);
    the missing-libass path is tested explicitly below."""
    monkeypatch.setattr(cli, "has_filter", lambda name: True)


@pytest.fixture
def fake(monkeypatch) -> FakePipeline:
    fake = FakePipeline()
    monkeypatch.setattr(cli, "build_stages", fake.build_stages)
    monkeypatch.setattr(cli, "run_pipeline", fake.run_pipeline)
    return fake


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"not really a video")
    return path


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)  # nothing leaks into the repo even if --jobs-dir is missed
    return tmp_path / "jobs"


def invoke(*args: str):
    return runner.invoke(cli.app, [str(a) for a in args])


def run(video: Path, jobs_dir: Path, *args: str):
    return invoke("run", video, "--jobs-dir", jobs_dir, *args)


# --- help / version ---------------------------------------------------------------


def test_help_lists_every_option():
    result = invoke("run", "--help")
    assert result.exit_code == 0
    for option in [
        "--job-id",
        "--jobs-dir",
        "--force",
        "--from-stage",
        "--whisper-model",
        "--language",
        "--llm-provider",
        "--llm-model",
        "--min-duration",
        "--max-duration",
        "--max-clips",
        "--aspect",
        "--crop-x",
        "--background",
        "--captions",
        "--no-captions",
        "--verbose",
        "-v",
    ]:
        assert option in result.stdout, option


def test_no_captions(fake, video, jobs_dir):
    result = run(video, jobs_dir, "--no-captions")
    assert result.exit_code == 0, result.output
    assert fake.config.caption.enabled is False
    assert run(video, jobs_dir).exit_code == 0
    assert fake.config.caption.enabled is True


def test_ctrl_c_exits_130(fake, video, jobs_dir):
    fake.error = KeyboardInterrupt()
    result = run(video, jobs_dir)
    assert result.exit_code == 130
    assert "interrupted" in result.output


def test_version():
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.startswith("echo-clips ")


# --- config mapping ---------------------------------------------------------------


def test_defaults_come_from_pipeline_config(fake, video, jobs_dir):
    result = run(video, jobs_dir)
    assert result.exit_code == 0, result.output
    assert fake.config == PipelineConfig(jobs_dir=jobs_dir)
    assert fake.stages_config is fake.config
    assert fake.calls[0]["stages"] is FAKE_STAGES
    assert fake.calls[0]["force"] is False
    assert fake.calls[0]["from"] is None


def test_options_map_to_config_fields(fake, video, jobs_dir):
    result = run(
        video,
        jobs_dir,
        "--whisper-model", "medium",
        "--language", "fr",
        "--llm-provider", "groq",
        "--llm-model", "openai/gpt-oss-120b",
        "--llm-concurrency", "2",
        "--min-duration", "20",
        "--max-duration", "45",
        "--max-clips", "3",
        "--aspect", "source",
        "--crop-x", "0.25",
        "--background", "black",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    c = fake.config
    assert c.jobs_dir == jobs_dir
    assert (c.transcribe.model_size, c.transcribe.language) == ("medium", "fr")
    assert (c.llm.provider, c.llm.model) == ("groq", "openai/gpt-oss-120b")
    assert c.llm.max_concurrency == 2
    assert (c.refine.min_duration, c.refine.max_duration) == (20.0, 45.0)
    assert c.refine.max_clips == 3
    assert (c.clip.aspect, c.clip.crop_x, c.clip.background) == ("source", 0.25, "black")
    # untouched fields keep their defaults
    defaults = PipelineConfig()
    assert c.llm.host == defaults.llm.host
    assert c.refine.padding == defaults.refine.padding
    assert c.clip.crf == defaults.clip.crf


def test_force_and_from_stage_are_passed_through(fake, video, jobs_dir):
    result = run(video, jobs_dir, "--force", "--from-stage", "refine")
    assert result.exit_code == 0, result.output
    assert fake.calls[0]["force"] is True
    assert fake.calls[0]["from"] is StageName.REFINE


# --- usage errors (exit 2) --------------------------------------------------------


def test_invalid_from_stage_is_a_usage_error(fake, video, jobs_dir):
    result = run(video, jobs_dir, "--from-stage", "render")
    assert result.exit_code == 2
    assert not fake.calls


def test_min_greater_than_max_is_a_usage_error(fake, video, jobs_dir):
    result = run(video, jobs_dir, "--min-duration", "50", "--max-duration", "40")
    assert result.exit_code == 2
    assert "max_duration < min_duration" in result.stderr
    assert "Traceback" not in result.output
    assert not fake.calls
    assert not jobs_dir.exists()


def test_out_of_range_crop_x_is_a_usage_error(fake, video, jobs_dir):
    result = run(video, jobs_dir, "--crop-x", "1.5")
    assert result.exit_code == 2
    assert "crop_x" in result.stderr
    assert not fake.calls


# --- failures (exit 1) ------------------------------------------------------------


def test_pipeline_error_exits_1_without_traceback(fake, video, jobs_dir):
    fake.error = PipelineError(StageName.LLM, LLMError("ollama is not running"))
    result = run(video, jobs_dir)
    assert result.exit_code == 1
    assert "stage 'llm' failed: ollama is not running" in result.stderr
    assert "Traceback" not in result.output


def test_verbose_shows_traceback(fake, video, jobs_dir):
    fake.error = PipelineError(StageName.LLM, LLMError("ollama is not running"))
    result = run(video, jobs_dir, "--verbose")
    assert result.exit_code == 1
    assert "Traceback" in result.stderr
    assert "stage 'llm' failed: ollama is not running" in result.stderr


def test_missing_video_is_a_clean_error(fake, tmp_path, jobs_dir):
    result = run(tmp_path / "nope.mp4", jobs_dir)
    assert result.exit_code == 1
    assert "input video not found" in result.stderr
    assert "Traceback" not in result.output
    assert not fake.calls


# --- output -----------------------------------------------------------------------


def test_progress_lines(fake, video, jobs_dir):
    # As #12 emits them: RUNNING → DONE/FAILED; a skipped stage sends only SKIPPED.
    fake.events = [
        StageEvent(StageName.INGEST, StageStatus.RUNNING),
        StageEvent(StageName.INGEST, StageStatus.DONE, elapsed=2.14),
        StageEvent(StageName.TRANSCRIBE, StageStatus.SKIPPED),
        StageEvent(StageName.LLM, StageStatus.RUNNING),
        StageEvent(StageName.LLM, StageStatus.FAILED, elapsed=0.5, error="boom"),
    ]
    result = run(video, jobs_dir)
    lines = result.stdout.splitlines()
    assert lines[:3] == [
        "[1/6] ingest      done      2.1s",
        "[2/6] transcribe  skipped",
        "[3/6] llm         failed    0.5s",
    ]


def test_failed_without_running_event(fake, video, jobs_dir):
    """A stage whose ``outputs()`` raises sends FAILED with no RUNNING before it."""
    fake.events = [StageEvent(StageName.REFINE, StageStatus.FAILED, error="ValueError: x")]
    result = run(video, jobs_dir)
    assert result.stdout.splitlines()[0] == "[5/6] refine      failed"


def test_lists_job_dir_and_clip_paths(fake, video, jobs_dir):
    fake.write_clips = True
    result = run(video, jobs_dir, "--job-id", "ep42")
    assert result.exit_code == 0, result.output
    job = Job(jobs_dir / "ep42")
    assert str(job.root) in result.stdout
    assert str(job.clip_video_path(1, "Solar power is now the cheapest electricity")) in (
        result.stdout
    )
    assert str(job.clip_video_path(2, "Sleep and productivity")) in result.stdout


def _with_golden_scores(fake: FakePipeline) -> None:
    """The fake run also writes the golden topics.json and scores.json (#21)."""
    run_pipeline = fake.run_pipeline

    def run_and_score(job, *args, **kwargs):
        meta = run_pipeline(job, *args, **kwargs)
        for name in ("topics.json", "scores.json"):
            shutil.copy(GOLDEN_CLIPS.with_name(name), job.root / name)
        return meta

    fake.run_pipeline = run_and_score  # the test patches cli.run_pipeline with it


def test_lists_clips_best_first_with_scores(fake, video, jobs_dir, monkeypatch):
    fake.write_clips = True
    _with_golden_scores(fake)
    monkeypatch.setattr(cli, "run_pipeline", fake.run_pipeline)
    result = run(video, jobs_dir, "--job-id", "ep42")
    assert result.exit_code == 0, result.output
    job = Job(jobs_dir / "ep42")
    lines = result.stdout.splitlines()
    start = lines.index("clips (2, best first):")
    sleep, solar = (
        job.clip_video_path(2, "Sleep and productivity"),
        job.clip_video_path(1, "Solar power is now the cheapest electricity"),
    )
    assert lines[start + 1 : start + 5] == [
        f"   7.6  {sleep}",
        "        Opens on a relatable claim and lands a concrete takeaway.",
        f"   6.2  {solar}",
        "        Clear, surprising fact, but a flat delivery.",
    ]


def test_no_score_lists_clips_in_time_order(fake, video, jobs_dir, monkeypatch):
    fake.write_clips = True
    _with_golden_scores(fake)
    monkeypatch.setattr(cli, "run_pipeline", fake.run_pipeline)
    result = run(video, jobs_dir, "--job-id", "ep42", "--no-score")
    assert result.exit_code == 0, result.output
    assert fake.config.score.enabled is False
    assert "clips (2):" in result.stdout and "best first" not in result.stdout


def test_score_is_on_by_default(fake, video, jobs_dir):
    assert run(video, jobs_dir).exit_code == 0
    assert fake.config.score.enabled is True
    assert fake.stages_config.score.enabled is True


def test_no_clips_json(fake, video, jobs_dir):
    result = run(video, jobs_dir)
    assert result.exit_code == 0, result.output
    assert "no clips" in result.stdout


def test_job_id_reuse_resumes_same_job(fake, video, jobs_dir):
    assert run(video, jobs_dir, "--job-id", "ep42").exit_code == 0
    first = fake.calls[-1]["job"]
    created_at = first.read_meta().created_at
    assert run(video, jobs_dir, "--job-id", "ep42", "--from-stage", "clip").exit_code == 0
    second = fake.calls[-1]["job"]
    assert second.root == first.root == jobs_dir / "ep42"
    assert second.read_meta().created_at == created_at
    assert [p.name for p in jobs_dir.iterdir()] == ["ep42"]


def test_job_id_for_another_video_is_rejected(fake, video, tmp_path, jobs_dir):
    assert run(video, jobs_dir, "--job-id", "ep42").exit_code == 0
    other = tmp_path / "other.mp4"
    other.write_bytes(b"x")
    result = run(other, jobs_dir, "--job-id", "ep42")
    assert result.exit_code == 1
    assert "was created for" in result.stderr


def test_random_job_id(fake, video, jobs_dir):
    assert run(video, jobs_dir).exit_code == 0
    (job_dir,) = jobs_dir.iterdir()
    assert len(job_dir.name) == 12
    int(job_dir.name, 16)


def test_dotenv_is_loaded_without_overriding_the_shell(fake, video, jobs_dir, monkeypatch):
    import os

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    (jobs_dir.parent / ".env").write_text("MISTRAL_API_KEY=from-dotenv\nGROQ_API_KEY=from-dotenv\n")
    seen = {}
    real = fake.run_pipeline

    def capture(*args, **kwargs):
        seen.update(mistral=os.environ.get("MISTRAL_API_KEY"), groq=os.environ.get("GROQ_API_KEY"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_pipeline", capture)
    try:
        assert run(video, jobs_dir).exit_code == 0
    finally:
        os.environ.pop("MISTRAL_API_KEY", None)  # load_dotenv sets it outside monkeypatch
    assert seen == {"mistral": "from-dotenv", "groq": "from-shell"}


def test_url_is_downloaded_into_a_job(fake, jobs_dir, monkeypatch, tmp_path):
    seen = {}

    def fake_job_from_url(jobs_dir_, url, job_id=None, *, on_progress=None):
        seen.update(jobs_dir=jobs_dir_, url=url, job_id=job_id)
        on_progress(0.5)
        on_progress(1.0)
        video = tmp_path / "downloaded.mp4"
        video.write_bytes(b"x")
        return Job.create(jobs_dir_, video, job_id="youtube-abc")

    monkeypatch.setattr(cli, "job_from_url", fake_job_from_url)
    result = invoke("run", "https://youtu.be/abc", "--jobs-dir", jobs_dir)
    assert result.exit_code == 0, result.output
    assert seen == {"jobs_dir": jobs_dir, "url": "https://youtu.be/abc", "job_id": None}
    assert "[download]  40%" in result.output and "[download] 100%" in result.output
    assert fake.calls[0]["job"].job_id == "youtube-abc"


def test_url_download_error_exits_1(fake, jobs_dir, monkeypatch):
    def failing(*args, **kwargs):
        raise InvalidInputError("could not download https://youtu.be/x: Private video")

    monkeypatch.setattr(cli, "job_from_url", failing)
    result = invoke("run", "https://youtu.be/x", "--jobs-dir", jobs_dir)
    assert result.exit_code == 1
    assert "Private video" in result.output
    assert fake.calls == []


def _clear_quiet_env(monkeypatch):
    # setenv first so monkeypatch restores (removes) them afterwards even if unset now.
    for name in cli.QUIET_ENV:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_quiet_model_libraries_by_default(fake, video, jobs_dir, monkeypatch):
    _clear_quiet_env(monkeypatch)
    monkeypatch.setenv("CT2_VERBOSE", "2")  # the user's own setting wins
    assert run(video, jobs_dir).exit_code == 0
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["CT2_VERBOSE"] == "2"


def test_verbose_leaves_model_libraries_alone(fake, video, jobs_dir, monkeypatch):
    _clear_quiet_env(monkeypatch)
    assert run(video, jobs_dir, "--verbose").exit_code == 0
    assert not any(name in os.environ for name in cli.QUIET_ENV)


def test_missing_libass_fails_before_running_any_stage(fake, video, jobs_dir, monkeypatch):
    monkeypatch.setattr(cli, "has_filter", lambda name: False)
    result = run(video, jobs_dir)
    assert result.exit_code == 1
    assert "--no-captions" in result.stderr
    assert fake.calls == [], "the pipeline never started"
    assert run(video, jobs_dir, "--no-captions").exit_code == 0


def test_libass_check_skipped_when_ffmpeg_is_missing(fake, video, jobs_dir, monkeypatch):
    def no_ffmpeg(name):
        raise cli.ExternalToolError("ffmpeg not found on PATH")

    monkeypatch.setattr(cli, "has_filter", no_ffmpeg)
    assert run(video, jobs_dir).exit_code == 0  # left to the ingest stage to report


def test_max_clips_defaults_to_all_and_rejects_zero(fake, video, jobs_dir):
    assert run(video, jobs_dir).exit_code == 0
    assert fake.config.refine.max_clips is None
    assert run(video, jobs_dir, "--max-clips", "0").exit_code == 2


def _running(stage: StageName, fraction: float | None, detail: str = "") -> StageEvent:
    return StageEvent(stage, StageStatus.RUNNING, progress=fraction, detail=detail)


def test_piped_progress_is_plain_lines_every_10_percent(fake, video, jobs_dir):
    fake.events = [
        _running(StageName.TRANSCRIBE, None),
        *(_running(StageName.TRANSCRIBE, i / 100, f"{i}s / 100s") for i in range(0, 101, 4)),
        StageEvent(StageName.TRANSCRIBE, StageStatus.DONE, elapsed=3.0),
    ]
    result = run(video, jobs_dir)
    assert "\r" not in result.stdout and "\x1b" not in result.stdout
    lines = result.stdout.splitlines()
    progress = [line for line in lines if "%" in line]
    # 4% steps pass 10%, 20%, ... 100%: one line each.
    assert [line.split()[2] for line in progress] == [
        f"{p}%" for p in (12, 20, 32, 40, 52, 60, 72, 80, 92, 100)
    ]
    assert progress[0].startswith("[2/6] transcribe   12%  12s / 100s  ")
    assert lines[len(progress)] == "[2/6] transcribe  done      3.0s"


def test_tty_line_redraws_in_place():
    event = _running(StageName.TRANSCRIBE, 0.215, "12:30 / 58:10")
    line = cli._tty_line(event, 5, 31.04, width=80)
    assert line == "\r[2/5] transcribe   22%  12:30 / 58:10  31.0s\x1b[K"
    assert cli._tty_line(event, 5, 31.04, width=12) == "\r[2/5] trans\x1b[K"


def test_tty_output_is_throttled_and_cleared_before_the_final_line(capsys):
    now = [0.0]
    out = cli._StageOutput(tty=True, clock=lambda: now[0], width=lambda: 80)
    out(_running(StageName.LLM, None))
    for i in range(1, 11):  # 10 reports in 0.6 s → redrawn every 0.1 s at most
        now[0] = i * 0.06
        out(_running(StageName.LLM, i / 10, f"window {i}/10"))
    out(StageEvent(StageName.LLM, StageStatus.DONE, elapsed=0.5))
    written = capsys.readouterr().out
    assert written.count("\r[3/6] llm") == 5
    assert written.endswith("\r\x1b[K[3/6] llm         done      0.5s\n")


def test_speakers_flags(fake, video, jobs_dir):
    assert run(video, jobs_dir).exit_code == 0
    assert fake.config.diarize.enabled is False
    result = run(video, jobs_dir, "--speakers", "--num-speakers", "2")
    assert result.exit_code == 0, result.output
    assert fake.config.diarize.enabled is True and fake.config.diarize.num_speakers == 2
    assert run(video, jobs_dir, "--num-speakers", "0").exit_code == 2


def test_stage_numbering_follows_the_built_stages(fake, video, jobs_dir, monkeypatch):
    """With --speakers, diarize is stage 3 of 6; without it, there are 5 stages."""
    from types import SimpleNamespace

    names = [StageName.INGEST, StageName.TRANSCRIBE, StageName.DIARIZE, StageName.LLM]
    monkeypatch.setattr(
        fake, "build_stages", lambda config: [SimpleNamespace(name=n) for n in names]
    )
    monkeypatch.setattr(cli, "build_stages", fake.build_stages)
    fake.events = [StageEvent(StageName.DIARIZE, StageStatus.DONE, elapsed=1.0)]
    result = run(video, jobs_dir, "--speakers")
    assert "[3/4] diarize     done      1.0s" in result.stdout
