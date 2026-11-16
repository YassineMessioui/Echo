"""``echo-clips`` CLI. ``run_pipeline`` and ``build_stages`` are faked, so no stage runs."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from echo import cli
from echo.config import PipelineConfig
from echo.errors import LLMError
from echo.job import Job
from echo.models import StageName, StageStatus
from echo.pipeline import PipelineError, StageEvent

GOLDEN_CLIPS = Path(__file__).parent / "fixtures" / "golden" / "clips.json"

# Wide, colourless help so Rich doesn't wrap or style option names.
runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


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
        return ["fake-stages"]

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
    assert fake.calls[0]["stages"] == ["fake-stages"]
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
        "--min-duration", "20",
        "--max-duration", "45",
        "--aspect", "source",
        "--crop-x", "0.25",
        "--background", "black",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    c = fake.config
    assert c.jobs_dir == jobs_dir
    assert (c.transcribe.model_size, c.transcribe.language) == ("medium", "fr")
    assert (c.llm.provider, c.llm.model) == ("groq", "openai/gpt-oss-120b")
    assert (c.refine.min_duration, c.refine.max_duration) == (20.0, 45.0)
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
        "[1/5] ingest      done      2.1s",
        "[2/5] transcribe  skipped",
        "[3/5] llm         failed    0.5s",
    ]


def test_failed_without_running_event(fake, video, jobs_dir):
    """A stage whose ``outputs()`` raises sends FAILED with no RUNNING before it."""
    fake.events = [StageEvent(StageName.REFINE, StageStatus.FAILED, error="ValueError: x")]
    result = run(video, jobs_dir)
    assert result.stdout.splitlines()[0] == "[4/5] refine      failed"


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
