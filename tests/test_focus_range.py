"""Focus range (#59): ``--range`` parsing, ingest of a span, the transcript offset, the
re-run on a changed range, and the acceptance test on the podcast fixture."""

from __future__ import annotations

import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from echo import cli
from echo.config import IngestConfig, PipelineConfig, RefineConfig
from echo.errors import InvalidInputError
from echo.ffmpeg import ffmpeg, probe
from echo.ingest import RANGE_KEY, IngestStage, extract_audio, resolve_range
from echo.job import Job
from echo.models import Segment, StageName, Topic, Transcript, Word
from echo.pipeline import run_pipeline
from echo.refine import refine_clips
from echo.testing import FakeTranscriber
from echo.transcribe import INGEST_RANGE_KEY, TranscribeStage, in_original_time

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


def _range(start: str | None = None, end: str | None = None) -> IngestConfig:
    return IngestConfig(start=start, end=end)


# --- resolving the range ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("15%", "85%", (15.0, 85.0)),
        ("12", "30.5", (12.0, 30.5)),
        (None, "85%", (0.0, 85.0)),
        ("10%", None, (10.0, 100.0)),
        ("0", "100%", (0.0, 100.0)),
        ("20", "50%", (20.0, 50.0)),
    ],
)
def test_resolve_range(start, end, expected):
    assert resolve_range(_range(start, end), 100.0) == expected


def test_no_range_is_the_whole_input():
    assert resolve_range(IngestConfig(), 100.0) is None


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        ("50", "50", "empty or inverted"),
        ("60", "40%", "empty or inverted"),
        ("100", None, "past the end"),
        ("100%", None, "past the end"),
        (None, "120", "past the end"),
    ],
)
def test_bad_ranges_are_clear_errors(start, end, match):
    with pytest.raises(InvalidInputError, match=match):
        resolve_range(_range(start, end), 100.0)


def test_key_is_shared_by_ingest_and_transcribe():
    assert RANGE_KEY == INGEST_RANGE_KEY


def test_settings_fingerprint_the_range():
    stage = IngestStage()
    assert stage.settings(PipelineConfig()) == "range=:"
    ranged = PipelineConfig(ingest=_range("15%", "85%"))
    assert stage.settings(ranged) == "range=15%:85%"


# --- CLI --range ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("15%:85%", ("15%", "85%")),
        ("120:3300", ("120", "3300")),
        (":85%", (None, "85%")),
        ("10%:", ("10%", None)),
        (" 90.5 : 600 ", ("90.5", "600")),
        ("30:50%", ("30", "50%")),
    ],
)
def test_parse_range(value, expected):
    assert cli.parse_range(value) == {"start": expected[0], "end": expected[1]}
    assert cli.parse_range(None) == {"start": None, "end": None}


@pytest.fixture
def captured(monkeypatch, tmp_path: Path) -> list[PipelineConfig]:
    """``run_pipeline`` faked: records the config the CLI built."""
    configs: list[PipelineConfig] = []

    def fake_run(job, stages, config, **kwargs):
        configs.append(config)
        return job.read_meta()

    monkeypatch.setattr(cli, "has_filter", lambda name: True)
    monkeypatch.setattr(cli, "build_stages", lambda config: [])
    monkeypatch.setattr(cli, "run_pipeline", fake_run)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ep.mp4").write_bytes(b"not really a video")
    return configs


def _run_cli(*args: str):
    return runner.invoke(cli.app, ["run", "ep.mp4", "--jobs-dir", "jobs", *args])


def test_cli_range_maps_to_ingest_config(captured):
    result = _run_cli("--range", "15%:85%")
    assert result.exit_code == 0, result.output
    assert (captured[0].ingest.start, captured[0].ingest.end) == ("15%", "85%")


def test_cli_without_range_processes_everything(captured):
    assert _run_cli().exit_code == 0
    assert (captured[0].ingest.start, captured[0].ingest.end) == (None, None)


@pytest.mark.parametrize(
    "value", ["abc", "15%", ":", "85%:15%", "300:120", "150%:", "1:2:3", "-5:10", "a:b"]
)
def test_cli_invalid_range_exits_2(captured, value):
    result = _run_cli(f"--range={value}")
    assert result.exit_code == 2, result.output
    assert "range" in result.stderr or "start" in result.stderr or "end" in result.stderr
    assert "Traceback" not in result.output
    assert not captured


def test_cli_help_lists_range():
    result = runner.invoke(cli.app, ["run", "--help"])
    assert "--range" in result.stdout


# --- ingest extracts the span (real ffmpeg) -----------------------------------------------


def _wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)


def _ramp_audio(path: Path, seconds: float = 10.0) -> Path:
    """Audio whose loudness grows linearly with time (amplitude t / seconds), so the level
    at the start of an extract tells where it was cut from."""
    ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc=t/{seconds}*sin(2*PI*440*t):s=16000:d={seconds}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(path),
        ]
    )
    return path


def _level(samples: np.ndarray) -> float:
    """Peak amplitude of a short stretch, as a fraction of full scale."""
    return float(np.abs(samples).max()) / 32768.0


@pytest.mark.ffmpeg
def test_extracts_only_the_span(make_video, tmp_path: Path):
    video = make_video(duration=6.0)
    out = tmp_path / "audio.wav"
    extract_audio(video, out, _range("25%", "4.5"))  # 1.5 s .. 4.5 s
    with wave.open(str(out), "rb") as w:
        assert w.getnframes() / w.getframerate() == pytest.approx(3.0, abs=0.05)


@pytest.mark.ffmpeg
def test_extract_cuts_at_the_right_place(tmp_path: Path):
    source = _ramp_audio(tmp_path / "ramp.m4a")
    out = tmp_path / "audio.wav"
    extract_audio(source, out, _range("40%", "8"))  # 4 s .. 8 s
    samples = _wav(out)
    rate = 16_000
    assert len(samples) / rate == pytest.approx(4.0, abs=0.05)
    assert _level(samples[rate // 10 : rate // 5]) == pytest.approx(0.415, abs=0.03)
    assert _level(samples[-rate // 5 : -rate // 10]) == pytest.approx(0.785, abs=0.03)


@pytest.mark.ffmpeg
def test_range_past_the_end_fails_before_encoding(make_video, tmp_path: Path):
    out = tmp_path / "audio.wav"
    with pytest.raises(InvalidInputError, match="past the end"):
        extract_audio(make_video(duration=2.0), out, _range("1", "10"))
    assert not out.exists()


@pytest.mark.ffmpeg
def test_stage_records_the_range_and_clears_it(make_video, tmp_path: Path):
    job = Job.create(tmp_path / "jobs", make_video(duration=4.0), job_id="r")
    IngestStage().run(job, PipelineConfig(ingest=_range("25%", "75%")))
    assert job.read_meta().models[RANGE_KEY] == "1.000:3.000"

    IngestStage().run(job, PipelineConfig())
    assert RANGE_KEY not in job.read_meta().models


@pytest.mark.ffmpeg
def test_link_job_ranges_the_downloaded_audio(job_copy: Job, tmp_path: Path):
    shutil.copyfile(_ramp_audio(tmp_path / "ramp.m4a"), job_copy.input_audio)
    assert not job_copy.input_video.exists()
    IngestStage().run(job_copy, PipelineConfig(ingest=_range("50%", None)))
    assert (
        job_copy.read_meta().models[RANGE_KEY]
        == f"5.000:{probe(job_copy.input_audio).duration:.3f}"
    )
    assert len(_wav(job_copy.audio_path)) / 16_000 == pytest.approx(5.0, abs=0.1)


# --- transcribe shifts into original time ---------------------------------------------


def _relative_transcript() -> Transcript:
    """What whisper returns for a 30 s ``audio.wav`` (times from 0)."""
    return Transcript(
        language="en",
        duration=30.0,
        segments=[
            Segment(
                id=0,
                start=0.5,
                end=4.0,
                text="Hello there.",
                words=[
                    Word(start=0.5, end=1.0, text="Hello"),
                    Word(start=1.1, end=4.0, text="there."),
                ],
            ),
            Segment(id=1, start=4.5, end=29.0, text="Second.", words=[]),
        ],
    )


def test_transcribe_stage_shifts_by_the_recorded_offset(job_copy: Job):
    job_copy.transcript_path.unlink()
    job_copy.audio_path.write_bytes(b"wav")  # the fake transcriber doesn't read it
    meta = job_copy.read_meta()
    meta.models[INGEST_RANGE_KEY] = "100.000:130.000"
    job_copy.write_meta(meta)

    TranscribeStage(FakeTranscriber(_relative_transcript())).run(job_copy, PipelineConfig())

    t = job_copy.read_transcript()
    assert t.offset == 100.0
    assert t.duration == 130.0
    assert [(s.start, s.end) for s in t.segments] == [(100.5, 104.0), (104.5, 129.0)]
    assert [(w.start, w.end) for w in t.segments[0].words] == [(100.5, 101.0), (101.1, 104.0)]
    assert [s.text for s in t.segments] == ["Hello there.", "Second."]


def test_no_recorded_range_leaves_the_transcript_alone(job_copy: Job):
    job_copy.transcript_path.unlink()
    job_copy.audio_path.write_bytes(b"wav")
    transcript = _relative_transcript()
    TranscribeStage(FakeTranscriber(transcript)).run(job_copy, PipelineConfig())
    assert job_copy.read_transcript() == transcript
    assert in_original_time(transcript, 0.0) is transcript


# --- refine pads within [offset, duration] ---------------------------------------------


def test_refine_never_pads_before_the_offset():
    transcript = in_original_time(_relative_transcript(), 100.0)
    topics = [Topic(title="All", summary="s", start_segment=0, end_segment=1)]
    config = RefineConfig(padding=2.0, min_duration=5.0, max_duration=60.0)
    (clip,) = refine_clips(transcript, topics, config)  # speech spans 100.5 .. 129 s
    assert clip.start == 100.0, "padded only back to the start of the transcribed span"
    assert clip.end == 130.0, "and only up to its end"


# --- changing the range re-runs from ingest -------------------------------------------


@dataclass
class _Later:
    """A stage after ingest that just records that it ran."""

    name: StageName
    log: list[StageName]

    def outputs(self, job: Job) -> list[Path]:
        return [job.root / f"{self.name}.out"]

    def run(self, job: Job, config: PipelineConfig) -> None:
        self.log.append(self.name)
        self.outputs(job)[0].write_text("ok")


@pytest.mark.ffmpeg
def test_changing_the_range_reruns_ingest_and_later_stages(make_video, tmp_path: Path):
    job = Job.create(tmp_path / "jobs", make_video(duration=4.0), job_id="rerun")
    log: list[StageName] = []

    @dataclass
    class LoggedIngest(IngestStage):
        def run(self, job: Job, config: PipelineConfig) -> None:
            log.append(self.name)
            super().run(job, config)

    stages = [LoggedIngest(), *(_Later(name, log) for name in list(StageName)[1:])]

    run_pipeline(job, stages, PipelineConfig())
    assert log == list(StageName)

    log.clear()
    run_pipeline(job, stages, PipelineConfig())
    assert log == [], "same (no) range: everything skipped"

    log.clear()
    ranged = PipelineConfig(ingest=_range("25%", "75%"))
    run_pipeline(job, stages, ranged)
    assert log == list(StageName)
    meta = job.read_meta()
    assert meta.stages[StageName.INGEST].settings == "range=25%:75%"
    assert meta.models[RANGE_KEY] == "1.000:3.000"

    log.clear()
    run_pipeline(job, stages, ranged)
    assert log == []

    log.clear()
    run_pipeline(job, stages, PipelineConfig())  # back to the whole video
    assert log == list(StageName)
    assert RANGE_KEY not in job.read_meta().models
