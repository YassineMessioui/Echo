import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from echo import diarize as diarize_module
from echo.config import DiarizeConfig, PipelineConfig
from echo.diarize import DiarizeStage, SherpaDiarizer, ensure_models, relabel
from echo.errors import ArtifactError
from echo.job import Job
from echo.models import SpeakerTurn, StageName, Transcript
from echo.pipeline import build_stages, run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "podcast"


@dataclass
class FakeStage:
    """Writes ``<name>.out`` into the job and logs that it ran."""

    name: StageName
    log: list[StageName]

    def outputs(self, job: Job) -> list[Path]:
        return [job.root / f"{self.name}.out"]

    def run(self, job: Job, config: PipelineConfig) -> None:
        self.log.append(self.name)
        self.outputs(job)[0].write_text("ok")


@dataclass
class FakeDiarizer:
    turns: list[SpeakerTurn]
    model_name: str = "fake/diarizer"
    calls: list[Path] = field(default_factory=list)

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        self.calls.append(audio)
        return self.turns


def _turn(start: float, end: float, speaker: str) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


def test_relabel_orders_speakers_by_first_appearance_and_shifts_time():
    turns = [_turn(10, 20, "7"), _turn(0, 9, "3"), _turn(21, 30, "3")]
    speakers = relabel(turns, offset=100.0)
    assert [(t.start, t.end, t.speaker) for t in speakers.turns] == [
        (100.0, 109.0, "S1"),
        (110.0, 120.0, "S2"),
        (121.0, 130.0, "S1"),
    ]


def _job(job_copy: Job, offset: float = 0.0) -> Job:
    job_copy.audio_path.write_bytes(b"wav")
    transcript = job_copy.read_transcript()
    job_copy.write_transcript(
        Transcript.model_validate(transcript.model_dump() | {"offset": offset})
    )
    return job_copy


def test_stage_writes_speakers_in_original_time(job_copy: Job):
    job = _job(job_copy, offset=0.0)
    fake = FakeDiarizer([_turn(0, 30, "1"), _turn(30, 60, "0")])
    DiarizeStage(fake).run(job, PipelineConfig())
    speakers = job.read_speakers()
    assert speakers.labels == ["S1", "S2"]
    assert fake.calls == [job.audio_path]
    assert job.read_meta().models["diarizer"] == "fake/diarizer"


def test_stage_requires_audio(tmp_path: Path):
    job = Job(tmp_path / "j")
    job.root.mkdir()
    with pytest.raises(ArtifactError, match="audio.wav"):
        DiarizeStage(FakeDiarizer([])).run(job, PipelineConfig())


def test_settings_fingerprint():
    stage = DiarizeStage(FakeDiarizer([]))
    assert stage.settings(PipelineConfig()) == "fake/diarizer|speakers=auto|t=0.5"
    config = PipelineConfig(diarize=DiarizeConfig(enabled=True, num_speakers=2))
    assert stage.settings(config) == "fake/diarizer|speakers=2|t=0.5"


def test_build_stages_only_includes_diarize_when_enabled(monkeypatch):
    monkeypatch.setattr(diarize_module, "ensure_models", lambda *a: pytest.fail("loaded early"))
    assert StageName.DIARIZE not in [s.name for s in build_stages(PipelineConfig())]
    on = build_stages(PipelineConfig(diarize=DiarizeConfig(enabled=True)))
    names = [s.name for s in on]
    assert names.index(StageName.DIARIZE) == names.index(StageName.TRANSCRIBE) + 1


def test_enabling_diarization_on_an_existing_job_runs_it_and_later_stages(job_copy: Job):
    """Turning ``--speakers`` on later diarizes and re-runs llm/refine/clip (recolour)."""
    job = _job(job_copy)
    log: list[StageName] = []
    stages = [FakeStage(n, log) for n in StageName if n is not StageName.DIARIZE]
    run_pipeline(job, stages, PipelineConfig())
    log.clear()
    with_diarize = [*stages[:2], DiarizeStage(FakeDiarizer([_turn(0, 5, "a")])), *stages[2:]]
    run_pipeline(job, with_diarize, PipelineConfig())
    assert log == [StageName.LLM, StageName.REFINE, StageName.CLIP]
    assert job.read_speakers().labels == ["S1"]


def test_ensure_models_extracts_and_caches(tmp_path: Path, monkeypatch):
    """Downloads once (atomically) and extracts the segmentation archive safely."""
    src = tmp_path / "src"
    (src / "sherpa-onnx-pyannote-segmentation-3-0").mkdir(parents=True)
    (src / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx").write_bytes(b"seg")
    archive = tmp_path / "seg.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        tar.add(
            src / "sherpa-onnx-pyannote-segmentation-3-0", "sherpa-onnx-pyannote-segmentation-3-0"
        )
    downloads: list[str] = []

    def fake_download(url: str, out: Path) -> None:
        downloads.append(url)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(archive.read_bytes() if url.endswith(".tar.bz2") else b"emb")

    monkeypatch.setattr(diarize_module, "_download", fake_download)
    models = tmp_path / "models"
    segmentation, embedding = ensure_models(models)
    assert segmentation.read_bytes() == b"seg" and embedding.read_bytes() == b"emb"
    assert sorted(p.name for p in models.iterdir()) == [
        "nemo_en_titanet_small.onnx",
        "sherpa-onnx-pyannote-segmentation-3-0",
    ]
    ensure_models(models)
    assert len(downloads) == 2, "cached: no second download"


def test_models_dir_can_be_overridden(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ECHO_MODELS_DIR", str(tmp_path))
    assert diarize_module.models_dir() == tmp_path


@pytest.mark.model
@pytest.mark.ffmpeg
def test_real_diarization_finds_the_two_fixture_voices(tmp_path: Path):
    """The fixture is Samantha / Daniel / Samantha (see tests/fixtures/podcast/script.txt)."""
    from echo.config import IngestConfig
    from echo.ingest import extract_audio

    audio = tmp_path / "audio.wav"
    extract_audio(FIXTURE / "podcast.mp4", audio, IngestConfig())
    turns = relabel(SherpaDiarizer(DiarizeConfig(enabled=True)).diarize(audio))
    sections = json.loads((FIXTURE / "sections.json").read_text())["sections"]
    assert turns.labels == ["S1", "S2"]
    for section, voice in zip(sections, ["S1", "S2", "S1"], strict=True):
        mid = (section["start"] + section["end"]) / 2
        assert turns.speaker_at(mid - 1, mid + 1) == voice, section["title"]
