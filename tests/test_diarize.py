import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from echo import diarize as diarize_module
from echo.config import DiarizeConfig, PipelineConfig
from echo.diarize import (
    DiarizeStage,
    SherpaDiarizer,
    ensure_models,
    fold_minor_speakers,
    merge_similar_speakers,
    relabel,
    speech_per_speaker,
)
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


def _spans(turns: list[SpeakerTurn]) -> list[tuple[float, float, str]]:
    return [(t.start, t.end, t.speaker) for t in turns]


# --- #119: tiny speakers folded into the nearest main speaker -----------------------------


def test_speech_per_speaker_in_order_of_first_appearance():
    turns = [_turn(10, 14, "b"), _turn(0, 5, "a"), _turn(20, 21, "b")]
    assert speech_per_speaker(turns) == {"a": 5.0, "b": 5.0}


def test_fold_gives_tiny_speakers_to_the_nearest_main_speaker_in_time():
    turns = [
        _turn(0, 40, "a"),
        _turn(40.5, 41, "x"),  # 0.5 s after a's turn, 9 s before b's: goes to a
        _turn(50, 90, "b"),
        _turn(89, 90.5, "y"),  # overlaps b's turn: goes to b, and is joined to it
        _turn(100, 140, "a"),
    ]
    folded = fold_minor_speakers(turns, min_share=0.05)
    assert _spans(folded) == [(0, 40, "a"), (40.5, 41, "a"), (50, 90.5, "b"), (100, 140, "a")]
    assert _spans(relabel(folded).turns) == [
        (0, 40, "S1"),
        (40.5, 41, "S1"),
        (50, 90.5, "S2"),
        (100, 140, "S1"),
    ]


def test_fold_threshold_is_a_share_of_all_speech():
    # 100 s of speech: "c" has 3 % of it, kept at 2 %, folded at 5 %
    turns = [_turn(0, 60, "a"), _turn(60, 97, "b"), _turn(97, 100, "c")]
    assert speech_per_speaker(fold_minor_speakers(turns, 0.02)) == {"a": 60, "b": 37, "c": 3}
    assert speech_per_speaker(fold_minor_speakers(turns, 0.05)) == {"a": 60, "b": 40}


def test_fold_tie_goes_to_the_earlier_turn():
    turns = [_turn(0, 30, "a"), _turn(31, 32, "x"), _turn(33, 63, "b")]
    assert _spans(fold_minor_speakers(turns, 0.1))[1] == (31, 32, "a")


def test_fold_keeps_the_biggest_speaker_and_first_appearance_order():
    # The first voice is tiny; labels still follow first appearance of what remains
    turns = [_turn(0, 1, "x"), _turn(5, 50, "b"), _turn(50, 52, "y"), _turn(60, 100, "a")]
    folded = relabel(fold_minor_speakers(turns, 0.1))
    assert folded.labels == ["S1", "S2"]
    assert _spans(folded.turns) == [(0, 1, "S1"), (5, 52, "S1"), (60, 100, "S2")]
    # Everyone is under the share: the one with the most speech stays
    many = [_turn(i * 10, i * 10 + 1 + (i == 2), str(i)) for i in range(5)]
    assert set(speech_per_speaker(fold_minor_speakers(many, 0.5))) == {"2"}


def test_fold_is_a_no_op_when_off_or_nothing_is_tiny():
    turns = [_turn(0, 10, "a"), _turn(10, 10.1, "b")]
    assert fold_minor_speakers(turns, 0) == turns
    assert fold_minor_speakers([], 0.02) == []
    big = [_turn(0, 10, "a"), _turn(10, 20, "b")]
    assert fold_minor_speakers(big, 0.02) == big


# --- #119: second clustering pass on per-speaker mean voices ------------------------------


def test_merge_joins_speakers_with_close_mean_voices():
    turns = [_turn(0, 10, "a"), _turn(10, 20, "b"), _turn(20, 30, "c"), _turn(30, 40, "a")]
    voices = {
        "a": np.array([1.0, 0.0]) * 20,
        "b": np.array([0.0, 1.0]) * 10,
        "c": np.array([0.95, 0.31]) * 10,  # cosine distance ~0.05 to "a"
    }
    merged = merge_similar_speakers(turns, voices, max_distance=0.3)
    assert [t.speaker for t in merged] == ["a", "b", "a", "a"]
    assert merge_similar_speakers(turns, voices, max_distance=0.01) == turns
    assert merge_similar_speakers(turns, voices, max_distance=0) == turns
    # A huge distance merges everyone into the speaker heard first
    assert {t.speaker for t in merge_similar_speakers(turns, voices, 1.9)} == {"a"}


def test_merge_keeps_the_label_heard_first_and_chains_through_merged_voices():
    # "c" speaks first and absorbs "b"; "a" is far from both
    turns = [_turn(0, 5, "c"), _turn(5, 10, "a"), _turn(10, 15, "b")]
    voices = {
        "c": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.9, 0.44, 0.0]),
        "a": np.array([0.0, 0.0, 1.0]),
    }
    assert [t.speaker for t in merge_similar_speakers(turns, voices, 0.2)] == ["c", "a", "c"]
    # Speakers without a voice (nothing embedded) are left as they are
    lone = merge_similar_speakers(turns, {"c": voices["c"]}, 0.5)
    assert [t.speaker for t in lone] == ["c", "a", "b"]


def test_merge_tie_goes_to_the_pair_heard_first():
    turns = [_turn(0, 5, "a"), _turn(5, 10, "b"), _turn(10, 15, "c")]
    same = np.array([1.0, 0.0])
    voices = {"a": same, "b": same, "c": same}
    assert [t.speaker for t in merge_similar_speakers(turns, voices, 0.1)] == ["a", "a", "a"]


def test_post_processing_only_when_estimating(monkeypatch):
    """``--num-speakers`` is an exact override: no merging or folding then."""
    raw = [_turn(0, 50, "0"), _turn(50, 50.5, "1"), _turn(51, 100, "2")]

    class Engine:
        def process(self, samples, callback):
            return self

        def sort_by_start_time(self):
            return [
                type("R", (), {"start": t.start, "end": t.end, "speaker": t.speaker}) for t in raw
            ]

    monkeypatch.setattr(diarize_module, "_load_wav", lambda path: np.zeros(16_000))
    seen: list[str] = []

    def run(config: DiarizeConfig) -> list[str]:
        diarizer = SherpaDiarizer(config)
        diarizer._engine = Engine()  # type: ignore[assignment]
        monkeypatch.setattr(diarizer, "_voices", lambda s, turns: seen.append("voices") or {})
        return [t.speaker for t in diarizer.diarize(Path("audio.wav"))]

    assert run(DiarizeConfig(num_speakers=3)) == ["0", "1", "2"] and seen == []
    assert run(DiarizeConfig()) == ["0", "2"] and seen == ["voices"]  # "1" folded into "0"


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
    auto = "fake/diarizer|speakers=auto|t=0.8|merge=0.3|min_share=0.02"
    assert stage.settings(PipelineConfig()) == auto
    folding_off = PipelineConfig(diarize=DiarizeConfig(min_speaker_share=0))
    assert stage.settings(folding_off) == auto.replace("min_share=0.02", "min_share=0.0")
    # An exact count ignores the estimation settings
    config = PipelineConfig(diarize=DiarizeConfig(enabled=True, num_speakers=2, threshold=0.4))
    assert stage.settings(config) == "fake/diarizer|speakers=2"


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
    assert log == [StageName.LLM, StageName.SCORE, StageName.REFINE, StageName.CLIP]
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
