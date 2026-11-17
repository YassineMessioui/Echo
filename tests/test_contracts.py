from pathlib import Path

import pytest
from pydantic import ValidationError

from echo.config import LLMConfig, PipelineConfig, RefineConfig
from echo.errors import ArtifactError, InvalidInputError, LLMError
from echo.interfaces import LLMClient
from echo.job import Job, slugify
from echo.models import (
    ClipSpec,
    Segment,
    StageName,
    StageStatus,
    Topic,
    Transcript,
    TranscriptWindow,
    Word,
)
from echo.testing import FakeLLMClient, FakeTranscriber


def _seg(i: int, start: float, end: float) -> Segment:
    return Segment(id=i, start=start, end=end, text=f"segment {i}")


# --- models -----------------------------------------------------------------------


def test_word_and_segment_reject_reversed_times():
    with pytest.raises(ValidationError):
        Word(start=2, end=1, text="x")
    with pytest.raises(ValidationError):
        _seg(0, 5, 4)


def test_transcript_requires_contiguous_ids():
    with pytest.raises(ValidationError, match="contiguous"):
        Transcript(language="en", duration=10, segments=[_seg(0, 0, 1), _seg(2, 1, 2)])


def test_transcript_rejects_segments_past_duration():
    with pytest.raises(ValidationError, match="duration"):
        Transcript(language="en", duration=1, segments=[_seg(0, 0, 5)])


def test_transcript_segment_lookup():
    t = Transcript(language="en", duration=2, segments=[_seg(0, 0, 1), _seg(1, 1, 2)])
    assert t.segment(1).start == 1
    with pytest.raises(KeyError):
        t.segment(2)


def test_topic_and_window_reject_reversed_spans():
    with pytest.raises(ValidationError):
        Topic(title="t", summary="s", start_segment=3, end_segment=2)
    with pytest.raises(ValidationError):
        TranscriptWindow(start_segment=3, end_segment=2, text="")


def test_clip_duration_must_match():
    with pytest.raises(ValidationError, match="duration"):
        ClipSpec(topic_title="t", start=0, end=10, duration=5)
    assert ClipSpec.between("t", 1.0004, 11.0).duration == 10.0


def test_artifacts_are_strict_and_frozen():
    with pytest.raises(ValidationError):
        Topic(title="t", summary="s", start_segment=0, end_segment=0, extra="nope")
    topic = Topic(title="t", summary="s", start_segment=0, end_segment=0)
    with pytest.raises(ValidationError):
        topic.title = "other"


def test_refine_config_rejects_inverted_range():
    with pytest.raises(ValidationError):
        RefineConfig(min_duration=60, max_duration=30)
    assert PipelineConfig().refine.min_duration == 15


# --- golden artifacts --------------------------------------------------------------


def test_golden_artifacts_validate(golden_job: Job):
    golden_job.read_transcript()
    golden_job.read_topics()
    golden_job.read_clips()
    golden_job.read_meta()


def test_golden_artifacts_are_mutually_consistent(golden_job: Job):
    transcript = golden_job.read_transcript()
    topics = golden_job.read_topics()
    clips = golden_job.read_clips()
    refine = RefineConfig()

    for topic in topics:
        transcript.segment(topic.end_segment)  # topic ids exist
    assert [c.topic_title for c in clips] == [t.title for t in topics]
    for prev, cur in zip(clips, clips[1:], strict=False):
        assert prev.end <= cur.start, "clips must not overlap"
    for clip in clips:
        assert 0 <= clip.start and clip.end <= transcript.duration
        assert refine.min_duration <= clip.duration <= refine.max_duration


# --- job directory ------------------------------------------------------------------


def test_job_create_is_idempotent(tmp_path: Path):
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"fake")
    job = Job.create(tmp_path / "jobs", video, job_id="abc")
    # A symlink, or a hardlink on Windows without symlink rights: same file either way.
    assert job.input_video.samefile(video)
    meta = job.read_meta()
    assert meta.job_id == "abc"
    assert set(meta.stages) == set(StageName)
    assert all(r.status is StageStatus.PENDING for r in meta.stages.values())

    again = Job.create(tmp_path / "jobs", video, job_id="abc")
    assert again.read_meta().created_at == meta.created_at
    assert Job.open(tmp_path / "jobs", "abc").root == job.root


@pytest.mark.parametrize("fail", [("symlink_to",), ("symlink_to", "link")])
def test_job_create_falls_back_without_symlinks(tmp_path: Path, monkeypatch, fail):
    """Windows without Developer Mode: symlink fails → hardlink; cross-volume → copy."""

    def denied(*args, **kwargs):
        raise OSError("WinError 1314: A required privilege is not held by the client")

    if "symlink_to" in fail:
        monkeypatch.setattr(Path, "symlink_to", denied)
    if "link" in fail:
        monkeypatch.setattr("echo.job.os.link", denied)
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"fake")
    job = Job.create(tmp_path / "jobs", video)
    assert not job.input_video.is_symlink()
    assert job.input_video.read_bytes() == b"fake"


def test_job_create_rejects_different_video(tmp_path: Path):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    Job.create(tmp_path / "jobs", first, job_id="abc")
    with pytest.raises(InvalidInputError, match="was created for"):
        Job.create(tmp_path / "jobs", second, job_id="abc")


def test_job_create_with_moved_source(tmp_path: Path):
    video = tmp_path / "ep.mp4"
    video.write_bytes(b"fake")
    Job.create(tmp_path / "jobs", video, job_id="abc")
    moved = video.rename(tmp_path / "moved.mp4")  # input.mp4 is now a broken symlink
    with pytest.raises(InvalidInputError):
        Job.create(tmp_path / "jobs", moved, job_id="abc")


def test_job_create_rejects_missing_video(tmp_path: Path):
    with pytest.raises(InvalidInputError):
        Job.create(tmp_path, tmp_path / "missing.mp4")


def test_job_open_unknown(tmp_path: Path):
    with pytest.raises(ArtifactError):
        Job.open(tmp_path, "nope")


def test_artifact_round_trip(job_copy: Job, golden_job: Job):
    for read, write in [
        ("read_transcript", "write_transcript"),
        ("read_topics", "write_topics"),
        ("read_clips", "write_clips"),
        ("read_meta", "write_meta"),
    ]:
        original = getattr(golden_job, read)()
        getattr(job_copy, write)(original)
        assert getattr(job_copy, read)() == original
    assert not list(job_copy.root.glob(".*.tmp")), "atomic writes leave no temp files"


def test_reading_missing_or_invalid_artifact(tmp_path: Path):
    job = Job(tmp_path)
    with pytest.raises(ArtifactError, match="missing"):
        job.read_topics()
    job.topics_path.write_text('[{"title": ""}]')
    with pytest.raises(ArtifactError, match="invalid"):
        job.read_topics()


def test_clip_video_path(golden_job: Job):
    path = golden_job.clip_video_path(3, "Sleep & Productivity!")
    assert path == golden_job.clips_dir / "03-sleep-productivity.mp4"


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Hello, World!", "hello-world"),
        ("Café déjà vu", "cafe-deja-vu"),
        ("???", "clip"),
        ("a" * 80, "a" * 50),
    ],
)
def test_slugify(text: str, slug: str):
    assert slugify(text) == slug


# --- fakes --------------------------------------------------------------------------


def test_fake_llm_client(golden_job: Job):
    llm = FakeLLMClient(["first"])
    assert isinstance(llm, LLMClient)
    assert llm.complete("sys", "prompt") == "first"
    assert llm.calls[0].prompt == "prompt"
    with pytest.raises(LLMError):
        llm.complete("sys", "again")


def test_fake_transcriber(golden_job: Job):
    transcript = golden_job.read_transcript()
    fake = FakeTranscriber(transcript)
    assert fake.transcribe(Path("audio.wav")) == transcript


def test_caption_config_defaults_and_validation():
    from echo.config import CaptionConfig

    caption = PipelineConfig().caption
    assert caption.enabled and caption.max_words == 3
    assert caption.highlight_colour == "#FFD400"
    for bad in ({"colour": "yellow"}, {"highlight_colour": "#FFF"}, {"max_words": 0}):
        with pytest.raises(ValidationError):
            CaptionConfig(**bad)


def test_job_meta_source_fields_are_optional(golden_job: Job):
    """job.json files written before ``source_url`` / ``source_info`` existed still load."""
    meta = golden_job.read_meta()
    assert meta.source_url is None and meta.source_info == {}
    linked = meta.model_copy(
        update={"source_url": "https://youtu.be/x", "source_info": {"id": "x", "duration": 61.5}}
    )
    assert type(meta).model_validate_json(linked.model_dump_json()) == linked


def test_refine_max_clips_and_caption_font_defaults():
    from echo.config import CaptionConfig

    assert RefineConfig().max_clips is None
    assert RefineConfig(max_clips=3).max_clips == 3
    with pytest.raises(ValidationError):
        RefineConfig(max_clips=0)
    assert CaptionConfig().font == "Montserrat ExtraBold"


def test_llm_max_concurrency_defaults_to_auto():
    assert LLMConfig().max_concurrency is None  # 4 hosted, 1 Ollama (resolved by the stage)
    assert LLMConfig(max_concurrency=1).max_concurrency == 1
    with pytest.raises(ValidationError):
        LLMConfig(max_concurrency=0)
