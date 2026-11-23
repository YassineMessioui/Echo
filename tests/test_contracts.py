import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from echo.config import LLMConfig, PipelineConfig, PublishConfig, RefineConfig, TranscribeConfig
from echo.errors import ArtifactError, InvalidInputError, LLMError
from echo.interfaces import LLMClient, Publisher
from echo.job import Job, slugify
from echo.models import (
    Approval,
    ClipPost,
    ClipSpec,
    Platform,
    PostText,
    PublishResult,
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


def test_link_job_paths_and_remote_video(tmp_path: Path):
    from echo.interfaces import RemoteVideo

    job = Job(tmp_path / "youtube-x")
    assert job.input_audio == job.root / "input-audio.m4a"
    remote = RemoteVideo(url="https://example/v", audio=job.input_audio, width=1920, height=1080)
    assert remote.headers == {}


def test_transcribe_batching_and_vad_defaults():
    config = TranscribeConfig()
    assert (config.batch_size, config.vad_filter, config.vad_min_silence_ms) == (0, True, None)
    assert TranscribeConfig(batch_size=8).batch_size == 8  # batched pipeline
    with pytest.raises(ValidationError):
        TranscribeConfig(batch_size=-1)
    with pytest.raises(ValidationError):
        TranscribeConfig(vad_min_silence_ms=-1)


def test_llm_max_concurrency_defaults_to_auto():
    assert LLMConfig().max_concurrency is None  # 4 hosted, 1 Ollama (resolved by the stage)
    assert LLMConfig(max_concurrency=1).max_concurrency == 1
    for bad in (0, 17):  # 17+: a burst that free tiers answer with 429s
        with pytest.raises(ValidationError):
            LLMConfig(max_concurrency=bad)


def test_focus_range_and_auto_window_config():
    from echo.config import ChunkConfig, IngestConfig

    assert IngestConfig().start is None and IngestConfig().end is None
    assert IngestConfig(start="15%", end="3300").end == "3300"
    for bad in ({"start": "150%"}, {"end": "1:30"}, {"start": "-5"}):
        with pytest.raises(ValidationError):
            IngestConfig(**bad)
    assert ChunkConfig().max_tokens == "auto"
    assert ChunkConfig(max_tokens=24_000).max_tokens == 24_000
    with pytest.raises(ValidationError):
        ChunkConfig(max_tokens=0)


def test_transcript_offset_defaults_to_zero(golden_job: Job):
    assert golden_job.read_transcript().offset == 0.0


@pytest.mark.parametrize(("start", "end"), [("50%", "20%"), ("90", "30"), ("40%", "40%")])
def test_focus_range_rejects_inverted_same_unit_pairs(start, end):
    from echo.config import IngestConfig

    with pytest.raises(ValidationError, match="empty or inverted"):
        IngestConfig(start=start, end=end)
    IngestConfig(start="600", end="5%")  # mixed units: checked by ingest instead


# --- diarization (#22, #68) -------------------------------------------------------------


def test_old_job_json_without_diarize_stage_loads(golden_job: Job):
    from echo.models import StageName

    meta = golden_job.read_meta()  # written before the diarize stage existed
    assert meta.stages[StageName.DIARIZE].status.value == "pending"


def test_speakers_validation_and_lookup():
    from echo.models import Speakers, SpeakerTurn

    with pytest.raises(ValidationError):
        SpeakerTurn(start=5, end=5, speaker="S1")
    with pytest.raises(ValidationError):
        Speakers(
            turns=[
                SpeakerTurn(start=5, end=9, speaker="S1"),
                SpeakerTurn(start=0, end=4, speaker="S2"),
            ]
        )
    speakers = Speakers(
        turns=[
            SpeakerTurn(start=0, end=10, speaker="S1"),
            SpeakerTurn(start=10, end=20, speaker="S2"),
            SpeakerTurn(start=25, end=30, speaker="S1"),
        ]
    )
    assert speakers.labels == ["S1", "S2"]
    assert speakers.speaker_at(8, 12) == "S1"  # 2 s vs 2 s: first wins
    assert speakers.speaker_at(9, 15) == "S2"
    assert speakers.speaker_at(21, 22) == "S2"  # in a gap: nearest turn
    assert Speakers(turns=[]).speaker_at(0, 1) is None


def test_speakers_round_trip(job_copy: Job):
    from echo.models import Speakers, SpeakerTurn

    speakers = Speakers(turns=[SpeakerTurn(start=0.5, end=3.0, speaker="S1")])
    job_copy.write_speakers(speakers)
    assert job_copy.read_speakers() == speakers


def test_diarize_and_speaker_caption_config():
    from echo.config import CaptionConfig, DiarizeConfig, PipelineConfig

    assert PipelineConfig().diarize.enabled is False
    with pytest.raises(ValidationError):
        DiarizeConfig(num_speakers=0)
    # #119: estimating the count, with a second pass on mean voices and tiny speakers folded
    d = DiarizeConfig()
    assert (d.threshold, d.merge_threshold, d.min_speaker_share) == (0.8, 0.3, 0.02)
    assert DiarizeConfig(merge_threshold=0, min_speaker_share=0).min_speaker_share == 0
    for bad in ({"min_speaker_share": 1}, {"min_speaker_share": -0.1}, {"merge_threshold": -1}):
        with pytest.raises(ValidationError):
            DiarizeConfig(**bad)
    assert CaptionConfig().speaker_colours[0] == "#FFFFFF"
    with pytest.raises(ValidationError):
        CaptionConfig(speaker_colours=[])
    with pytest.raises(ValidationError):
        CaptionConfig(speaker_colours=["blue"])


# --- clip scores (#21) ------------------------------------------------------------------


def test_old_topics_without_scores_still_load(golden_job: Job):
    """topics.json from before #21 has no ``score``; it loads as None."""
    assert all(topic.score is None for topic in golden_job.read_topics())


def test_golden_scores_match_the_golden_topics(golden_job: Job):
    scored = golden_job.read_scores()
    topics = golden_job.read_topics()
    assert [t.model_copy(update={"score": None}) for t in scored] == topics
    assert all(t.score is not None and 0 <= t.score.overall <= 10 for t in scored)


def test_score_criteria_are_between_0_and_10():
    from echo.models import Score

    ok = {"overall": 5, "hook": 5, "payoff": 5, "clarity": 5, "energy": 5}
    assert Score(**ok).reason == ""
    for name in ok:
        with pytest.raises(ValidationError):
            Score(**(ok | {name: 11}))
        with pytest.raises(ValidationError):
            Score(**(ok | {name: -1}))


def test_scores_round_trip_and_topics_json_has_no_null_score(job_copy: Job):
    scored = job_copy.read_scores()
    job_copy.write_scores(scored)
    assert job_copy.read_scores() == scored
    job_copy.write_topics(job_copy.read_topics())
    assert "score" not in job_copy.topics_path.read_text()


def test_old_job_json_without_score_stage_loads(golden_job: Job):
    assert golden_job.read_meta().stages[StageName.SCORE].status is StageStatus.PENDING


def test_score_config_defaults_on():
    assert PipelineConfig().score.enabled is True


# --- clips/publish.json (#67, #100) ----------------------------------------------------

PUBLISH_FIXTURE = Path(__file__).parent / "fixtures" / "publish.json"


def _posts_job(job_copy: Job) -> Job:
    job_copy.clips_dir.mkdir(exist_ok=True)
    shutil.copy(PUBLISH_FIXTURE, job_copy.publish_path)
    return job_copy


def test_publish_json_matches_the_golden_clips(job_copy: Job):
    job = _posts_job(job_copy)
    posts = job.read_posts()
    clips = job.read_clips()
    assert [p.clip for p in posts] == [
        job.clip_video_path(i, c.topic_title).name for i, c in enumerate(clips, start=1)
    ]
    assert [p.topic_title for p in posts] == [c.topic_title for c in clips]


def test_publish_json_defaults_and_round_trip(job_copy: Job):
    job = _posts_job(job_copy)
    solar, sleep = job.read_posts()
    assert sleep.approval is Approval.PENDING and sleep.overrides == {} and sleep.results == {}
    assert solar.published_on(Platform.YOUTUBE)
    assert not solar.published_on(Platform.TIKTOK)  # an error result is not published
    assert not solar.published_on(Platform.INSTAGRAM)
    job.write_posts([solar, sleep])
    assert job.read_posts() == [solar, sleep]
    # Defaults are left out, so a fresh post stays short and readable for hand edits.
    assert '"approval": "pending"' not in job.publish_path.read_text()


def test_old_jobs_have_no_publish_json(golden_job: Job):
    assert not golden_job.publish_path.exists()
    with pytest.raises(ArtifactError, match="missing"):
        golden_job.read_posts()


def test_text_for_applies_platform_overrides(job_copy: Job):
    solar = _posts_job(job_copy).read_posts()[0]
    tiktok = solar.text_for(Platform.TIKTOK)
    assert tiktok.title == "Solar just got cheap"
    assert tiktok.description == solar.description and tiktok.hashtags == solar.hashtags
    assert solar.text_for(Platform.YOUTUBE).title == solar.title
    empty = solar.model_copy(update={"overrides": {Platform.YOUTUBE: PostText(hashtags=[])}})
    assert empty.text_for(Platform.YOUTUBE).hashtags == []  # [] overrides, None inherits


def _post(**changes) -> dict:
    return {"clip": "01-a.mp4", "topic_title": "A", "title": "A title"} | changes


@pytest.mark.parametrize(
    "changes",
    [
        {"clip": "clips/01-a.mp4"},
        {"clip": "..\\01-a.mp4"},
        {"clip": "01-a.mov"},
        {"title": ""},
        {"title": "x" * 101},
        {"hashtags": ["#solar"]},
        {"hashtags": ["two words"]},
        {"approval": "maybe"},
        {"overrides": {"myspace": {"title": "x"}}},
        {"results": {"youtube": {}}},
        {"results": {"youtube": {"id": "1", "error": "and failed"}}},
    ],
)
def test_clip_post_rejects_invalid_values(changes):
    with pytest.raises(ValidationError):
        ClipPost.model_validate(_post(**changes))


def test_a_clip_is_listed_once(job_copy: Job):
    post = ClipPost.model_validate(_post())
    with pytest.raises(ArtifactError, match="twice"):
        job_copy.write_posts([post, post])
    job_copy.clips_dir.mkdir(exist_ok=True)
    job_copy.publish_path.write_text(f"[{post.model_dump_json()}, {post.model_dump_json()}]")
    with pytest.raises(ArtifactError, match="twice"):
        job_copy.read_posts()


def test_publish_config_is_private_opt_in_and_needs_approval():
    config = PipelineConfig().publish
    assert config.platforms == [] and config.privacy == "private"
    assert config.require_approval is True
    with pytest.raises(ValidationError):
        PublishConfig(require_approval=False)
    with pytest.raises(ValidationError):
        PublishConfig(privacy="friends")
    assert PublishConfig(platforms=["youtube"]).platforms == [Platform.YOUTUBE]


def test_publish_is_not_a_pipeline_stage():
    assert "publish" not in {name.value for name in StageName}


def test_publisher_protocol_shape(tmp_path: Path):
    class FakePublisher:
        platform = Platform.YOUTUBE

        def publish(self, video: Path, post: ClipPost, config: PublishConfig) -> PublishResult:
            return PublishResult(id="abc", privacy=config.privacy)

    publisher: Publisher = FakePublisher()
    result = publisher.publish(tmp_path / "01-a.mp4", ClipPost(**_post()), PublishConfig())
    assert result.ok and result.privacy == "private"


def test_webcam_config_defaults_and_validation():
    """#126: on by default (auto), top-left, 40 % of the width."""
    from echo.config import ClipConfig

    config = ClipConfig()
    assert (config.webcam, config.webcam_corner, config.webcam_size) == ("auto", "top-left", 0.4)
    for bad in (
        {"webcam": "on"},
        {"webcam_corner": "middle"},
        {"webcam_size": 0.1},
        {"webcam_size": 0.7},
    ):
        with pytest.raises(ValidationError):
            ClipConfig(**bad)
