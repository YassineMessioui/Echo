"""Post text for clips (#101): parsing, limits, retries, batching and publish.json updates."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from echo import cli
from echo.config import ChunkConfig, LLMConfig
from echo.errors import LLMError, TopicExtractionError
from echo.job import Job
from echo.models import Approval, ClipPost, Platform, PostText, PublishResult
from echo.publish import prompts
from echo.publish.text import (
    CLIP_WORDS,
    DESCRIPTION_MAX,
    TITLE_MAX,
    ClipText,
    InvalidPostsError,
    LLMPostWriter,
    batches,
    clip_text,
    parse_posts,
    update_posts,
)
from echo.testing import FakeLLMClient

CHUNK = ChunkConfig()
SOLAR = "01-solar-power-is-now-the-cheapest-electricity.mp4"
SLEEP = "02-sleep-and-productivity.mp4"


def entry(clip: int, title="A good title", description="What you learn.", hashtags=None) -> dict:
    return {
        "clip": clip,
        "title": title,
        "description": description,
        "hashtags": ["one", "two", "three"] if hashtags is None else hashtags,
    }


def answer(*entries: dict) -> str:
    return json.dumps({"posts": list(entries)})


def writer(*responses: str) -> LLMPostWriter:
    return LLMPostWriter(FakeLLMClient(responses), LLMConfig())


# --- parse_posts ----------------------------------------------------------------------


def test_parses_posts_in_clip_order_and_normalises():
    drafts = parse_posts(
        answer(
            entry(1, title="  Second   clip "),
            entry(0, hashtags=["#Sleep", "focus", "sleep", "health"], description="a\n\n b "),
        ),
        2,
    )
    assert drafts[0].hashtags == ["Sleep", "focus", "health"]  # '#' and duplicates removed
    assert drafts[0].description == "a\nb"
    assert drafts[1].title == "Second clip"


@pytest.mark.parametrize("raw", ["```json\n" + answer(entry(0)) + "\n```", json.dumps([entry(0)])])
def test_accepts_fences_and_bare_lists(raw):
    assert parse_posts(raw, 1)[0].title == "A good title"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("nope", "not valid JSON"),
        ('{"topics": []}', "of the form"),
        (answer(entry(0)), "no entry for clip(s) 1"),
        (answer(entry(0), entry(0), entry(1)), "more than one entry"),
        (answer(entry(0), entry(5)), "there is no clip 5"),
        (answer(entry(0), entry("1")), '"clip" must be an integer'),
        (answer(entry(0), entry(1, title="x" * (TITLE_MAX + 1))), "the most is 70"),
        (answer(entry(0), entry(1, title=" ")), '"title" is empty'),
        (answer(entry(0), entry(1, description="")), '"description" is empty'),
        (answer(entry(0), entry(1, description="a\nb\nc\nd")), "more than 3 lines"),
        (
            answer(entry(0), entry(1, description="x" * (DESCRIPTION_MAX + 1))),
            f"the most is {DESCRIPTION_MAX}",
        ),
        (answer(entry(0), entry(1, hashtags=["a", "b"])), "3 to 5 different hashtags, not 2"),
        (answer(entry(0), entry(1, hashtags=["a", "b", "c", "d", "e", "f"])), "not 6"),
        (answer(entry(0), entry(1, hashtags=["a", "A", "b"])), "not 2"),
        (answer(entry(0), entry(1, hashtags=["two words", "b", "c"])), "without spaces"),
        (answer(entry(0), entry(1, hashtags="a b c")), "list of strings"),
        (answer(entry(0), entry(1, title=3)), '"title" must be a string'),
        (json.dumps({"posts": ["x", "y"]}), "not an object"),
    ],
)
def test_rejects_invalid_answers(raw, message):
    with pytest.raises(InvalidPostsError) as e:
        parse_posts(raw, 2)
    assert message in str(e.value)


# --- LLMPostWriter --------------------------------------------------------------------


def test_writer_sends_numbered_clips_with_topic_and_text():
    w = writer(answer(entry(0), entry(1)))
    clips = [ClipText("Solar", "Cheap power.", "Solar is cheap."), ClipText("Sleep", "", "Sleep.")]
    assert len(w.write(clips)) == 2
    (call,) = w.client.calls
    assert "[0] Topic: Solar. Cheap power.\nSaid: Solar is cheap." in call.prompt
    assert "[1] Topic: Sleep.\nSaid: Sleep." in call.prompt
    assert call.json_schema == prompts.POSTS_SCHEMA
    assert f"at most {TITLE_MAX} characters" in call.system


def test_writer_retries_with_the_error_then_gives_up():
    bad = answer(entry(0, hashtags=["a"]))
    w = writer(bad, answer(entry(0)))
    assert w.write([ClipText("t", "", "x")])[0].title == "A good title"
    assert "not 1" in w.client.calls[1].prompt and bad in w.client.calls[1].prompt

    w = LLMPostWriter(FakeLLMClient([bad] * 3), LLMConfig(max_retries=2))
    with pytest.raises(TopicExtractionError, match="after 3 attempts"):
        w.write([ClipText("t", "", "x")])


def test_writer_does_not_retry_llm_errors():
    w = writer()  # no responses: the fake raises LLMError
    with pytest.raises(LLMError):
        w.write([ClipText("t", "", "x")])
    assert len(w.client.calls) == 1


def test_no_clips_no_call():
    w = writer()
    assert w.write([]) == []
    assert w.client.calls == []


# --- clip text and batching -------------------------------------------------------------


def test_clip_text_is_what_the_clip_says(golden_job: Job):
    transcript = golden_job.read_transcript()
    solar, sleep = golden_job.read_clips()
    text = clip_text(transcript, solar, "summary")
    inside = [s.text for s in transcript.segments if s.end > solar.start and s.start < solar.end]
    assert text.said == " ".join(" ".join(inside).split())
    assert text.topic_title == solar.topic_title and text.summary == "summary"
    assert clip_text(transcript, sleep).said != text.said


def test_clip_text_is_capped(golden_job: Job):
    transcript = golden_job.read_transcript()
    long = transcript.model_copy(
        update={
            "segments": [
                s.model_copy(update={"text": "word " * CLIP_WORDS}) for s in transcript.segments
            ]
        }
    )
    said = clip_text(long, golden_job.read_clips()[1]).said
    assert len(said.split()) == CLIP_WORDS + 1 and said.endswith("…")


def test_batches_fit_the_budget():
    clips = [ClipText("t", "", "word " * 400) for _ in range(5)]  # ~600 tokens each
    assert batches(clips, 100_000) == [range(0, 5)]
    groups = batches(clips, 2_000)
    assert [i for g in groups for i in g] == list(range(5)) and len(groups) > 1
    assert batches(clips, 10) == [range(i, i + 1) for i in range(5)]  # one each at least


# --- update_posts -----------------------------------------------------------------------


def test_writes_a_pending_post_per_clip(job_copy: Job):
    update = update_posts(job_copy, writer(answer(entry(0, "Solar"), entry(1, "Sleep"))), CHUNK)
    posts = job_copy.read_posts()
    assert posts == update.posts
    assert [(p.clip, p.title, p.approval) for p in posts] == [
        (SOLAR, "Solar", Approval.PENDING),
        (SLEEP, "Sleep", Approval.PENDING),
    ]
    assert update.written == [SOLAR, SLEEP] and update.dropped == []
    assert job_copy.read_meta().models["posts_prompt"] == prompts.POSTS_PROMPT_VERSION


def test_rerun_keeps_reviewed_posts_and_only_writes_new_clips(job_copy: Job):
    update_posts(job_copy, writer(answer(entry(0, "Solar"), entry(1, "Sleep"))), CHUNK)
    solar, sleep = job_copy.read_posts()
    edited = solar.model_copy(update={"title": "My own title", "approval": Approval.APPROVED})
    job_copy.write_posts([edited])  # sleep's post is gone, as if it were new

    w = writer(answer(entry(0, "Sleep again")))
    update = update_posts(job_copy, w, CHUNK)
    assert update.written == [SLEEP]
    assert len(w.client.calls) == 1 and "Solar" not in w.client.calls[0].prompt
    assert [p.title for p in job_copy.read_posts()] == ["My own title", "Sleep again"]

    w = writer()  # everything has a post: no LLM call at all
    assert update_posts(job_copy, w, CHUNK).written == []
    assert w.client.calls == []


def test_regenerate_rewrites_only_pending_text(job_copy: Job):
    update_posts(job_copy, writer(answer(entry(0, "Solar"), entry(1, "Sleep"))), CHUNK)
    solar, sleep = job_copy.read_posts()
    overrides = {Platform.TIKTOK: PostText(title="Short")}
    job_copy.write_posts(
        [
            solar.model_copy(update={"approval": Approval.REJECTED}),
            sleep.model_copy(update={"overrides": overrides}),
        ]
    )
    update = update_posts(job_copy, writer(answer(entry(0, "Sleep v2"))), CHUNK, regenerate=True)
    assert update.written == [SLEEP]
    solar, sleep = job_copy.read_posts()
    assert solar.title == "Solar" and solar.approval is Approval.REJECTED
    assert sleep.title == "Sleep v2" and sleep.overrides == overrides


def test_posts_of_removed_clips_are_dropped_unless_published(job_copy: Job):
    published = ClipPost(
        clip="07-old.mp4",
        topic_title="Old",
        title="Old",
        approval=Approval.APPROVED,
        results={Platform.YOUTUBE: PublishResult(id="abc")},
    )
    failed = ClipPost(
        clip="08-gone.mp4",
        topic_title="Gone",
        title="Gone",
        results={Platform.YOUTUBE: PublishResult(error="quota")},
    )
    job_copy.clips_dir.mkdir(exist_ok=True)
    job_copy.write_posts([published, failed])
    update = update_posts(job_copy, writer(answer(entry(0), entry(1))), CHUNK)
    assert [p.clip for p in job_copy.read_posts()] == [SOLAR, SLEEP, "07-old.mp4"]
    assert update.dropped == ["08-gone.mp4"]


def test_a_renamed_topic_gets_new_text(job_copy: Job):
    """Same file name, different topic: refine changed the clip, so its text is stale."""
    job_copy.clips_dir.mkdir(exist_ok=True)
    job_copy.write_posts([ClipPost(clip=SOLAR, topic_title="Something else", title="Stale")])
    update_posts(job_copy, writer(answer(entry(0, "Solar"), entry(1, "Sleep"))), CHUNK)
    assert [p.title for p in job_copy.read_posts()] == ["Solar", "Sleep"]


# --- CLI --------------------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


@pytest.fixture
def jobs(tmp_path: Path, golden_job: Job) -> Path:
    shutil.copytree(golden_job.root, tmp_path / "ep42")
    return tmp_path


def test_cli_writes_and_lists_posts(jobs: Path, monkeypatch):
    client = FakeLLMClient([answer(entry(0, "Solar", hashtags=["a", "b", "c"]), entry(1, "Sleep"))])
    monkeypatch.setattr("echo.llm.client.make_llm_client", lambda config: client)
    result = runner.invoke(cli.app, ["posts", "ep42", "--jobs-dir", str(jobs)])
    assert result.exit_code == 0, result.output
    assert f"  [new] {SOLAR}\n        Solar\n        #a #b #c\n" in result.stdout
    assert Job(jobs / "ep42").publish_path.is_file()

    # By directory too; nothing new, so no LLM call and the status is shown.
    result = runner.invoke(cli.app, ["posts", str(jobs / "ep42")])
    assert result.exit_code == 0, result.output
    assert f"  [pending] {SLEEP}" in result.stdout
    assert len(client.calls) == 1


def test_cli_reports_errors(jobs: Path, monkeypatch):
    monkeypatch.setattr("echo.llm.client.make_llm_client", lambda config: FakeLLMClient([]))
    result = runner.invoke(cli.app, ["posts", "nope", "--jobs-dir", str(jobs)])
    assert result.exit_code == 1 and "no job 'nope'" in result.stderr
    result = runner.invoke(cli.app, ["posts", "ep42", "--jobs-dir", str(jobs)])
    assert result.exit_code == 1 and "error:" in result.stderr


def test_output_never_crashes_on_unencodable_text():
    """A redirected stdout on Windows is cp1252: the LLM's non-breaking hyphens (seen in
    a real run) must not crash the listing."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    cli._tolerant_output([stream])
    stream.write("re‑positioning \U0001f525")
    stream.flush()
    assert buffer.getvalue() == b"re?positioning ?"


# --- source context (posts-v2) ------------------------------------------------------------


def test_the_prompt_names_the_source_video(job_copy: Job):
    """A link job's title and channel tell the LLM what the clips are (the game, the
    creator), so titles and hashtags can say so."""
    from echo.publish.text import source_of

    meta = job_copy.read_meta()
    meta.source_info = {"title": "PLAYING GTA 6 EARLY ACCESS", "uploader": "IShowSpeed"}
    job_copy.write_meta(meta)
    assert source_of(job_copy) == '"PLAYING GTA 6 EARLY ACCESS" by IShowSpeed'
    w = writer(answer(entry(0), entry(1)))
    update_posts(job_copy, w, CHUNK)
    assert 'Source video: "PLAYING GTA 6 EARLY ACCESS" by IShowSpeed' in w.client.calls[0].prompt
    assert "the game, show or creator" in w.client.calls[0].system


def test_a_local_job_names_its_file(job_copy: Job):
    from echo.publish.text import source_of

    meta = job_copy.read_meta()
    meta.source = "/videos/My Podcast Ep 42.mp4"
    meta.source_info = {}
    job_copy.write_meta(meta)
    assert source_of(job_copy) == 'a video file named "My Podcast Ep 42"'


def test_no_source_no_intro():
    w = writer(answer(entry(0)))
    w.write([ClipText("t", "", "x")])
    assert w.client.calls[0].prompt.startswith("Clips 0 to 0:")
