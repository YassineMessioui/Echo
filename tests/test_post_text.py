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
    assert "context_hashtags" in w.client.calls[0].system


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


# --- context hashtags and names (#129) ------------------------------------------------------


def test_context_hashtags_go_first_in_every_post():
    raw = json.dumps(
        {
            "context_hashtags": ["#GTA6", "ishowspeed"],
            "posts": [
                entry(0, hashtags=["lucia", "gta6", "IShowSpeed"]),
                entry(1, hashtags=["ishowspeed", "jason", "gta6", "vicecity"]),
            ],
        }
    )
    first, second = parse_posts(raw, 2)
    assert first.hashtags == ["GTA6", "ishowspeed", "lucia"]
    assert second.hashtags == ["GTA6", "ishowspeed", "jason", "vicecity"]


def test_a_post_without_the_context_hashtags_is_sent_back():
    raw = json.dumps(
        {
            "context_hashtags": ["gta6", "ishowspeed"],
            "posts": [entry(0, hashtags=["gta6", "a", "b"])],
        }
    )
    with pytest.raises(InvalidPostsError, match=r"clip 0: .*context hashtags \(ishowspeed\)"):
        parse_posts(raw, 1)


@pytest.mark.parametrize(
    ("context", "message"),
    [(["a", "b", "c", "d"], "at most 3"), (["two words"], "without spaces"), ("gta6", "list")],
)
def test_invalid_context_hashtags(context, message):
    raw = json.dumps({"context_hashtags": context, "posts": [entry(0)]})
    with pytest.raises(InvalidPostsError, match=message):
        parse_posts(raw, 1)


def test_no_context_is_fine():
    raw = json.dumps({"context_hashtags": [], "posts": [entry(0)]})
    assert parse_posts(raw, 1)[0].hashtags == ["one", "two", "three"]


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("title", "One person gets grilled"),
        ("description", "A heated exchange between two friends."),
        ("description", "Someone searches for a car."),
        ("title", "What the speaker thinks"),
    ],
)
def test_vague_people_are_sent_back(field, text):
    with pytest.raises(InvalidPostsError, match="name them instead"):
        parse_posts(answer(entry(0, **{field: text})), 1)


def test_named_people_pass():
    item = entry(0, title="IShowSpeed reacts to Lucia confronting Jason in GTA 6")
    assert parse_posts(answer(item), 1)[0].title.startswith("IShowSpeed reacts")


def test_retry_mentions_the_missing_context():
    bad = json.dumps({"context_hashtags": ["gta6"], "posts": [entry(0)]})
    good = json.dumps(
        {"context_hashtags": ["gta6"], "posts": [entry(0, hashtags=["gta6", "a", "b"])]}
    )
    w = writer(bad, good)
    assert w.write([ClipText("t", "", "x")])[0].hashtags == ["gta6", "a", "b"]
    assert "context hashtags (gta6)" in w.client.calls[1].prompt


# --- frames for vision models (#129, posts-v4) ----------------------------------------------


def test_frames_are_sent_as_numbered_images():
    w = writer(answer(entry(0), entry(1), entry(2)))
    clips = [
        ClipText("Cutscene", "", "Just tell me the truth!", frame=b"jpeg-a"),
        ClipText("No frame", "", "words"),
        ClipText("Map", "", "where is my car", frame=b"jpeg-b"),
    ]
    assert len(w.write(clips)) == 3
    (call,) = w.client.calls
    assert call.images == (b"jpeg-a", b"jpeg-b")
    assert "[0] Topic: Cutscene.\nOn screen: image 1\nSaid:" in call.prompt
    assert "[1] Topic: No frame.\nSaid:" in call.prompt
    assert "[2] Topic: Map.\nOn screen: image 2\nSaid:" in call.prompt
    assert "a cutscene shows the game's characters" in call.system


def test_a_model_without_vision_falls_back_to_text_once():
    class TextOnly(FakeLLMClient):
        def complete(self, system, prompt, *, json_schema=None, images=()):
            if images:
                self.calls.append(None)
                raise LLMError("model does not support image input")
            return super().complete(system, prompt, json_schema=json_schema)

    client = TextOnly([answer(entry(0)), answer(entry(0))])
    w = LLMPostWriter(client, LLMConfig())
    framed = [ClipText("t", "", "x", frame=b"jpeg")]
    assert w.write(framed)[0].title == "A good title"
    assert w.vision is False and "image input" in w.vision_error
    assert "On screen" not in client.calls[1].prompt
    w.write(framed)  # no second try with images
    assert client.calls[2] is not None and client.calls[2].images == ()


def test_vision_off_sends_no_images():
    w = LLMPostWriter(FakeLLMClient([answer(entry(0))]), LLMConfig(), vision=False)
    w.write([ClipText("t", "", "x", frame=b"jpeg")])
    assert w.client.calls[0].images == ()


def test_batches_cap_the_images_per_call():
    from echo.publish.text import MAX_IMAGES

    clips = [ClipText("t", "", "x", frame=b"j") for _ in range(MAX_IMAGES * 2 + 1)]
    groups = batches(clips, 1_000_000)
    assert [len(g) for g in groups] == [MAX_IMAGES, MAX_IMAGES, 1]
    assert batches([ClipText("t", "", "x")] * 12, 1_000_000) == [range(0, 12)]


def test_thinking_before_the_answer_is_ignored():
    raw = "<think>Clip 0 is a cutscene with Lucia.</think>\n" + answer(entry(0))
    assert parse_posts(raw, 1)[0].title == "A good title"


@pytest.mark.ffmpeg
def test_clip_frame_is_two_moments_side_by_side(tmp_path, make_video):
    from echo.publish.text import clip_frame

    video = make_video(duration=4.0, size="1080x1920", audio=False)
    jpeg = clip_frame(video, 4.0)
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    out = tmp_path / "f.jpg"
    out.write_bytes(jpeg)
    from echo.ffmpeg import probe

    assert probe(out).frame_size == (576, 512)
    assert clip_frame(tmp_path / "missing.mp4", 4.0) is None


def test_update_posts_sends_frames_of_the_cut_clips(job_copy: Job, monkeypatch):
    import echo.publish.text as text_module

    seen: list[str] = []
    monkeypatch.setattr(
        text_module, "clip_frame", lambda path, d: seen.append(path.name) or b"jpeg"
    )
    w = writer(answer(entry(0), entry(1)))
    update_posts(job_copy, w, CHUNK)
    assert seen == [SOLAR, SLEEP]
    assert w.client.calls[0].images == (b"jpeg", b"jpeg")
    seen.clear()
    w = LLMPostWriter(FakeLLMClient([answer(entry(0), entry(1))]), LLMConfig(), vision=False)
    update_posts(job_copy, w, CHUNK, regenerate=True)
    assert seen == []


def test_a_too_large_request_is_split_not_given_up():
    """Groq's free tier caps a request at 7000 input tokens: with frames, three clips can
    be too much at once. The batch is split (still with frames), and vision stays on."""

    class Capped(FakeLLMClient):
        def complete(self, system, prompt, *, json_schema=None, images=()):
            if len(images) > 1:
                self.calls.append(None)
                raise LLMError("groq returned HTTP 413: Request too large for model")
            return super().complete(system, prompt, json_schema=json_schema, images=images)

    client = Capped([answer(entry(0)), answer(entry(0)), answer(entry(0))])
    w = LLMPostWriter(client, LLMConfig())
    framed = [ClipText(f"t{i}", "", "x", frame=b"jpeg") for i in range(3)]
    assert len(w.write(framed)) == 3
    assert w.vision is True and w.vision_error is None
    assert [c.images for c in client.calls if c is not None] == [(b"jpeg",)] * 3
