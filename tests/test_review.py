"""Review (#102): approve, reject or edit clips; only approved clips can be published."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from echo import cli
from echo.errors import PublishError
from echo.job import Job
from echo.models import Approval, ClipPost, Platform, PublishResult
from echo.publish.review import (
    ReviewError,
    edit_text,
    parse_numbers,
    ready_to_publish,
    set_approval,
)

SOLAR = "01-solar-power-is-now-the-cheapest-electricity.mp4"
SLEEP = "02-sleep-and-productivity.mp4"

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


def post(clip: str = SOLAR, **changes) -> ClipPost:
    base = {"clip": clip, "topic_title": "T", "title": "A title", "hashtags": ["a"]}
    return ClipPost.model_validate(base | changes)


# --- parse_numbers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("1", [0]), ("3,1", [0, 2]), ("2-4", [1, 2, 3]), (" 1, 1-2 ,", [0, 1]), ("5", [4])],
)
def test_parse_numbers(spec, expected):
    assert parse_numbers(spec, 5) == expected


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("0", "numbered 1 to 5"),
        ("6", "no clip 6"),
        ("3-2", "no clip 3-2"),
        ("x", "not a clip"),
        ("", "no clip numbers"),
    ],
)
def test_parse_numbers_rejects(spec, message):
    with pytest.raises(ReviewError, match=message):
        parse_numbers(spec, 5)


# --- approval and edits ----------------------------------------------------------------------


def test_approve_needs_the_video(tmp_path: Path):
    with pytest.raises(ReviewError, match="video is missing"):
        set_approval(post(), Approval.APPROVED, tmp_path)
    (tmp_path / SOLAR).write_bytes(b"")
    assert set_approval(post(), Approval.APPROVED, tmp_path).approval is Approval.APPROVED
    # Rejecting (or resetting) a clip whose video is gone is fine.
    assert set_approval(post(SLEEP), Approval.REJECTED, tmp_path).approval is Approval.REJECTED


def test_edit_text_validates_and_normalises():
    edited = edit_text(
        post(), title="  New   title ", description=" Line\nTwo ", hashtags=["#x", " y", ""]
    )
    assert (edited.title, edited.description, edited.hashtags) == (
        "New title",
        "Line\nTwo",
        ["x", "y"],
    )
    assert edit_text(post(), description="d").title == "A title"  # None keeps a field
    with pytest.raises(ReviewError, match="title"):
        edit_text(post(), title="x" * 101)
    with pytest.raises(ReviewError, match="hashtags"):
        edit_text(post(), hashtags=["two words"])


# --- the publish gate ----------------------------------------------------------------------


def test_only_approved_unpublished_posts_are_ready():
    posts = [
        post("01-a.mp4", approval="approved"),
        post("02-b.mp4"),  # pending
        post("03-c.mp4", approval="rejected"),
        post("04-d.mp4", approval="approved", results={"youtube": {"id": "x"}}),
        post("05-e.mp4", approval="approved", results={"youtube": {"error": "quota"}}),
    ]
    ready = ready_to_publish(posts, Platform.YOUTUBE)
    assert [p.clip for p in ready] == ["01-a.mp4", "05-e.mp4"]  # a failed attempt is retried
    assert [p.clip for p in ready_to_publish(posts, Platform.TIKTOK)] == [
        "01-a.mp4",
        "04-d.mp4",
        "05-e.mp4",
    ]


@pytest.mark.parametrize("approval", ["pending", "rejected"])
def test_an_unapproved_clip_cant_be_published(approval):
    posts = [post("01-a.mp4", approval="approved"), post("02-b.mp4", approval=approval)]
    with pytest.raises(PublishError, match="not approved.*02-b.mp4"):
        ready_to_publish(posts, Platform.YOUTUBE, only=[0, 1])
    assert [p.clip for p in ready_to_publish(posts, Platform.YOUTUBE, only=[0])] == ["01-a.mp4"]


def test_published_ones_are_skipped_even_when_asked_for():
    done = post(approval="approved", results={"youtube": PublishResult(id="x").model_dump()})
    assert ready_to_publish([done], Platform.YOUTUBE, only=[0]) == []


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def job(tmp_path: Path, golden_job: Job) -> Job:
    """A golden job copy with both clip videos (empty files) and pending posts."""
    shutil.copytree(golden_job.root, tmp_path / "ep42")
    job = Job(tmp_path / "ep42")
    job.clips_dir.mkdir(exist_ok=True)
    for name in (SOLAR, SLEEP):
        (job.clips_dir / name).write_bytes(b"")
    job.write_posts(
        [
            post(SOLAR, topic_title="Solar power is now the cheapest electricity", title="Solar"),
            post(
                SLEEP,
                topic_title="Sleep and productivity",
                title="Sleep",
                description="Line one\nLine two",
            ),
        ]
    )
    return job


def review(job: Job, *args: str, input: str | None = None):
    return runner.invoke(cli.app, ["review", str(job.root), *args], input=input)


def test_lists_posts_without_changing_them(job: Job):
    before = job.publish_path.read_bytes()
    result = review(job)
    assert result.exit_code == 0, result.output
    assert f"  1 [pending] {SOLAR}  25.7s\n      Solar\n      #a\n" in result.stdout
    assert "      Line one\n      Line two\n" in result.stdout
    assert "0 approved, 2 pending, 0 rejected" in result.stdout
    assert job.publish_path.read_bytes() == before


def test_approve_and_reject_with_flags(job: Job):
    result = review(job, "--approve", "1", "--reject", "2")
    assert result.exit_code == 0, result.output
    assert [p.approval for p in job.read_posts()] == [Approval.APPROVED, Approval.REJECTED]
    assert "1 approved, 0 pending, 1 rejected" in result.stdout
    assert review(job, "--reset", "1-2").exit_code == 0
    assert {p.approval for p in job.read_posts()} == {Approval.PENDING}


def test_edit_with_flags(job: Job):
    result = review(
        job, "--clip", "2", "--title", "Better", "--description", "One\\nTwo", "--hashtags", "#x,y"
    )
    assert result.exit_code == 0, result.output
    sleep = job.read_posts()[1]
    assert (sleep.title, sleep.description, sleep.hashtags) == ("Better", "One\nTwo", ["x", "y"])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--approve", "3"], "no clip 3"),
        (["--approve", "1", "--reject", "1"], "two different decisions"),
        (["--title", "x"], "need --clip N"),
        (["--clip", "3", "--title", "x"], "no clip 3"),
        (["--clip", "1", "--title", "x" * 101], "invalid text"),
    ],
)
def test_bad_requests_change_nothing(job: Job, args, message):
    before = job.publish_path.read_bytes()
    result = review(job, *args)
    assert result.exit_code == 1 and message in result.stderr
    assert job.publish_path.read_bytes() == before


def test_cant_approve_a_missing_video(job: Job):
    (job.clips_dir / SLEEP).unlink()
    result = review(job, "--approve", "2")
    assert result.exit_code == 1 and "video is missing" in result.stderr


def test_needs_post_text_first(job: Job):
    job.publish_path.unlink()
    result = review(job)
    assert result.exit_code == 1 and "echo-clips posts" in result.stderr


def test_interactive_review(job: Job):
    # Clip 1: edit, then approve. Clip 2: reject.
    keys = "e\nEdited title\nNew line one\\nline two\nx,y\na\nr\n"
    result = review(job, "-i", input=keys)
    assert result.exit_code == 0, result.output
    solar, sleep = job.read_posts()
    assert solar.approval is Approval.APPROVED and solar.title == "Edited title"
    assert solar.description == "New line one\nline two" and solar.hashtags == ["x", "y"]
    assert sleep.approval is Approval.REJECTED


def test_interactive_quit_keeps_decisions_so_far(job: Job):
    result = review(job, "-i", input="a\nq\n")
    assert result.exit_code == 0, result.output
    assert [p.approval for p in job.read_posts()] == [Approval.APPROVED, Approval.PENDING]


def test_interactive_shows_errors_and_asks_again(job: Job):
    (job.clips_dir / SOLAR).unlink()
    result = review(job, "-i", input="a\ns\ns\n")
    assert result.exit_code == 0, result.output
    assert "video is missing" in result.output
    assert {p.approval for p in job.read_posts()} == {Approval.PENDING}
