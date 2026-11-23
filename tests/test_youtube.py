"""YouTube publishing (#103) on a fake HTTP transport: no network, no real account."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.parse
import urllib.request
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from echo import cli
from echo.config import PublishConfig
from echo.errors import PublishError
from echo.job import Job
from echo.models import Approval, ClipPost, Platform, PostText, PublishResult
from echo.publish import youtube
from echo.publish.runner import publish_job
from echo.publish.youtube import ClientSecrets, YouTubePublisher, video_metadata

CLIENT = ClientSecrets(client_id="cid.apps.googleusercontent.com", client_secret="shh")
SESSION = "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=xyz"


@pytest.fixture(autouse=True)
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config"
    monkeypatch.setenv("ECHO_CONFIG_DIR", str(path))
    monkeypatch.delenv("ECHO_YOUTUBE_CLIENT_SECRETS", raising=False)
    return path


def signed_in() -> None:
    youtube._write_private(youtube.token_path(), json.dumps({"refresh_token": "r1"}))


def post(**changes) -> ClipPost:
    base = {
        "clip": "01-a.mp4",
        "topic_title": "A",
        "title": "A title",
        "description": "Line one.",
        "hashtags": ["sleep", "focus"],
        "approval": "approved",
    }
    return ClipPost.model_validate(base | changes)


class FakeGoogle:
    """Token endpoint + resumable upload, recording every request."""

    def __init__(
        self,
        *,
        upload_status: int = 200,
        upload_body: dict | None = None,
        token_status: int = 200,
        token_body: dict | None = None,
        start_status: int = 200,
        start_body: dict | None = None,
        resume_once: bool = False,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.upload_status, self.upload_body = upload_status, upload_body or {"id": "vid123"}
        self.token_status = token_status
        self.token_body = token_body or {"access_token": "a1", "expires_in": 3599}
        self.start_status, self.start_body = start_status, start_body or {}
        self.resume_once = resume_once
        self.received = b""

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(self.token_status, json=self.token_body)
        if request.method == "POST":
            if self.start_status != 200:
                return httpx.Response(self.start_status, json=self.start_body)
            return httpx.Response(200, headers={"Location": SESSION})
        self.received += request.content
        if self.resume_once:
            self.resume_once = False
            return httpx.Response(308, headers={"Range": f"bytes=0-{len(self.received) - 1}"})
        return httpx.Response(self.upload_status, json=self.upload_body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def publisher(google: FakeGoogle, **kwargs) -> YouTubePublisher:
    return YouTubePublisher(CLIENT, http=google.client(), **kwargs)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "01-a.mp4"
    path.write_bytes(b"x" * 1000)
    return path


# --- metadata -------------------------------------------------------------------------------


def test_metadata_is_private_short_with_tags():
    body = video_metadata(post(), "private")
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    assert body["snippet"]["title"] == "A title"
    assert body["snippet"]["description"] == "Line one.\n\n#sleep #focus #Shorts"
    assert body["snippet"]["tags"] == ["sleep", "focus"]


def test_metadata_uses_youtube_overrides_and_caps_tags():
    p = post(
        hashtags=["shorts", *(f"tag{i:03d}x" * 5 for i in range(40))],
        overrides={"youtube": {"title": "YT title", "description": ""}},
    )
    body = video_metadata(p, "unlisted")
    assert body["snippet"]["title"] == "YT title"
    assert body["snippet"]["description"].startswith("#shorts #tag000")  # no duplicate #Shorts
    assert "#Shorts" not in body["snippet"]["description"]
    assert sum(len(t) + 1 for t in body["snippet"]["tags"]) <= youtube.TAGS_MAX_CHARS + 1
    assert body["status"]["privacyStatus"] == "unlisted"


# --- upload ---------------------------------------------------------------------------------


def test_uploads_and_returns_the_short_link(video: Path):
    signed_in()
    google = FakeGoogle()
    result = publisher(google).publish(video, post(), PublishConfig())
    assert result.ok and result.id == "vid123"
    assert result.url == "https://youtube.com/shorts/vid123" and result.privacy == "private"
    token, start, upload = google.requests
    assert dict(urllib.parse.parse_qsl(token.content.decode())) == {
        "client_id": CLIENT.client_id,
        "client_secret": "shh",
        "refresh_token": "r1",
        "grant_type": "refresh_token",
    }
    assert start.url.params["uploadType"] == "resumable"
    assert start.headers["Authorization"] == "Bearer a1"
    assert start.headers["X-Upload-Content-Length"] == "1000"
    assert json.loads(start.content)["status"]["privacyStatus"] == "private"
    assert str(upload.url) == SESSION and upload.headers["Content-Range"] == "bytes 0-999/1000"
    assert google.received == video.read_bytes()


def test_uploads_in_chunks_and_resumes(video: Path):
    signed_in()
    google = FakeGoogle(resume_once=True)
    result = publisher(google, chunk_size=400).publish(video, post(), PublishConfig())
    assert result.ok
    ranges = [r.headers["Content-Range"] for r in google.requests if r.method == "PUT"]
    assert ranges == ["bytes 0-399/1000", "bytes 400-799/1000"][: len(ranges)]


def test_the_access_token_is_fetched_once(video: Path):
    signed_in()
    google = FakeGoogle()
    p = publisher(google)
    p.publish(video, post(), PublishConfig())
    p.publish(video, post(), PublishConfig())
    assert sum(r.url.host == "oauth2.googleapis.com" for r in google.requests) == 1


def test_not_signed_in(video: Path):
    with pytest.raises(PublishError, match="echo-clips auth youtube"):
        publisher(FakeGoogle()).publish(video, post(), PublishConfig())


def test_revoked_sign_in(video: Path):
    signed_in()
    google = FakeGoogle(token_status=400, token_body={"error": "invalid_grant"})
    with pytest.raises(PublishError, match="expired or was revoked"):
        publisher(google).publish(video, post(), PublishConfig())


def _api_error(status: int, reason: str) -> dict:
    return {
        "error": {"code": status, "message": f"{reason} happened", "errors": [{"reason": reason}]}
    }


@pytest.mark.parametrize(
    ("status", "reason", "message"),
    [
        (403, "quotaExceeded", "quota used up"),
        (400, "uploadLimitExceeded", "quota used up"),
        (401, "authError", "refused the sign-in"),
        (401, "youtubeSignupRequired", "no YouTube channel"),
    ],
)
def test_errors_that_stop_publishing(video: Path, status, reason, message):
    signed_in()
    google = FakeGoogle(start_status=status, start_body=_api_error(status, reason))
    with pytest.raises(PublishError, match=message):
        publisher(google).publish(video, post(), PublishConfig())


def test_a_rejected_clip_is_a_result_not_an_exception(video: Path):
    signed_in()
    google = FakeGoogle(start_status=400, start_body=_api_error(400, "invalidTitle"))
    result = publisher(google).publish(video, post(), PublishConfig())
    assert not result.ok and "invalidTitle happened" in result.error


def test_missing_video_is_a_result(tmp_path: Path):
    signed_in()
    result = publisher(FakeGoogle()).publish(tmp_path / "gone.mp4", post(), PublishConfig())
    assert not result.ok and "missing" in result.error


def test_network_failure_is_a_result(video: Path):
    signed_in()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "a"})
        raise httpx.ConnectError("offline")

    p = YouTubePublisher(CLIENT, http=httpx.Client(transport=httpx.MockTransport(handler)))
    result = p.publish(video, post(), PublishConfig())
    assert not result.ok and "offline" in result.error


# --- client secrets and sign-in ----------------------------------------------------------------


def test_client_secrets_from_the_console_file(tmp_path: Path, config_dir: Path):
    source = tmp_path / "client_secret_123.json"
    source.write_text(json.dumps({"installed": {"client_id": "id", "client_secret": "s"}}))
    saved = youtube.install_client_secrets(source)
    assert saved == config_dir / "youtube-client.json"
    assert ClientSecrets.load() == ClientSecrets("id", "s")
    with pytest.raises(PublishError, match="no YouTube app credentials"):
        ClientSecrets.load(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(PublishError, match="not a Google OAuth client"):
        youtube.install_client_secrets(bad)


def test_sign_in_flow_stores_only_the_refresh_token(config_dir: Path):
    """The browser is simulated: it follows the consent URL's redirect_uri with a code."""
    google = FakeGoogle(token_body={"access_token": "a", "refresh_token": "r9", "scope": "s"})
    opened: list[str] = []

    def browser(url: str) -> bool:
        opened.append(url)
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        back = f"{query['redirect_uri']}/?code=c1&state={query['state']}"
        threading.Thread(target=lambda: urllib.request.urlopen(back, timeout=10).read()).start()
        return True

    path = youtube.sign_in(CLIENT, http=google.client(), open_browser=browser, timeout=10)
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(opened[0]).query))
    assert query["scope"] == youtube.SCOPE and query["access_type"] == "offline"
    assert query["prompt"] == "select_account consent"  # the channel chooser is always shown
    assert query["code_challenge_method"] == "S256"
    exchange = dict(urllib.parse.parse_qsl(google.requests[0].content.decode()))
    assert exchange["code"] == "c1" and exchange["code_verifier"]
    assert exchange["redirect_uri"] == query["redirect_uri"]
    assert json.loads(path.read_text()) == {"refresh_token": "r9", "scope": "s"}


def test_sign_in_rejects_a_wrong_state():
    def browser(url: str) -> bool:
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        back = f"{query['redirect_uri']}/?code=c1&state=forged"
        threading.Thread(target=lambda: urllib.request.urlopen(back, timeout=10).read()).start()
        return True

    with pytest.raises(PublishError, match="doesn't match"):
        youtube.sign_in(CLIENT, http=FakeGoogle().client(), open_browser=browser, timeout=10)
    assert not youtube.token_path().exists()


# --- publish_job ---------------------------------------------------------------------------------


class FakePublisher:
    platform = Platform.YOUTUBE

    def __init__(self, results: list[PublishResult | Exception]) -> None:
        self.results = results
        self.sent: list[str] = []

    def publish(self, video: Path, post: ClipPost, config: PublishConfig) -> PublishResult:
        self.sent.append(video.name)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def job(tmp_path: Path, golden_job: Job) -> Job:
    shutil.copytree(golden_job.root, tmp_path / "ep42")
    job = Job(tmp_path / "ep42")
    job.clips_dir.mkdir(exist_ok=True)
    job.write_posts(
        [
            post(clip="01-a.mp4"),
            post(clip="02-b.mp4", approval="pending"),
            post(clip="03-c.mp4"),
            post(clip="04-d.mp4", results={"youtube": {"id": "old"}}),
        ]
    )
    return job


def test_publishes_only_approved_unposted_clips_and_saves_each(job: Job):
    fake = FakePublisher([PublishResult(id="v1"), PublishResult(error="rejected")])
    done = publish_job(job, fake, PublishConfig())
    assert fake.sent == ["01-a.mp4", "03-c.mp4"]
    posts = job.read_posts()
    assert posts[0].results[Platform.YOUTUBE].id == "v1"
    assert posts[1].results == {}
    assert posts[2].results[Platform.YOUTUBE].error == "rejected"
    assert posts[3].results[Platform.YOUTUBE].id == "old"
    assert [r.ok for _, r in done] == [True, False]
    # A re-run retries only the failed one.
    fake = FakePublisher([PublishResult(id="v3")])
    publish_job(job, fake, PublishConfig())
    assert fake.sent == ["03-c.mp4"]


def test_a_stop_keeps_results_so_far(job: Job):
    fake = FakePublisher([PublishResult(id="v1"), PublishError("quota used up")])
    with pytest.raises(PublishError):
        publish_job(job, fake, PublishConfig())
    posts = job.read_posts()
    assert posts[0].published_on(Platform.YOUTUBE) and posts[2].results == {}


def test_asking_for_an_unapproved_clip_sends_nothing(job: Job):
    fake = FakePublisher([])
    with pytest.raises(PublishError, match="not approved"):
        publish_job(job, fake, PublishConfig(), only=[0, 1])
    assert fake.sent == []


# --- CLI ------------------------------------------------------------------------------

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


def test_cli_publish(job: Job, monkeypatch: pytest.MonkeyPatch):
    fake = FakePublisher(
        [
            PublishResult(id="v1", url="https://youtube.com/shorts/v1", privacy="unlisted"),
            PublishResult(error="rejected"),
        ]
    )
    fake.close = lambda: None
    monkeypatch.setattr(youtube, "YouTubePublisher", lambda: fake)
    result = runner.invoke(cli.app, ["publish", str(job.root), "--privacy", "unlisted"])
    assert result.exit_code == 1  # one clip failed
    assert "posted 01-a.mp4 (unlisted): https://youtube.com/shorts/v1" in result.stdout
    assert "failed 03-c.mp4: rejected" in result.stderr


def test_cli_publish_refuses_unapproved_and_other_platforms(job: Job):
    result = runner.invoke(cli.app, ["publish", str(job.root), "--clips", "2"])
    assert result.exit_code == 1 and "not approved" in result.stderr
    result = runner.invoke(cli.app, ["publish", str(job.root), "--platform", "tiktok"])
    assert result.exit_code == 1 and "isn't supported yet" in result.stderr
    result = runner.invoke(cli.app, ["publish", str(job.root), "--privacy", "friends"])
    assert result.exit_code == 1 and "--privacy" in result.stderr


def test_cli_publish_nothing_to_do(job: Job, monkeypatch: pytest.MonkeyPatch):
    posts = [p.model_copy(update={"approval": Approval.REJECTED}) for p in job.read_posts()]
    job.write_posts(posts)
    fake = FakePublisher([])
    fake.close = lambda: None
    monkeypatch.setattr(youtube, "YouTubePublisher", lambda: fake)
    result = runner.invoke(cli.app, ["publish", str(job.root)])
    assert result.exit_code == 0 and "nothing to post" in result.stdout


def test_overrides_reach_the_upload(video: Path):
    signed_in()
    google = FakeGoogle()
    p = post(overrides={Platform.YOUTUBE: PostText(title="Only on YT").model_dump()})
    publisher(google).publish(video, p, PublishConfig())
    assert json.loads(google.requests[1].content)["snippet"]["title"] == "Only on YT"
