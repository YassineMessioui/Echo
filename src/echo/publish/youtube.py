"""YouTube Shorts (#103): sign in once (OAuth, installed app) and upload clips.

Plain ``httpx`` against Google's REST endpoints, so publishing adds no dependency and
the tests run on a fake transport.

- **App credentials** (a Google Cloud "Desktop app" OAuth client, #99): the JSON file
  downloaded from the Cloud console, at ``$ECHO_YOUTUBE_CLIENT_SECRETS`` or
  ``<config dir>/youtube-client.json``. ``echo-clips auth youtube --client-secrets F``
  copies it there.
- **Sign-in** (``echo-clips auth youtube``): the browser opens Google's consent page and
  redirects back to a one-shot server on ``127.0.0.1`` (PKCE, a random ``state``). The
  refresh token is stored in ``<config dir>/youtube-token.json`` (owner-only where the
  OS allows), never in the repo, the job or the logs.
- **Upload**: a resumable ``videos.insert`` in chunks, at ``PublishConfig.privacy``
  (private by default). ``#Shorts`` is added to the description; with a vertical video
  of 3 minutes or less, YouTube files it as a Short.

The config dir is ``$ECHO_CONFIG_DIR``, else ``~/.config/echo``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import urllib.parse
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

from echo.config import PublishConfig
from echo.errors import PublishError
from echo.models import ClipPost, Platform, PublishResult

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"

CHUNK_SIZE = 8 * 2**20
"""Bytes per upload request (a multiple of 256 KiB, as the resumable protocol requires)."""
TAGS_MAX_CHARS = 450
"""YouTube allows 500 characters of tags in all; keep a margin for its own counting."""
CATEGORY = "22"
"""People & Blogs: a safe default for podcast clips."""
SHORTS_TAG = "#Shorts"

AUTH_HINT = "sign in with: echo-clips auth youtube"


def config_dir() -> Path:
    return Path(os.environ.get("ECHO_CONFIG_DIR") or Path.home() / ".config" / "echo")


def client_secrets_path() -> Path:
    return Path(
        os.environ.get("ECHO_YOUTUBE_CLIENT_SECRETS") or config_dir() / "youtube-client.json"
    )


def token_path() -> Path:
    return config_dir() / "youtube-token.json"


@dataclass(frozen=True)
class ClientSecrets:
    client_id: str
    client_secret: str

    @classmethod
    def load(cls, path: Path | None = None) -> ClientSecrets:
        """The OAuth client from a Cloud console JSON file (``installed`` or ``web``)."""
        path = path or client_secrets_path()
        if not path.is_file():
            raise PublishError(
                f"no YouTube app credentials at {path}: create a Desktop OAuth client in "
                "Google Cloud (see docs/publishing.md), then run "
                "echo-clips auth youtube --client-secrets <downloaded file>"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            app = data.get("installed") or data.get("web") or data
            return cls(client_id=app["client_id"], client_secret=app["client_secret"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise PublishError(f"{path} is not a Google OAuth client file: {e}") from e


def _write_private(path: Path, data: str) -> None:
    """Write ``data`` atomically, readable by the owner only where the OS allows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def install_client_secrets(source: Path) -> Path:
    """Validate ``source`` and copy it to ``client_secrets_path()``."""
    ClientSecrets.load(source)
    target = client_secrets_path()
    _write_private(target, source.read_text(encoding="utf-8"))
    return target


# --- sign-in ---------------------------------------------------------------------------


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def auth_url(client: ClientSecrets, redirect_uri: str, state: str, challenge: str) -> str:
    query = {
        "client_id": client.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",  # a refresh token, so later uploads need no browser
        # Always show the account/channel chooser (a Brand Account channel is picked
        # there) and the consent screen, so a refresh token is always returned.
        "prompt": "select_account consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(query)}"


def _wait_for_redirect(server: HTTPServer, timeout: float) -> dict[str, str]:
    """Serve the one redirect from Google and return its query parameters."""
    received: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server's name)
            query = urllib.parse.urlparse(self.path).query
            received.update({k: v[0] for k, v in urllib.parse.parse_qs(query).items()})
            ok = "code" in received
            body = (
                "Echo is signed in to YouTube. You can close this tab."
                if ok
                else "Sign-in was cancelled or failed. You can close this tab."
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args: object) -> None:  # keep the code out of the console
            pass

    server.RequestHandlerClass = Handler
    server.timeout = timeout
    server.handle_request()
    if not received:
        raise PublishError(f"no answer from the Google sign-in page within {timeout:.0f} s")
    return received


def sign_in(
    client: ClientSecrets,
    *,
    http: httpx.Client | None = None,
    open_browser: Callable[[str], object] = webbrowser.open,
    timeout: float = 300,
) -> Path:
    """Run the installed-app OAuth flow and store the refresh token. Returns its path."""
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        redirect_uri = f"http://127.0.0.1:{server.server_address[1]}"
        url = auth_url(client, redirect_uri, state, challenge)
        if not open_browser(url):
            print(f"Open this link to sign in:\n{url}")
        answer = _wait_for_redirect(server, timeout)
    finally:
        server.server_close()
    if answer.get("state") != state:
        raise PublishError("sign-in answer doesn't match this request (state); try again")
    if "code" not in answer:
        raise PublishError(f"sign-in failed: {answer.get('error', 'no code returned')}")
    data = _token_request(
        {
            "code": answer["code"],
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        },
        http,
    )
    if "refresh_token" not in data:
        raise PublishError("Google returned no refresh token; run echo-clips auth youtube again")
    path = token_path()
    _write_private(
        path,
        json.dumps({"refresh_token": data["refresh_token"], "scope": data.get("scope", SCOPE)}),
    )
    return path


def _token_request(form: dict[str, str], http: httpx.Client | None) -> dict[str, Any]:
    client = http or httpx.Client(timeout=30)
    try:
        response = client.post(TOKEN_URL, data=form)
    except httpx.HTTPError as e:
        raise PublishError(f"can't reach Google sign-in: {e}") from e
    finally:
        if http is None:
            client.close()
    data = _json(response)
    if response.status_code == 400 and data.get("error") == "invalid_grant":
        raise PublishError(f"the YouTube sign-in expired or was revoked; {AUTH_HINT}")
    if response.is_error:
        raise PublishError(
            f"Google sign-in failed ({response.status_code}): "
            f"{data.get('error_description') or data.get('error') or response.text[:200]}"
        )
    return data


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# --- upload ---------------------------------------------------------------------------


def video_metadata(post: ClipPost, privacy: str) -> dict[str, Any]:
    """The ``videos.insert`` body for ``post``'s YouTube text."""
    text = post.text_for(Platform.YOUTUBE)
    hashtags = [f"#{t}" for t in text.hashtags or []]
    if SHORTS_TAG.lower() not in (h.lower() for h in hashtags):
        hashtags.append(SHORTS_TAG)
    description = "\n\n".join(part for part in (text.description, " ".join(hashtags)) if part)
    tags: list[str] = []
    for tag in text.hashtags or []:
        if sum(len(t) + 1 for t in tags) + len(tag) > TAGS_MAX_CHARS:
            break
        tags.append(tag)
    return {
        "snippet": {
            "title": text.title,
            "description": description[:5000],
            "tags": tags,
            "categoryId": CATEGORY,
        },
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }


def _api_error(response: httpx.Response) -> tuple[str, str]:
    """(reason, message) from a Google API error response."""
    error = _json(response).get("error")
    if not isinstance(error, dict):
        return "", response.text[:200]
    errors = error.get("errors") or [{}]
    reason = errors[0].get("reason", "") if isinstance(errors[0], dict) else ""
    return reason, str(error.get("message", ""))


_QUOTA_REASONS = {"quotaExceeded", "uploadLimitExceeded", "rateLimitExceeded"}
_AUTH_REASONS = {"authError", "unauthorized", "forbidden", "youtubeSignupRequired"}


class YouTubePublisher:
    """Implements ``echo.interfaces.Publisher`` for YouTube Shorts.

    Raises PublishError when nothing more can be uploaded (sign-in expired or missing,
    quota used up, no channel); returns a ``PublishResult`` with ``error`` when only this
    clip failed (e.g. YouTube rejected its title), so the next clip is still tried.
    """

    platform = Platform.YOUTUBE

    def __init__(
        self,
        client: ClientSecrets | None = None,
        *,
        http: httpx.Client | None = None,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self._client = client
        self._http = http
        self._chunk_size = chunk_size
        self._access_token: str | None = None

    def _get_http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=httpx.Timeout(30, write=300))
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def _token(self) -> str:
        if self._access_token is None:
            path = token_path()
            if not path.is_file():
                raise PublishError(f"not signed in to YouTube; {AUTH_HINT}")
            try:
                refresh = json.loads(path.read_text(encoding="utf-8"))["refresh_token"]
            except (ValueError, KeyError, TypeError) as e:
                raise PublishError(f"{path} is damaged; {AUTH_HINT}") from e
            client = self._client or ClientSecrets.load()
            data = _token_request(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret,
                    "refresh_token": refresh,
                    "grant_type": "refresh_token",
                },
                self._get_http(),
            )
            self._access_token = str(data["access_token"])
        return self._access_token

    def publish(self, video: Path, post: ClipPost, config: PublishConfig) -> PublishResult:
        if not video.is_file():
            return PublishResult(error=f"the clip video is missing: {video}")
        size = video.stat().st_size
        http = self._get_http()
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(size),
        }
        try:
            start = http.post(
                UPLOAD_URL,
                params={"uploadType": "resumable", "part": "snippet,status"},
                json=video_metadata(post, config.privacy),
                headers=headers,
            )
            if start.is_error:
                return self._failed(start)
            session = start.headers.get("Location")
            if not session:
                return PublishResult(error="YouTube didn't start the upload (no session URL)")
            done = self._send(http, session, video, size)
        except httpx.HTTPError as e:
            return PublishResult(error=f"upload failed: {e}")
        if done.is_error:
            return self._failed(done)
        video_id = _json(done).get("id")
        if not video_id:
            return PublishResult(error="YouTube accepted the upload but returned no video id")
        return PublishResult(
            id=str(video_id),
            url=f"https://youtube.com/shorts/{video_id}",
            published_at=datetime.now(UTC),
            privacy=config.privacy,
        )

    def _send(self, http: httpx.Client, session: str, video: Path, size: int) -> httpx.Response:
        """PUT the file in chunks; 308 means "go on from the byte after Range"."""
        auth = {"Authorization": f"Bearer {self._token()}"}
        offset, stalled = 0, 0
        with video.open("rb") as f:
            while True:
                f.seek(offset)
                chunk = f.read(self._chunk_size)
                end = offset + len(chunk) - 1
                headers = auth | {"Content-Length": str(len(chunk))}
                if size:
                    headers["Content-Range"] = f"bytes {offset}-{end}/{size}"
                response = http.put(session, content=chunk, headers=headers)
                if response.status_code != 308:
                    return response
                uploaded = response.headers.get("Range")  # e.g. "bytes=0-8388607"
                next_offset = int(uploaded.rsplit("-", 1)[1]) + 1 if uploaded else 0
                stalled = stalled + 1 if next_offset <= offset else 0
                if stalled >= 3:
                    raise httpx.TransportError("YouTube keeps asking for the same bytes")
                offset = next_offset

    def _failed(self, response: httpx.Response) -> PublishResult:
        reason, message = _api_error(response)
        if response.status_code == 401 or reason in _AUTH_REASONS:
            if reason == "youtubeSignupRequired":
                raise PublishError("this Google account has no YouTube channel yet; create one")
            raise PublishError(f"YouTube refused the sign-in ({message or reason}); {AUTH_HINT}")
        if reason in _QUOTA_REASONS:
            raise PublishError(
                f"YouTube upload quota used up ({reason}); try again tomorrow (the daily "
                "quota resets at midnight Pacific time)"
            )
        return PublishResult(
            error=f"YouTube rejected the upload ({response.status_code}): {message or reason}"
        )
