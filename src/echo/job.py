"""Job directory: the on-disk contract between stages.

jobs/<job_id>/
  input.mp4              # symlink to the source video (local files)
  input-audio.m4a        # link jobs: only the audio is downloaded (#60); see RemoteVideo
  audio.wav              # stage: ingest
  transcript.json        # stage: transcribe
  speakers.json          # stage: diarize (optional, #22)
  topics.json            # stage: llm
  clips.json             # stage: refine
  clips/<nn>-<slug>.mp4  # stage: clip
  clips/publish.json     # command: publish (#67): text, approval, results per clip
  job.json               # orchestrator
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from echo.errors import ArtifactError, InvalidInputError
from echo.models import ClipPost, ClipSpec, JobMeta, Speakers, Topic, Transcript

_TOPICS = TypeAdapter(list[Topic])
_CLIPS = TypeAdapter(list[ClipSpec])
_POSTS = TypeAdapter(list[ClipPost])

SLUG_MAX_LEN = 50


def slugify(text: str, max_len: int = SLUG_MAX_LEN) -> str:
    """ASCII, lowercase, dash-separated slug. Never empty."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "clip"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _unique_clips(posts: list[ClipPost], path: Path) -> None:
    seen: set[str] = set()
    for post in posts:
        if post.clip in seen:
            raise ArtifactError(f"invalid artifact {path}: clip {post.clip!r} is listed twice")
        seen.add(post.clip)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink, falling back to a hardlink, then a copy.

    Windows needs admin or Developer Mode for symlinks, and hardlinks only work
    on the same volume.
    """
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


class Job:
    """Typed access to one job directory. Stages use these paths and read/write methods only."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # --- lifecycle --------------------------------------------------------------

    @classmethod
    def create(cls, jobs_dir: Path, video: Path, job_id: str | None = None) -> Job:
        """Create (or reopen) a job for ``video``. Idempotent for an existing job_id.

        Raises InvalidInputError if the video is missing, or if ``job_id`` already
        belongs to a different video.
        """
        if not video.is_file():
            raise InvalidInputError(f"input video not found: {video}")
        source = video.resolve()
        job = cls(jobs_dir / (job_id or uuid.uuid4().hex[:12]))
        if job.meta_path.exists() and job.read_meta().source != str(source):
            raise InvalidInputError(
                f"job {job.job_id!r} was created for {job.read_meta().source}, not {source}"
            )
        job.root.mkdir(parents=True, exist_ok=True)
        if not (job.input_video.is_symlink() or job.input_video.exists()):
            _link_or_copy(source, job.input_video)
        if not job.meta_path.exists():
            job.write_meta(
                JobMeta(job_id=job.job_id, source=str(source), created_at=datetime.now(UTC))
            )
        return job

    @classmethod
    def open(cls, jobs_dir: Path, job_id: str) -> Job:
        job = cls(jobs_dir / job_id)
        if not job.meta_path.is_file():
            raise ArtifactError(f"no job {job_id!r} in {jobs_dir}")
        return job

    @property
    def job_id(self) -> str:
        return self.root.name

    # --- paths ------------------------------------------------------------------

    @property
    def input_video(self) -> Path:
        return self.root / "input.mp4"

    @property
    def input_audio(self) -> Path:
        """Link jobs only: the downloaded audio track. The video stays remote and only the
        clip sections are read (``echo.interfaces.RemoteVideo``). Content may be any
        container ffmpeg reads, whatever the extension."""
        return self.root / "input-audio.m4a"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

    @property
    def speakers_path(self) -> Path:
        return self.root / "speakers.json"

    @property
    def topics_path(self) -> Path:
        return self.root / "topics.json"

    @property
    def clips_path(self) -> Path:
        return self.root / "clips.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def publish_path(self) -> Path:
        """``clips/publish.json``: one ``ClipPost`` per clip (#67). Absent until the
        clips' text is generated (#101)."""
        return self.clips_dir / "publish.json"

    @property
    def meta_path(self) -> Path:
        return self.root / "job.json"

    def clip_video_path(self, index: int, topic_title: str) -> Path:
        """``clips/<nn>-<slug>.mp4``, with ``index`` starting at 1."""
        return self.clips_dir / f"{index:02d}-{slugify(topic_title)}.mp4"

    # --- artifacts --------------------------------------------------------------

    def read_transcript(self) -> Transcript:
        return self._read_model(self.transcript_path, Transcript)

    def write_transcript(self, transcript: Transcript) -> None:
        _atomic_write(self.transcript_path, transcript.model_dump_json(indent=2).encode())

    def read_speakers(self) -> Speakers:
        return self._read_model(self.speakers_path, Speakers)

    def write_speakers(self, speakers: Speakers) -> None:
        _atomic_write(self.speakers_path, speakers.model_dump_json(indent=2).encode())

    def read_topics(self) -> list[Topic]:
        return self._read_list(self.topics_path, _TOPICS)

    def write_topics(self, topics: list[Topic]) -> None:
        _atomic_write(self.topics_path, _TOPICS.dump_json(topics, indent=2))

    def read_clips(self) -> list[ClipSpec]:
        return self._read_list(self.clips_path, _CLIPS)

    def write_clips(self, clips: list[ClipSpec]) -> None:
        _atomic_write(self.clips_path, _CLIPS.dump_json(clips, indent=2))

    def read_posts(self) -> list[ClipPost]:
        posts = self._read_list(self.publish_path, _POSTS)
        _unique_clips(posts, self.publish_path)
        return posts

    def write_posts(self, posts: list[ClipPost]) -> None:
        _unique_clips(posts, self.publish_path)
        _atomic_write(self.publish_path, _POSTS.dump_json(posts, indent=2, exclude_defaults=True))

    def read_meta(self) -> JobMeta:
        return self._read_model(self.meta_path, JobMeta)

    def write_meta(self, meta: JobMeta) -> None:
        _atomic_write(self.meta_path, meta.model_dump_json(indent=2).encode())

    # --- helpers ----------------------------------------------------------------

    @staticmethod
    def _read_model[M: BaseModel](path: Path, model: type[M]) -> M:
        if not path.is_file():
            raise ArtifactError(f"missing artifact: {path}")
        try:
            return model.model_validate_json(path.read_bytes())
        except ValidationError as e:
            raise ArtifactError(f"invalid artifact {path}: {e}") from e

    @staticmethod
    def _read_list[T](path: Path, adapter: TypeAdapter[list[T]]) -> list[T]:
        if not path.is_file():
            raise ArtifactError(f"missing artifact: {path}")
        try:
            return adapter.validate_json(path.read_bytes())
        except ValidationError as e:
            raise ArtifactError(f"invalid artifact {path}: {e}") from e
