"""Job directory: the on-disk contract between stages.

jobs/<job_id>/
  input.mp4              # symlink to the source video
  audio.wav              # stage: ingest
  transcript.json        # stage: transcribe
  topics.json            # stage: llm
  clips.json             # stage: refine
  clips/<nn>-<slug>.mp4  # stage: clip
  job.json               # orchestrator
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from echo.errors import ArtifactError, InvalidInputError
from echo.models import ClipSpec, JobMeta, Topic, Transcript

_TOPICS = TypeAdapter(list[Topic])
_CLIPS = TypeAdapter(list[ClipSpec])

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


class Job:
    """Typed access to one job directory. Stages use these paths and read/write methods only."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # --- lifecycle --------------------------------------------------------------

    @classmethod
    def create(cls, jobs_dir: Path, video: Path, job_id: str | None = None) -> Job:
        """Create (or reopen) a job for ``video``. Idempotent for an existing job_id."""
        if not video.is_file():
            raise InvalidInputError(f"input video not found: {video}")
        job = cls(jobs_dir / (job_id or uuid.uuid4().hex[:12]))
        job.root.mkdir(parents=True, exist_ok=True)
        if not job.input_video.exists():
            job.input_video.symlink_to(video.resolve())
        if not job.meta_path.exists():
            job.write_meta(
                JobMeta(
                    job_id=job.root.name,
                    source=str(video.resolve()),
                    created_at=datetime.now(UTC),
                )
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
    def audio_path(self) -> Path:
        return self.root / "audio.wav"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

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

    def read_topics(self) -> list[Topic]:
        return self._read_list(self.topics_path, _TOPICS)

    def write_topics(self, topics: list[Topic]) -> None:
        _atomic_write(self.topics_path, _TOPICS.dump_json(topics, indent=2))

    def read_clips(self) -> list[ClipSpec]:
        return self._read_list(self.clips_path, _CLIPS)

    def write_clips(self, clips: list[ClipSpec]) -> None:
        _atomic_write(self.clips_path, _CLIPS.dump_json(clips, indent=2))

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
