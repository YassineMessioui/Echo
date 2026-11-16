"""Pipeline configuration. Each layer reads only its own section."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IngestConfig(_Section):
    sample_rate: int = 16_000
    channels: int = 1


class TranscribeConfig(_Section):
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = Field(default=None, description="None = auto-detect.")


class ChunkConfig(_Section):
    max_tokens: int = Field(default=3000, gt=0)
    overlap_segments: int = Field(default=5, ge=0)


class LLMConfig(_Section):
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = Field(default=2, ge=0)


class RefineConfig(_Section):
    min_duration: float = Field(default=15.0, gt=0)
    max_duration: float = Field(default=60.0, gt=0)
    padding: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def _range(self) -> "RefineConfig":
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration < min_duration")
        return self


class ClipConfig(_Section):
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "veryfast"
    aspect: Literal["9:16", "source"] = "9:16"
    """'9:16' outputs 1080x1920 for Reels/Shorts/TikTok; 'source' keeps the original framing."""
    background: Literal["blur", "black"] = "blur"
    """Fill for sources narrower than 9:16, which are padded instead of cropped."""
    crop_x: float = Field(default=0.5, ge=0.0, le=1.0)
    """Horizontal crop position for 9:16 (0 = left edge, 0.5 = centre, 1 = right edge)."""


class PipelineConfig(_Section):
    jobs_dir: Path = Path("jobs")
    ingest: IngestConfig = IngestConfig()
    transcribe: TranscribeConfig = TranscribeConfig()
    chunk: ChunkConfig = ChunkConfig()
    llm: LLMConfig = LLMConfig()
    refine: RefineConfig = RefineConfig()
    clip: ClipConfig = ClipConfig()
