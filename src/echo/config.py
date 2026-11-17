"""Pipeline configuration. Each layer reads only its own section."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IngestConfig(_Section):
    # The transcribe stage only accepts 16 kHz mono 16-bit PCM (``transcribe.load_wav``);
    # other values only make sense once another speech model needs them.
    sample_rate: int = 16_000
    channels: int = 1


class TranscribeConfig(_Section):
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    language: str | None = Field(default=None, description="None = auto-detect.")
    batch_size: int = Field(
        default=0,
        ge=0,
        description="Decode this many ~30 s chunks at once (faster-whisper's batched "
        "pipeline, best on a GPU); 0 = the sequential decoder, more accurate on a CPU (#58).",
    )
    vad_filter: bool = Field(
        default=True,
        description="Skip silence and music with Silero VAD before decoding (new default "
        "in #58; set false for the earlier transcripts).",
    )
    vad_min_silence_ms: int | None = Field(
        default=None,
        ge=0,
        description="Silence (ms) that ends a speech chunk; only used with vad_filter. "
        "None = faster-whisper's default for the mode (160 ms batched, 2000 ms sequential).",
    )


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
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Transcript windows sent to the LLM at once (#56); 1 = one after "
        "another. None = a per-provider default (see ``echo.llm.merge.concurrency``).",
    )


class RefineConfig(_Section):
    min_duration: float = Field(default=15.0, gt=0)
    max_duration: float = Field(default=60.0, gt=0)
    padding: float = Field(default=0.2, ge=0)
    max_clips: int | None = Field(default=None, ge=1)
    """Keep at most this many clips (None = all). Until clip scoring (#21) exists, the
    ones closest to ~37 s are kept, in chronological order."""

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
    max_workers: int | None = Field(default=None, ge=1, le=16)
    """Clips encoded at once (#57); each ffmpeg gets a share of the cores. None =
    ``min(4, cpu_count // 2)`` (see ``echo.clip.clip_workers``). It doesn't change the
    output, so it isn't part of the clip manifest's fingerprint."""


HexColour = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


class CaptionConfig(_Section):
    """Burned-in captions (#17): a few words at a time, the spoken word highlighted.

    Sizes are in pixels of the 1080x1920 output; captions scale with other sizes.
    """

    enabled: bool = True
    font: str = "Montserrat ExtraBold"
    """Bundled with Echo (echo/fonts, SIL OFL), so captions look the same everywhere;
    any installed system font name works too."""
    font_size: int = Field(default=84, gt=0)
    colour: HexColour = "#FFFFFF"
    highlight_colour: HexColour = "#FFD400"
    outline_colour: HexColour = "#000000"
    outline: int = Field(default=6, ge=0)
    margin_bottom: int = Field(default=420, ge=0)
    """Distance from the bottom edge; the default keeps clear of TikTok/Reels buttons."""
    max_words: int = Field(default=3, ge=1)
    """Words on screen at once."""
    max_gap: float = Field(default=0.6, ge=0)
    """A pause longer than this (seconds) starts a new caption line."""
    uppercase: bool = True


class PipelineConfig(_Section):
    jobs_dir: Path = Path("jobs")
    ingest: IngestConfig = IngestConfig()
    transcribe: TranscribeConfig = TranscribeConfig()
    chunk: ChunkConfig = ChunkConfig()
    llm: LLMConfig = LLMConfig()
    refine: RefineConfig = RefineConfig()
    clip: ClipConfig = ClipConfig()
    caption: CaptionConfig = CaptionConfig()
