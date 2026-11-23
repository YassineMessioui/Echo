"""Pipeline configuration. Each layer reads only its own section."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


_RANGE_POINT = r"^(\d+(\.\d+)?%?)$"
"""Seconds (``"120"``, ``"90.5"``) or a percentage of the video (``"15%"``)."""


class IngestConfig(_Section):
    # The transcribe stage only accepts 16 kHz mono 16-bit PCM (``transcribe.load_wav``);
    # other values only make sense once another speech model needs them.
    sample_rate: int = 16_000
    channels: int = 1
    start: str | None = Field(default=None, pattern=_RANGE_POINT)
    """Focus range (#59): only process from here (seconds or ``"15%"``); None = start."""
    end: str | None = Field(default=None, pattern=_RANGE_POINT)
    """Focus range end (seconds or ``"85%"``); None = end of the video."""

    @model_validator(mode="after")
    def _percent_range(self) -> "IngestConfig":
        for value in (self.start, self.end):
            if value is not None and value.endswith("%") and float(value[:-1]) > 100:
                raise ValueError(f"focus range {value!r} is over 100%")
        # Same-unit pairs can be checked here; mixed units (seconds vs %) need the
        # video's duration, so ingest checks those.
        if self.start is not None and self.end is not None:
            if self.start.endswith("%") == self.end.endswith("%"):
                if float(self.start.rstrip("%")) >= float(self.end.rstrip("%")):
                    raise ValueError(f"focus range {self.start}:{self.end} is empty or inverted")
        return self


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


AUTO_MAX_TOKENS = 3000
"""What ``ChunkConfig.max_tokens="auto"`` resolves to when nothing more specific applies."""


class ChunkConfig(_Section):
    max_tokens: Annotated[int, Field(gt=0)] | Literal["auto"] = "auto"
    """Window budget in tokens. ``"auto"`` picks one for the model
    (``echo.llm.chunking.resolve_max_tokens``, #63): 12k for long-context hosted models,
    otherwise ``AUTO_MAX_TOKENS``."""
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
    """Keep at most this many clips (None = all), in chronological order: the highest
    scored ones when topics have scores (#21), else the ones closest to ~37 s."""

    @model_validator(mode="after")
    def _range(self) -> "RefineConfig":
        if self.max_duration < self.min_duration:
            raise ValueError("max_duration < min_duration")
        return self


class ScoreConfig(_Section):
    """Clip scoring (#21): the LLM rates each topic's clip text for hook, payoff,
    standalone clarity and energy, so ``RefineConfig.max_clips`` keeps the best ones."""

    enabled: bool = True
    """Off: no score stage, and ``max_clips`` keeps the clips closest to ~37 s."""


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
    reframe: Literal["centre", "face", "speaker"] = "centre"
    """How the 9:16 crop window is placed on sources wide enough to be cropped (#16):

    - ``"centre"``: fixed at ``crop_x`` for the whole clip.
    - ``"face"``: on the faces in each shot (camera cuts are detected), falling back to
      ``crop_x`` in shots without a face.
    - ``"speaker"``: like ``"face"``, but in shots where the faces don't all fit, the
      window follows whoever speaks (needs diarization, ``DiarizeConfig.enabled``;
      otherwise it behaves like ``"face"``).

    Ignored with ``aspect="source"`` and for sources that are padded instead of cropped."""


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
    speaker_colours: list[HexColour] = Field(
        default_factory=lambda: ["#FFFFFF", "#7FDBFF", "#B8F2A6", "#FFB3D9"], min_length=1
    )
    """Caption colour per speaker (#68), in order of first appearance; reused cyclically.
    Only applies when diarization (``DiarizeConfig.enabled``) found two or more speakers;
    otherwise captions use ``colour``."""
    speaker_tags: bool = False
    """Prefix a caption line with its speaker's label when the speaker changes (#68)."""


class DiarizeConfig(_Section):
    """Speaker diarization (#22): who speaks when. Off by default: it adds ~5 min of CPU
    per hour of audio."""

    enabled: bool = False
    num_speakers: int | None = Field(default=None, ge=1)
    """None = estimate the number of speakers."""
    threshold: float = Field(default=0.5, gt=0, lt=2)
    """Clustering distance when estimating: lower splits into more speakers."""


class PipelineConfig(_Section):
    jobs_dir: Path = Path("jobs")
    ingest: IngestConfig = IngestConfig()
    transcribe: TranscribeConfig = TranscribeConfig()
    chunk: ChunkConfig = ChunkConfig()
    llm: LLMConfig = LLMConfig()
    score: ScoreConfig = ScoreConfig()
    refine: RefineConfig = RefineConfig()
    clip: ClipConfig = ClipConfig()
    caption: CaptionConfig = CaptionConfig()
    diarize: DiarizeConfig = DiarizeConfig()
