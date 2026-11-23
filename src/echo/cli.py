"""``echo-clips`` command line (spec: ``docs/cli.md``).

``echo-clips run VIDEO`` builds a ``PipelineConfig`` from the options, creates (or
reopens) the job and hands it to ``run_pipeline``. Option defaults are read from
``PipelineConfig()`` so the CLI and the config can't drift.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, get_args

import typer
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from echo.clip import clip_paths
from echo.config import (
    CaptionConfig,
    ClipConfig,
    DiarizeConfig,
    IngestConfig,
    LLMConfig,
    PipelineConfig,
    RefineConfig,
    ScoreConfig,
    TranscribeConfig,
)
from echo.download import is_url, job_from_url
from echo.errors import EchoError, ExternalToolError
from echo.ffmpeg import has_filter
from echo.job import Job
from echo.models import ClipSpec, Score, StageName, StageStatus
from echo.pipeline import StageEvent, build_stages, run_pipeline
from echo.refine import scored_topics

_D = PipelineConfig()
_STAGES = [name for name in StageName if name is not StageName.DIARIZE]
"""Stage order for ``[n/N]`` when diarization (optional) isn't part of the run."""


def _choices(name: str, field: str) -> type[StrEnum]:
    """A StrEnum of a ``Literal`` config field's values, so Typer shows them as choices."""
    values = get_args(ClipConfig.model_fields[field].annotation)
    return StrEnum(name, {v: v for v in values})


Aspect = _choices("Aspect", "aspect")
Background = _choices("Background", "background")
_DEFAULT_ASPECT = Aspect(_D.clip.aspect)
_DEFAULT_BACKGROUND = Background(_D.clip.background)

app = typer.Typer(
    help="Turn a long video into short vertical clips.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _package_version() -> str:
    try:
        return version("echo")
    except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
        return "unknown"


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"echo-clips {_package_version()}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Turn a long video into short vertical clips."""
    # API keys (GROQ_API_KEY, MISTRAL_API_KEY, ...) can live in a git-ignored .env, found
    # from the current directory upwards. Variables already set in the shell win.
    load_dotenv(find_dotenv(usecwd=True), override=False)


QUIET_ENV = {
    # Hugging Face: the Windows "symlinks not supported" warning and the
    # "unauthenticated requests" notice look like errors on a first run (#65).
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_VERBOSITY": "error",
    # CTranslate2: its float16->float32 note on CPU (-1 = errors only).
    "CT2_VERBOSE": "-1",
}


def _quiet_model_libraries() -> None:
    """Quiet the model libraries for a normal (non --verbose) run. Values the user set
    win, and nothing has imported huggingface_hub/ctranslate2 yet (they read these at
    import), since faster-whisper is imported lazily by the transcribe stage."""
    for name, value in QUIET_ENV.items():
        os.environ.setdefault(name, value)


def _check_captions_supported(config: PipelineConfig) -> None:
    """With captions on, check for libass now rather than after transcription and the
    LLM have run. If ffmpeg itself is missing, the ingest stage reports that."""
    if not config.caption.enabled:
        return
    try:
        supported = has_filter("subtitles")
    except ExternalToolError:
        return
    if not supported:
        typer.echo(
            "error: this ffmpeg has no 'subtitles' filter (built without libass), so captions "
            "can't be burned in. Install an ffmpeg build with libass, or run with --no-captions.",
            err=True,
        )
        raise typer.Exit(1)


def _usage_error(message: str) -> typer.Exit:
    # Typer 0.29 vendors Click and doesn't export UsageError; exit 2 like one.
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(2)


def _normalise_point(point: str | None, *, is_end: bool) -> str | None:
    """One spelling per point, so the same span keeps the same ingest fingerprint:
    ``15.0%`` → ``15%``, ``0`` / ``0%`` start → none, ``100%`` end → none."""
    if point is None:
        return None
    pct = point.endswith("%")
    number = point.rstrip("%")
    try:
        value = float(number)
    except ValueError:
        return point  # malformed: IngestConfig reports it
    if (value == 0 and not is_end) or (pct and value == 100 and is_end):
        return None
    text = str(int(value)) if value.is_integer() else repr(value)
    return f"{text}%" if pct else text


def parse_range(value: str | None) -> dict[str, str | None]:
    """``--range START:END`` → ``IngestConfig`` ``start`` / ``end`` (either may be empty).

    Each side is seconds (``120``) or a percentage (``15%``); ``IngestConfig`` validates
    the forms. Two points of the same unit must be in order here; a mix of seconds and
    percent is checked by the ingest stage, once the duration is known.
    """
    if value is None:
        return {"start": None, "end": None}
    start, sep, end = value.strip().partition(":")
    start, end = start.strip() or None, end.strip() or None
    if not sep or ":" in (end or "") or (start is None and end is None):
        raise _usage_error(
            f"invalid --range {value!r}: expected START:END, e.g. 15%:85%, 120:3300 or :85%"
        )
    start, end = _normalise_point(start, is_end=False), _normalise_point(end, is_end=True)
    if start is None and end is None:
        return {"start": None, "end": None}  # e.g. 0%:100%: the whole video
    if start and end and start.endswith("%") == end.endswith("%"):
        try:
            inverted = float(start.rstrip("%")) >= float(end.rstrip("%"))
        except ValueError:
            inverted = False  # a malformed point: IngestConfig reports it
        if inverted:
            raise _usage_error(f"invalid --range {value!r}: START must be before END")
    return {"start": start, "end": end}


def _build_config(
    *,
    jobs_dir: Path,
    ingest: dict[str, Any] | None = None,
    transcribe: dict[str, Any],
    llm: dict[str, Any],
    refine: dict[str, Any],
    clip: dict[str, Any],
    caption: dict[str, Any],
    diarize: dict[str, Any] | None = None,
    score: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Overlay the CLI values on the defaults, validating every section.

    ``model_copy(update=...)`` skips validation, so each section is re-validated from
    its dumped defaults plus the overrides; a ``ValidationError`` is a usage error (exit 2).
    """
    try:
        return PipelineConfig.model_validate(
            _D.model_dump()
            | {
                "jobs_dir": jobs_dir,
                "ingest": IngestConfig.model_validate(_D.ingest.model_dump() | (ingest or {})),
                "transcribe": TranscribeConfig.model_validate(
                    _D.transcribe.model_dump() | transcribe
                ),
                "llm": LLMConfig.model_validate(_D.llm.model_dump() | llm),
                "refine": RefineConfig.model_validate(_D.refine.model_dump() | refine),
                "clip": ClipConfig.model_validate(_D.clip.model_dump() | clip),
                "caption": CaptionConfig.model_validate(_D.caption.model_dump() | caption),
                "diarize": DiarizeConfig.model_validate(_D.diarize.model_dump() | (diarize or {})),
                "score": ScoreConfig.model_validate(_D.score.model_dump() | (score or {})),
            }
        )
    except ValidationError as e:
        problems = "; ".join(_describe(err) for err in e.errors())
        # Typer 0.29 vendors Click and doesn't export UsageError; exit 2 like one.
        typer.echo(f"Error: invalid options: {problems}", err=True)
        raise typer.Exit(2) from e


def _describe(err: Any) -> str:
    loc = ".".join(str(p) for p in err["loc"])
    msg = err["msg"].removeprefix("Value error, ")
    return f"{loc}: {msg}" if loc else msg


def _index(stage: StageName, order: Sequence[StageName]) -> int:
    return order.index(stage) + 1 if stage in order else list(StageName).index(stage) + 1


def _progress_line(event: StageEvent, total: int, order: Sequence[StageName] = _STAGES) -> str:
    index = _index(event.stage, order)
    elapsed = f"{event.elapsed:.1f}s" if event.elapsed is not None else ""
    return f"[{index}/{total}] {event.stage.value:<11} {event.status.value:<9} {elapsed}".rstrip()


CLEAR_LINE = "\x1b[K"
REDRAW_INTERVAL = 0.1
"""Seconds between redraws of the running line on a terminal (at most ~10 per second)."""


def _running_line(
    event: StageEvent, total: int, elapsed: float, order: Sequence[StageName] = _STAGES
) -> str:
    """``[2/5] transcribe   42%  12:30 / 58:10  31.0s`` for a RUNNING event with progress."""
    index = _index(event.stage, order)
    percent = f"{(event.progress or 0.0) * 100:.0f}%"
    detail = f"{event.detail}  " if event.detail else ""
    return f"[{index}/{total}] {event.stage.value:<11} {percent:>4}  {detail}{elapsed:.1f}s"


def _tty_line(
    event: StageEvent,
    total: int,
    elapsed: float,
    width: int,
    order: Sequence[StageName] = _STAGES,
) -> str:
    """The running line redrawn in place: a carriage return, the line cut to the terminal width (a
    wrapped line can't be redrawn), then clear to the end of the line."""
    line = _running_line(event, total, elapsed, order)
    return "\r" + line[: max(width - 1, 1)] + CLEAR_LINE


class _StageOutput:
    """Turns ``StageEvent``s into the CLI's stage lines (#61).

    DONE / SKIPPED / FAILED print one final line per stage, as always. RUNNING events
    with ``progress`` show where a long stage is: on a terminal, one line redrawn in
    place (throttled); otherwise (piped, or ``--verbose``, whose log lines would break
    an in-place line) a plain line each time the stage passes another 10%.
    """

    def __init__(
        self,
        *,
        tty: bool,
        order: Sequence[StageName] = tuple(_STAGES),
        clock: Callable[[], float] = time.monotonic,
        width: Callable[[], int] = lambda: shutil.get_terminal_size().columns,
    ) -> None:
        self._tty = tty
        self._order = list(order)
        self._total = len(self._order)
        self._clock = clock
        self._width = width
        self._started: dict[StageName, float] = {}
        self._steps: dict[StageName, int] = {}
        self._last_draw = -REDRAW_INTERVAL
        self._drawn = False

    def __call__(self, event: StageEvent) -> None:
        now = self._clock()
        if event.status is StageStatus.RUNNING:
            started = self._started.setdefault(event.stage, now)
            if event.progress is not None:
                self._show(event, now - started, now)
            return
        if self._drawn:
            typer.echo("\r" + CLEAR_LINE, nl=False, color=True)
            self._drawn = False
        typer.echo(_progress_line(event, self._total, self._order))

    def _show(self, event: StageEvent, elapsed: float, now: float) -> None:
        if self._tty:
            if now - self._last_draw < REDRAW_INTERVAL:
                return
            self._last_draw = now
            # color=True: keep the clear-line code (only ever written to a terminal).
            line = _tty_line(event, self._total, elapsed, self._width(), self._order)
            typer.echo(line, nl=False, color=True)
            self._drawn = True
            return
        step = int((event.progress or 0.0) * 10)
        if step > self._steps.get(event.stage, 0):
            self._steps[event.stage] = step
            typer.echo(_running_line(event, self._total, elapsed))


def _is_terminal() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


class _DownloadProgress:
    """Prints ``[download]  40%`` lines in 20% steps (per stream: video, then audio)."""

    def __init__(self) -> None:
        self._last = -1

    def __call__(self, fraction: float) -> None:
        step = int(fraction * 100) // 20 * 20
        if step < self._last:  # a new stream started
            self._last = -1
        if step > self._last:
            self._last = step
            typer.echo(f"[download] {step:3d}%")


@app.command()
def run(
    video: Annotated[
        str,
        typer.Argument(
            help="The source video: a local file, or a link (YouTube etc.) to download.",
            show_default=False,
        ),
    ],
    job_id: Annotated[
        str | None,
        typer.Option(
            help="Job id (default: random 12-char hex). Reusing an id resumes that job.",
            show_default=False,
        ),
    ] = None,
    jobs_dir: Annotated[
        Path, typer.Option(help="Directory holding the job directories.")
    ] = _D.jobs_dir,
    force: Annotated[bool, typer.Option("--force", help="Re-run every stage.")] = False,
    from_stage: Annotated[
        StageName | None,
        typer.Option(help="Re-run this stage and every later one.", show_default=False),
    ] = None,
    focus_range: Annotated[
        str | None,
        typer.Option(
            "--range",
            metavar="START:END",
            help="Only process this part of the video, in seconds or percent "
            "(15%:85%, 120:3300, :85%). Clips keep original-video times.",
            show_default=False,
        ),
    ] = None,
    whisper_model: Annotated[
        str, typer.Option(help="faster-whisper model size.")
    ] = _D.transcribe.model_size,
    language: Annotated[
        str | None,
        typer.Option(help="Spoken language code (default: auto-detect).", show_default=False),
    ] = _D.transcribe.language,
    llm_provider: Annotated[
        str, typer.Option(help="LLM provider (see docs/llm-providers.md).")
    ] = _D.llm.provider,
    llm_model: Annotated[str, typer.Option(help="LLM model name.")] = _D.llm.model,
    llm_concurrency: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=16,
            help="Transcript windows sent to the LLM at once "
            "(default: 4 for hosted providers, 1 for Ollama).",
            show_default=False,
        ),
    ] = _D.llm.max_concurrency,
    min_duration: Annotated[
        float, typer.Option(help="Shortest clip, in seconds.")
    ] = _D.refine.min_duration,
    max_duration: Annotated[
        float, typer.Option(help="Longest clip, in seconds (60 is the platform cap).")
    ] = _D.refine.max_duration,
    max_clips: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Keep at most this many clips: the best scored (or, with --no-score, "
            "those closest to ~37 s).",
        ),
    ] = _D.refine.max_clips,
    score: Annotated[
        bool,
        typer.Option(
            "--score/--no-score",
            help="Rate each clip for hook, payoff, clarity and energy with the LLM "
            "(one extra call), to rank clips and pick the best for --max-clips.",
        ),
    ] = _D.score.enabled,
    aspect: Annotated[
        Aspect, typer.Option(help="Output framing: 9:16 vertical, or keep the source framing.")
    ] = _DEFAULT_ASPECT,
    crop_x: Annotated[
        float, typer.Option(help="Horizontal crop position for 9:16 (0 = left, 1 = right).")
    ] = _D.clip.crop_x,
    background: Annotated[
        Background, typer.Option(help="Fill for sources narrower than 9:16.")
    ] = _DEFAULT_BACKGROUND,
    captions: Annotated[
        bool, typer.Option("--captions/--no-captions", help="Burn word-highlighted captions in.")
    ] = _D.caption.enabled,
    speakers: Annotated[
        bool,
        typer.Option(
            "--speakers/--no-speakers",
            help="Detect who speaks when (speaker diarization, ~5 min per hour of audio); "
            "captions get a colour per speaker.",
        ),
    ] = _D.diarize.enabled,
    num_speakers: Annotated[
        int | None,
        typer.Option(
            min=1, help="How many speakers there are (default: estimate).", show_default=False
        ),
    ] = _D.diarize.num_speakers,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug logging and full tracebacks.")
    ] = False,
) -> None:
    """Run the pipeline on VIDEO, resuming any stages already done."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        _quiet_model_libraries()

    config = _build_config(
        jobs_dir=jobs_dir,
        ingest=parse_range(focus_range),
        transcribe={"model_size": whisper_model, "language": language},
        llm={"provider": llm_provider, "model": llm_model, "max_concurrency": llm_concurrency},
        refine={
            "min_duration": min_duration,
            "max_duration": max_duration,
            "max_clips": max_clips,
        },
        clip={"aspect": aspect.value, "crop_x": crop_x, "background": background.value},
        caption={"enabled": captions},
        diarize={"enabled": speakers, "num_speakers": num_speakers},
        score={"enabled": score},
    )
    _check_captions_supported(config)

    stages = build_stages(config)
    on_event = _StageOutput(tty=_is_terminal() and not verbose, order=[s.name for s in stages])

    try:
        if is_url(video):
            job = job_from_url(
                config.jobs_dir, video, job_id=job_id, on_progress=_DownloadProgress()
            )
        else:
            job = Job.create(config.jobs_dir, Path(video), job_id=job_id)
        run_pipeline(
            job,
            stages,
            config,
            force=force,
            from_stage=from_stage,
            on_event=on_event,
        )
        clips = job.read_clips() if job.clips_path.is_file() else []
        scores = _clip_scores(job, config) if clips else {}
    except EchoError as e:
        if verbose:
            typer.echo(traceback.format_exc(), err=True, nl=False)
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        # run_pipeline has already recorded the stage as failed ("interrupted").
        typer.echo("interrupted", err=True)
        raise typer.Exit(130) from None

    typer.echo(f"job: {job.root}")
    _print_clips(clips, clip_paths(job, clips), scores)


def _clip_scores(job: Job, config: PipelineConfig) -> dict[str, Score]:
    """Each clip's score by topic title, from the topics refine used (#21); empty when
    scoring is off or didn't run."""
    if not config.score.enabled or not job.topics_path.is_file():
        return {}
    return {t.title: t.score for t in scored_topics(job, config) if t.score is not None}


def _print_clips(clips: list[ClipSpec], paths: list[Path], scores: dict[str, Score]) -> None:
    """The clip paths; with scores, best first, each with its score and reason."""
    if not clips:
        typer.echo("no clips")
        return
    if not scores:
        typer.echo(f"clips ({len(clips)}):")
        for path in paths:
            typer.echo(f"  {path}")
        return
    typer.echo(f"clips ({len(clips)}, best first):")
    rows = sorted(
        zip(clips, paths, strict=True),
        key=lambda row: -(s.overall if (s := scores.get(row[0].topic_title)) else -1.0),
    )
    for clip, path in rows:
        score = scores.get(clip.topic_title)
        typer.echo(f"  {score.overall:4.1f}  {path}" if score else f"     -  {path}")
        if score and score.reason:
            typer.echo(f"        {score.reason}")
