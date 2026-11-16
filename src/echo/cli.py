"""``echo-clips`` command line (spec: ``docs/cli.md``).

``echo-clips run VIDEO`` builds a ``PipelineConfig`` from the options, creates (or
reopens) the job and hands it to ``run_pipeline``. Option defaults are read from
``PipelineConfig()`` so the CLI and the config can't drift.
"""

from __future__ import annotations

import logging
import traceback
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
    LLMConfig,
    PipelineConfig,
    RefineConfig,
    TranscribeConfig,
)
from echo.errors import EchoError
from echo.job import Job
from echo.models import StageName, StageStatus
from echo.pipeline import StageEvent, build_stages, run_pipeline

_D = PipelineConfig()
_STAGES = list(StageName)


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


def _build_config(
    *,
    jobs_dir: Path,
    transcribe: dict[str, Any],
    llm: dict[str, Any],
    refine: dict[str, Any],
    clip: dict[str, Any],
    caption: dict[str, Any],
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
                "transcribe": TranscribeConfig.model_validate(
                    _D.transcribe.model_dump() | transcribe
                ),
                "llm": LLMConfig.model_validate(_D.llm.model_dump() | llm),
                "refine": RefineConfig.model_validate(_D.refine.model_dump() | refine),
                "clip": ClipConfig.model_validate(_D.clip.model_dump() | clip),
                "caption": CaptionConfig.model_validate(_D.caption.model_dump() | caption),
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


def _progress_line(event: StageEvent, total: int) -> str:
    index = _STAGES.index(event.stage) + 1
    elapsed = f"{event.elapsed:.1f}s" if event.elapsed is not None else ""
    return f"[{index}/{total}] {event.stage.value:<11} {event.status.value:<9} {elapsed}".rstrip()


@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="The source video.", show_default=False)],
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
    min_duration: Annotated[
        float, typer.Option(help="Shortest clip, in seconds.")
    ] = _D.refine.min_duration,
    max_duration: Annotated[
        float, typer.Option(help="Longest clip, in seconds (60 is the platform cap).")
    ] = _D.refine.max_duration,
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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Debug logging and full tracebacks.")
    ] = False,
) -> None:
    """Run the pipeline on VIDEO, resuming any stages already done."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    config = _build_config(
        jobs_dir=jobs_dir,
        transcribe={"model_size": whisper_model, "language": language},
        llm={"provider": llm_provider, "model": llm_model},
        refine={"min_duration": min_duration, "max_duration": max_duration},
        clip={"aspect": aspect.value, "crop_x": crop_x, "background": background.value},
        caption={"enabled": captions},
    )

    def on_event(event: StageEvent) -> None:
        if event.status is not StageStatus.RUNNING:
            typer.echo(_progress_line(event, len(_STAGES)))

    try:
        job = Job.create(config.jobs_dir, video, job_id=job_id)
        run_pipeline(
            job,
            build_stages(config),
            config,
            force=force,
            from_stage=from_stage,
            on_event=on_event,
        )
        paths = clip_paths(job, job.read_clips()) if job.clips_path.is_file() else []
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
    if paths:
        typer.echo(f"clips ({len(paths)}):")
        for path in paths:
            typer.echo(f"  {path}")
    else:
        typer.echo("no clips")
