"""Orchestrator: run the stages in order against one job directory.

This module is the contract between the orchestrator (#12), the CLI (#13) and the
end-to-end test (#14). The signatures and documented behaviour here are fixed; the
bodies are filled in by #12.

Behaviour of ``run_pipeline``:

- Stages run in ``StageName`` order (ingest → transcribe → llm → refine → clip).
- A stage is **skipped** when every path in ``stage.outputs(job)`` exists, unless it is
  forced. ``force=True`` forces every stage; ``from_stage=X`` forces X and every later
  stage (earlier ones are still skipped if done). A forced stage that has a ``force``
  dataclass field (``ClipStage``) is run with ``force=True``.
- ``job.json`` is updated around every stage: ``running`` with ``started_at``, then
  ``done`` / ``skipped`` / ``failed`` with ``finished_at`` and ``error``.
- An ``EchoError`` from a stage (from ``run()`` or ``outputs()``) marks it ``failed``,
  stops the run (later stages stay as they were) and is re-raised as ``PipelineError``.
  Any other exception, including ``KeyboardInterrupt``, is still recorded as ``failed``
  in ``job.json`` (``error="interrupted"`` for Ctrl+C, ``"<Type>: <message>"`` for bugs)
  and then propagates, so no stage is left ``running``.
- Re-running after a failure resumes at the failed stage, because the completed stages'
  outputs exist.
- A stage may define ``settings(config) -> str | None``, a fingerprint of the settings
  its outputs depend on. It's recorded in ``job.json`` (``StageRecord.settings``), and a
  stage whose fingerprint changed re-runs even though its outputs exist (and so forces
  the later stages). A failed re-run keeps the old fingerprint, so it's retried next
  time. A job recorded before a stage had settings keeps its outputs and adopts the
  current fingerprint, but only if that stage last finished (done or skipped).
- While a stage runs, its ``echo.progress.report(fraction, detail)`` calls become
  ``RUNNING`` events with ``progress`` / ``detail`` (non-decreasing, ending at 1.0).
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from echo import progress
from echo.clip import ClipStage
from echo.config import LLMConfig, PipelineConfig, TranscribeConfig
from echo.errors import EchoError
from echo.ingest import IngestStage
from echo.interfaces import Stage
from echo.job import Job
from echo.llm.chunking import WindowChunker
from echo.llm.client import make_llm_client
from echo.llm.merge import LLMStage
from echo.llm.prompts import PROMPT_VERSION
from echo.llm.topics import LLMTopicExtractor
from echo.models import (
    JobMeta,
    StageName,
    StageRecord,
    StageStatus,
    Topic,
    Transcript,
    TranscriptWindow,
)
from echo.refine import RefineStage
from echo.transcribe import TranscribeStage, WhisperTranscriber


@dataclass(frozen=True)
class StageEvent:
    """Progress notification sent to ``run_pipeline``'s ``on_event`` callback.

    A stage that runs sends ``RUNNING`` then ``DONE`` or ``FAILED``. A skipped stage
    sends only ``SKIPPED``, and one whose ``outputs()`` fails sends only ``FAILED``.
    ``elapsed`` (seconds) is set on ``DONE`` / ``FAILED``, ``error`` on ``FAILED``.

    In between, a running stage may send more ``RUNNING`` events with ``progress``
    (0..1, non-decreasing) and a short ``detail`` (``"12:30 / 58:10"``, ``"window
    4/15"``); the first ``RUNNING`` event has ``progress=None``.
    """

    stage: StageName
    status: StageStatus
    elapsed: float | None = None
    error: str | None = None
    progress: float | None = None
    detail: str = ""


class PipelineError(EchoError):
    """A stage failed with an expected error; ``job.json`` records it."""

    def __init__(self, stage: StageName, cause: EchoError) -> None:
        super().__init__(f"stage {stage.value!r} failed: {cause}")
        self.stage = stage
        self.cause = cause


def build_stages(config: PipelineConfig) -> list[Stage]:
    """The real stages for ``config``, in ``StageName`` order.

    Heavy resources (the whisper model, LLM client) must be created lazily, only when
    their stage actually runs, so a resumed or partly-skipped run doesn't load them.
    """
    extractor = _LazyTopicExtractor(config.llm)
    return [
        IngestStage(),
        TranscribeStage(_LazyTranscriber(config.transcribe)),
        LLMStage(WindowChunker(), extractor, models=_LLMModels(extractor)),
        RefineStage(),
        ClipStage(),
    ]


class _LazyTranscriber:
    """``Transcriber`` that creates the ``WhisperTranscriber`` on first use, so building
    the stages never touches faster-whisper."""

    def __init__(self, config: TranscribeConfig) -> None:
        self._config = config
        self._real: WhisperTranscriber | None = None

    def _get(self) -> WhisperTranscriber:
        if self._real is None:
            self._real = WhisperTranscriber(self._config)
        return self._real

    @property
    def model_name(self) -> str:
        return self._get().model_name

    def transcribe(self, audio: Path) -> Transcript:
        return self._get().transcribe(audio)


class _LazyTopicExtractor:
    """``TopicExtractor`` that builds the LLM client on first use. A hosted client needs
    its API key, which a run that skips the llm stage shouldn't require."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._real: LLMTopicExtractor | None = None
        self._lock = threading.Lock()  # windows are extracted from several threads (#56)

    def _get(self) -> LLMTopicExtractor:
        with self._lock:
            if self._real is None:
                self._real = LLMTopicExtractor(make_llm_client(self._config), self._config)
            return self._real

    @property
    def model_name(self) -> str:
        return self._get().client.model_name

    def extract(self, window: TranscriptWindow) -> list[Topic]:
        return self._get().extract(window)


class _LLMModels(Mapping[str, str]):
    """``LLMStage.models``, resolved when the stage records them (after extraction), so
    the client's ``model_name`` is known without building the client up front.

    Note: ``LLMStage.run``'s ``if self.models:`` calls ``__len__``, which builds the
    client; that's fine because it runs after extraction already built it.
    """

    def __init__(self, extractor: _LazyTopicExtractor) -> None:
        self._extractor = extractor

    def _models(self) -> dict[str, str]:
        return {"llm": self._extractor.model_name, "llm_prompt": PROMPT_VERSION}

    def __getitem__(self, key: str) -> str:
        return self._models()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models())

    def __len__(self) -> int:
        return len(self._models())


def run_pipeline(
    job: Job,
    stages: Sequence[Stage],
    config: PipelineConfig,
    *,
    force: bool = False,
    from_stage: StageName | None = None,
    on_event: Callable[[StageEvent], None] | None = None,
) -> JobMeta:
    """Run ``stages`` on ``job`` (see the module docstring) and return the final ``JobMeta``.

    Raises PipelineError when a stage fails with an EchoError.
    """
    order = list(StageName)
    first_forced = len(order) if from_stage is None else order.index(from_stage)

    def emit(event: StageEvent) -> None:
        if on_event is not None:
            on_event(event)

    for stage in sorted(stages, key=lambda s: order.index(s.name)):
        name = stage.name
        forced = force or order.index(name) >= first_forced
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        last = job.read_meta().stages[name]
        # The fingerprint of the outputs on disk. It's kept through RUNNING / FAILED, so a
        # re-run that fails can't make old outputs look current on the next run.
        kept = last.settings
        try:
            settings = _settings(stage, config)
            changed = settings is not None and (
                settings != kept
                if kept is not None
                # Unknown provenance: only a finished stage may adopt the current settings;
                # after a failed or interrupted run, its outputs can't be trusted.
                else last.status in (StageStatus.RUNNING, StageStatus.FAILED)
            )
            if not forced and not changed and all(p.exists() for p in stage.outputs(job)):
                _record_skipped(job, name, started, settings)
                emit(StageEvent(name, StageStatus.SKIPPED))
                continue
            if (forced or changed) and _has_force_field(stage):
                stage = dataclasses.replace(stage, force=True)
            running = StageRecord(status=StageStatus.RUNNING, started_at=started, settings=kept)
            _record(job, name, running)
            emit(StageEvent(name, StageStatus.RUNNING))
            with progress.capture(_progress_sink(name, emit)):
                stage.run(job, config)
        except EchoError as e:
            _fail(job, name, started, t0, str(e), emit, kept)
            raise PipelineError(name, e) from e
        except KeyboardInterrupt:
            _fail(job, name, started, t0, "interrupted", emit, kept)
            raise
        except BaseException as e:  # a bug: record it, then let it propagate
            error = f"{type(e).__name__}: {e}".removesuffix(": ")
            _fail(job, name, started, t0, error, emit, kept)
            raise
        done = StageRecord(
            status=StageStatus.DONE,
            started_at=started,
            finished_at=datetime.now(UTC),
            settings=settings,
        )
        _record(job, name, done)
        emit(StageEvent(name, StageStatus.DONE, elapsed=time.perf_counter() - t0))
        # This stage produced new outputs, so every later stage's outputs are stale.
        first_forced = min(first_forced, order.index(name) + 1)

    return job.read_meta()


def _settings(stage: Stage, config: PipelineConfig) -> str | None:
    """The stage's optional settings fingerprint (see the module docstring)."""
    fingerprint = getattr(stage, "settings", None)
    return fingerprint(config) if callable(fingerprint) else None


def _progress_sink(name: StageName, emit: Callable[[StageEvent], None]) -> progress.ProgressSink:
    """``echo.progress`` reports → RUNNING events, kept non-decreasing."""
    best = [0.0]

    def sink(fraction: float, detail: str) -> None:
        best[0] = max(best[0], fraction)
        emit(StageEvent(name, StageStatus.RUNNING, progress=best[0], detail=detail))

    return sink


def _has_force_field(stage: Stage) -> bool:
    return dataclasses.is_dataclass(stage) and any(
        f.name == "force" for f in dataclasses.fields(stage)
    )


def _record(job: Job, name: StageName, record: StageRecord) -> None:
    # Re-read first: stages write job.json themselves (``meta.models``).
    meta = job.read_meta()
    meta.stages[name] = record
    job.write_meta(meta)


def _record_skipped(job: Job, name: StageName, now: datetime, settings: str | None = None) -> None:
    """Mark ``name`` skipped, keeping the times of the run that produced its outputs.

    ``settings`` is kept from that run; a job recorded before the stage had settings
    adopts the current fingerprint.
    """
    previous = job.read_meta().stages[name]
    settings = previous.settings or settings
    if previous.finished_at is not None and previous.status in (
        StageStatus.DONE,
        StageStatus.SKIPPED,
    ):
        record = StageRecord(
            status=StageStatus.SKIPPED,
            started_at=previous.started_at,
            finished_at=previous.finished_at,
            settings=settings,
        )
    else:
        record = StageRecord(status=StageStatus.SKIPPED, finished_at=now, settings=settings)
    _record(job, name, record)


def _fail(
    job: Job,
    name: StageName,
    started: datetime,
    t0: float,
    error: str,
    emit: Callable[[StageEvent], None],
    settings: str | None = None,
) -> None:
    record = StageRecord(
        status=StageStatus.FAILED,
        started_at=started,
        finished_at=datetime.now(UTC),
        error=error,
        settings=settings,
    )
    _record(job, name, record)
    emit(StageEvent(name, StageStatus.FAILED, elapsed=time.perf_counter() - t0, error=error))
