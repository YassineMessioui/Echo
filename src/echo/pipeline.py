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
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    """

    stage: StageName
    status: StageStatus
    elapsed: float | None = None
    error: str | None = None


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

    def _get(self) -> LLMTopicExtractor:
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
    the client's ``model_name`` is known without building the client up front."""

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
        try:
            if not forced and all(path.exists() for path in stage.outputs(job)):
                _record(job, name, StageRecord(status=StageStatus.SKIPPED, finished_at=started))
                emit(StageEvent(name, StageStatus.SKIPPED))
                continue
            if forced and _has_force_field(stage):
                stage = dataclasses.replace(stage, force=True)
            _record(job, name, StageRecord(status=StageStatus.RUNNING, started_at=started))
            emit(StageEvent(name, StageStatus.RUNNING))
            stage.run(job, config)
        except EchoError as e:
            _fail(job, name, started, t0, str(e), emit)
            raise PipelineError(name, e) from e
        except KeyboardInterrupt:
            _fail(job, name, started, t0, "interrupted", emit)
            raise
        except BaseException as e:  # a bug: record it, then let it propagate
            _fail(job, name, started, t0, f"{type(e).__name__}: {e}".removesuffix(": "), emit)
            raise
        done = StageRecord(
            status=StageStatus.DONE, started_at=started, finished_at=datetime.now(UTC)
        )
        _record(job, name, done)
        emit(StageEvent(name, StageStatus.DONE, elapsed=time.perf_counter() - t0))

    return job.read_meta()


def _has_force_field(stage: Stage) -> bool:
    return dataclasses.is_dataclass(stage) and any(
        f.name == "force" for f in dataclasses.fields(stage)
    )


def _record(job: Job, name: StageName, record: StageRecord) -> None:
    # Re-read first: stages write job.json themselves (``meta.models``).
    meta = job.read_meta()
    meta.stages[name] = record
    job.write_meta(meta)


def _fail(
    job: Job,
    name: StageName,
    started: datetime,
    t0: float,
    error: str,
    emit: Callable[[StageEvent], None],
) -> None:
    record = StageRecord(
        status=StageStatus.FAILED, started_at=started, finished_at=datetime.now(UTC), error=error
    )
    _record(job, name, record)
    emit(StageEvent(name, StageStatus.FAILED, elapsed=time.perf_counter() - t0, error=error))
