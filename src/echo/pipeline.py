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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from echo.config import PipelineConfig
from echo.errors import EchoError
from echo.interfaces import Stage
from echo.job import Job
from echo.models import JobMeta, StageName, StageStatus


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
    raise NotImplementedError("#12")


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
    raise NotImplementedError("#12")
