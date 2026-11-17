"""Progress inside a running stage (#61).

Stages call ``report(fraction, detail)`` while they work; ``run_pipeline`` turns that
into ``StageEvent(status=RUNNING, progress=..., detail=...)`` for the CLI. Outside a
pipeline run (tests, library use) it does nothing.

The sink lives in a ``ContextVar``, so it follows the stage's thread but not worker
threads: report from the thread that called ``Stage.run`` (e.g. as futures complete).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ProgressSink = Callable[[float, str], None]

_sink: ContextVar[ProgressSink | None] = ContextVar("echo_progress_sink", default=None)


def report(fraction: float, detail: str = "") -> None:
    """Report that the current stage is ``fraction`` (0..1) done, e.g. ``"window 4/15"``."""
    if (sink := _sink.get()) is not None:
        sink(min(1.0, max(0.0, fraction)), detail)


@contextmanager
def capture(sink: ProgressSink) -> Iterator[None]:
    """Send ``report`` calls made inside the block to ``sink``."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
