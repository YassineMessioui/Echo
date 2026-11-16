"""Shared fixtures.

``golden_job`` is a read-only, mutually consistent set of artifacts (transcript →
topics → clips) for one fake episode. Use it as the input to the layer you're
building instead of waiting for the upstream layer.
"""

import shutil
from pathlib import Path

import pytest

from echo.job import Job

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


@pytest.fixture
def golden_job() -> Job:
    return Job(GOLDEN_DIR)


@pytest.fixture
def job_copy(tmp_path: Path) -> Job:
    """A writable copy of the golden job, for tests that write artifacts."""
    root = tmp_path / "golden"
    shutil.copytree(GOLDEN_DIR, root)
    return Job(root)
