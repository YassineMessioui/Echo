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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.ffmpeg`` tests when ffmpeg isn't installed (CI always has it)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not installed")
    for item in items:
        if "ffmpeg" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def make_video(tmp_path: Path):
    """Generate a small test video with ffmpeg's lavfi sources (no fixture files needed).

    ``make_video(duration=3.0, size="320x180", audio=True, fps=25, name="in.mp4")``
    returns the path. Video is a test pattern with a visible timestamp; audio is a sine.
    """
    from echo.ffmpeg import ffmpeg

    def _make(
        duration: float = 3.0,
        size: str = "320x180",
        audio: bool = True,
        fps: int = 25,
        name: str = "in.mp4",
    ) -> Path:
        out = tmp_path / name
        args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={duration}"]
        if audio:
            args += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={duration}",
            ]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps * 2)]
        if audio:
            args += ["-c:a", "aac", "-shortest"]
        ffmpeg([*args, str(out)])
        return out

    return _make
