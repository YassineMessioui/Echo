"""End to end: the real stages on a real ~2 min podcast, through ``run_pipeline``.

    uv run pytest -m model tests/test_e2e.py

Runs real ffmpeg and faster-whisper (``tiny``, downloaded on first use). The LLM is a
``FakeLLMClient`` that reads the numbered transcript lines from each prompt and answers
with the segments that fall inside each topic of ``tests/fixtures/podcast/sections.json``
(the spoken topics' time spans), so its segment ids are always valid for whatever
whisper produced.

``test_live_pipeline`` swaps in a real LLM; it is skipped unless ``ECHO_LIVE_LLM`` is
set, as in ``tests/test_llm_live.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echo.config import CaptionConfig, IngestConfig, LLMConfig, PipelineConfig, TranscribeConfig
from echo.ffmpeg import ffmpeg, probe
from echo.job import Job
from echo.llm.prompts import PROMPT_VERSION
from echo.llm.topics import LLMTopicExtractor
from echo.models import StageName, StageStatus
from echo.pipeline import StageEvent, build_stages, run_pipeline
from echo.testing import FakeLLMClient

pytestmark = [pytest.mark.model, pytest.mark.ffmpeg]

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "podcast"
SECTIONS = json.loads((FIXTURE_DIR / "sections.json").read_text(encoding="utf-8"))["sections"]

# One transcript line, as rendered by ``echo.llm.chunking.format_segment``.
_LINE = re.compile(r"^\[(\d+)\] (\d+):(\d\d):(\d\d)–(\d+):(\d\d):(\d\d) ", re.MULTILINE)


class SectionLLM(FakeLLMClient):
    """A ``FakeLLMClient`` whose answer is computed from the prompt it receives.

    Each segment is assigned to the section containing its midpoint; every section
    with segments becomes one topic spanning its first to last segment id.
    """

    def __init__(self, sections: list[dict[str, Any]]) -> None:
        super().__init__([])
        self.sections = sections

    def complete(
        self, system: str, prompt: str, *, json_schema: dict[str, Any] | None = None
    ) -> str:
        self._responses.append(self.answer(prompt))
        return super().complete(system, prompt, json_schema=json_schema)

    def answer(self, prompt: str) -> str:
        ids: dict[str, list[int]] = {}
        for m in _LINE.finditer(prompt):
            h1, m1, s1, h2, m2, s2 = (int(g) for g in m.groups()[1:])
            # Timestamps are truncated to whole seconds, so the end is up to 1 s later.
            mid = (h1 * 3600 + m1 * 60 + s1 + h2 * 3600 + m2 * 60 + s2 + 1) / 2
            section = min(self.sections, key=lambda s: _distance(mid, s))
            ids.setdefault(section["title"], []).append(int(m.group(1)))
        topics = [
            {
                "title": s["title"],
                "summary": f"The hosts talk about {s['title'].lower()}.",
                "start_segment": min(ids[s["title"]]),
                "end_segment": max(ids[s["title"]]),
            }
            for s in self.sections
            if s["title"] in ids
        ]
        return json.dumps({"topics": topics})


def _distance(t: float, section: dict[str, Any]) -> float:
    return max(section["start"] - t, t - section["end"], 0.0)


def _config(tmp_path: Path, **llm: Any) -> PipelineConfig:
    return PipelineConfig(
        jobs_dir=tmp_path / "jobs",
        transcribe=TranscribeConfig(
            model_size=os.environ.get("ECHO_TEST_WHISPER", "tiny"), device="cpu"
        ),
        llm=LLMConfig(**llm),
    )


def _job(tmp_path: Path, config: PipelineConfig) -> Job:
    video = tmp_path / "podcast.mp4"
    shutil.copyfile(FIXTURE_DIR / "podcast.mp4", video)
    return Job.create(config.jobs_dir, video, job_id="e2e")


def _with_llm_client(stages: list, client: FakeLLMClient, config: PipelineConfig) -> list:
    """``stages`` with the LLM stage's extractor rebuilt around ``client``; its chunker,
    merger and every other stage stay as ``build_stages`` made them."""
    models = {"llm": client.model_name, "llm_prompt": PROMPT_VERSION}

    def swap(stage):
        if stage.name != StageName.LLM:
            return stage
        return replace(stage, extractor=LLMTopicExtractor(client, config.llm), models=models)

    return [swap(stage) for stage in stages]


def _check_artifacts(job: Job) -> None:
    """Every artifact exists, validates through ``Job`` and every clip is a 9:16 video
    whose length matches ``clips.json``."""
    for path in (job.audio_path, job.transcript_path, job.topics_path, job.clips_path):
        assert path.is_file(), f"missing {path.name}"

    audio = probe(job.audio_path)
    assert audio.has_audio and audio.sample_rate == 16_000 and audio.channels == 1

    transcript = job.read_transcript()
    assert transcript.segments, "whisper found no speech in the fixture"
    assert [s.id for s in transcript.segments] == list(range(len(transcript.segments)))
    assert transcript.duration == pytest.approx(SECTIONS[-1]["end"], abs=1.0)

    for topic in job.read_topics():
        assert topic.end_segment < len(transcript.segments)

    clips = job.read_clips()
    assert clips, "no clip survived refinement"
    for i, clip in enumerate(clips, start=1):
        path = job.clip_video_path(i, clip.topic_title)
        assert path.is_file(), f"missing {path.name}"
        info = probe(path)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio
        assert info.duration <= 60.0 + 0.1
        assert info.duration == pytest.approx(clip.duration, abs=0.1)
        assert 15.0 <= clip.duration <= 60.0


def _check_resume(job: Job, stages: list, config: PipelineConfig) -> None:
    """A second run skips every stage (one SKIPPED event each, nothing runs) and records
    every stage as skipped in ``job.json``."""
    events: list[StageEvent] = []
    meta = run_pipeline(job, stages, config, on_event=events.append)

    ran = [stage.name for stage in stages]  # diarize is optional and off here
    assert [(e.stage, e.status) for e in events] == [(name, StageStatus.SKIPPED) for name in ran]
    for record in (meta, job.read_meta()):
        for name in ran:
            assert record.stages[name].status == StageStatus.SKIPPED
            assert record.stages[name].finished_at is not None
            assert record.stages[name].error is None


def test_pipeline_end_to_end(tmp_path: Path):
    config = _config(tmp_path)
    job = _job(tmp_path, config)
    client = SectionLLM(SECTIONS)
    stages = _with_llm_client(build_stages(config), client, config)

    meta = run_pipeline(job, stages, config)

    ran = [stage.name for stage in stages]
    assert StageName.DIARIZE not in ran, "diarization is off by default"
    assert {name: meta.stages[name].status for name in ran} == {
        name: StageStatus.DONE for name in ran
    }
    assert meta.stages[StageName.DIARIZE].status == StageStatus.PENDING
    assert job.read_meta() == meta
    assert meta.models["whisper"] == f"faster-whisper/{config.transcribe.model_size}"
    assert meta.models["llm"] == client.model_name
    assert client.calls, "the LLM stage never called the client"

    _check_artifacts(job)
    topics = job.read_topics()
    # The fake answers one topic per spoken section, and each is long enough to clip.
    assert [t.title for t in topics] == [s["title"] for s in SECTIONS]
    assert [c.topic_title for c in job.read_clips()] == [s["title"] for s in SECTIONS]

    calls = len(client.calls)
    _check_resume(job, stages, config)
    assert len(client.calls) == calls, "a skipped LLM stage must not call the client"


def test_focus_range_clips_stay_inside_it_in_original_time(tmp_path: Path):
    """#59 acceptance: with ``--range 30%:70%`` only that span is transcribed, and the
    clips lie inside it, timed (and cut) in original-video time."""
    # Captions off: they don't depend on the range, and this runs without libass.
    config = _config(tmp_path).model_copy(
        update={
            "ingest": IngestConfig(start="30%", end="70%"),
            "caption": CaptionConfig(enabled=False),
        }
    )
    job = _job(tmp_path, config)
    stages = _with_llm_client(build_stages(config), SectionLLM(SECTIONS), config)

    run_pipeline(job, stages, config)

    duration = probe(job.input_video).duration
    lo, hi = round(0.3 * duration, 3), round(0.7 * duration, 3)  # recorded to the ms
    assert job.read_meta().models["ingest_range"] == f"{lo:.3f}:{hi:.3f}"
    assert probe(job.audio_path).duration == pytest.approx(hi - lo, abs=0.1)

    transcript = job.read_transcript()
    assert transcript.offset == pytest.approx(lo, abs=0.001)
    assert transcript.duration == pytest.approx(hi, abs=0.1)
    assert all(lo <= s.start and s.end <= hi + 0.1 for s in transcript.segments)
    # Timestamps are in original time: the bicycle section starts at 35.4 s in the video.
    bikes = next(s for s in transcript.segments if "bicycle" in s.text.lower())
    assert bikes.start == pytest.approx(SECTIONS[1]["start"], abs=1.5)

    clips = job.read_clips()
    # Only the bicycle section lies (almost) wholly inside 30–70 %; the coffee section's
    # tail and the ocean section's head are too short to clip.
    assert [c.topic_title for c in clips] == [SECTIONS[1]["title"]]
    for i, clip in enumerate(clips, start=1):
        assert lo <= clip.start < clip.end <= hi
        assert clip.start == pytest.approx(SECTIONS[1]["start"], abs=1.5)
        assert clip.end == pytest.approx(SECTIONS[1]["end"], abs=1.5)
        path = job.clip_video_path(i, clip.topic_title)
        assert probe(path).duration == pytest.approx(clip.duration, abs=0.1)
        # Cut from the original at clip.start: its loudness envelope matches the
        # original's there, and not a few seconds earlier.
        clip_env = _envelope(path, 0.0, 10.0, tmp_path / "a.wav")
        same = _envelope(job.input_video, clip.start, 10.0, tmp_path / "b.wav")
        shifted = _envelope(job.input_video, clip.start - 3.0, 10.0, tmp_path / "c.wav")
        assert np.corrcoef(clip_env, same)[0, 1] > 0.9
        assert np.corrcoef(clip_env, shifted)[0, 1] < 0.6


def _envelope(media: Path, start: float, seconds: float, wav: Path) -> np.ndarray:
    """RMS loudness of ``media`` in 50 ms frames, from ``start`` for ``seconds``."""
    ffmpeg(
        ["-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(media)]
        + ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    )
    with wave.open(str(wav), "rb") as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    frames = samples[: len(samples) // 800 * 800].reshape(-1, 800)
    return np.sqrt((frames**2).mean(axis=1))


LIVE_PROVIDER = os.environ.get("ECHO_LIVE_LLM")


@pytest.mark.skipif(not LIVE_PROVIDER, reason="set ECHO_LIVE_LLM=<provider> to run")
def test_live_pipeline(tmp_path: Path):
    """Fully live: the stages exactly as ``build_stages`` makes them, with a real LLM.

    ECHO_LIVE_LLM=ollama uv run pytest -m model tests/test_e2e.py
    """
    overrides = {"provider": LIVE_PROVIDER}
    if model := os.environ.get("ECHO_LIVE_LLM_MODEL"):
        overrides["model"] = model
    config = _config(tmp_path, **overrides)
    job = _job(tmp_path, config)
    stages = build_stages(config)

    run_pipeline(job, stages, config)

    _check_artifacts(job)
    _check_resume(job, stages, config)
