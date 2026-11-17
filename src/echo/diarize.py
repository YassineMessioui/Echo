"""Speaker diarization (#22): ``audio.wav`` + ``transcript.json`` → ``speakers.json``.

Who speaks when, with sherpa-onnx: pyannote segmentation 3.0 finds speech turns, NeMo
TitaNet-small embeds each turn's voice, and clustering groups turns by speaker. It
runs offline on CPU (~5 min per hour of audio), with no Hugging Face token and no
PyTorch. The two models (~46 MB) download on first use into ``models_dir()``.

Turns are written in original-video time (shifted by ``Transcript.offset``, so a focus
range works) and relabelled ``S1``, ``S2``, ... in order of first appearance.

Optional: ``build_stages`` only includes this stage when ``DiarizeConfig.enabled``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import httpx
import numpy as np

from echo import progress
from echo.config import DiarizeConfig, PipelineConfig
from echo.errors import ArtifactError, TranscriptionError
from echo.interfaces import Diarizer
from echo.job import Job
from echo.models import Speakers, SpeakerTurn, StageName

if TYPE_CHECKING:
    import sherpa_onnx

log = logging.getLogger(__name__)

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = (
    f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
"""pyannote segmentation 3.0 (MIT), exported to ONNX by sherpa-onnx."""
EMBEDDING_URL = f"{_RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx"
EMBEDDING_MODEL = "nemo_en_titanet_small.onnx"
"""NVIDIA NeMo TitaNet-small speaker embeddings (CC-BY-4.0). Chosen over WeSpeaker
ResNet34 (8.2 s vs 12.6 s on the e2e fixture, same turns) and 3D-Speaker CAM++ (split
one voice into extra speakers)."""

MODEL_NAME = "sherpa-onnx/pyannote-3.0+titanet-small"
SAMPLE_RATE = 16_000


def models_dir() -> Path:
    """Where diarization models are cached: ``$ECHO_MODELS_DIR``, else ``~/.cache/echo/models``."""
    return Path(os.environ.get("ECHO_MODELS_DIR") or Path.home() / ".cache" / "echo" / "models")


def _download(url: str, out: Path) -> None:
    """Download ``url`` to ``out`` atomically (a failed download leaves nothing behind)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as f,
            httpx.stream("GET", url, follow_redirects=True, timeout=60) as r,
        ):
            r.raise_for_status()
            for chunk in r.iter_bytes():
                f.write(chunk)
        os.replace(tmp, out)
    except (httpx.HTTPError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise TranscriptionError(f"could not download diarization model {url}: {e}") from e


def ensure_models(directory: Path | None = None) -> tuple[Path, Path]:
    """Paths of the segmentation and embedding models, downloading any that are missing."""
    directory = directory or models_dir()
    segmentation, embedding = directory / SEGMENTATION_MODEL, directory / EMBEDDING_MODEL
    if not segmentation.is_file():
        log.warning("downloading speaker diarization models (~46 MB, once) to %s", directory)
        archive = directory / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive)
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            with tarfile.open(archive) as tar:
                tar.extractall(tmp, filter="data")
            target = directory / Path(SEGMENTATION_MODEL).parent
            shutil.rmtree(target, ignore_errors=True)
            os.replace(Path(tmp) / Path(SEGMENTATION_MODEL).parent, target)
        archive.unlink(missing_ok=True)
    if not embedding.is_file():
        _download(EMBEDDING_URL, embedding)
    return segmentation, embedding


def _load_wav(path: Path) -> np.ndarray:
    """The ingest stage's 16 kHz 16-bit PCM WAV as mono float32 samples in [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as wav:
            rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as e:
        raise TranscriptionError(f"not a PCM WAV file: {path} ({e})") from e
    if rate != SAMPLE_RATE or width != 2:
        raise TranscriptionError(f"{path}: expected 16 kHz 16-bit PCM (re-run the ingest stage)")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples


class SherpaDiarizer:
    """``Diarizer`` backed by sherpa-onnx. Models load on the first ``diarize`` call."""

    model_name = MODEL_NAME

    def __init__(self, config: DiarizeConfig, models: Path | None = None) -> None:
        self.config = config
        self._models = models
        self._engine: sherpa_onnx.OfflineSpeakerDiarization | None = None

    def _get(self) -> sherpa_onnx.OfflineSpeakerDiarization:
        if self._engine is None:
            import sherpa_onnx

            segmentation, embedding = ensure_models(self._models)
            threads = os.cpu_count() or 1
            engine_config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=str(segmentation)
                    ),
                    num_threads=threads,
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding), num_threads=threads
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=self.config.num_speakers or -1,
                    threshold=self.config.threshold,
                ),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not engine_config.validate():
                raise TranscriptionError("invalid speaker diarization model configuration")
            self._engine = sherpa_onnx.OfflineSpeakerDiarization(engine_config)
        return self._engine

    def diarize(self, audio: Path) -> list[SpeakerTurn]:
        samples = _load_wav(audio)
        engine = self._get()

        def on_chunk(done: int, total: int) -> int:
            progress.report(done / total if total else 1.0, f"chunk {done}/{total}")
            return 0  # 0 = keep going

        try:
            result = engine.process(samples, callback=on_chunk).sort_by_start_time()
        except RuntimeError as e:
            raise TranscriptionError(f"speaker diarization failed: {e}") from e
        return [
            SpeakerTurn(start=r.start, end=r.end, speaker=str(r.speaker))
            for r in result
            if r.end > r.start
        ]


def relabel(turns: list[SpeakerTurn], offset: float = 0.0) -> Speakers:
    """Shift turns by ``offset`` (into original-video time) and rename speakers ``S1``,
    ``S2``, ... in order of first appearance, so labels don't depend on the engine."""
    names: dict[str, str] = {}
    out = []
    for turn in sorted(turns, key=lambda t: t.start):
        label = names.setdefault(turn.speaker, f"S{len(names) + 1}")
        out.append(SpeakerTurn(start=turn.start + offset, end=turn.end + offset, speaker=label))
    return Speakers(turns=out)


@dataclass
class DiarizeStage:
    """Stage adapter: ``audio.wav`` (+ ``transcript.json`` for the offset) → ``speakers.json``."""

    diarizer: Diarizer

    name: ClassVar[StageName] = StageName.DIARIZE

    def outputs(self, job: Job) -> list[Path]:
        return [job.speakers_path]

    def settings(self, config: PipelineConfig) -> str:
        """Re-diarize when the engine or the clustering settings change."""
        d = config.diarize
        return f"{self.diarizer.model_name}|speakers={d.num_speakers or 'auto'}|t={d.threshold}"

    def run(self, job: Job, config: PipelineConfig) -> None:
        if not job.audio_path.is_file():
            raise ArtifactError(f"missing artifact: {job.audio_path}")
        offset = job.read_transcript().offset if job.transcript_path.is_file() else 0.0
        speakers = relabel(self.diarizer.diarize(job.audio_path), offset)
        job.write_speakers(speakers)
        meta = job.read_meta()
        meta.models["diarizer"] = self.diarizer.model_name
        job.write_meta(meta)
        progress.report(1.0, f"{len(speakers.labels)} speakers")
