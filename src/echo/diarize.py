"""Speaker diarization (#22): ``audio.wav`` + ``transcript.json`` → ``speakers.json``.

Who speaks when, with sherpa-onnx: pyannote segmentation 3.0 finds speech turns, NeMo
TitaNet-small embeds each turn's voice, and clustering groups turns by speaker. It
runs offline on CPU (~5 min per hour of audio), with no Hugging Face token and no
PyTorch. The two models (~46 MB) download on first use into ``models_dir()``.

When the number of speakers is estimated (no ``num_speakers``), one clustering pass
over-splits long audio (#119: 171 "speakers" on a 68-min two-person podcast), so two
clean-up steps follow: ``merge_similar_speakers`` merges speakers whose mean voices are
close, and ``fold_minor_speakers`` gives speakers with a tiny share of the speech
(laughter, crosstalk, "mm-hm") to the nearest main speaker in time. An exact
``num_speakers`` skips both.

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
VOICE_SAMPLE = 60.0
"""Seconds of each speaker's longest turns embedded for the second clustering pass."""
MIN_EMBED_SAMPLES = SAMPLE_RATE // 4
"""Shorter turns are zero-padded to this length before embedding."""


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
        self._extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None

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
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(engine_config.embedding)
        return self._engine

    def _voices(self, samples: np.ndarray, turns: list[SpeakerTurn]) -> dict[str, np.ndarray]:
        """Each speaker's mean voice for ``merge_similar_speakers``: the duration-weighted
        sum of the unit embeddings of its longest turns, up to ``VOICE_SAMPLE`` seconds
        (so the cost stays small on long audio)."""
        assert self._extractor is not None
        by_speaker: dict[str, list[SpeakerTurn]] = {}
        for turn in turns:
            by_speaker.setdefault(turn.speaker, []).append(turn)
        voices: dict[str, np.ndarray] = {}
        for speaker, own in by_speaker.items():
            total = 0.0
            for turn in sorted(own, key=lambda t: t.start - t.end):  # longest first
                if total >= VOICE_SAMPLE:
                    break
                chunk = samples[int(turn.start * SAMPLE_RATE) : int(turn.end * SAMPLE_RATE)]
                if len(chunk) < MIN_EMBED_SAMPLES:
                    chunk = np.pad(chunk, (0, MIN_EMBED_SAMPLES - len(chunk)))
                stream = self._extractor.create_stream()
                stream.accept_waveform(SAMPLE_RATE, chunk)
                stream.input_finished()
                vector = np.asarray(self._extractor.compute(stream), dtype=np.float64)
                norm = np.linalg.norm(vector)
                if norm:
                    duration = turn.end - turn.start
                    voices[speaker] = voices.get(speaker, 0.0) + vector / norm * duration
                    total += duration
        return voices

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
        turns = [
            SpeakerTurn(start=r.start, end=r.end, speaker=str(r.speaker))
            for r in result
            if r.end > r.start
        ]
        if self.config.num_speakers:  # an exact count: keep what the engine found
            return turns
        if self.config.merge_threshold > 0:
            voices = self._voices(samples, turns)
            turns = merge_similar_speakers(turns, voices, self.config.merge_threshold)
        return fold_minor_speakers(turns, self.config.min_speaker_share)


def speech_per_speaker(turns: list[SpeakerTurn]) -> dict[str, float]:
    """Seconds of speech per speaker, in order of first appearance."""
    speech: dict[str, float] = {}
    for turn in sorted(turns, key=lambda t: t.start):
        speech[turn.speaker] = speech.get(turn.speaker, 0.0) + turn.end - turn.start
    return speech


def merge_similar_speakers(
    turns: list[SpeakerTurn], voices: dict[str, np.ndarray], max_distance: float
) -> list[SpeakerTurn]:
    """Second clustering pass: merge speakers whose mean voices are close.

    ``voices`` maps each speaker to the duration-weighted sum of its turns' (unit)
    embeddings. The closest pair (cosine distance) is merged while it is under
    ``max_distance``; a merged voice is the sum of both. Ties go to the pair that appears
    first, and a merged speaker keeps the label of the one that spoke first. Speakers
    without a voice are left alone; ``max_distance <= 0`` changes nothing."""
    order = [s for s in speech_per_speaker(turns) if s in voices]
    if max_distance <= 0 or len(order) < 2:
        return turns
    sums = {s: np.asarray(voices[s], dtype=np.float64) for s in order}
    into = {s: s for s in order}
    while len(order) > 1:
        unit = np.stack([sums[s] / (np.linalg.norm(sums[s]) or 1.0) for s in order])
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -np.inf)
        i, j = divmod(int(np.argmax(similarity)), len(order))  # first pair wins a tie
        if 1.0 - similarity[i, j] >= max_distance:
            break
        keep, drop = order[min(i, j)], order[max(i, j)]
        sums[keep] = sums[keep] + sums.pop(drop)
        order.remove(drop)
        for s, target in into.items():
            if target == drop:
                into[s] = keep
    return [t.model_copy(update={"speaker": into.get(t.speaker, t.speaker)}) for t in turns]


def fold_minor_speakers(turns: list[SpeakerTurn], min_share: float) -> list[SpeakerTurn]:
    """Give the turns of speakers holding under ``min_share`` of all speech (laughter,
    crosstalk, "mm-hm" split off as their own speaker) to the nearest real speaker in
    time: the main speaker whose turn is closest (overlap = 0 s; a tie goes to the
    earlier turn). The speaker with the most speech always stays. Same-speaker turns that
    then overlap or touch are joined. ``min_share <= 0`` changes nothing."""
    speech = speech_per_speaker(turns)
    total = sum(speech.values())
    if min_share <= 0 or not total:
        return turns
    top = max(speech, key=lambda s: speech[s])  # first appearance wins a tie
    main = {s for s, secs in speech.items() if secs / total >= min_share} | {top}
    if len(main) == len(speech):
        return turns
    ordered = sorted(turns, key=lambda t: t.start)
    anchors = [t for t in ordered if t.speaker in main]

    def nearest(turn: SpeakerTurn) -> str:
        def gap(other: SpeakerTurn) -> float:
            return max(0.0, other.start - turn.end, turn.start - other.end)

        return min(anchors, key=lambda a: (gap(a), a.start)).speaker

    folded = [
        t if t.speaker in main else t.model_copy(update={"speaker": nearest(t)}) for t in ordered
    ]
    out: list[SpeakerTurn] = []
    last: dict[str, int] = {}  # speaker -> index of its latest turn in ``out``
    for turn in folded:
        i = last.get(turn.speaker)
        if i is not None and turn.start <= out[i].end:
            out[i] = out[i].model_copy(update={"end": max(out[i].end, turn.end)})
            continue
        last[turn.speaker] = len(out)
        out.append(turn)
    return out


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
        """Re-diarize when the engine or the clustering settings change. An exact
        ``num_speakers`` ignores the estimation settings, so they're left out then."""
        d = config.diarize
        if d.num_speakers:
            return f"{self.diarizer.model_name}|speakers={d.num_speakers}"
        return (
            f"{self.diarizer.model_name}|speakers=auto|t={d.threshold}"
            f"|merge={d.merge_threshold}|min_share={d.min_speaker_share}"
        )

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
