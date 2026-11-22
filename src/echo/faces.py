"""Face detection for reframing (#16): YuNet, run with onnxruntime.

YuNet (``face_detection_yunet_2023mar.onnx``, 230 KB, MIT licence, from OpenCV's
opencv_zoo) is a small anchor-free face detector. It runs here with ``onnxruntime``,
which faster-whisper already installs, so face reframing adds no dependency: no OpenCV.
The model downloads once into ``echo.diarize.models_dir()`` (``~/.cache/echo/models``),
atomically, and is checked against its SHA-256.

The decoding below follows OpenCV's ``FaceDetectorYN`` (``face_detect.cpp``): the
model takes a 640x640 BGR image (0-255, no normalisation) and outputs, for each stride
(8, 16, 32), a class score, an objectness score, a box and five landmarks per grid cell.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx
import numpy as np

from echo.diarize import models_dir
from echo.errors import ExternalToolError

if TYPE_CHECKING:
    import onnxruntime

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_NAME = "yunet-2023mar"

INPUT_SIZE = 640
"""The ONNX model's fixed input: 640x640. Frames are padded (bottom/right) to it."""
STRIDES = (8, 16, 32)
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3


@dataclass(frozen=True)
class Face:
    """A detected face, in fractions of the frame (0..1): centre, size, confidence."""

    cx: float
    cy: float
    w: float
    h: float
    score: float = 1.0
    mouth: tuple[float, float] | None = None
    """Centre of the two mouth-corner landmarks, when the detector gives them."""

    @property
    def left(self) -> float:
        return self.cx - self.w / 2

    @property
    def right(self) -> float:
        return self.cx + self.w / 2


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Face]:
        """Faces in ``frame`` (H x W x 3 BGR uint8, at most 640 x 640).

        Raises ExternalToolError if the model can't be loaded.
        """
        ...


def ensure_model(directory: Path | None = None) -> Path:
    """Path of the YuNet model, downloading it (atomically, SHA-256 checked) if missing."""
    path = (directory or models_dir()) / MODEL_FILE
    if path.is_file():
        return path
    log.info("downloading the face detection model (230 KB, once) to %s", path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with (
            os.fdopen(fd, "wb") as f,
            httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as r,
        ):
            r.raise_for_status()
            for chunk in r.iter_bytes():
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ExternalToolError(f"face detection model from {MODEL_URL} has the wrong hash")
        os.replace(tmp, path)
    except (httpx.HTTPError, OSError) as e:
        raise ExternalToolError(
            f"could not download the face detection model {MODEL_URL}: {e} "
            "(or reframe with --reframe centre)"
        ) from e
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Indices kept by greedy non-maximum suppression (boxes as x1, y1, x2, y2)."""
    order = np.argsort(-scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode(
    outputs: dict[str, np.ndarray],
    width: int,
    height: int,
    *,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[Face]:
    """Faces from YuNet's raw outputs, for a ``width`` x ``height`` frame padded to 640."""
    boxes, scores, mouths = [], [], []
    for stride in STRIDES:
        cols = INPUT_SIZE // stride
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0, 1)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0, 1)
        score = np.sqrt(cls * obj)
        idx = np.nonzero(score >= score_threshold)[0]
        if not idx.size:
            continue
        row, col = (idx // cols).astype(np.float32), (idx % cols).astype(np.float32)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)[idx]
        kps = outputs[f"kps_{stride}"].reshape(-1, 10)[idx]
        cx, cy = (col + bbox[:, 0]) * stride, (row + bbox[:, 1]) * stride
        w, h = np.exp(bbox[:, 2]) * stride, np.exp(bbox[:, 3]) * stride
        boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
        scores.append(score[idx])
        # Landmarks: right eye, left eye, nose tip, right / left mouth corner.
        mx = ((kps[:, 6] + col) + (kps[:, 8] + col)) * stride / 2
        my = ((kps[:, 7] + row) + (kps[:, 9] + row)) * stride / 2
        mouths.append(np.stack([mx, my], axis=1))
    if not boxes:
        return []
    all_boxes, all_scores = np.concatenate(boxes), np.concatenate(scores)
    all_mouths = np.concatenate(mouths)
    faces = []
    for i in _nms(all_boxes, all_scores, NMS_THRESHOLD):
        x1, y1, x2, y2 = (float(v) for v in all_boxes[i])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (0 <= cx < width and 0 <= cy < height):
            continue  # in the padding
        faces.append(
            Face(
                cx=cx / width,
                cy=cy / height,
                w=(x2 - x1) / width,
                h=(y2 - y1) / height,
                score=float(all_scores[i]),
                mouth=(float(all_mouths[i, 0]) / width, float(all_mouths[i, 1]) / height),
            )
        )
    return faces


class YuNetDetector:
    """``FaceDetector`` backed by YuNet on onnxruntime. The model loads on first use."""

    model_name = MODEL_NAME

    def __init__(self, model: Path | None = None) -> None:
        self._model = model
        self._session: onnxruntime.InferenceSession | None = None

    def _get(self) -> onnxruntime.InferenceSession:
        if self._session is None:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3  # errors only
            try:
                self._session = onnxruntime.InferenceSession(
                    str(self._model or ensure_model()),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as e:  # onnxruntime raises its own (non-Echo) errors
                raise ExternalToolError(f"could not load the face detection model: {e}") from e
        return self._session

    def detect(self, frame: np.ndarray) -> list[Face]:
        height, width = frame.shape[:2]
        if width > INPUT_SIZE or height > INPUT_SIZE:
            raise ValueError(f"frame {width}x{height} is larger than {INPUT_SIZE}x{INPUT_SIZE}")
        padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
        padded[:height, :width] = frame
        blob = padded.transpose(2, 0, 1)[np.newaxis]
        session = self._get()
        names = [o.name for o in session.get_outputs()]
        values = session.run(names, {session.get_inputs()[0].name: blob})
        return decode(dict(zip(names, values, strict=True)), width, height)


@functools.cache
def default_detector() -> YuNetDetector:
    """One shared detector, so its model loads once per process."""
    return YuNetDetector()
