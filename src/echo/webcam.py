"""Streamer webcam overlays (#126): find the webcam in a stream clip, keep it in a corner.

A stream is gameplay with the streamer's webcam pasted on top, usually small and in a
corner. The 9:16 crop keeps the middle of the gameplay and cuts the webcam, and with it
the streamer's reactions. So, for each clip:

1. **Sample a few moments** (``sample_frames``): ``SAMPLES`` single frames spread over
   the clip, each from a fast seek (no decoding of everything in between, so it's cheap
   on 4K and remote sources), scaled to 640 wide. Being seconds apart, gameplay changes
   a lot between them, while the webcam's room stays put.
2. **Find the webcam** (``find_webcam``): a small face (YuNet, ``echo.faces``) in the
   same spot in most samples, while the picture around it changes. A game character's
   face moves with the camera and the cuts, so it doesn't qualify; a still face on a
   still picture is a talking-head video, not an overlay.
3. **Find its box:** the webcam is the still region around that face, inside moving
   gameplay. The box grows from the face until the pixels start changing from sample to
   sample. Without a clear boundary, a typical webcam shape around the face is used.
4. **Compose** (``overlay_filter``): the clip is framed as usual, and the webcam is
   cropped from the source, scaled to ``ClipConfig.webcam_size`` of the clip's width,
   given a thin border and laid over ``ClipConfig.webcam_corner``. One encode, and the
   captions are burned in after it.

A stream's webcam stays in one place, so once a clip's webcam is known it's reused for
the job's other clips whose still face sits inside it (e.g. a calm menu scene, where the
"is the gameplay moving?" check alone would fail).

Boxes are fractions of the frame (0..1) until ``overlay_filter`` turns them into pixels.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo.config import ClipConfig
from echo.faces import Face, FaceDetector
from echo.ffmpeg import ffmpeg
from echo.framing import sample_size
from echo.reframe import TARGET

log = logging.getLogger(__name__)

SAMPLES = 8
"""Frames sampled per clip."""
PARALLEL_SEEKS = 4
"""Seeks run at once (each is a short ffmpeg run; remote ones mostly wait on the network)."""
MIN_FRAMES = 4
"""Fewer sampled frames than this: too little evidence, no webcam."""
MIN_PRESENCE = 0.6
"""The webcam face must be in at least this share of the samples..."""
MAX_SPREAD = 0.02
"""...within this distance (fraction of the frame) of its usual spot..."""
MAX_FACE_WIDTH = 0.2
"""...and narrower than this share of the frame (a webcam is a small overlay)."""
MIN_BACKGROUND_CHANGE = 8.0
"""Mean change (0-255) outside the webcam between samples: below it, the picture is
still (a talking-head video, or a menu), so the webcam isn't confirmed by this clip."""

MOVING = 15.0
"""A pixel whose brightness varies more than this (std over the samples) is moving."""
BOUNDARY = 0.5
"""A column or row of the search band is outside the webcam once this share of it moves."""
SEARCH = (1.4, 1.1, 6.0)
"""Where the webcam's edges are searched for, in face widths from the face's centre:
from 1.4 sideways and 1.1 up or down (a webcam is never tighter than that around a face,
and a streamer moving right next to their face must not look like the boundary), out to 6."""
TYPICAL_WIDTH = 3.4
"""Webcam width in face widths when an edge can't be found (a face fills ~30 % of a
typical webcam's width)."""
TYPICAL_ASPECT = 4 / 3

MARGIN = 36
"""Pixels between the webcam and the clip's sides."""
TOP = 150
"""Top corners: pixels from the top (clear of the platforms' top bar)."""
BOTTOM = 600
"""Bottom corners: pixels from the bottom (above the captions, 420 px up)."""
BORDER = 4


@dataclass(frozen=True)
class Box:
    """A rectangle in fractions of the frame."""

    x: float
    y: float
    w: float
    h: float

    def contains(self, cx: float, cy: float) -> bool:
        return self.x <= cx <= self.x + self.w and self.y <= cy <= self.y + self.h


InputAt = Callable[[float], list[str]]
"""ffmpeg input arguments that open the video with a fast seek to a time (seconds)."""


def sample_frames(
    input_at: InputAt, start: float, duration: float, width: int, height: int, workdir: Path
) -> list[np.ndarray]:
    """``SAMPLES`` frames spread over ``[start, start + duration]``, 640 wide, BGR.

    ``width`` / ``height`` are the displayed frame size, so the frames have its shape.
    """
    sw, sh = sample_size(width, height)
    times = [start + duration * (k + 0.5) / SAMPLES for k in range(SAMPLES)]

    def grab(k: int) -> np.ndarray | None:
        out = workdir / f"sample{k}.raw"
        ffmpeg(
            [*input_at(times[k]), "-frames:v", "1", "-an", "-sn",
             "-vf", f"scale={sw}:{sh},format=bgr24", "-f", "rawvideo", str(out)]
        )  # fmt: skip
        data = np.fromfile(out, dtype=np.uint8)
        return data[: sw * sh * 3].reshape(sh, sw, 3) if data.size >= sw * sh * 3 else None

    with ThreadPoolExecutor(PARALLEL_SEEKS) as pool:
        frames = list(pool.map(grab, range(SAMPLES)))
    return [f for f in frames if f is not None]


def still_face(detections: Sequence[Sequence[Face]]) -> Face | None:
    """The small face that stays put in most samples, if any (its median position)."""
    clusters: list[dict[int, Face]] = []  # sample index -> that sample's face in the spot
    for k, faces in enumerate(detections):
        for face in faces:
            if face.w > MAX_FACE_WIDTH:
                continue
            for cluster in clusters:
                ref = next(iter(cluster.values()))
                if abs(ref.cx - face.cx) <= MAX_SPREAD and abs(ref.cy - face.cy) <= MAX_SPREAD:
                    cluster.setdefault(k, face)
                    break
            else:
                clusters.append({k: face})
    found = max(clusters, key=len, default=None)
    if found is None or len(found) < MIN_PRESENCE * len(detections):
        return None
    faces = list(found.values())
    return Face(
        cx=float(np.median([f.cx for f in faces])),
        cy=float(np.median([f.cy for f in faces])),
        w=float(np.median([f.w for f in faces])),
        h=float(np.median([f.h for f in faces])),
    )


def _grey(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([f.astype(np.float32).mean(axis=2) for f in frames])


def _boundary(moving_share: np.ndarray, start: int, stop: int) -> int | None:
    """First index from ``start`` towards ``stop`` (either way) where two neighbours in a
    row are mostly moving: the webcam's edge. None if there's none in range."""
    step = 1 if stop >= start else -1
    n = len(moving_share)
    for i in range(start, stop, step):
        j = i + step
        if 0 <= i < n and 0 <= j < n and moving_share[i] >= BOUNDARY <= moving_share[j]:
            return i
    return None


def webcam_box(face: Face, frames: Sequence[np.ndarray]) -> Box:
    """The still region around ``face`` inside moving surroundings (fractions), or a
    typical webcam shape around it on any side whose edge can't be found."""
    h, w = frames[0].shape[:2]
    fx, fy, fw = face.cx * w, face.cy * h, max(face.w * w, 4.0)
    moving = _grey(frames).std(axis=0) > MOVING
    side, vertical, far = (k * fw for k in SEARCH)
    rows = slice(max(0, int(fy - fw)), min(h, int(fy + fw) + 1))
    cols = slice(max(0, int(fx - fw)), min(w, int(fx + fw) + 1))
    by_col = moving[rows].mean(axis=0)
    by_row = moving[:, cols].mean(axis=1)
    left = _boundary(by_col, int(fx - side), int(fx - far) - 1)
    right = _boundary(by_col, int(fx + side), int(fx + far) + 1)
    top = _boundary(by_row, int(fy - vertical), int(fy - far) - 1)
    bottom = _boundary(by_row, int(fy + vertical), int(fy + far) + 1)
    tw = TYPICAL_WIDTH * fw
    th = tw / TYPICAL_ASPECT
    x0 = left + 1 if left is not None else fx - tw / 2
    x1 = right if right is not None else fx + tw / 2
    y0 = top + 1 if top is not None else fy - th * 0.45
    y1 = bottom if bottom is not None else fy + th * 0.55
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    return Box(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def _background_change(frames: Sequence[np.ndarray], box: Box) -> float:
    """Mean brightness change between consecutive samples, outside ``box``."""
    grey = _grey(frames)
    _, h, w = grey.shape
    outside = np.ones((h, w), dtype=bool)
    outside[
        int(box.y * h) : int((box.y + box.h) * h), int(box.x * w) : int((box.x + box.w) * w)
    ] = False
    return float(np.abs(np.diff(grey, axis=0))[:, outside].mean())


def find_webcam(
    frames: Sequence[np.ndarray], detector: FaceDetector, known: Box | None = None
) -> Box | None:
    """The webcam overlay in these samples of one clip, or None.

    ``known`` is a webcam already found in another clip of the same video: a still face
    inside it confirms it, even when this clip's picture is calm.
    """
    if len(frames) < MIN_FRAMES:
        return None
    face = still_face([detector.detect(f) for f in frames])
    if face is None:
        return None
    if known is not None and known.contains(face.cx, face.cy):
        return known
    box = webcam_box(face, frames)
    if _background_change(frames, box) < MIN_BACKGROUND_CHANGE:
        return None
    return box


def detect_webcam(
    input_at: InputAt,
    start: float,
    duration: float,
    width: int,
    height: int,
    detector: FaceDetector,
    known: Box | None = None,
) -> Box | None:
    """Sample one clip and find its webcam overlay (``find_webcam``)."""
    with tempfile.TemporaryDirectory(prefix="echo-webcam-") as tmp:
        frames = sample_frames(input_at, start, duration, width, height, Path(tmp))
    box = find_webcam(frames, detector, known)
    log.info("webcam: %s (%d samples)", box, len(frames))
    return box


class ExcludingDetector:
    """A ``FaceDetector`` that ignores faces inside ``box``: with a webcam overlay, face
    framing follows the game, not the streamer's face in the overlay."""

    def __init__(self, detector: FaceDetector, box: Box) -> None:
        self._detector, self._box = detector, box

    def detect(self, frame: np.ndarray) -> list[Face]:
        return [f for f in self._detector.detect(frame) if not self._box.contains(f.cx, f.cy)]


def _even(n: float) -> int:
    return max(2, int(round(n)) // 2 * 2)


def overlay_filter(framing: str, box: Box, width: int, height: int, config: ClipConfig) -> str:
    """``-vf`` graph: the clip framed by ``framing`` (a 1080x1920 filter chain), with the
    webcam ``box`` of the ``width`` x ``height`` source laid over a corner."""
    cw, ch = _even(box.w * width), _even(box.h * height)
    cx = min(max(0, _even(box.x * width)), width - cw)
    cy = min(max(0, _even(box.y * height)), height - ch)
    ow = _even(TARGET[0] * config.webcam_size)
    oh = _even(ow * ch / cw)
    fw, fh = ow + 2 * BORDER, oh + 2 * BORDER
    left = "left" in config.webcam_corner
    x = MARGIN if left else TARGET[0] - fw - MARGIN
    y = TOP if config.webcam_corner.startswith("top") else TARGET[1] - fh - BOTTOM
    return (
        f"split=2[echo_bg][echo_cam];[echo_bg]{framing}[echo_frame];"
        f"[echo_cam]crop={cw}:{ch}:{cx}:{cy},scale={ow}:{oh}:flags=lanczos,"
        f"pad={fw}:{fh}:{BORDER}:{BORDER}:white,setsar=1[echo_webcam];"
        f"[echo_frame][echo_webcam]overlay={x}:{y}"
    )
