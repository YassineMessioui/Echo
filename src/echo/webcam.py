"""Streamer webcam overlays (#126): find the webcam in a stream clip, keep it in a corner.

A stream is gameplay with the streamer's webcam pasted on top, usually small and in a
corner. The 9:16 crop keeps the middle of the gameplay and cuts the webcam, and with it
the streamer's reactions. So, for each clip:

1. **Sample cheaply:** only the clip's keyframes are decoded (``-skip_frame nokey``),
   scaled to 640 wide: a handful of frames, even from a 4K source.
2. **Find the webcam:** a face (YuNet, ``echo.faces``) that sits in the same small spot
   in most frames, while the picture around it changes. A game character's face moves
   with the camera and the cuts, so it doesn't qualify.
3. **Find its box:** overlays have straight, still edges, while gameplay edges move. The
   time-averaged edge map makes the overlay's border stand out, so the box is grown from
   the face to the strongest still edges around it. Without clear edges, a typical
   webcam shape around the face is used.
4. **Compose (``overlay_filter``):** the clip is framed as usual, and the webcam is
   cropped from the source, scaled to ``ClipConfig.webcam_size`` of the clip's width,
   given a thin border and laid over ``ClipConfig.webcam_corner``. One encode, and the
   captions are burned in after it.

Boxes are fractions of the frame (0..1) until ``overlay_filter`` turns them into pixels.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo.config import ClipConfig
from echo.faces import Face, FaceDetector
from echo.ffmpeg import ffmpeg
from echo.framing import sample_size
from echo.reframe import TARGET

log = logging.getLogger(__name__)

MIN_FRAMES = 4
"""Fewer sampled frames than this: too little evidence, no webcam."""
MIN_PRESENCE = 0.6
"""The webcam face must be in at least this share of the frames..."""
MAX_SPREAD = 0.02
"""...within this distance (fraction of the frame) of its usual spot..."""
MAX_FACE_WIDTH = 0.2
"""...and smaller than this share of the frame's width (a webcam is a small overlay)."""
MIN_BACKGROUND_CHANGE = 6.0
"""Mean pixel change (0-255) outside the webcam between frames: below it, the whole
picture is still (a talking-head video, not a stream), so there's no overlay to keep."""

TYPICAL_WIDTH = 3.4
"""Webcam width in face widths when its edges can't be found (a face fills ~30 % of a
typical webcam's width)."""
TYPICAL_ASPECT = 4 / 3
EDGE_SEARCH = (1.2, 5.0)
"""How far from the face an edge is searched for, in face widths (nearest, farthest)."""
MIN_EDGE_RATIO = 3.0
"""A still edge must be this many times stronger than the typical still edge."""

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


def sample_keyframes(
    input_args: list[str], duration: float, width: int, height: int, workdir: Path
) -> list[np.ndarray]:
    """The clip's keyframes, 640 wide, as BGR arrays (``input_args`` seek to its start).

    Decoding only keyframes is what keeps this cheap on 4K sources. If the clip has too
    few (long GOPs), one frame per second is decoded instead.
    """
    sw, sh = sample_size(width, height)
    out = workdir / "keyframes.raw"
    scale = f"scale={sw}:{sh},format=bgr24"
    t = ["-t", f"{duration:.3f}"]
    ffmpeg(["-skip_frame", "nokey", *input_args, "-an", "-sn", *t, "-vf", scale,
            "-fps_mode", "passthrough", "-f", "rawvideo", str(out)])  # fmt: skip
    frames = _read(out, sw, sh)
    if len(frames) < MIN_FRAMES:
        ffmpeg([*input_args, "-an", "-sn", *t, "-vf", f"fps=1,{scale}",
                "-f", "rawvideo", str(out)])  # fmt: skip
        frames = _read(out, sw, sh)
    return frames


def _read(path: Path, w: int, h: int) -> list[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    size = w * h * 3
    return list(data[: data.size // size * size].reshape(-1, h, w, 3))


def _still_face(detections: Sequence[Sequence[Face]]) -> Face | None:
    """The face that stays put in most frames, if any (the median of its detections)."""
    clusters: list[dict[int, Face]] = []  # frame index -> that frame's face in the spot
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
    best = list(found.values())
    return Face(
        cx=float(np.median([f.cx for f in best])),
        cy=float(np.median([f.cy for f in best])),
        w=float(np.median([f.w for f in best])),
        h=float(np.median([f.h for f in best])),
    )


def _still_edges(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged |d/dx| and |d/dy| of the frames' brightness: still edges (overlay
    borders, HUD) keep their strength, moving gameplay edges average out."""
    grey = np.stack([f.astype(np.float32).mean(axis=2) for f in frames])
    gx = np.abs(np.diff(grey, axis=2)).mean(axis=0)  # vertical edges, H x (W-1)
    gy = np.abs(np.diff(grey, axis=1)).mean(axis=0)  # horizontal edges, (H-1) x W
    return gx, gy


def _edge(profile: np.ndarray, lo: int, hi: int, typical: float) -> int | None:
    """Index of the strongest edge in ``profile[lo:hi]``, if clearly above ``typical``."""
    lo, hi = max(lo, 0), min(hi, len(profile))
    if hi - lo < 1:
        return None
    i = lo + int(np.argmax(profile[lo:hi]))
    return i if profile[i] >= MIN_EDGE_RATIO * max(typical, 1e-3) else None


def _box(face: Face, frames: Sequence[np.ndarray]) -> Box:
    """The webcam's box around ``face``: the still edges nearest around it, else a
    typical webcam shape."""
    h, w = frames[0].shape[:2]
    fx, fy, fw = face.cx * w, face.cy * h, max(face.w * w, 4.0)
    gx, gy = _still_edges(frames)
    near, far = (round(k * fw) for k in EDGE_SEARCH)
    rows = slice(max(0, int(fy - fw)), min(h, int(fy + fw)))
    cols = slice(max(0, int(fx - fw)), min(w, int(fx + fw)))
    vertical = gx[rows].mean(axis=0)  # per column
    horizontal = gy[:, cols].mean(axis=1)  # per row
    typical_v, typical_h = float(np.median(gx)), float(np.median(gy))
    left = _edge(vertical, int(fx - far), int(fx - near / 2), typical_v)
    right = _edge(vertical, int(fx + near / 2), int(fx + far), typical_v)
    top = _edge(horizontal, int(fy - far), int(fy - near / 2), typical_h)
    bottom = _edge(horizontal, int(fy + near / 2), int(fy + far), typical_h)
    if None in (left, right, top, bottom):
        bw = TYPICAL_WIDTH * fw
        bh = bw / TYPICAL_ASPECT
        x0, y0 = fx - bw / 2, fy - bh * 0.45
        x0, y0 = min(max(x0, 0), w - bw), min(max(y0, 0), h - bh)
        return Box(x0 / w, y0 / h, min(bw, w) / w, min(bh, h) / h)
    x0, x1 = left + 1, right + 1  # gx[i] is the edge between columns i and i+1
    y0, y1 = top + 1, bottom + 1
    return Box(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def find_webcam(frames: Sequence[np.ndarray], detector: FaceDetector) -> Box | None:
    """The webcam overlay in these frames of one clip, or None."""
    if len(frames) < MIN_FRAMES:
        return None
    face = _still_face([detector.detect(f) for f in frames])
    if face is None:
        return None
    box = _box(face, frames)
    # A still face on a still picture is a talking-head video, not a webcam overlay.
    h, w = frames[0].shape[:2]
    outside = np.ones((h, w), dtype=bool)
    outside[
        int(box.y * h) : int((box.y + box.h) * h), int(box.x * w) : int((box.x + box.w) * w)
    ] = False
    grey = [f.astype(np.float32).mean(axis=2) for f in frames]
    change = np.mean([np.abs(b - a)[outside].mean() for a, b in zip(grey, grey[1:], strict=False)])
    if change < MIN_BACKGROUND_CHANGE:
        return None
    return box


def detect_webcam(
    input_args: list[str], duration: float, width: int, height: int, detector: FaceDetector
) -> Box | None:
    """Sample one clip's keyframes and find its webcam overlay (``find_webcam``)."""
    with tempfile.TemporaryDirectory(prefix="echo-webcam-") as tmp:
        frames = sample_keyframes(input_args, duration, width, height, Path(tmp))
    box = find_webcam(frames, detector)
    log.info("webcam: %s (%d keyframes)", box, len(frames))
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
