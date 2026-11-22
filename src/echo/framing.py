"""Where to put the 9:16 window, shot by shot (#16): ``ClipConfig.reframe = "face"``.

For one clip, before it is cut:

1. **Sample** (one ffmpeg run over the clip, local or ``RemoteVideo``): small BGR frames
   at ``SAMPLE_FPS`` for face detection, and tiny greyscale frames at ``CUT_FPS`` to find
   camera cuts.
2. **Shots:** a cut is a frame much different from the one before it. Podcasts are
   multi-camera, and each camera frames its subject at a different x, so the window is
   placed per shot, not per clip.
3. **Faces** in each sampled frame (``echo.faces``), grouped per shot into people by
   position.
4. **Choose x per shot:** centred on all the faces if they fit in the window, else on
   the most present (then largest) face; ``crop_x`` if the shot has no face.
5. **Expression:** the x positions become one ``crop`` x expression of ``t`` (seconds
   from the clip start), e.g. ``657+300*gte(t\\,4.2)``, so the cut stays a single encode.

All positions here are fractions of the frame width (0..1) until ``crop_expression``
turns them into pixels.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from echo.faces import Face, FaceDetector
from echo.ffmpeg import ffmpeg

log = logging.getLogger(__name__)

SAMPLE_FPS = 3
"""Face-detection frames per second of clip."""
SAMPLE_WIDTH = 640
"""Width of the face-detection frames (YuNet's input is 640x640)."""
CUT_FPS = 30
"""Rate of the tiny frames used to find camera cuts: a cut is placed within 1/30 s."""
CUT_SIZE = (64, 36)

CUT_THRESHOLD = 20.0
"""Mean absolute difference (0-255) between consecutive tiny frames that counts as a cut."""
CUT_RATIO = 3.0
"""...and it must be this many times the typical difference around it (motion isn't a cut)."""
MIN_SHOT = 0.5
"""Shots shorter than this (seconds) are merged into the previous one (flashes)."""

MIN_PRESENCE = 0.25
"""A face must be in at least this fraction of a shot's frames to count."""
MIN_RELATIVE_SIZE = 0.4
"""...and at least this fraction of the largest face's width (drops background faces)."""
FIT_MARGIN = 0.9
"""Faces "fit" if they span at most this fraction of the window (some room on the sides)."""
DEAD_ZONE = 0.02
"""Consecutive shots whose x differs by less than this (fraction of the frame) keep the
same x, so the window doesn't twitch between two shots of the same camera."""


@dataclass(frozen=True)
class Samples:
    """What ``sample`` read from one clip."""

    frames: list[np.ndarray]
    """BGR face-detection frames, ``SAMPLE_FPS`` per second from the clip start."""
    cut_frames: np.ndarray
    """N x H x W greyscale frames, ``CUT_FPS`` per second from the clip start."""
    fps: float = SAMPLE_FPS
    cut_fps: float = CUT_FPS


@dataclass(frozen=True)
class Person:
    """One face position within a shot, across its frames."""

    cx: float
    left: float
    right: float
    width: float
    presence: float
    """Fraction of the shot's sampled frames it was detected in."""


@dataclass(frozen=True)
class Shot:
    start: float
    end: float
    people: tuple[Person, ...] = ()
    frames: int = 0


def sample_size(width: int, height: int) -> tuple[int, int]:
    """Even (w, h) of the face-detection frames for a ``width`` x ``height`` (display) source."""
    w = SAMPLE_WIDTH
    h = max(2, min(SAMPLE_WIDTH, round(SAMPLE_WIDTH * height / width / 2) * 2))
    return w, h


def sample(
    input_args: list[str], duration: float, width: int, height: int, workdir: Path
) -> Samples:
    """Read the clip once with ffmpeg (``input_args`` already seek to its start) and
    return its face-detection and cut-detection frames.

    ``width`` / ``height`` are the displayed frame size (after rotation and SAR), so the
    scaled frames have the shape viewers see.
    """
    sw, sh = sample_size(width, height)
    cw, ch = CUT_SIZE
    faces_raw, cuts_raw = workdir / "faces.raw", workdir / "cuts.raw"
    graph = (
        f"[0:v]split=2[a][b];"
        f"[a]fps={SAMPLE_FPS},scale={sw}:{sh},format=bgr24[faces];"
        f"[b]fps={CUT_FPS},scale={cw}:{ch},format=gray[cuts]"
    )
    t = ["-t", f"{duration:.3f}"]  # an output option: needed on both outputs
    args = [*input_args, "-an", "-sn", "-filter_complex", graph]
    args += ["-map", "[faces]", *t, "-f", "rawvideo", str(faces_raw)]
    args += ["-map", "[cuts]", *t, "-f", "rawvideo", str(cuts_raw)]
    ffmpeg(args)
    faces = np.fromfile(faces_raw, dtype=np.uint8)
    cuts = np.fromfile(cuts_raw, dtype=np.uint8)
    frames = list(faces[: faces.size // (sw * sh * 3) * sw * sh * 3].reshape(-1, sh, sw, 3))
    cut_frames = cuts[: cuts.size // (cw * ch) * cw * ch].reshape(-1, ch, cw)
    return Samples(frames=frames, cut_frames=cut_frames)


def find_cuts(cut_frames: np.ndarray, fps: float = CUT_FPS) -> list[float]:
    """Times (seconds from the clip start) of camera cuts: frames much different from the
    previous one, both in absolute terms and relative to the motion around them."""
    if len(cut_frames) < 2:
        return []
    frames = cut_frames.astype(np.float32)
    diffs = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2))
    cuts = []
    for i, d in enumerate(diffs):
        if d < CUT_THRESHOLD:
            continue
        around = np.concatenate([diffs[max(0, i - 15) : i], diffs[i + 1 : i + 16]])
        typical = float(np.median(around)) if around.size else 0.0
        if d >= CUT_RATIO * typical:
            cuts.append((i + 1) / fps)
    return cuts


def shots(cuts: Sequence[float], duration: float) -> list[tuple[float, float]]:
    """(start, end) of each shot; shots under ``MIN_SHOT`` join the previous one."""
    bounds = [0.0]
    for t in sorted(cuts):
        if MIN_SHOT <= t <= duration - MIN_SHOT and t - bounds[-1] >= MIN_SHOT:
            bounds.append(t)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def group_people(per_frame: Sequence[Sequence[Face]]) -> tuple[Person, ...]:
    """Group the faces of one shot's frames into people by horizontal position."""
    clusters: list[list[Face]] = []
    for faces in per_frame:
        for face in faces:
            best = None
            for cluster in clusters:
                cx = float(np.median([f.cx for f in cluster]))
                width = max(f.w for f in cluster)
                if abs(cx - face.cx) <= 0.6 * max(width, face.w):
                    if best is None or abs(cx - face.cx) < abs(
                        float(np.median([f.cx for f in best])) - face.cx
                    ):
                        best = cluster
            if best is None:
                clusters.append([face])
            else:
                best.append(face)
    n = max(1, len(per_frame))
    people = []
    for cluster in clusters:
        people.append(
            Person(
                cx=float(np.median([f.cx for f in cluster])),
                left=float(np.median([f.left for f in cluster])),
                right=float(np.median([f.right for f in cluster])),
                width=float(np.median([f.w for f in cluster])),
                presence=min(1.0, len(cluster) / n),
            )
        )
    return tuple(sorted(people, key=lambda p: p.cx))


def main_people(people: Sequence[Person]) -> list[Person]:
    """The people that matter for framing: present enough, and not tiny next to the others."""
    present = [p for p in people if p.presence >= MIN_PRESENCE]
    if not present:
        return []
    largest = max(p.width for p in present)
    return [p for p in present if p.width >= MIN_RELATIVE_SIZE * largest]


def choose_x(people: Sequence[Person], window: float) -> float | None:
    """Centre (fraction of the frame width) of the window for a shot with these people,
    or None for no face. ``window`` is the window's width as a fraction of the frame."""
    main = main_people(people)
    if not main:
        return None
    left, right = min(p.left for p in main), max(p.right for p in main)
    if right - left <= FIT_MARGIN * window:
        return (left + right) / 2
    # Too far apart to fit: the most present face, then the largest.
    return max(main, key=lambda p: (round(p.presence, 1), p.width)).cx


def analyse(samples: Samples, duration: float, detector: FaceDetector) -> list[Shot]:
    """Split the clip into shots and find the people in each."""
    detections = [detector.detect(frame) for frame in samples.frames]
    result = []
    for start, end in shots(find_cuts(samples.cut_frames, samples.cut_fps), duration):
        # Frame k is at k / fps; skip the ones right at a cut, which may still be blended.
        idx = [
            k
            for k in range(len(detections))
            if start + 0.1 <= k / samples.fps < end - 0.05
            or (start <= k / samples.fps < end and end - start < 0.3)
        ]
        people = group_people([detections[k] for k in idx])
        result.append(Shot(start=start, end=end, people=people, frames=len(idx)))
    return result


def plan(shot_list: Sequence[Shot], window: float, default: float) -> list[tuple[float, float]]:
    """(start time, window centre) per change of position, from the shots.

    Shots without a face use ``default`` (the window centre ``crop_x`` gives). Positions
    are clamped so the window stays inside the frame, and changes smaller than
    ``DEAD_ZONE`` are dropped.
    """
    half = window / 2
    keyframes: list[tuple[float, float]] = []
    for shot in shot_list:
        x = choose_x(shot.people, window)
        x = default if x is None else min(max(x, half), 1 - half)
        if keyframes and abs(x - keyframes[-1][1]) < DEAD_ZONE:
            continue
        keyframes.append((shot.start, x))
    return keyframes or [(0.0, default)]


def crop_expression(keyframes: Sequence[tuple[float, float]], width: int, crop_w: int) -> str:
    """ffmpeg ``crop`` x for these (time, window centre) keyframes on a ``width``-pixel
    frame with a ``crop_w``-pixel window: a constant, or a sum of steps, e.g.
    ``657+300*gte(t\\,4.200)``. Commas are escaped for a ``-vf`` filtergraph.
    """
    pixels = [(t, max(0, min(width - crop_w, round(x * width - crop_w / 2)))) for t, x in keyframes]
    expr = str(pixels[0][1])
    for (_, before), (t, x) in zip(pixels, pixels[1:], strict=False):
        if x != before:
            expr += f"{x - before:+d}*gte(t\\,{t:.3f})"
    return expr


def frame_crop(
    input_args: list[str],
    duration: float,
    width: int,
    height: int,
    crop_w: int,
    default_x: float,
    detector: FaceDetector,
) -> str:
    """The ``crop`` x expression framing the faces of one clip, shot by shot.

    ``input_args`` are ffmpeg input arguments that seek to the clip start (local file or
    remote URL); ``width`` x ``height`` is the displayed frame size and ``crop_w`` the
    window width in those pixels; ``default_x`` is ``ClipConfig.crop_x``.
    """
    window = crop_w / width
    default = default_x * (1 - window) + window / 2  # crop_x -> window centre
    with tempfile.TemporaryDirectory(prefix="echo-framing-") as tmp:
        samples = sample(input_args, duration, width, height, Path(tmp))
    shot_list = analyse(samples, duration, detector)
    keyframes = plan(shot_list, window, default)
    log.debug(
        "framing: %d shots, %d with faces, x %s",
        len(shot_list),
        sum(1 for s in shot_list if main_people(s.people)),
        [(round(t, 2), round(x, 3)) for t, x in keyframes],
    )
    return crop_expression(keyframes, width, crop_w)
