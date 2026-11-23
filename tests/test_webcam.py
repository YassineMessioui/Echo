"""Streamer webcam overlays (#126): detection on synthetic stream frames, the filter graph,
and one real encode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echo.clip import cut_clips
from echo.config import ClipConfig
from echo.faces import Face
from echo.ffmpeg import ffmpeg, probe
from echo.models import ClipSpec
from echo.webcam import (
    BORDER,
    MARGIN,
    TOP,
    Box,
    ClipWebcam,
    ExcludingDetector,
    analyse,
    consolidate,
    find_webcam,
    overlay_filter,
    webcam_box,
)

W, H = 640, 360
CAM = (20, 150, 110, 82)  # x, y, w, h in pixels: a webcam overlay on the left, like a stream


def stream_frames(n: int = 8, *, still: bool = False, border: bool = True) -> list[np.ndarray]:
    """Gameplay that changes every frame (or doesn't), with a bordered webcam pasted on."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    frames = []
    for k in range(n):
        frame = base.copy() if still else rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        x, y, w, h = CAM
        frame[y : y + h, x : x + w] = 60 + 10 * (k % 3)  # the streamer moves a little
        if border:
            frame[y - 2 : y, x - 2 : x + w + 2] = 255
            frame[y + h : y + h + 2, x - 2 : x + w + 2] = 255
            frame[y - 2 : y + h + 2, x - 2 : x] = 255
            frame[y - 2 : y + h + 2, x + w : x + w + 2] = 255
        frames.append(frame)
    return frames


def webcam_face(jitter: float = 0.0) -> Face:
    """The streamer's face inside CAM (fractions of the frame)."""
    x, y, w, h = CAM
    return Face(cx=(x + w / 2) / W + jitter, cy=(y + h * 0.45) / H, w=30 / W, h=36 / H)


class Faces:
    """A fake detector: ``per_frame(k)`` gives frame k's faces."""

    def __init__(self, per_frame) -> None:
        self.per_frame, self.k = per_frame, 0

    def detect(self, frame: np.ndarray) -> list[Face]:
        faces = self.per_frame(self.k)
        self.k += 1
        return faces


def test_finds_the_webcam_box_from_its_still_border():
    detector = Faces(lambda k: [webcam_face(0.003 * (k % 2))])
    box = find_webcam(stream_frames(), detector)
    assert box is not None
    x, y, w, h = CAM
    assert box.x == pytest.approx(x / W, abs=3 / W)
    assert box.y == pytest.approx(y / H, abs=3 / H)
    assert box.w == pytest.approx(w / W, abs=5 / W)
    assert box.h == pytest.approx(h / H, abs=5 / H)


def test_the_box_is_the_still_region_even_without_a_border():
    box = find_webcam(stream_frames(border=False), Faces(lambda k: [webcam_face()]))
    assert box is not None
    x, y, w, h = CAM
    assert box.x == pytest.approx(x / W, abs=3 / W) and box.w == pytest.approx(w / W, abs=5 / W)
    assert box.y == pytest.approx(y / H, abs=3 / H) and box.h == pytest.approx(h / H, abs=5 / H)


def test_a_typical_shape_where_no_edge_is_found():
    """Nothing moves around the face (e.g. a static HUD band): a typical webcam shape."""
    face = webcam_face()
    box = webcam_box(face, stream_frames(still=True))
    assert box.contains(face.cx, face.cy)
    assert box.w == pytest.approx(3.4 * face.w, rel=0.05)
    assert box.h == pytest.approx(box.w * W / H * 3 / 4, rel=0.05)


def test_consolidate_uses_the_median_webcam_for_every_clip_with_the_face_in_it():
    """A stream's webcam doesn't move: one clip's too-tight box (the streamer moved a
    lot) and a calm clip (a menu: face found, box not confirmed) both get the median."""
    face = webcam_face()
    good = Box(0.03, 0.42, 0.17, 0.22)
    tight = Box(0.06, 0.47, 0.09, 0.11)
    good2 = Box(0.035, 0.43, 0.165, 0.21)
    elsewhere = Face(cx=0.8, cy=0.2, w=0.05, h=0.08)
    clips = [
        ClipWebcam(face, tight),
        ClipWebcam(face, good),
        ClipWebcam(face, good2),
        ClipWebcam(face, None),  # calm clip
        ClipWebcam(elsewhere, None),  # a different still face: no webcam
        ClipWebcam(),  # no face at all
    ]
    boxes = consolidate(clips)
    median = good2  # by area: tight < good2 < good
    assert boxes == [median, median, median, median, None, None]


def test_consolidate_without_any_webcam():
    assert consolidate([ClipWebcam(), ClipWebcam(webcam_face(), None)]) == [None, None]


def test_analyse_reports_the_face_even_on_a_calm_clip():
    result = analyse(stream_frames(still=True), Faces(lambda k: [webcam_face()]))
    assert result.face is not None and result.box is None


def test_a_moving_game_character_is_not_a_webcam():
    def moving(k: int) -> list[Face]:
        return [Face(cx=0.3 + 0.05 * k, cy=0.5, w=0.06, h=0.1)]

    assert find_webcam(stream_frames(), Faces(moving)) is None


def test_a_face_in_too_few_frames_is_not_a_webcam():
    detector = Faces(lambda k: [webcam_face()] if k < 3 else [])
    assert find_webcam(stream_frames(), detector) is None


def test_a_big_face_is_a_person_on_camera_not_an_overlay():
    big = Face(cx=0.5, cy=0.4, w=0.3, h=0.5)
    assert find_webcam(stream_frames(), Faces(lambda k: [big])) is None


def test_a_still_picture_is_a_talking_head_not_a_stream():
    """A still face on a still background (a podcast camera) has no overlay to keep."""
    detector = Faces(lambda k: [webcam_face()])
    assert find_webcam(stream_frames(still=True), detector) is None


def test_too_few_frames():
    assert find_webcam(stream_frames(3), Faces(lambda k: [webcam_face()])) is None


def test_excluding_detector_drops_faces_in_the_webcam():
    box = Box(0.0, 0.0, 0.3, 0.3)
    inner, outer = Face(cx=0.1, cy=0.1, w=0.05, h=0.05), Face(cx=0.6, cy=0.5, w=0.1, h=0.1)
    assert ExcludingDetector(Faces(lambda k: [inner, outer]), box).detect(np.zeros(1)) == [outer]


# --- filter graph -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("corner", "x", "y"),
    [
        ("top-left", MARGIN, TOP),
        ("top-right", 1080 - (432 + 2 * BORDER) - MARGIN, TOP),
    ],
)
def test_overlay_filter(corner, x, y):
    box = Box(0.03, 0.42, 0.17, 0.22)
    graph = overlay_filter(
        "crop=606:1080:657:0,scale=1080:1920,setsar=1",
        box,
        1920,
        1080,
        ClipConfig(webcam_corner=corner),
    )
    assert graph.startswith("split=2[echo_bg][echo_cam];[echo_bg]crop=606:1080:657:0,")
    assert "[echo_cam]crop=326:238:58:454,scale=432:314:flags=lanczos," in graph
    assert graph.endswith(f"[echo_frame][echo_webcam]overlay={x}:{y}")


def test_bottom_corners_stay_above_the_captions():
    graph = overlay_filter(
        "null", Box(0, 0, 0.2, 0.2), 1920, 1080, ClipConfig(webcam_corner="bottom-left")
    )
    y = int(graph.rsplit(":", 1)[1])
    height = 242 + 2 * BORDER  # a 384x216 webcam scaled to 432 wide, plus its border
    assert y + height <= 1920 - 420  # its bottom stays above the captions (420 px up)


# --- real ffmpeg ------------------------------------------------------------------------


@pytest.mark.ffmpeg
def test_webcam_is_kept_in_the_corner(tmp_path: Path):
    """A synthetic stream: noisy, always-changing "gameplay" + a still grey webcam with a
    white border.
    The clip keeps the webcam (white border) at the top-left corner."""
    src = tmp_path / "stream.mp4"
    ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=4",
            "-f", "lavfi", "-i", "color=c=gray:size=220x164:rate=25:duration=4",
            "-filter_complex",
            "[0]noise=alls=80:allf=t[game];[1]pad=228:172:4:4:white[cam];[game][cam]overlay=40:300",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12", "-an", str(src),
        ]
    )  # fmt: skip
    # The detector sees the face in the webcam's spot on every (640-wide) frame.
    face = Face(cx=(44 + 110) / 1280, cy=(304 + 70) / 720, w=60 / 1280, h=70 / 720)
    out = tmp_path / "clip.mp4"
    cut_clips(
        src,
        [ClipSpec.between("t", 0.0, 3.5)],
        [out],
        ClipConfig(),
        detector=Faces(lambda k: [face]),
    )
    info = probe(out)
    assert info.frame_size == (1080, 1920)
    frame = tmp_path / "f.png"
    ffmpeg(["-ss", "1", "-i", str(out), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray",
            str(frame)])  # fmt: skip
    grey = np.fromfile(frame, dtype=np.uint8).reshape(1920, 1080)
    # The white border at the overlay's top-left corner, and the grey webcam inside.
    assert grey[TOP + 1, MARGIN + 50] > 200
    assert 100 < grey[TOP + 60, MARGIN + 60] < 160


# --- review fixes (#134) ---------------------------------------------------------------------


def test_offline_the_clips_are_cut_without_a_webcam(tmp_path, monkeypatch):
    """No face model yet and no network: the webcam is skipped, the clip stage goes on."""
    import echo.clip as clip_module
    from echo.errors import ExternalToolError

    def offline(*args, **kwargs):
        raise ExternalToolError("could not download the face detection model")

    monkeypatch.setattr(clip_module, "detect_webcams", offline)
    boxes = clip_module._search_webcams(
        tmp_path / "v.mp4", [ClipSpec.between("t", 0, 5)], (1920, 1080, 606), None, 0.5
    )
    assert boxes == [None]


def test_a_webcam_inside_the_crop_is_not_shown_twice():
    from echo.webcam import inside_crop

    # A 606 px centre crop of 1920 spans x 0.342-0.658.
    assert inside_crop(Box(0.45, 0.8, 0.1, 0.15), 1920, 606, 0.5)  # bottom-centre cam
    assert not inside_crop(Box(0.03, 0.42, 0.17, 0.22), 1920, 606, 0.5)  # left cam
    assert not inside_crop(Box(0.6, 0.1, 0.1, 0.1), 1920, 606, 0.5)  # half outside


def test_a_box_much_taller_than_wide_is_cut_back_around_the_face():
    """A still name plate under the cam would make the box grow down: max 1:1."""
    frames = stream_frames()
    for f in frames:  # a still "name plate" right under the webcam
        x, y, w, h = CAM
        f[y + h : y + h + 120, x - 2 : x + w + 2] = 30
    box = webcam_box(webcam_face(), frames)
    assert box.h * H <= box.w * W + 1
    face = webcam_face()
    assert box.contains(face.cx, face.cy)


def test_a_tall_webcam_stays_above_the_captions():
    config = ClipConfig(webcam_size=0.6, webcam_corner="top-left")
    graph = overlay_filter("null", Box(0.0, 0.0, 0.1, 0.4), 1920, 1080, config)
    ow, oh = (int(v) for v in graph.split("scale=")[1].split(":")[:2])
    assert TOP + oh + 2 * BORDER <= 1440
    assert ow < 648  # shrunk from 0.6 * 1080 to keep the shape


def test_non_square_pixels_keep_the_webcam_shape():
    """Anamorphic 1440x1080 stored as 4:3 pixels: the cam's display aspect is used."""
    box = Box(0.0, 0.0, 0.25, 0.25)  # 360x270 stored = 480x270 displayed (16:9)
    graph = overlay_filter("null", box, 1440, 1080, ClipConfig(), sar=4 / 3)
    ow, oh = (int(v) for v in graph.split("scale=")[1].split(":")[:2])
    assert abs(ow / oh - 16 / 9) < 0.02
