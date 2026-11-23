"""Face framing (#16, #108): shots, crop choice, the crop expression, and real runs."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from echo import faces as faces_module
from echo.clip import cut_clips
from echo.config import ClipConfig
from echo.errors import ExternalToolError
from echo.faces import Face, YuNetDetector, decode, ensure_model
from echo.ffmpeg import ffmpeg, probe
from echo.framing import (
    Person,
    Samples,
    analyse,
    choose_x,
    crop_expression,
    find_cuts,
    group_people,
    plan,
    sample,
    shots,
)
from echo.models import ClipSpec
from echo.reframe import TARGET, crop_window, video_filter

WINDOW = 606 / 1920  # a 9:16 window on 1920x1080


def _person(cx: float, width: float = 0.08, presence: float = 1.0) -> Person:
    return Person(cx=cx, left=cx - width / 2, right=cx + width / 2, width=width, presence=presence)


@dataclass
class FakeDetector:
    """Returns ``faces[k]`` for the k-th frame it sees (frames are only counted)."""

    faces: list[list[Face]]
    seen: list[np.ndarray] = field(default_factory=list)

    def detect(self, frame: np.ndarray) -> list[Face]:
        self.seen.append(frame)
        return self.faces[len(self.seen) - 1] if len(self.seen) <= len(self.faces) else []


# --- choosing x ----------------------------------------------------------------------


def test_one_face_is_centred():
    assert choose_x([_person(0.3)], WINDOW) == pytest.approx(0.3)


def test_faces_that_fit_are_centred_together():
    # Two faces 0.2 apart, 0.08 wide: a 0.28 span fits in 0.9 * 0.316.
    assert choose_x([_person(0.4), _person(0.6)], WINDOW) == pytest.approx(0.5)


def test_faces_too_far_apart_use_the_most_present():
    people = [_person(0.2, presence=0.5), _person(0.8, presence=1.0)]
    assert choose_x(people, WINDOW) == pytest.approx(0.8)


def test_equally_present_faces_too_far_apart_use_the_largest():
    people = [_person(0.2, width=0.15), _person(0.8, width=0.1)]
    assert choose_x(people, WINDOW) == pytest.approx(0.2)


def test_no_face_gives_none():
    assert choose_x([], WINDOW) is None


def test_fleeting_and_small_background_faces_are_ignored():
    people = [
        _person(0.45, width=0.15),
        _person(0.9, width=0.15, presence=0.1),  # seen in one frame
        _person(0.1, width=0.03),  # a face on a poster
    ]
    assert choose_x(people, WINDOW) == pytest.approx(0.45)


def test_group_people_by_position():
    frames = [
        [Face(0.30, 0.4, 0.1, 0.2), Face(0.70, 0.4, 0.1, 0.2)],
        [Face(0.32, 0.4, 0.1, 0.2)],
        [Face(0.31, 0.4, 0.1, 0.2), Face(0.69, 0.4, 0.1, 0.2)],
        [],
    ]
    people = group_people(frames)
    assert [round(p.cx, 2) for p in people] == [0.31, 0.69]
    assert [p.presence for p in people] == [0.75, 0.5]


# --- shots ----------------------------------------------------------------------------


def _cut_frames(levels: list[int], noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    frames = np.array([np.full((36, 64), v, dtype=np.float32) for v in levels])
    frames += rng.normal(0, noise, frames.shape) if noise else 0
    return np.clip(frames, 0, 255).astype(np.uint8)


def test_find_cuts():
    levels = [40] * 30 + [160] * 45 + [80] * 15
    assert find_cuts(_cut_frames(levels, noise=3)) == pytest.approx([1.0, 2.5])


def test_steady_motion_is_not_a_cut():
    # Big frame-to-frame changes everywhere (fast motion), none standing out.
    levels = [50 if i % 2 else 150 for i in range(60)]
    assert find_cuts(_cut_frames(levels)) == []


def test_shots_merge_flashes():
    assert shots([1.0, 1.2, 3.0], 4.0) == [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]
    assert shots([0.2, 3.9], 4.0) == [(0.0, 4.0)], "cuts at the very edges are ignored"
    assert shots([], 2.0) == [(0.0, 2.0)]


def test_analyse_frames_per_shot():
    # 3 s clip, cut at 1.5 s: face on the left, then on the right.
    left, right = Face(0.3, 0.4, 0.1, 0.2), Face(0.75, 0.4, 0.1, 0.2)
    detector = FakeDetector([[left]] * 5 + [[right]] * 4)
    frames = [np.zeros((360, 640, 3), np.uint8)] * 9
    samples = Samples(frames=frames, cut_frames=_cut_frames([30] * 45 + [200] * 45))
    result = analyse(samples, 3.0, detector)
    assert [(s.start, s.end) for s in result] == [(0.0, 1.5), (1.5, 3.0)]
    assert [round(s.people[0].cx, 2) for s in result] == [0.3, 0.75]


# --- plan + expression ------------------------------------------------------------------


@dataclass(frozen=True)
class _Shot:
    start: float
    end: float
    people: tuple[Person, ...] = ()


def test_plan_falls_back_to_default_without_faces():
    shot_list = [_Shot(0, 2, (_person(0.3),)), _Shot(2, 4), _Shot(4, 6, (_person(0.3),))]
    assert plan(shot_list, WINDOW, 0.5) == [(0, 0.3), (2, 0.5), (4, 0.3)]


def test_plan_clamps_to_the_frame_and_ignores_small_moves():
    half = WINDOW / 2
    shot_list = [
        _Shot(0, 2, (_person(0.02),)),
        _Shot(2, 4, (_person(0.05),)),  # still clamped to the left edge
        _Shot(4, 6, (_person(0.61),)),
        _Shot(6, 8, (_person(0.62),)),  # within the dead zone: no change
    ]
    assert plan(shot_list, WINDOW, 0.5) == [(0, half), (4, 0.61)]


def test_crop_expression():
    assert crop_expression([(0.0, 0.5)], 1920, 606) == "657"
    expr = crop_expression([(0.0, 0.5), (4.2, 0.8), (9.0, 0.5)], 1920, 606)
    assert expr == "657+576*gte(t\\,4.200)-576*gte(t\\,9.000)"
    assert crop_expression([(0.0, 0.0), (1.0, 1.0)], 1920, 606) == "0+1314*gte(t\\,1.000)"


def test_crop_window_matches_video_filter():
    assert crop_window(1920, 1080) == (1920, 1080, 606)
    assert crop_window(1080, 1920) is None, "already 9:16"
    assert crop_window(1080, 1080) is None, "padded, not cropped"
    assert crop_window(1440, 1080, sar=4 / 3) == (1920, 1080, 606)
    vf = video_filter(1920, 1080, ClipConfig(), crop_x_expr="10+5*gte(t\\,1.000)")
    assert vf == "crop=606:1080:10+5*gte(t\\,1.000):0,scale=1080:1920,setsar=1"


# --- YuNet decoding + model download -----------------------------------------------------


def _outputs(cells: dict[int, tuple[float, list[float]]]) -> dict[str, np.ndarray]:
    """Fake YuNet outputs: ``{stride-32 cell index: (score, [dx, dy, log w, log h])}``."""
    out = {}
    for stride in (8, 16, 32):
        n = (640 // stride) ** 2
        out[f"cls_{stride}"] = np.zeros((1, n, 1), np.float32)
        out[f"obj_{stride}"] = np.zeros((1, n, 1), np.float32)
        out[f"bbox_{stride}"] = np.zeros((1, n, 4), np.float32)
        out[f"kps_{stride}"] = np.zeros((1, n, 10), np.float32)
    for idx, (score, box) in cells.items():
        out["cls_32"][0, idx, 0] = out["obj_32"][0, idx, 0] = score
        out["bbox_32"][0, idx] = box
    return out


def test_decode_yunet_outputs():
    # Cell (row 3, col 5) at stride 32, box 64x96 centred on the cell's corner + (0.5, 0.5).
    box = [0.5, 0.5, float(np.log(2)), float(np.log(3))]
    faces = decode(_outputs({3 * 20 + 5: (0.9, box), 3 * 20 + 12: (0.7, box)}), 640, 360)
    assert len(faces) == 2
    first = max(faces, key=lambda f: f.score)
    assert first.cx == pytest.approx(5.5 * 32 / 640)
    assert first.cy == pytest.approx(3.5 * 32 / 360)
    assert (first.w, first.h) == pytest.approx((64 / 640, 96 / 360))
    assert first.score == pytest.approx(0.9)


def test_decode_drops_overlaps_low_scores_and_padding():
    box = [0.5, 0.5, float(np.log(2)), float(np.log(2))]
    same = [-0.4, 0.5, float(np.log(2)), float(np.log(2))]  # the next cell, nearly same box
    cells = {
        0: (0.3, box),
        2 * 20 + 2: (0.95, box),
        2 * 20 + 3: (0.8, same),
        5 * 20 + 10: (0.9, box),
    }
    faces = decode(_outputs(cells), 640, 360)
    # The low score is dropped, the overlapping box merged into the better one.
    assert sorted((round(f.cx * 640), round(f.score, 2)) for f in faces) == [(80, 0.95), (336, 0.9)]
    faces = decode(_outputs({15 * 20 + 2: (0.9, box)}), 640, 360)  # row 15: y = 496 > 360
    assert faces == []


def test_model_download_checks_the_hash(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"not the model"

    monkeypatch.setattr(faces_module.httpx, "stream", lambda *a, **k: Response())
    with pytest.raises(ExternalToolError, match="wrong hash"):
        ensure_model(tmp_path)
    assert list(tmp_path.iterdir()) == [], "nothing left behind"
    (tmp_path / faces_module.MODEL_FILE).write_bytes(b"cached")
    assert ensure_model(tmp_path) == tmp_path / faces_module.MODEL_FILE


# --- real ffmpeg ------------------------------------------------------------------------


def _scene_video(path: Path, duration: float = 3.0) -> None:
    """A 320x180 video: dark for the first half, a bright frame from the middle on."""
    half = duration / 2
    ffmpeg(
        [
            "-f", "lavfi", "-i", f"color=0x202020:s=320x180:d={half}:r=30",
            "-f", "lavfi", "-i", f"color=0xd0d0d0:s=320x180:d={half}:r=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ]
    )  # fmt: skip


@pytest.mark.ffmpeg
def test_sample_reads_frames_and_finds_the_cut(tmp_path):
    video = tmp_path / "scene.mp4"
    _scene_video(video)
    samples = sample(["-ss", "0.5", "-i", str(video)], 2.0, 320, 180, tmp_path)
    assert samples.frames[0].shape == (360, 640, 3)
    assert len(samples.frames) == pytest.approx(6, abs=1)
    assert len(samples.cut_frames) == pytest.approx(60, abs=1)
    assert find_cuts(samples.cut_frames) == pytest.approx([1.0], abs=0.04)


@pytest.mark.ffmpeg
def test_face_reframed_clip_is_vertical_with_the_right_duration(tmp_path):
    video = tmp_path / "scene.mp4"
    _scene_video(video, duration=4.0)
    face = Face(0.8, 0.4, 0.1, 0.2)
    detector = FakeDetector([[face]] * 100)
    out = tmp_path / "out.mp4"
    clip = ClipSpec(topic_title="t", start=0.5, end=3.5, duration=3.0)
    cut_clips(video, [clip], [out], ClipConfig(reframe="face"), detector=detector)
    info = probe(out)
    assert (info.width, info.height) == TARGET
    assert info.duration == pytest.approx(3.0, abs=0.1)
    assert detector.seen, "the clip was sampled for faces"


@pytest.mark.ffmpeg
def test_face_framing_moves_the_window(tmp_path):
    """A left-dark / right-bright source: framing a face on the right shows the bright side."""
    video = tmp_path / "halves.mp4"
    ffmpeg(
        [
            "-f", "lavfi", "-i", "color=black:s=320x180:d=2:r=30",
            "-f", "lavfi", "-i", "color=white:s=320x180:d=2:r=30",
            "-filter_complex", "[0:v][1:v]hstack,format=yuv420p[v]", "-map", "[v]",
            "-c:v", "libx264", str(video),
        ]
    )  # fmt: skip
    clip = ClipSpec(topic_title="t", start=0.0, end=1.0, duration=1.0)
    brightness = {}
    for mode in ("centre", "face"):
        out = tmp_path / f"{mode}.mp4"
        detector = FakeDetector([[Face(0.85, 0.5, 0.05, 0.1)]] * 100)
        cut_clips(video, [clip], [out], ClipConfig(reframe=mode), detector=detector)
        png = tmp_path / f"{mode}.png"
        ffmpeg(["-i", str(out), "-frames:v", "1", "-vf", "scale=54:96", "-pix_fmt", "gray",
                "-f", "rawvideo", str(png)])  # fmt: skip
        brightness[mode] = np.fromfile(png, np.uint8).mean()
    assert brightness["centre"] == pytest.approx(127, abs=40), "half dark, half bright"
    assert brightness["face"] > 240, "window fully on the bright right half"


# --- real detector ----------------------------------------------------------------------

PORTRAIT = Path(__file__).parent / "fixtures" / "faces" / "portrait.jpg"


@pytest.mark.model
@pytest.mark.ffmpeg
def test_yunet_finds_a_face_at_a_known_position(tmp_path):
    """The public-domain portrait (see fixtures/faces/README.md), composited onto a wide
    grey canvas with its centre at x = 0.75: YuNet finds one face there."""
    raw = tmp_path / "frame.raw"
    ffmpeg(
        [
            "-f", "lavfi", "-i", "color=gray:s=640x360",
            "-i", str(PORTRAIT),
            "-filter_complex", "[1:v]scale=-2:300[p];[0:v][p]overlay=x=480-overlay_w/2:y=30",
            "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", str(raw),
        ]
    )  # fmt: skip
    frame = np.fromfile(raw, np.uint8).reshape(360, 640, 3)
    faces = YuNetDetector().detect(frame)
    assert len(faces) == 1
    assert faces[0].cx == pytest.approx(0.75, abs=0.05)
    assert faces[0].score > 0.8
