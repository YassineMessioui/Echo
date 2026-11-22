"""Following the active speaker (#16, #109): timeline smoothing, speaker -> seat, plan."""

import json
import logging
from dataclasses import dataclass, field

import pytest

from echo.clip import MANIFEST, ClipStage, clip_turns, cut_clips
from echo.config import CaptionConfig, ClipConfig, DiarizeConfig, PipelineConfig
from echo.ffmpeg import probe
from echo.follow import MIN_HOLD, PAN, map_speakers, plan_speakers, timeline
from echo.framing import Person, Shot, crop_expression
from echo.models import ClipSpec, Speakers, SpeakerTurn
from echo.reframe import TARGET, video_filter

WINDOW = 606 / 1920
LEFT, RIGHT = 0.15, 0.85


def _person(cx: float, width: float = 0.05) -> Person:
    return Person(cx=cx, left=cx - width / 2, right=cx + width / 2, width=width, presence=1.0)


def _wide_shot(start: float, end: float, turns, fps: float = 3.0) -> Shot:
    """A two-shot whose mouths move only for whoever ``turns`` says is speaking."""
    mouths = []
    k = int(start * fps) + 1
    while k / fps < end:
        t = k / fps
        speaker = next((s for a, b, s in turns if a <= t < b), None)
        left, right = (0.08 if speaker == s else 0.01 for s in ("S1", "S2"))
        mouths.append((t, ((LEFT, left), (RIGHT, right))))
        k += 1
    people = (_person(LEFT), _person(RIGHT))
    return Shot(start=start, end=end, people=people, frames=len(mouths), mouths=tuple(mouths))


# --- timeline -----------------------------------------------------------------------------


def test_timeline_follows_turns():
    turns = [(0, 5, "S1"), (5, 12, "S2"), (12, 20, "S1")]
    assert timeline(turns, 0, 20) == [(0, "S1"), (5, "S2"), (12, "S1")]


def test_short_interjections_do_not_move_the_window():
    turns = [(0, 5, "S1"), (5, 5.8, "S2"), (5.8, 12, "S1"), (12, 20, "S2")]
    assert timeline(turns, 0, 20) == [(0, "S1"), (12, "S2")]


def test_silence_keeps_the_last_speaker_and_minimum_hold_applies():
    turns = [(0, 4, "S1"), (6, 6 + MIN_HOLD - 0.1, "S2"), (9, 15, "S1")]
    assert timeline(turns, 0, 15) == [(0, "S1")]


def test_short_run_at_the_shot_start_goes_to_the_next_speaker():
    turns = [(0, 3, "S1"), (3, 10, "S2")]
    assert timeline(turns, 2.5, 10) == [(2.5, "S2")]


def test_timeline_without_turns():
    assert timeline([], 0, 10) == []
    assert timeline([(20, 30, "S1")], 0, 10) == []


# --- speaker -> seat --------------------------------------------------------------------


def test_speakers_are_mapped_to_the_seat_whose_mouth_moves():
    turns = [(0, 6, "S2"), (6, 12, "S1"), (12, 20, "S2")]
    shots = [_wide_shot(0, 20, turns)]
    assert map_speakers(shots, turns, WINDOW) == {"S1": (2, 0), "S2": (2, 1)}


def test_no_mapping_without_mouth_evidence():
    turns = [(0, 10, "S1"), (10, 20, "S2")]
    shot = _wide_shot(0, 20, [])  # nobody's mouth moves more than anyone's
    assert map_speakers([shot], turns, WINDOW) == {}


def test_close_ups_give_no_mapping():
    turns = [(0, 10, "S1")]
    shot = Shot(0, 10, people=(_person(0.4, width=0.15),), mouths=((1.0, ((0.4, 0.1),)),))
    assert map_speakers([shot], turns, WINDOW) == {}


# --- plan -------------------------------------------------------------------------------


def test_wide_shot_follows_the_speaker_with_pans():
    turns = [(0, 6, "S2"), (6, 12, "S1"), (12, 20, "S2")]
    keyframes = plan_speakers([_wide_shot(0, 20, turns)], turns, WINDOW, 0.5)
    half = WINDOW / 2
    assert keyframes == [(0, 1 - half, 0.0), (6, half, PAN), (12, 1 - half, PAN)]
    expr = crop_expression(keyframes, 1920, 606)
    assert expr == "1314-1314*clip((t-6.000)/0.400\\,0\\,1)+1314*clip((t-12.000)/0.400\\,0\\,1)"


def test_close_ups_are_framed_like_face_mode():
    turns = [(0, 10, "S1")]
    close = Shot(0, 10, people=(_person(0.4, width=0.15),))
    assert plan_speakers([close], turns, WINDOW, 0.5) == [(0, 0.4, 0.0)]


def test_unknown_speakers_fall_back_to_face_framing():
    turns = [(0, 10, "S1"), (10, 20, "S2")]
    shot = _wide_shot(0, 20, [])  # no evidence: same as reframe = "face"
    (keyframe,) = plan_speakers([shot], turns, WINDOW, 0.5)
    assert keyframe[0] == 0 and keyframe[2] == 0.0


def test_new_shot_cuts_instead_of_panning():
    turns = [(0, 6, "S2"), (6, 12, "S1"), (12, 30, "S2")]
    shots = [_wide_shot(0, 15, turns), Shot(15, 30, people=(_person(0.45, width=0.15),))]
    keyframes = plan_speakers(shots, turns, WINDOW, 0.5)
    assert keyframes[-1] == pytest.approx((15, 0.45, 0.0))


def test_clip_turns_are_relative_to_the_clip():
    speakers = Speakers(
        turns=[
            SpeakerTurn(start=95, end=105, speaker="S1"),
            SpeakerTurn(start=105, end=150, speaker="S2"),
            SpeakerTurn(start=200, end=210, speaker="S1"),
        ]
    )
    clip = ClipSpec(topic_title="t", start=100, end=130, duration=30)
    assert clip_turns(speakers, clip) == [(0.0, 5.0, "S1"), (5.0, 30.0, "S2")]
    assert clip_turns(None, clip) == []


# --- clip stage -----------------------------------------------------------------------


@dataclass
class SpeakerClipper:
    """Fake ``Clipper`` that records the ``speakers`` it gets."""

    got: list[object] = field(default_factory=list)

    def __call__(self, video, clips, outputs, config, *, extra_filter=None, speakers=None):
        self.got.append(speakers)
        for out in outputs:
            out.write_bytes(b"clip")


def _config(reframe: str, diarize: bool = True) -> PipelineConfig:
    return PipelineConfig(
        clip=ClipConfig(reframe=reframe),
        caption=CaptionConfig(enabled=False),
        diarize=DiarizeConfig(enabled=diarize),
    )


def _diarized(job):
    job.input_video.write_bytes(b"video")
    clip = job.read_clips()[0]
    turns = [SpeakerTurn(start=clip.start, end=clip.start + 5, speaker="S1")]
    job.write_speakers(Speakers(turns=turns))
    return job


def test_speaker_framing_gets_the_turns_and_rediarizing_recuts(job_copy):
    job = _diarized(job_copy)
    clipper = SpeakerClipper()
    ClipStage(clipper=clipper).run(job, _config("speaker"))
    assert all(isinstance(s, Speakers) for s in clipper.got)
    assert all("speakers" in e for e in json.loads((job.clips_dir / MANIFEST).read_text()).values())

    again = SpeakerClipper()
    ClipStage(clipper=again).run(job, _config("speaker"))
    assert again.got == [], "same turns: nothing to re-cut"

    clip = job.read_clips()[0]
    job.write_speakers(Speakers(turns=[SpeakerTurn(start=clip.start, end=clip.end, speaker="S2")]))
    again = SpeakerClipper()
    ClipStage(clipper=again).run(job, _config("speaker"))
    assert len(again.got) == 1, "only the clip whose turns changed"


def test_other_modes_do_not_get_speakers(job_copy):
    job = _diarized(job_copy)
    clipper = SpeakerClipper()
    ClipStage(clipper=clipper).run(job, _config("face"))
    assert clipper.got and all(s is None for s in clipper.got)
    assert not any(
        "speakers" in e for e in json.loads((job.clips_dir / MANIFEST).read_text()).values()
    )


def test_speaker_framing_without_diarization_warns(job_copy, caplog):
    job = _diarized(job_copy)
    clipper = SpeakerClipper()
    with caplog.at_level(logging.WARNING):
        ClipStage(clipper=clipper).run(job, _config("speaker", diarize=False))
    assert all(s is None for s in clipper.got)
    assert "needs diarization" in caplog.text


@pytest.mark.ffmpeg
def test_pan_expression_encodes_to_vertical(make_video, tmp_path):
    src = make_video(duration=3.0, size="640x360")
    expr = crop_expression([(0, 0.2, 0), (1.0, 0.8, PAN)], 640, 202)
    vf = video_filter(640, 360, ClipConfig(), crop_x_expr=expr)
    out = tmp_path / "pan.mp4"
    clip = ClipSpec(topic_title="t", start=0.5, end=2.5, duration=2.0)
    cut_clips(src, [clip], [out], ClipConfig(), video_filter=vf)
    info = probe(out)
    assert (info.width, info.height) == TARGET
    assert info.duration == pytest.approx(2.0, abs=0.1)
