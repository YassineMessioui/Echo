"""Per-speaker caption colours (#68)."""

from pathlib import Path

import numpy as np
import pytest

from echo import ffmpeg
from echo.captions import CaptionStyle, build_ass, speaker_palette, subtitles_filter
from echo.config import CaptionConfig, ClipConfig, DiarizeConfig, PipelineConfig
from echo.models import ClipSpec, Segment, Speakers, SpeakerTurn, Transcript, Word

WHITE, BLUE = "&H00FFFFFF", "&H00FFDB7F"  # default speaker_colours[0], [1] in ASS (BGR)
CLIP = ClipSpec.between("t", 0.0, 10.0)


def _transcript(words: list[tuple[float, float, str]]) -> Transcript:
    segment = Segment(
        id=0,
        start=words[0][0],
        end=words[-1][1],
        text=" ".join(w for _, _, w in words),
        words=[Word(start=s, end=e, text=w) for s, e, w in words],
    )
    return Transcript(language="en", duration=10.0, segments=[segment])


def _speakers(*turns: tuple[float, float, str]) -> Speakers:
    return Speakers(turns=[SpeakerTurn(start=s, end=e, speaker=x) for s, e, x in turns])


def _dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


TWO_VOICES = _transcript(
    [(0.0, 0.5, "hello"), (0.5, 1.0, "there"), (1.2, 1.6, "hi"), (1.6, 2.0, "back")]
)
A_THEN_B = _speakers((0.0, 1.1, "S1"), (1.1, 2.5, "S2"))


def test_two_speakers_get_two_base_colours():
    yellow = "&H0000D4FF"  # highlight_colour #FFD400
    events = [
        e.split(",,", 2)[-1].removeprefix("0,0,0,,")
        for e in _dialogues(build_ass(TWO_VOICES, CLIP, speakers=A_THEN_B))
    ]
    assert events == [
        # Each line opens in its speaker's colour; after the highlighted (yellow) word
        # the speaker's colour comes back, instead of {\r} resetting to the default.
        f"{{\\c{WHITE}}}{{\\c{yellow}}}HELLO{{\\c{WHITE}}} THERE",
        f"{{\\c{WHITE}}}HELLO {{\\c{yellow}}}THERE{{\\c{WHITE}}}",
        f"{{\\c{BLUE}}}{{\\c{yellow}}}HI{{\\c{BLUE}}} BACK",
        f"{{\\c{BLUE}}}HI {{\\c{yellow}}}BACK{{\\c{BLUE}}}",
    ]


def test_a_line_never_mixes_speakers():
    # max_words=3 would put "hello there hi" on one line; the speaker change splits it.
    lines = [
        e.rsplit(",,", 1)[-1] for e in _dialogues(build_ass(TWO_VOICES, CLIP, speakers=A_THEN_B))
    ]
    assert all("HELLO" not in line or "HI" not in line for line in lines)
    assert sum("HELLO" in line for line in lines) == 2  # one event per word of line 1


def test_single_speaker_output_is_unchanged():
    one = _speakers((0.0, 3.0, "S1"))
    assert build_ass(TWO_VOICES, CLIP, speakers=one) == build_ass(TWO_VOICES, CLIP)
    assert build_ass(TWO_VOICES, CLIP, speakers=Speakers(turns=[])) == build_ass(TWO_VOICES, CLIP)


def test_colours_follow_first_appearance_in_the_whole_job():
    """A clip where only S2 speaks still uses S2's colour, so colours are stable."""
    job_speakers = _speakers((0.0, 5.0, "S1"), (5.0, 9.0, "S2"))
    later = _transcript([(6.0, 6.5, "late"), (6.5, 7.0, "words")])
    events = _dialogues(build_ass(later, CLIP, speakers=job_speakers))
    assert events and all(f"{{\\c{BLUE}}}" in e for e in events)


def test_palette_cycles_and_needs_two_speakers():
    style = CaptionStyle(speaker_colours=("#111111", "#222222"))
    three = _speakers((0, 1, "S1"), (1, 2, "S2"), (2, 3, "S3"))
    assert speaker_palette(three, style) == {"S1": "#111111", "S2": "#222222", "S3": "#111111"}
    assert speaker_palette(_speakers((0, 1, "S1")), style) == {}
    assert speaker_palette(None, style) == {}


def test_speaker_tags_mark_each_change_of_speaker():
    style = CaptionStyle(speaker_tags=True)
    lines = [
        e.rsplit(",,", 1)[-1] for e in _dialogues(build_ass(TWO_VOICES, CLIP, style, A_THEN_B))
    ]
    assert all("S1: " in line for line in lines[:2])
    assert all("S2: " in line for line in lines[2:])
    assert "S1: " not in build_ass(TWO_VOICES, CLIP, CaptionStyle(speaker_tags=True))


def test_outline_is_kept_for_readability():
    ass = build_ass(TWO_VOICES, CLIP, speakers=A_THEN_B)
    style_line = next(line for line in ass.splitlines() if line.startswith("Style: Caption"))
    assert ",&H00000000," in style_line and style_line.split(",")[16] == "6"  # black, 6 px


# --- clip stage -----------------------------------------------------------------------------


def test_clip_stage_settings_cover_captions_and_speakers():
    from echo.clip import ClipStage

    stage, base = ClipStage(), PipelineConfig()
    variants = [
        PipelineConfig(caption=CaptionConfig(enabled=False)),
        PipelineConfig(diarize=DiarizeConfig(enabled=True)),
        PipelineConfig(caption=CaptionConfig(speaker_colours=["#000000", "#FFFFFF"])),
        PipelineConfig(clip=ClipConfig(aspect="source")),
    ]
    assert len({stage.settings(base), *(stage.settings(v) for v in variants)}) == 5


def test_clip_stage_colours_captions_only_with_diarization_on(job_copy, monkeypatch):
    from echo import clip as clip_module
    from echo.clip import ClipStage

    job = job_copy
    job.input_video.write_bytes(b"video")
    clips = job.read_clips()
    turns = [
        SpeakerTurn(start=0.0, end=34.0, speaker="S1"),
        SpeakerTurn(start=34.0, end=81.0, speaker="S2"),
    ]
    job.write_speakers(Speakers(turns=turns))
    tracks: list[str] = []

    def clipper(video, clips_, outputs, config, *, extra_filter=None):
        tracks.append(
            Path(
                extra_filter.split("filename=", 1)[1].split(":fontsdir")[0].replace("\\", "")
            ).read_text("utf-8-sig")
        )
        for out in outputs:
            out.write_bytes(b"clip")

    monkeypatch.setattr(clip_module, "has_filter", lambda name: True)
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert tracks and not any(BLUE in t for t in tracks), "diarization off: one colour"
    tracks.clear()
    ClipStage(clipper=clipper).run(job, PipelineConfig(diarize=DiarizeConfig(enabled=True)))
    assert len(tracks) == len(clips), "colours changed the tracks: every clip re-cut"
    assert any(BLUE in t for t in tracks)


def test_toggling_captions_recuts_an_existing_job(job_copy, monkeypatch):
    """The bug from the first captioned run: captions on later didn't re-cut anything."""
    from echo import clip as clip_module
    from echo.clip import ClipStage
    from echo.pipeline import run_pipeline

    job = job_copy
    job.input_video.write_bytes(b"video")
    cuts: list[bool] = []

    def clipper(video, clips_, outputs, config, *, extra_filter=None):
        cuts.append(extra_filter is not None)
        for out in outputs:
            out.write_bytes(b"clip")

    monkeypatch.setattr(clip_module, "has_filter", lambda name: True)
    stages = [ClipStage(clipper=clipper)]
    run_pipeline(job, stages, PipelineConfig(caption=CaptionConfig(enabled=False)))
    assert cuts and not any(cuts)
    cuts.clear()
    run_pipeline(job, stages, PipelineConfig())  # captions on: must re-cut with captions
    assert cuts and all(cuts)


# --- real render ------------------------------------------------------------------------------


def _frame(video: Path, vf: str, at: float, out: Path) -> np.ndarray:
    ffmpeg.ffmpeg(
        ["-i", str(video), "-vf", vf, "-ss", f"{at}", "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", str(out)]
    )
    return np.frombuffer(out.read_bytes(), dtype=np.uint8).reshape(480, 270, 3).astype(int)


def _count(frame: np.ndarray, rgb: tuple[int, int, int], tolerance: int = 40) -> int:
    return int((np.abs(frame - np.array(rgb)).max(axis=2) <= tolerance).sum())


@pytest.mark.ffmpeg
def test_real_render_gives_each_speaker_a_different_colour(tmp_path: Path):
    if not ffmpeg.has_filter("subtitles"):
        pytest.skip("this ffmpeg has no libass ('subtitles' filter)")
    video = tmp_path / "black.mp4"
    ffmpeg.ffmpeg(
        ["-f", "lavfi", "-i", "color=black:size=270x480:rate=25:duration=4"]
        + ["-pix_fmt", "yuv420p", str(video)]
    )
    # Two-word lines: at each sampled time the second word shows in the speaker's colour.
    words = [(0.0, 1.5, "one"), (1.5, 1.8, "two"), (2.0, 3.5, "three"), (3.5, 3.8, "four")]
    speakers = _speakers((0.0, 1.9, "S1"), (1.9, 4.0, "S2"))
    style = CaptionStyle(max_words=2, font_size=120, margin_bottom=200)
    ass = tmp_path / "speakers.ass"
    ass.write_text(
        build_ass(_transcript(words), ClipSpec.between("t", 0.0, 4.0), style, speakers), "utf-8-sig"
    )

    vf = subtitles_filter(ass)
    first = _frame(video, vf, 0.5, tmp_path / "a.raw")  # S1 line: "TWO" in white
    second = _frame(video, vf, 2.5, tmp_path / "b.raw")  # S2 line: "FOUR" in light blue
    white, light_blue = (255, 255, 255), (127, 219, 255)
    assert _count(first, white) > 200 and _count(first, light_blue) < 20
    assert _count(second, light_blue) > 200 and _count(second, white) < 20
