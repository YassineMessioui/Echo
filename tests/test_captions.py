from pathlib import Path

import pytest

from echo import ffmpeg
from echo.captions import (
    CaptionStyle,
    _ass_colour,
    _ass_time,
    build_ass,
    chain_filters,
    clip_words,
    group_lines,
    subtitles_filter,
    write_captions,
)
from echo.job import Job
from echo.models import ClipSpec, Segment, Transcript, Word


def _transcript(words: list[tuple[float, float, str]], duration: float = 30.0) -> Transcript:
    segment = Segment(
        id=0,
        start=words[0][0],
        end=words[-1][1],
        text=" ".join(w for _, _, w in words),
        words=[Word(start=s, end=e, text=w) for s, e, w in words],
    )
    return Transcript(language="en", duration=duration, segments=[segment])


def _dialogues(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


# --- formatting ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0:00:00.00"), (1.234, "0:00:01.23"), (61.5, "0:01:01.50"), (3725.999, "1:02:06.00")],
)
def test_ass_time(seconds: float, text: str):
    assert _ass_time(seconds) == text


def test_ass_colour_is_bgr():
    assert _ass_colour("#FFD400") == "&H0000D4FF"
    with pytest.raises(ValueError):
        _ass_colour("yellow")


# --- words and lines ----------------------------------------------------------------------


def test_clip_words_are_shifted_and_clamped(golden_job: Job):
    transcript = golden_job.read_transcript()
    clip = golden_job.read_clips()[0]  # 8.3 → 34.0
    words = clip_words(transcript, clip)
    assert words[0].text == "Let's"
    assert words[0].start == pytest.approx(8.5 - 8.3)
    assert all(0 <= w.start <= w.end <= clip.duration for w in words)
    assert words[-1].end <= clip.duration
    # Nothing from the previous segment ("…a lot to cover.") leaks in.
    assert "cover." not in [w.text for w in words]


def test_words_straddling_the_clip_edge_are_clamped():
    transcript = _transcript([(0.0, 1.0, "before"), (1.8, 2.6, "edge"), (3.0, 3.5, "in")])
    words = clip_words(transcript, ClipSpec.between("t", 2.0, 10.0))
    assert [(w.text, w.start, w.end) for w in words] == [("edge", 0.0, 0.6), ("in", 1.0, 1.5)]
    # Mostly outside the clip (midpoint 1.9 < 2.0): left out.
    grazing = _transcript([(1.6, 2.2, "grazing"), (3.0, 3.5, "in")])
    assert [w.text for w in clip_words(grazing, ClipSpec.between("t", 2.0, 10.0))] == ["in"]


def test_lines_respect_max_words_punctuation_and_pauses():
    transcript = _transcript(
        [
            (0.0, 0.3, "one"),
            (0.3, 0.6, "two"),
            (0.6, 0.9, "three"),
            (0.9, 1.2, "four."),
            (1.2, 1.5, "five"),
            (3.0, 3.3, "after"),
            (3.3, 3.6, "pause"),
        ]
    )
    words = clip_words(transcript, ClipSpec.between("t", 0.0, 10.0))
    lines = group_lines(words, CaptionStyle(max_words=3, max_gap=0.6))
    assert [[w.text for w in line] for line in lines] == [
        ["one", "two", "three"],
        ["four."],
        ["five"],
        ["after", "pause"],
    ]


# --- ASS track ----------------------------------------------------------------------------


def test_build_ass_highlights_each_word_in_turn():
    transcript = _transcript([(10.0, 10.4, "hello"), (10.5, 10.9, "there"), (11.0, 11.6, "friend")])
    ass = build_ass(transcript, ClipSpec.between("t", 10.0, 30.0))

    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    events = _dialogues(ass)
    assert len(events) == 3
    hl = "{\\c&H0000D4FF}"
    assert events[0] == (
        f"Dialogue: 0,0:00:00.00,0:00:00.50,Caption,,0,0,0,,{hl}HELLO{{\\r}} THERE FRIEND"
    )
    assert events[1].startswith("Dialogue: 0,0:00:00.50,0:00:01.00,")
    assert events[1].endswith(f"HELLO {hl}THERE{{\\r}} FRIEND")
    assert events[2].startswith("Dialogue: 0,0:00:01.00,0:00:01.60,")


def test_style_options():
    transcript = _transcript([(0.0, 0.5, "Hi"), (0.5, 1.0, "{x}\\y")])
    ass = build_ass(
        transcript,
        ClipSpec.between("t", 0.0, 5.0),
        CaptionStyle(uppercase=False, font="Montserrat", font_size=70),
    )
    assert "Style: Caption,Montserrat,70," in ass
    assert "(x)/y" in _dialogues(ass)[0], "braces and backslashes can't inject ASS overrides"
    assert "Hi" in _dialogues(ass)[0]


def test_clip_without_words_has_no_events():
    transcript = _transcript([(0.0, 1.0, "early")])
    assert _dialogues(build_ass(transcript, ClipSpec.between("t", 5.0, 20.0))) == []


def test_write_captions(golden_job: Job, tmp_path: Path):
    transcript, clips = golden_job.read_transcript(), golden_job.read_clips()
    outs = [tmp_path / "clips" / f"{i:02d}.ass" for i in range(1, len(clips) + 1)]
    write_captions(transcript, clips, outs)
    for out in outs:
        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM"
        assert _dialogues(raw.decode("utf-8-sig"))
    with pytest.raises(ValueError):
        write_captions(transcript, clips, outs[:1])


# --- ffmpeg filter ------------------------------------------------------------------------


def test_subtitles_filter_escapes_windows_paths():
    path = Path("C:/Users/a b/it's [1], x.ass")
    assert subtitles_filter(path, fonts_dir=None) == (
        "subtitles=filename=C\\\\:/Users/a b/it\\\\\\'s \\[1\\]\\, x.ass"
    )


def test_chain_filters():
    assert chain_filters(None, None) is None
    assert chain_filters("crop=1:1", None, "subtitles=filename=a.ass") == (
        "crop=1:1,subtitles=filename=a.ass"
    )


def _first_frame(video: Path, out: Path, vf: str | None) -> bytes:
    args = ["-i", str(video)]
    if vf:
        args += ["-vf", vf]
    ffmpeg.ffmpeg([*args, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", str(out)])
    return out.read_bytes()


@pytest.mark.ffmpeg
def test_captions_are_burned_in(make_video, tmp_path: Path):
    """Real ffmpeg + libass, with a path that needs every kind of escaping."""
    video = make_video(duration=1.0, size="270x480")
    transcript = _transcript([(0.0, 0.8, "Caption"), (0.8, 1.0, "test")], duration=1.0)
    ass = tmp_path / "it's a [test], ok" / "01-clip.ass"
    write_captions(transcript, [ClipSpec.between("t", 0.0, 1.0)], [ass])

    plain = _first_frame(video, tmp_path / "plain.raw", None)
    captioned = _first_frame(video, tmp_path / "captioned.raw", subtitles_filter(ass))
    assert len(plain) == len(captioned) == 270 * 480 * 3
    assert plain != captioned, "the caption should change the frame"


def test_subtitles_filter_points_libass_at_the_bundled_fonts():
    from echo.captions import FONTS_DIR

    assert (FONTS_DIR / "Montserrat-ExtraBold.ttf").is_file()
    assert (FONTS_DIR / "OFL.txt").is_file()
    vf = subtitles_filter(Path("C:/jobs/a.ass"), fonts_dir=Path("C:/fonts dir"))
    # The ":" between the two options is ffmpeg's separator, so it isn't escaped.
    assert vf == "subtitles=filename=C\\\\:/jobs/a.ass:fontsdir=C\\\\:/fonts dir"


def test_default_caption_font_is_bundled():
    from echo.config import CaptionConfig

    assert CaptionConfig().font == "Montserrat ExtraBold"
    assert "Style: Caption,Montserrat ExtraBold," in build_ass(
        _transcript([(0.0, 0.5, "hi")]), ClipSpec.between("t", 0.0, 5.0)
    )


@pytest.mark.ffmpeg
def test_bundled_font_is_used_by_libass(make_video, tmp_path: Path):
    """Real ffmpeg: libass picks Montserrat ExtraBold from the bundled folder, not a fallback."""
    if " subtitles " not in ffmpeg.run("ffmpeg", ["-hide_banner", "-filters"]).stdout:
        pytest.skip("this ffmpeg has no libass ('subtitles' filter)")
    video = make_video(duration=1.0, size="270x480")
    ass = tmp_path / "c.ass"
    write_captions(
        _transcript([(0.0, 0.9, "Montserrat")], duration=1.0),
        [ClipSpec.between("t", 0.0, 1.0)],
        [ass],
    )
    result = ffmpeg.run(
        "ffmpeg",
        [
            "-hide_banner",
            "-v",
            "verbose",
            "-nostdin",
            "-i",
            str(video),
            "-vf",
            subtitles_filter(ass),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
    )
    chosen = [line for line in result.stderr.splitlines() if "fontselect" in line]
    assert chosen, "libass reported no font selection"
    assert all("Montserrat" in line for line in chosen), chosen
