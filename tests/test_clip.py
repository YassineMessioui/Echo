import http.server
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from echo import clip as clip_module
from echo import ffmpeg as ffmpeg_module
from echo import progress
from echo.captions import subtitles_filter
from echo.clip import (
    MANIFEST,
    ClipStage,
    build_ffmpeg_args,
    captions_path,
    clip_paths,
    cut_clips,
)
from echo.config import CaptionConfig, ClipConfig, PipelineConfig
from echo.errors import ArtifactError, ExternalToolError, InvalidInputError
from echo.ffmpeg import probe
from echo.interfaces import RemoteVideo
from echo.job import Job
from echo.models import ClipSpec, StageName
from echo.reframe import TARGET


@dataclass
class RecordingClipper:
    """Fake ``Clipper``: records calls and writes a placeholder file per output."""

    calls: list[tuple[list[ClipSpec], list[Path]]] = field(default_factory=list)
    fail_on: int | None = None
    """Raise on this call number (0-based), like an ffmpeg crash mid-run."""

    extra_filters: list[str | None] = field(default_factory=list)

    def __call__(self, video, clips, outputs, config, *, extra_filter=None) -> None:
        if self.fail_on == len(self.calls):
            raise ExternalToolError("boom")
        self.calls.append((clips, outputs))
        self.extra_filters.append(extra_filter)
        for out in outputs:
            out.write_bytes(b"clip")

    @property
    def cut(self) -> list[tuple[ClipSpec, Path]]:
        """Every (clip, output) cut, across calls."""
        return [pair for clips, outs in self.calls for pair in zip(clips, outs, strict=True)]


def _job_with_video(job_copy: Job) -> Job:
    job_copy.input_video.write_bytes(b"not really a video")
    return job_copy


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


# --- ffmpeg arguments -------------------------------------------------------------


def test_args_seek_accurately_and_reencode(tmp_path):
    spec = ClipSpec.between("t", 1.5, 4.25)
    config = ClipConfig(crf=23, preset="fast")
    args = build_ffmpeg_args(tmp_path / "in.mp4", spec, tmp_path / "out.tmp", config)
    assert args[: args.index("-i")] == ["-accurate_seek", "-ss", "1.500"]
    assert args[args.index("-t") + 1] == "2.750"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-c:a") + 1] == "aac"
    assert args[args.index("-crf") + 1] == "23"
    assert args[args.index("-preset") + 1] == "fast"
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-movflags") + 1] == "+faststart"
    assert "-vf" not in args
    assert args[-1] == str(tmp_path / "out.tmp")


def test_args_video_filter_hook(tmp_path):
    spec = ClipSpec.between("t", 0, 2)
    args = build_ffmpeg_args(
        tmp_path / "in.mp4", spec, tmp_path / "o", ClipConfig(), video_filter="scale=1080:1920"
    )
    assert args[args.index("-vf") + 1] == "scale=1080:1920"
    assert args.index("-vf") > args.index("-i")


# --- cut_clips ----------------------------------------------------------------------


def test_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="2 clips but 1 output"):
        cut_clips(
            tmp_path / "in.mp4",
            [ClipSpec.between("a", 0, 1), ClipSpec.between("b", 1, 2)],
            [tmp_path / "a.mp4"],
            ClipConfig(),
        )


def test_missing_video(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        cut_clips(tmp_path / "nope.mp4", [], [], ClipConfig())


def test_failure_cleans_up_temp_file(tmp_path, monkeypatch):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")

    def failing_ffmpeg(args: list[str]) -> None:
        Path(args[-1]).write_bytes(b"partial")  # ffmpeg dies mid-write
        raise ExternalToolError("ffmpeg exited with code 1:\nboom")

    monkeypatch.setattr(clip_module, "ffmpeg", failing_ffmpeg)
    out_dir = tmp_path / "clips"
    with pytest.raises(ExternalToolError, match="boom"):
        cut_clips(
            video,
            [ClipSpec.between("a", 0, 1)],
            [out_dir / "01-a.mp4"],
            ClipConfig(aspect="source"),
        )
    assert list(out_dir.iterdir()) == []


@pytest.mark.ffmpeg
@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0.0, 3.0),
        (1.23, 4.56),  # keyframes are every 2 s (GOP 50 at 25 fps): not on one
        (5.0, 9.9),  # near the end of the source
    ],
)
def test_clip_duration_is_accurate(make_video, tmp_path, start, end):
    video = make_video(duration=10, fps=25)
    spec = ClipSpec.between("t", start, end)
    out = tmp_path / "clips" / "01-t.mp4"
    cut_clips(video, [spec], [out], ClipConfig())
    info = probe(out)
    assert info.duration == pytest.approx(spec.duration, abs=0.1)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == TARGET  # default aspect is 9:16
    assert _leftovers(out.parent) == []


@pytest.mark.ffmpeg
def test_cuts_several_clips(make_video, tmp_path):
    video = make_video(duration=10, fps=25)
    specs = [ClipSpec.between("a", 0.5, 2.5), ClipSpec.between("b", 3.37, 7.0)]
    outs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    cut_clips(video, specs, outs, ClipConfig())
    for spec, out in zip(specs, outs, strict=True):
        assert probe(out).duration == pytest.approx(spec.duration, abs=0.1)


@pytest.mark.ffmpeg
def test_video_without_audio(make_video, tmp_path):
    video = make_video(duration=4, audio=False)
    out = tmp_path / "a.mp4"
    cut_clips(video, [ClipSpec.between("a", 1, 3)], [out], ClipConfig())
    info = probe(out)
    assert info.has_video and not info.has_audio
    assert info.duration == pytest.approx(2, abs=0.1)


@pytest.mark.ffmpeg
def test_video_filter_is_applied(make_video, tmp_path):
    video = make_video(duration=3)
    out = tmp_path / "a.mp4"
    cut_clips(
        video, [ClipSpec.between("a", 0, 2)], [out], ClipConfig(), video_filter="scale=90:160"
    )
    info = probe(out)
    assert (info.width, info.height) == (90, 160)


@pytest.mark.ffmpeg
def test_real_ffmpeg_failure_surfaces(make_video, tmp_path):
    video = make_video(duration=3)
    out_dir = tmp_path / "clips"
    with pytest.raises(ExternalToolError, match="exited with code"):
        cut_clips(
            video,
            [ClipSpec.between("a", 0, 1)],
            [out_dir / "01-a.mp4"],
            ClipConfig(video_codec="no-such-codec"),
        )
    assert list(out_dir.iterdir()) == []


# --- ClipStage ----------------------------------------------------------------------


def test_stage_name():
    assert ClipStage.name == StageName.CLIP


def _paths(job: Job) -> list[Path]:
    return clip_paths(job, job.read_clips())


def test_clip_paths_use_index_and_slug(job_copy):
    assert [p.name for p in _paths(job_copy)] == [
        "01-solar-power-is-now-the-cheapest-electricity.mp4",
        "02-sleep-and-productivity.mp4",
    ]
    assert all(p.parent == job_copy.clips_dir for p in _paths(job_copy))


def test_outputs_include_a_missing_marker_until_cut(job_copy):
    job = _job_with_video(job_copy)
    assert not all(p.exists() for p in ClipStage().outputs(job))
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    assert ClipStage().outputs(job) == _paths(job)
    assert all(p.exists() for p in ClipStage().outputs(job))


def test_outputs_before_refine_ran(tmp_path):
    job = Job(tmp_path / "job")
    outputs = ClipStage().outputs(job)
    assert outputs and not any(p.exists() for p in outputs)


def test_titles_with_same_slug_get_distinct_files(job_copy):
    job_copy.write_clips([ClipSpec.between("Q&A", 0, 20), ClipSpec.between("Q & A!", 20, 40)])
    assert [p.name for p in _paths(job_copy)] == ["01-q-a.mp4", "02-q-a.mp4"]


def test_run_cuts_all_clips(job_copy):
    job = _job_with_video(job_copy)
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert clipper.cut == list(zip(job.read_clips(), _paths(job), strict=True))
    assert all(p.exists() for p in _paths(job))


def test_run_resumes_after_a_crash(job_copy):
    job = _job_with_video(job_copy)
    first, second = _paths(job)
    with pytest.raises(ExternalToolError):
        ClipStage(clipper=RecordingClipper(fail_on=1)).run(job, PipelineConfig())
    assert first.exists() and not second.exists()
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert clipper.cut == [(job.read_clips()[1], second)]


def test_unrecorded_file_is_recut(job_copy):
    """A file the manifest doesn't vouch for (e.g. a leftover) isn't trusted."""
    job = _job_with_video(job_copy)
    first, _ = _paths(job)
    job.clips_dir.mkdir()
    first.write_bytes(b"unknown")
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    assert first.read_bytes() == b"clip"


def test_moved_boundaries_are_recut(job_copy):
    """Re-running refine with new boundaries under the same title must not keep old files."""
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    a, b = job.read_clips()
    job.write_clips([a, ClipSpec.between(b.topic_title, b.start + 1, b.end - 5)])
    assert not all(p.exists() for p in ClipStage().outputs(job)), "stage must not be skipped"
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert clipper.cut == [(job.read_clips()[1], _paths(job)[1])]


def test_changed_encode_settings_are_recut(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig(clip=ClipConfig(crf=28)))
    assert len(clipper.cut) == 2


def test_clips_no_longer_wanted_are_deleted(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    old_second = _paths(job)[1]
    stranger = job.clips_dir / "my-own-edit.mp4"
    stranger.write_bytes(b"not ours")
    job.write_clips(job.read_clips()[:1])
    assert not all(p.exists() for p in ClipStage().outputs(job)), "stage must not be skipped"
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    assert not old_second.exists()
    assert stranger.exists(), "only files the stage wrote are deleted"
    assert sorted(p.name for p in job.clips_dir.glob("*.mp4")) == sorted(
        [_paths(job)[0].name, stranger.name]
    )
    assert (job.clips_dir / MANIFEST).is_file()


def test_run_with_everything_done_does_nothing(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert clipper.calls == []
    assert job.clips_dir.is_dir()


def test_force_recuts_everything(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    clipper = RecordingClipper()
    ClipStage(clipper=clipper, force=True).run(job, PipelineConfig())
    assert clipper.cut == list(zip(job.read_clips(), _paths(job), strict=True))


def test_run_without_input_video(job_copy):
    with pytest.raises(ArtifactError, match="input.mp4"):
        ClipStage(clipper=RecordingClipper()).run(job_copy, PipelineConfig())


def test_run_without_clips_json(tmp_path):
    job = Job(tmp_path / "job")
    job.root.mkdir()
    job.input_video.write_bytes(b"x")
    with pytest.raises(ArtifactError, match="clips.json"):
        ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())


def _no_captions(**sections) -> PipelineConfig:
    """Framing-only stage tests: these jobs have no transcript to caption from."""
    from echo.config import CaptionConfig

    return PipelineConfig(caption=CaptionConfig(enabled=False), **sections)


def test_captions_need_a_transcript(job_copy):
    job = _job_with_video(job_copy)
    job.transcript_path.unlink()
    with pytest.raises(ArtifactError, match="caption.enabled"):
        ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    ClipStage(clipper=RecordingClipper()).run(job, _no_captions())  # fine without captions


@pytest.mark.ffmpeg
def test_stage_end_to_end(make_video, tmp_path):
    job = Job(tmp_path / "job")
    job.root.mkdir()
    make_video(duration=10, name="src.mp4").replace(job.input_video)
    job.write_clips([ClipSpec.between("Intro", 0.4, 3.0), ClipSpec.between("Café ☕", 4.1, 8.0)])
    ClipStage().run(job, _no_captions(clip=ClipConfig(aspect="source")))
    outputs = ClipStage().outputs(job)
    assert outputs == _paths(job)
    assert [p.name for p in outputs] == ["01-intro.mp4", "02-cafe.mp4"]
    for spec, out in zip(job.read_clips(), outputs, strict=True):
        assert probe(out).duration == pytest.approx(spec.duration, abs=0.1)
    assert _leftovers(job.clips_dir) == []


@pytest.mark.ffmpeg
@pytest.mark.parametrize("size", ["320x180", "180x320", "240x240"])
def test_clips_are_vertical_by_default(make_video, tmp_path, size):
    video = make_video(duration=4, size=size)
    spec = ClipSpec.between("t", 0.5, 3.0)
    out = tmp_path / "clips" / "01-t.mp4"
    cut_clips(video, [spec], [out], ClipConfig())
    info = probe(out)
    assert (info.width, info.height) == TARGET
    assert info.duration == pytest.approx(spec.duration, abs=0.1)


@pytest.mark.ffmpeg
def test_source_aspect_keeps_framing(make_video, tmp_path):
    video = make_video(duration=3, size="320x180")
    out = tmp_path / "clips" / "01-t.mp4"
    cut_clips(video, [ClipSpec.between("t", 0, 2)], [out], ClipConfig(aspect="source"))
    assert (probe(out).width, probe(out).height) == (320, 180)


@pytest.mark.ffmpeg
def test_stage_outputs_vertical_clips_and_recuts_on_aspect_change(make_video, tmp_path):
    job = Job(tmp_path / "job")
    job.root.mkdir()
    make_video(duration=6, name="src.mp4").replace(job.input_video)
    job.write_clips([ClipSpec.between("Intro", 0.0, 2.5), ClipSpec.between("Outro", 3.0, 5.5)])
    ClipStage().run(job, _no_captions())
    assert all((probe(p).width, probe(p).height) == TARGET for p in _paths(job))
    ClipStage().run(job, _no_captions(clip=ClipConfig(aspect="source")))
    assert all((probe(p).width, probe(p).height) == (320, 180) for p in _paths(job))


def test_extra_filter_is_chained_after_reframing(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    seen: list[list[str]] = []
    monkeypatch.setattr(clip_module, "video_filter_for", lambda v, c: "crop=1:1,scale=1080:1920")
    monkeypatch.setattr(clip_module, "ffmpeg", lambda args: seen.append(args))
    monkeypatch.setattr(clip_module.os, "replace", lambda a, b: None)
    clip = ClipSpec.between("t", 0, 20)
    cut_clips(video, [clip], [tmp_path / "o.mp4"], ClipConfig(), extra_filter="eq=gamma=1.1")
    args = seen[0]
    assert args[args.index("-vf") + 1] == "crop=1:1,scale=1080:1920,eq=gamma=1.1"


def test_extra_filter_without_reframing(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    seen: list[list[str]] = []
    monkeypatch.setattr(clip_module, "ffmpeg", lambda args: seen.append(args))
    monkeypatch.setattr(clip_module.os, "replace", lambda a, b: None)
    clip = ClipSpec.between("t", 0, 20)
    cut_clips(
        video, [clip], [tmp_path / "o.mp4"], ClipConfig(aspect="source"), extra_filter="sub=x"
    )
    args = seen[0]
    assert args[args.index("-vf") + 1] == "sub=x"


# --- captions (#17) -----------------------------------------------------------------


def _caption_config(**caption) -> PipelineConfig:
    from echo.config import CaptionConfig

    return PipelineConfig(caption=CaptionConfig(**caption))


def test_captions_are_written_and_burned_in(job_copy):
    job = _job_with_video(job_copy)
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())

    for path, extra in zip(_paths(job), clipper.extra_filters, strict=True):
        ass = captions_path(path)
        assert ass.read_bytes().startswith(b"\xef\xbb\xbf")
        assert "Dialogue:" in ass.read_text(encoding="utf-8-sig")
        # Burned from a temp copy that was then moved into place.
        assert extra == subtitles_filter(ass.with_name(f".{ass.name}.tmp"))
    assert _leftovers(job.clips_dir) == []


def test_captions_disabled(job_copy):
    job = _job_with_video(job_copy)
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, _caption_config(enabled=False))
    assert clipper.extra_filters == [None, None]
    assert not any(captions_path(p).exists() for p in _paths(job))


def test_transcript_change_recuts_only_affected_clips(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())

    again = RecordingClipper()
    ClipStage(clipper=again).run(job, PipelineConfig())
    assert again.calls == [], "same captions: nothing re-cut"

    # Fix a word in the second clip's segment (e.g. a corrected transcript).
    transcript = job.read_transcript()
    seg = transcript.segments[5]
    word = seg.words[0].model_copy(update={"text": "EDITED"})
    seg = seg.model_copy(update={"words": [word, *seg.words[1:]]})
    segments = [*transcript.segments[:5], seg, *transcript.segments[6:]]
    job.write_transcript(transcript.model_copy(update={"segments": segments}))

    edited = RecordingClipper()
    ClipStage(clipper=edited).run(job, PipelineConfig())
    assert [p.name for _, p in edited.cut] == [_paths(job)[1].name]
    assert "EDITED" in captions_path(_paths(job)[1]).read_text(encoding="utf-8-sig")


def test_style_change_and_toggling_captions_recut(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())

    restyled = RecordingClipper()
    ClipStage(clipper=restyled).run(job, _caption_config(highlight_colour="#00FF00"))
    assert len(restyled.cut) == 2

    off = RecordingClipper()
    ClipStage(clipper=off).run(job, _caption_config(enabled=False))
    assert len(off.cut) == 2
    assert not any(captions_path(p).exists() for p in _paths(job)), "stale .ass removed"


def test_deleted_clips_take_their_captions_with_them(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    gone = _paths(job)[1]
    job.write_clips(job.read_clips()[:1])
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    assert not gone.exists() and not captions_path(gone).exists()


@pytest.mark.ffmpeg
def test_stage_burns_captions_into_vertical_clips(make_video, tmp_path):
    """Real ffmpeg: reframe + captions in one encode, 1080x1920 out, captions visible."""
    if not ffmpeg_module.has_filter("subtitles"):
        pytest.skip("this ffmpeg has no libass ('subtitles' filter)")
    from echo.models import Segment, Transcript, Word

    video = make_video(duration=3.0, size="320x180")
    words = [
        Word(start=0.2 + i * 0.4, end=0.55 + i * 0.4, text=w)
        for i, w in enumerate(["Burned", "in", "captions", "look", "like", "this."])
    ]
    transcript = Transcript(
        language="en",
        duration=3.0,
        segments=[Segment(id=0, start=0.2, end=2.55, text="…", words=words)],
    )

    def run(name: str, config: PipelineConfig) -> bytes:
        job = Job.create(tmp_path / name, video)
        job.write_transcript(transcript)
        job.write_clips([ClipSpec.between("Caption test", 0.0, 2.8)])
        ClipStage().run(job, config)
        out = _paths(job)[0]
        info = probe(out)
        assert (info.width, info.height) == TARGET
        assert info.duration == pytest.approx(2.8, abs=0.1)
        frame = tmp_path / f"{name}.raw"
        ffmpeg_module.ffmpeg(
            [
                "-ss",
                "0.5",
                "-i",
                str(out),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                str(frame),
            ]
        )
        return frame.read_bytes()

    with_captions = run("with", PipelineConfig())
    without = run("without", _caption_config(enabled=False))
    assert with_captions != without


def test_failed_encode_leaves_no_caption_file(job_copy):
    job = _job_with_video(job_copy)
    with pytest.raises(ExternalToolError):
        ClipStage(clipper=RecordingClipper(fail_on=0)).run(job, PipelineConfig())
    assert not captions_path(_paths(job)[0]).exists()
    assert _leftovers(job.clips_dir) == []


def test_ffmpeg_without_libass_gives_a_clear_error(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(clip_module, "has_filter", lambda name: False)
    with pytest.raises(ExternalToolError, match=r"without libass.*caption\.enabled = false"):
        cut_clips(
            video,
            [ClipSpec.between("t", 0, 20)],
            [tmp_path / "o.mp4"],
            ClipConfig(aspect="source"),
            extra_filter="subtitles=filename=a.ass",
        )


@pytest.mark.ffmpeg
@pytest.mark.parametrize("background", ["blur", "black"])
def test_extra_filter_attaches_after_the_blur_graph(make_video, tmp_path, background):
    """Real encode: a 1:1 source goes through the labelled blur/pad graph, and the extra
    filter (no libass needed) must attach to its final output pad."""
    video = make_video(duration=1.0, size="240x240")
    out = tmp_path / "clips" / "01-t.mp4"
    cut_clips(
        video,
        [ClipSpec.between("t", 0.0, 0.8)],
        [out],
        ClipConfig(background=background),
        extra_filter="crop=100:60:0:0",
    )
    assert (probe(out).width, probe(out).height) == (100, 60)


def test_video_filter_and_extra_filter_together(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"x")
    seen: list[list[str]] = []
    monkeypatch.setattr(clip_module, "video_filter_for", lambda v, c: pytest.fail("not probed"))
    monkeypatch.setattr(clip_module, "ffmpeg", lambda args: seen.append(args))
    monkeypatch.setattr(clip_module.os, "replace", lambda a, b: None)
    cut_clips(
        video,
        [ClipSpec.between("t", 0, 20)],
        [tmp_path / "o.mp4"],
        ClipConfig(),
        video_filter="scale=540:960",
        extra_filter="eq=gamma=1.1",
    )
    args = seen[0]
    assert args[args.index("-vf") + 1] == "scale=540:960,eq=gamma=1.1"


# --- remote video (link jobs, #75) -----------------------------------------------------


def test_args_for_remote_video_read_only_the_clip_and_local_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(
        clip_module, "http_options", lambda: frozenset({"request_size", "reconnect"})
    )
    remote = RemoteVideo(
        url="https://video.example/137",
        audio=tmp_path / "input-audio.m4a",
        width=1920,
        height=1080,
        headers={"User-Agent": "UA"},
    )
    args = build_ffmpeg_args(
        remote, ClipSpec.between("t", 12.5, 40.0), tmp_path / "o", ClipConfig()
    )
    assert args[args.index("-headers") + 1] == "User-Agent: UA\r\n"
    assert args[args.index("-request_size") + 1] == str(2**20)  # avoids YouTube throttling
    assert "-reconnect_on_network_error" not in args, "unsupported by this ffmpeg: not passed"
    inputs = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
    assert inputs == ["https://video.example/137", str(remote.audio)]
    assert args.count("-ss") == 2 and args[args.index("-ss") + 1] == "12.500"
    assert args[args.index("-map") + 1 : args.index("-map") + 4] == ["0:v:0", "-map", "1:a:0"]


def test_stage_resolves_the_source_once_and_only_when_cutting(job_copy):
    job = job_copy
    job.input_audio.write_bytes(b"audio")
    remote = RemoteVideo(url="https://v", audio=job.input_audio, width=1920, height=1080)
    resolved = []

    def source(j):
        resolved.append(j)
        return remote

    seen = []

    class Recording(RecordingClipper):
        def __call__(self, video, clips, outputs, config, **kwargs):
            seen.append(video)
            super().__call__(video, clips, outputs, config)

    config = PipelineConfig(caption=CaptionConfig(enabled=False))
    ClipStage(clipper=Recording(), source=source).run(job, config)
    assert len(resolved) == 1 and seen == [remote, remote]
    ClipStage(clipper=Recording(), source=source).run(job, config)  # all up to date
    assert len(resolved) == 1, "nothing to cut: the stream isn't resolved"


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves files with HTTP Range support, as video CDNs do (ffmpeg seeks with it)."""

    root: Path

    def do_GET(self):
        data = (self.root / self.path.lstrip("/")).read_bytes()
        start, end = 0, len(data) - 1
        if rng := self.headers.get("Range"):
            a, _, b = rng.removeprefix("bytes=").partition("-")
            start, end = int(a or 0), int(b) if b else len(data) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])

    def log_message(self, *args):
        pass


@pytest.mark.ffmpeg
def test_cut_from_a_remote_http_video(make_video, tmp_path):
    """Real ffmpeg reading the video over HTTP (local server) + a separate audio file."""
    from echo.ffmpeg import ffmpeg

    src = make_video(duration=6, size="320x180", name="remote.mp4")
    audio = tmp_path / "input-audio.m4a"
    ffmpeg(["-i", str(src), "-vn", "-c:a", "copy", str(audio)])
    handler = type("Handler", (_RangeHandler,), {"root": tmp_path})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/remote.mp4"
        remote = RemoteVideo(url=url, audio=audio, width=320, height=180)
        spec = ClipSpec.between("t", 1.3, 4.1)
        out = tmp_path / "clips" / "01-t.mp4"
        cut_clips(remote, [spec], [out], ClipConfig())
    finally:
        server.shutdown()
    info = probe(out)
    assert (info.width, info.height) == TARGET and info.has_audio
    assert info.duration == pytest.approx(spec.duration, abs=0.1)


def test_run_reports_clips_done(job_copy):
    job = _job_with_video(job_copy)
    n = len(job.read_clips())
    reports: list[tuple[float, str]] = []
    with progress.capture(lambda fraction, detail: reports.append((fraction, detail))):
        ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    assert reports == [(i / n, f"clip {i}/{n}") for i in range(1, n + 1)]


def test_up_to_date_clips_count_as_done(job_copy):
    job = _job_with_video(job_copy)
    ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())
    job.clip_video_path(1, job.read_clips()[0].topic_title).unlink()  # only the 1st is re-cut
    reports: list[tuple[float, str]] = []
    clipper = RecordingClipper()
    with progress.capture(lambda fraction, detail: reports.append((fraction, detail))):
        ClipStage(clipper=clipper).run(job, PipelineConfig())
    n = len(job.read_clips())
    assert len(clipper.cut) == 1
    assert [f for f, _ in reports] == sorted(f for f, _ in reports)
    assert reports[-1] == (1.0, f"clip {n}/{n}")
