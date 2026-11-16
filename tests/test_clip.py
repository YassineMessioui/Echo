from dataclasses import dataclass, field
from pathlib import Path

import pytest

from echo import clip as clip_module
from echo.clip import ClipStage, build_ffmpeg_args, cut_clips
from echo.config import ClipConfig, PipelineConfig
from echo.errors import ArtifactError, ExternalToolError, InvalidInputError
from echo.ffmpeg import probe
from echo.job import Job
from echo.models import ClipSpec, StageName


@dataclass
class RecordingClipper:
    """Fake ``Clipper``: records calls and writes a placeholder file per output."""

    calls: list[tuple[list[ClipSpec], list[Path]]] = field(default_factory=list)

    def __call__(self, video, clips, outputs, config) -> None:
        self.calls.append((clips, outputs))
        for out in outputs:
            out.write_bytes(b"clip")


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
        cut_clips(video, [ClipSpec.between("a", 0, 1)], [out_dir / "01-a.mp4"], ClipConfig())
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
    assert (info.width, info.height) == (320, 180)
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


def test_outputs_use_index_and_slug(job_copy):
    assert [p.name for p in ClipStage().outputs(job_copy)] == [
        "01-solar-power-is-now-the-cheapest-electricity.mp4",
        "02-sleep-and-productivity.mp4",
    ]
    assert all(p.parent == job_copy.clips_dir for p in ClipStage().outputs(job_copy))


def test_outputs_before_refine_ran(tmp_path):
    job = Job(tmp_path / "job")
    outputs = ClipStage().outputs(job)
    assert outputs and not any(p.exists() for p in outputs)


def test_titles_with_same_slug_get_distinct_files(job_copy):
    job_copy.write_clips([ClipSpec.between("Q&A", 0, 20), ClipSpec.between("Q & A!", 20, 40)])
    assert [p.name for p in ClipStage().outputs(job_copy)] == ["01-q-a.mp4", "02-q-a.mp4"]


def test_run_cuts_all_clips(job_copy):
    job = _job_with_video(job_copy)
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert len(clipper.calls) == 1
    clips, outputs = clipper.calls[0]
    assert clips == job.read_clips()
    assert outputs == ClipStage().outputs(job)
    assert all(p.exists() for p in outputs)


def test_run_skips_existing_clips(job_copy):
    job = _job_with_video(job_copy)
    first, second = ClipStage().outputs(job)
    job.clips_dir.mkdir()
    first.write_bytes(b"already cut")
    clipper = RecordingClipper()
    ClipStage(clipper=clipper).run(job, PipelineConfig())
    assert clipper.calls == [([job.read_clips()[1]], [second])]
    assert first.read_bytes() == b"already cut"


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
    assert clipper.calls == [(job.read_clips(), ClipStage().outputs(job))]


def test_run_without_input_video(job_copy):
    with pytest.raises(ArtifactError, match="input.mp4"):
        ClipStage(clipper=RecordingClipper()).run(job_copy, PipelineConfig())


def test_run_without_clips_json(tmp_path):
    job = Job(tmp_path / "job")
    job.root.mkdir()
    job.input_video.write_bytes(b"x")
    with pytest.raises(ArtifactError, match="clips.json"):
        ClipStage(clipper=RecordingClipper()).run(job, PipelineConfig())


@pytest.mark.ffmpeg
def test_stage_end_to_end(make_video, tmp_path):
    job = Job(tmp_path / "job")
    job.root.mkdir()
    make_video(duration=10, name="src.mp4").replace(job.input_video)
    job.write_clips([ClipSpec.between("Intro", 0.4, 3.0), ClipSpec.between("Café ☕", 4.1, 8.0)])
    ClipStage().run(job, PipelineConfig(clip=ClipConfig(aspect="source")))
    outputs = ClipStage().outputs(job)
    assert [p.name for p in outputs] == ["01-intro.mp4", "02-cafe.mp4"]
    for spec, out in zip(job.read_clips(), outputs, strict=True):
        assert probe(out).duration == pytest.approx(spec.duration, abs=0.1)
    assert _leftovers(job.clips_dir) == []
