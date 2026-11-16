"""Regenerate ``podcast.mp4`` and ``sections.json`` from ``script.txt``.

    uv run python tests/fixtures/podcast/make_podcast.py

macOS only: the speech comes from the built-in ``say`` command. Needs ffmpeg/ffprobe.
Each section of the script is spoken with its voice, sections are separated by
``GAP`` seconds of silence, and the audio is muxed over a 320x180 ffmpeg test
pattern. ``sections.json`` records each topic's time span, which the end-to-end test's
fake LLM uses to answer with segment ids that belong to that topic.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
GAP = 1.5  # seconds of silence between topics
RATE = 175  # words per minute for `say`


def parse_script(path: Path) -> list[tuple[str, str, str]]:
    """``(voice, title, text)`` for every ``## voice | title`` section."""
    sections: list[tuple[str, str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            voice, title = (part.strip() for part in line[3:].split("|", 1))
            sections.append((voice, title, []))
        elif line.strip() and not line.startswith("#"):
            sections[-1][2].append(line.strip())
    return [(voice, title, " ".join(text)) for voice, title, text in sections]


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return float(out)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    sections = parse_script(HERE / "script.txt")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        silence = tmp / "gap.wav"
        ffmpeg("-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono:d={GAP}", str(silence))

        parts: list[Path] = []
        spans: list[dict[str, object]] = []
        t = 0.0
        for i, (voice, title, text) in enumerate(sections):
            aiff, wav = tmp / f"{i}.aiff", tmp / f"{i}.wav"
            subprocess.run(["say", "-v", voice, "-r", str(RATE), "-o", str(aiff), text], check=True)
            ffmpeg("-i", str(aiff), "-ar", "22050", "-ac", "1", str(wav))
            if parts:
                parts.append(silence)
                t += GAP
            length = duration(wav)
            spans.append({"title": title, "start": round(t, 3), "end": round(t + length, 3)})
            parts.append(wav)
            t += length

        concat = tmp / "list.txt"
        concat.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        audio = tmp / "audio.wav"
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(audio))

        total = duration(audio)
        ffmpeg(
            "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=10:duration={total}",
            "-i", str(audio),
            "-c:v", "libx264", "-preset", "veryslow", "-crf", "36", "-pix_fmt", "yuv420p",
            "-g", "50", "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-shortest",
            "-map_metadata", "-1", "-movflags", "+faststart",
            str(HERE / "podcast.mp4"),
        )  # fmt: skip

    (HERE / "sections.json").write_text(
        json.dumps({"duration": round(total, 3), "sections": spans}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
