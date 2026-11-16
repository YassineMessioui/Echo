# Test fixtures

## `golden/`

Hand-written, mutually consistent artifacts for one fake episode (see
`tests/conftest.py`). No media.

## `podcast/`

A 110 s, 1.3 MB sample "podcast" for the end-to-end test (`tests/test_e2e.py`).

| File | What it is |
|---|---|
| `podcast.mp4` | 320x180, 10 fps H.264 test pattern + 48 kb/s mono AAC speech, 109.5 s |
| `script.txt` | The spoken text: three topics (coffee, bicycles, the deep ocean), ~35 s each |
| `sections.json` | Each topic's time span in the audio; the e2e test's fake LLM answers from it |
| `make_podcast.py` | Regenerates `podcast.mp4` and `sections.json` from `script.txt` |

**Provenance.** Everything is generated, nothing is recorded or downloaded. The script
was written for this repository. The speech is synthesized with the macOS `say` command
(voices Samantha and Daniel); the picture is ffmpeg's `testsrc2` pattern:

    uv run python tests/fixtures/podcast/make_podcast.py   # macOS, needs ffmpeg

The output is not byte-for-byte reproducible across macOS/ffmpeg versions; commit the
regenerated `podcast.mp4` and `sections.json` together.

**Licence.** The script, the generator and the test pattern are original work
released under CC0 1.0 (public domain dedication). The audio is machine speech from Apple's built-in voices. Apple's
macOS licence allows using the system voices to create content for personal,
non-commercial use; this file is used only as a non-commercial test fixture. If that is
ever a problem, regenerate the audio with an openly licensed TTS (e.g. `espeak-ng`,
whose output is unrestricted) by swapping the `say` call in `make_podcast.py`.
