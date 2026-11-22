# Echo

Echo takes a long video, finds the best moments, cuts them into short vertical clips (60 seconds max), and posts them to **TikTok, YouTube Shorts, Instagram Reels and Facebook Reels** automatically.

> **Status:** V0 and V1 are done. `echo-clips run <file or link>` turns a long video into vertical 9:16 clips (15–60 s) with word-highlighted captions, optionally coloured per speaker. It works on full-length episodes and YouTube links, and is resumable. Next: clip scoring (#21), speaker-tracking reframing (#16) and publishing (#67). The stack below can still change.

---

## How it works

```
 ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐
 │  1. Ingest │──▶│ 2. Transcribe│──▶│ 3. Find the  │──▶│ 4. Render  │──▶│  5. Publish  │
 │   video    │   │  + analyze   │   │   moments    │   │   clips    │   │  to platforms│
 └────────────┘   └──────────────┘   └──────────────┘   └────────────┘   └──────────────┘
```

1. **Ingest:** accept a local file or a URL, then normalize it (codec, frame rate, audio).
2. **Transcribe and analyze:** speech-to-text with word-level timestamps, plus signals such as scene changes, audio energy and silence.
3. **Find the moments:** score candidate segments (hooks, punchlines, complete thoughts) and pick the top N. Every clip is **≤ 60 s** and starts and ends on sentence boundaries.
4. **Render clips:**
   - reframe to 9:16 vertical, keeping the speaker or subject centered
   - burn in captions
   - optionally add a title, branding or an outro
5. **Publish:** generate a title, description and hashtags for each clip, then upload it to each enabled platform, either right away or on a schedule.

### The pipeline

Steps 1–4 work today; publishing (5) is next. Each stage reads the previous stage's artifact from the job directory and writes its own:

```mermaid
flowchart LR
  V["video or link"] -->|"input.mp4 / input-audio.m4a"| I["ingest"]
  I -->|"audio.wav"| T["transcribe"]
  T -->|"transcript.json"| D["diarize (optional)"]
  T -->|"transcript.json"| C
  subgraph L["llm"]
    direction LR
    C["chunk"] --> E["extract"] --> M["merge"]
  end
  M -->|"topics.json"| R["refine"]
  T -.->|"transcript.json"| R
  R -->|"clips.json"| K["clip"]
  D -.->|"speakers.json"| K
  V -.->|"video"| K
  K -->|"clips/nn-slug.mp4"| O["9:16 clips + captions"]
```

- **The LLM returns segment ids, never timestamps.** It sees numbered transcript lines and answers with `start_segment`/`end_segment`; refine turns those into times from the transcript. A model can't invent a timestamp, and every clip starts and ends on a segment boundary.
- **Retries:** an answer that isn't valid JSON or uses ids outside its window is sent back to the LLM with the error, up to `llm.max_retries` (2) times. Temporary provider errors (HTTP 429, 502, 503, 504, honouring `Retry-After`) are retried with backoff, so free-tier "high demand" errors don't fail the run. A bad key or unknown model fails at once, with a clear message.
- **Chunking:** long transcripts are split into overlapping windows (5 segments of overlap). The window size follows the model: 3k tokens for Ollama, 12k for long-context hosted models (bigger windows lost topics and misplaced spans in our tests, see [docs/llm-providers.md](docs/llm-providers.md)). Windows are extracted in parallel and topics found in several windows are merged by span overlap.
- **Links:** only the audio is downloaded; the clip stage reads just each clip's section of the video from the site (#75). On a 68-min podcast that's ~67 MB before transcription starts instead of ~240 MB.
- **Captions:** a few words at a time, the spoken word highlighted, in a bundled font (Montserrat ExtraBold). With `--speakers`, each speaker gets their own colour and a line never mixes speakers (#68).
- **Resumability:** a stage is skipped when its outputs exist and its settings haven't changed, so re-running a job resumes at the first stage that hasn't finished, and changing e.g. the whisper model or turning captions on re-runs exactly what it affects. `job.json` records each stage's status and settings. The clip stage re-cuts only the clips whose boundaries or output changed (`clips/manifest.json`). Ctrl+C stops at once (exit 130) and the next run picks up from there.

---

## Roadmap

**V0: video → vertical topic clips** ([milestone](https://github.com/YassineMessioui/Echo/milestone/1)): done
- [x] Data contracts and per-run job directory (#3)
- [x] Audio extraction (#4), transcription with word timestamps (#5)
- [x] Topic extraction with a local or hosted LLM: chunking, extraction, merging (#6–#9); see [docs/llm-providers.md](docs/llm-providers.md)
- [x] Boundary refinement: clips are **15–60 s**, never longer than 60 s (#10)
- [x] Clipping, **vertical 9:16 at 1080×1920** (#11, #26)
- [x] Resumable orchestrator, `echo-clips` CLI, end-to-end test, README (#12–#15)
- [x] Burned-in, word-highlighted captions (#17) in a bundled font (#64)

**V1: long videos and links** ([milestone](https://github.com/YassineMessioui/Echo/milestone/2)): done
- [x] Video links via yt-dlp (#60), downloading only the audio and reading clip sections remotely (#75)
- [x] Batched transcription + VAD (#58), CPU fallback without CUDA (#55)
- [x] Parallel topic extraction with retries (#56), window size per model (#63)
- [x] Focus range `--range` (#59), `--max-clips` (#62)
- [x] Progress inside long stages (#61), a quiet first run (#65), fast Ctrl+C (#83)
- [x] Re-run a stage when its settings change (#82)
- [x] Speaker diarization (#22) and a caption colour per speaker (#68)
- Measured and closed: cutting clips in parallel (#57); ffmpeg already keeps the cores busy

**Next**
- [ ] Clip scoring: keep only the most engaging moments (#21)
- [ ] Smart reframing that follows the speaker (#16)
- [ ] Publishing to YouTube Shorts, TikTok, Instagram Reels and Facebook Reels, with titles, hashtags, a review step and scheduling (epic #67)
- [ ] Web UI, job queue, S3 storage (#19, #20, #23)

---

## Proposed stack

| Layer            | Choice                                              |
|------------------|-----------------------------------------------------|
| Language         | Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Lint / tests     | ruff + pytest, run in GitHub Actions CI (Linux + Windows, plus a real-model e2e job) |
| Video processing | FFmpeg binaries called through `subprocess` (`echo.ffmpeg`) |
| Links            | [yt-dlp](https://github.com/yt-dlp/yt-dlp) (Python API): audio download + remote stream URLs |
| Transcription    | faster-whisper (local), word-level timestamps, optional batching + VAD |
| Speakers         | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): pyannote segmentation 3.0 + NeMo TitaNet-small, offline on CPU (no Hugging Face token, no PyTorch) |
| Moment selection | An LLM behind a swappable `LLMClient`: local through [Ollama](https://ollama.com) or hosted through any OpenAI-compatible API (Groq, Gemini, OpenRouter, Mistral, OpenAI). It returns segment IDs, never timestamps |
| Reframing        | Centre crop to 9:16 for now (position set by `--crop-x`; near-square sources are padded); speaker tracking later (#16) |
| Captions         | ASS subtitles burned in with FFmpeg + libass, bundled Montserrat ExtraBold, colour per speaker |
| Queue / jobs     | Synchronous for now; Celery/RQ + Redis later (#20) |
| Storage          | Local disk first; S3-compatible later (needed for IG/FB, which pull video from a URL) |
| Config           | CLI options → `PipelineConfig` (`echo.config`); API keys from environment variables or a git-ignored `.env` |

---

## Project layout

One module per pipeline stage. Each implements an interface from `echo.interfaces` and
reads/writes artifacts only through `echo.job.Job`; see [docs/contracts.md](docs/contracts.md).

```
Echo/
├── src/echo/
│   ├── models.py, interfaces.py, job.py, config.py, errors.py   # contracts
│   ├── pipeline.py      # orchestrator: order, resume, settings fingerprints
│   ├── progress.py      # progress inside a stage
│   ├── cli.py           # echo-clips
│   ├── download.py      # links: yt-dlp audio download, remote video (#60, #75)
│   ├── ffmpeg.py        # ffmpeg/ffprobe subprocess helpers
│   ├── ingest.py        # video → audio.wav, optional focus range (#4, #59)
│   ├── transcribe.py    # audio.wav → transcript.json (#5, #58)
│   ├── diarize.py       # audio.wav → speakers.json, optional (#22)
│   ├── llm/             # transcript.json → topics.json
│   │   ├── chunking.py  #   overlapping windows, size per model (#6, #63)
│   │   ├── client.py, hosted.py  #   Ollama + hosted LLM clients, retries (#7, #56)
│   │   ├── prompts.py, topics.py   #   topic extraction (#8)
│   │   └── merge.py     #   parallel windows, merge, LLM stage (#9, #56)
│   ├── refine.py        # topics → clips.json, 15–60 s (#10, #62)
│   ├── clip.py, reframe.py  # clips/<nn>-<slug>.mp4, 9:16 (#11, #26)
│   ├── captions.py      # word-highlighted, per-speaker captions (#17, #68)
│   ├── fonts/           # Montserrat ExtraBold (SIL OFL)
│   └── testing.py       # fakes for tests (FakeLLMClient, FakeTranscriber)
├── tests/               # incl. golden fixtures and a two-voice sample podcast
├── docs/
│   ├── cli.md           # echo-clips options
│   ├── contracts.md     # layer contracts
│   └── llm-providers.md # choosing a local or hosted LLM, window sizes
├── .github/workflows/   # CI (lint, test on Linux + Windows, e2e, build) + release
├── pyproject.toml
└── README.md
```

Next: `publish/` with one module per platform (#67).

---

## Platform notes

Every platform needs its own developer app and OAuth credentials. Check each one early, because approval can take weeks.

| Platform          | API                              | Notes / gotchas |
|-------------------|----------------------------------|-----------------|
| YouTube Shorts    | YouTube Data API v3 (`videos.insert`) | Vertical video counts as a Short. The default daily quota is limited, and an unverified app may have uploads locked to private. |
| TikTok            | Content Posting API              | Requires an app review. Until the app passes an audit, posts are restricted (e.g. private only). |
| Instagram Reels   | Instagram Graph API              | Needs a Business or Creator account linked to a Facebook Page. Video must be at a **public URL**. |
| Facebook Reels    | Facebook Graph API (Page videos/Reels) | Posts to a Page, not a personal profile. Needs Page permissions through app review. |

> Verify the current limits and requirements on each platform's developer docs before building. They change often.

---

## Getting started

### Prerequisites

- **Python 3.12** and [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv can install Python for you).
- **ffmpeg** and `ffprobe` on your PATH, built **with libass** for burned-in captions: `sudo apt install ffmpeg` (Debian/Ubuntu) and `choco install ffmpeg` (Windows) include it. Homebrew's default `ffmpeg` (macOS) does not; install `ffmpeg-full` (keg-only) and put it first on your PATH:
  ```bash
  brew install ffmpeg-full
  echo 'export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"' >> ~/.zshrc
  ```
  Or run without captions (`--no-captions`). ffmpeg 9+ also reads link clips faster (`-request_size`).
- **For links:** a JavaScript runtime (`deno`, `node` or `bun`) on PATH, which YouTube needs.
- **An LLM** for topic extraction. The default is local [Ollama](https://ollama.com):
  ```bash
  ollama pull llama3.2:3b
  ```
  Or a hosted provider (Groq, Gemini, OpenRouter, …) with an API key: copy `.env.example` to `.env` (git-ignored) and fill in your key, e.g. `GEMINI_API_KEY=...`; `echo-clips` loads it automatically. See [docs/llm-providers.md](docs/llm-providers.md). Tests don't need either.

The Whisper model (`small` by default) downloads on the first run. With `--speakers`, the diarization models (~46 MB) download once to `~/.cache/echo/models`.

**Optional: NVIDIA GPU.** Transcription uses the GPU when one is present and its CUDA libraries load; otherwise it runs on the CPU (with one warning line). faster-whisper needs **cuBLAS for CUDA 12** and **cuDNN 9**:
- **Linux:** `uv pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`, then put their `lib` folders on `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=$(.venv/bin/python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
  ```
- **Windows:** install the [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) and [cuDNN 9](https://developer.nvidia.com/cudnn-downloads) and make sure their `bin` folders are on `PATH` (`cublas64_12.dll`, `cudnn64_9.dll`).

Silence and music are skipped by voice activity detection (`TranscribeConfig.vad_filter`). On a GPU, `TranscribeConfig.batch_size = 8` decodes several ~30 s chunks at once. On a CPU it is ~2× faster but drops ~3 % of the words; a smaller model run sequentially is a better trade, so batching is off by default (benchmarks in #58). Changing the whisper settings (`model_size`, `language`, batched vs sequential, `vad_*`) on an existing job re-transcribes it automatically, and re-runs the later stages; `device`, `compute_type` and the batch size alone don't. `job.json` records the mode as `models.whisper_mode` (e.g. `"batched-8, vad"`).

### Setup

```bash
git clone https://github.com/YassineMessioui/Echo.git && cd Echo
uv sync                 # creates .venv and installs deps (incl. dev)
```

### Usage

The CLI is called `echo-clips` because `echo` clashes with the shell builtin:

```bash
uv run echo-clips run episode.mp4                          # → jobs/<job_id>/clips/*.mp4
uv run echo-clips run "https://youtu.be/VIDEO_ID" --llm-provider gemini   # a link
uv run echo-clips run "https://youtu.be/VIDEO_ID" --llm-provider gemini --speakers  # colour per speaker
uv run echo-clips run episode.mp4 --max-clips 5            # keep at most 5 clips
uv run echo-clips run episode.mp4 --range 10%:90%          # skip intro and outro
uv run echo-clips run episode.mp4 --job-id ep42            # name the job; re-run to resume it
uv run echo-clips run episode.mp4 --job-id ep42 --from-stage refine --max-duration 45
uv run echo-clips run episode.mp4 --job-id ep42 --force    # redo every stage
uv run echo-clips run episode.mp4 --aspect source          # keep the original framing
uv run echo-clips run episode.mp4 --no-captions            # no burned-in captions
```

Put quotes around links: they contain `?` and `&`, which the shell would otherwise interpret.

- **Links** (YouTube, Vimeo, …): only the audio is downloaded; clips are read straight from the site. Running the same link again resumes its job. Only clip videos you have the rights to use (see Open questions).
- **Clips** are **9:16 at 1080×1920** by default. `--aspect source` keeps the source framing; `--crop-x` moves the crop window (0 = left, 1 = right).
- **`--speakers`** detects who speaks when (about 5 min of CPU per hour of audio, so it's off by default) and gives each speaker's captions their own colour. `--num-speakers N` fixes the count.
- **`--range`** only processes part of the video; clips keep original-video times. It's opt-in: in our tests about a third of the best clips sat outside 15–85 %.
- **Re-running a job** skips the stages that are done; changed settings re-run what they affect. `--force` redoes every stage; `--from-stage <name>` redoes that stage and everything after it.
- **Progress:** long stages show a live line (`[2/5] transcribe   42%  12:30 / 58:10`); Ctrl+C stops at once.
- Every option and its default: [docs/cli.md](docs/cli.md).

---

## Job directory and data contracts

Each run gets a directory, written only through `echo.job.Job` (atomic writes, validated reads):

```
jobs/<job_id>/
├── input.mp4              # local file: link to (or copy of) the source video
├── input-audio.m4a        # link: the downloaded audio (the video stays remote)
├── audio.wav              # ingest: 16 kHz mono PCM
├── transcript.json        # transcribe
├── speakers.json          # diarize (only with --speakers)
├── topics.json            # llm
├── scores.json            # score: topics.json with an engagement score per topic
├── clips.json             # refine
├── clips/
│   ├── 01-<slug>.mp4      # clip, one per entry in clips.json
│   ├── 01-<slug>.ass      # its caption track (kept for inspection)
│   └── manifest.json      # clip: fingerprints used to skip unchanged clips
└── job.json               # orchestrator: stage status, settings and models used
```

The JSON files are the Pydantic models in [`echo.models`](src/echo/models.py); layer rules are in [docs/contracts.md](docs/contracts.md). Times are in seconds.

**`transcript.json`** (`Transcript`): segment ids run from 0 with no gaps; `language` is detected unless `--language` is set; `probability` may be `null`. Times are in original-video time: with `--range`, `offset` is where the audio starts.

```json
{
  "language": "en",
  "duration": 80.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 8.5,
      "text": "Welcome back to the show, today we have a lot to cover.",
      "words": [{ "start": 0.0, "end": 0.708, "text": "Welcome", "probability": 0.95 }]
    }
  ]
}
```

**`speakers.json`** (`Speakers`, only with `--speakers`): who speaks when, speakers named `S1`, `S2`, … in order of first appearance.

```json
{ "turns": [{ "start": 0.0, "end": 33.9, "speaker": "S1" }, { "start": 35.4, "end": 72.5, "speaker": "S2" }] }
```

**`topics.json`** (`list[Topic]`): inclusive segment spans.

```json
[
  {
    "title": "Sleep and productivity",
    "summary": "How even small sleep losses hurt focus and reaction time.",
    "start_segment": 4,
    "end_segment": 7
  }
]
```

**`scores.json`** (`list[Topic]`, #21): the topics of `topics.json`, same order and spans, with `score` set: each criterion 0–10, and `overall`, the weighted mean that `--max-clips` ranks by. `topics.json` itself has no `score`.

```json
[
  {
    "title": "Sleep and productivity",
    "summary": "How even small sleep losses hurt focus and reaction time.",
    "start_segment": 4,
    "end_segment": 7,
    "score": { "overall": 7.6, "hook": 8, "payoff": 8, "clarity": 8, "energy": 6, "reason": "Opens on a relatable claim and lands a concrete takeaway." }
  }
]
```

**`clips.json`** (`list[ClipSpec]`): `duration` is `end - start`, rounded to milliseconds.

```json
[{ "topic_title": "Sleep and productivity", "start": 34.0, "end": 80.2, "duration": 46.2 }]
```

**`job.json`** (`JobMeta`): `status` is `pending`, `running`, `done`, `skipped` or `failed`; `error` is set on failure (`"interrupted"` after Ctrl+C); `settings` fingerprints the config a stage's outputs depend on, so changing it re-runs the stage. Link jobs also record `source_url` and `source_info` (title, uploader, duration).

```json
{
  "job_id": "ep42",
  "source": "/videos/episode.mp4",
  "created_at": "2026-11-16T12:00:00Z",
  "stages": {
    "ingest": { "status": "done", "started_at": "2026-11-16T12:00:00Z", "finished_at": "2026-11-16T12:00:03Z", "error": null, "settings": "range=:" },
    "transcribe": { "status": "done", "started_at": "2026-11-16T12:00:03Z", "finished_at": "2026-11-16T12:01:10Z", "error": null, "settings": "small|lang=auto|sequential|vad(min_silence=2000ms)" },
    "diarize": { "status": "pending", "started_at": null, "finished_at": null, "error": null, "settings": null },
    "llm": { "status": "failed", "started_at": "2026-11-16T12:01:10Z", "finished_at": "2026-11-16T12:01:11Z", "error": "...", "settings": null },
    "refine": { "status": "pending", "started_at": null, "finished_at": null, "error": null, "settings": null },
    "clip": { "status": "pending", "started_at": null, "finished_at": null, "error": null, "settings": null }
  },
  "models": { "whisper": "faster-whisper/small", "whisper_mode": "sequential, vad" }
}
```

**`clips/manifest.json`**: file name → hashes of its `ClipSpec` and `ClipConfig`. A clip is re-cut when either changes; files no longer in `clips.json` are deleted.

```json
{
  "01-sleep-and-productivity.mp4": { "clip": "9c1e4f0a2b7d3e58", "config": "4a7b0c2d9e1f3a65" }
}
```

---

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format (CI runs --check)
uv run pytest           # tests; real-model tests are skipped by default
uv run pytest -m model  # also run real models: Whisper, diarization (slow, downloads weights)
```

Tests marked `ffmpeg` use the real binaries and are skipped if ffmpeg isn't installed. Develop against the golden fixtures and fakes described in [docs/contracts.md](docs/contracts.md).

**Pull requests:** every change goes through a PR to `main` from a branch. It can be merged once the required checks pass: `lint`, `test (ubuntu-latest)` and `test (windows-latest)`, with the branch up to date and review conversations resolved. The `e2e` job (real Whisper, diarization and ffmpeg on the sample podcast) runs on every PR too. Any collaborator can merge once green; ask the other owner for a review on anything non-trivial. Contract changes (`models.py`, `interfaces.py`, `job.py`, `errors.py`, `config.py`, golden fixtures) get their own PR.

**Releasing:** bump `version` in `pyproject.toml`, then push a tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The release workflow runs the tests, builds the package and publishes a GitHub Release with the wheel and sdist attached.

---

## Configuration

Echo is configured through `echo-clips` options, which map to `PipelineConfig` in [`echo.config`](src/echo/config.py); API keys come from environment variables or `.env`. A config file is planned for later (draft):

```yaml
clips:
  max_count: 5
  max_duration_s: 60
  min_duration_s: 15
  aspect: "9:16"
captions:
  enabled: true
  style: bold-center
publish:
  youtube:   { enabled: true,  privacy: private }
  tiktok:    { enabled: false }
  instagram: { enabled: false }
  facebook:  { enabled: false }
  require_approval: true
```

---

## Team

- **Imad Ait Ben Salah** ([Imdbn](https://github.com/Imdbn)): co-owner
- **Yassine Messioui** ([YassineMessioui](https://github.com/YassineMessioui)): co-owner

## Open questions

- Local processing or cloud (GPU for Whisper and face tracking)?
- Who owns the source videos? We need a rights and permissions policy before posting content we didn't create.
- Is this a CLI tool for ourselves, or a SaaS with user accounts?
- Which LLM gives the best topic boundaries? So far: Groq's `openai/gpt-oss-120b` found exactly the right three topics on a 2-minute test podcast, and Gemini's free `gemini-flash-lite-latest` works on hour-long episodes (its free tier is often overloaded; Echo retries). Still to compare on more real episodes, and locally with Ollama.
- How many clips should an episode give? A 68-min podcast gave 21; clip scoring (#21) should pick the best few.

## License

TBD
