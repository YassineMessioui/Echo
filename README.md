# Echo

Echo takes a long video, finds the best moments, cuts them into short vertical clips (60 seconds max), and posts them to **TikTok, YouTube Shorts, Instagram Reels and Facebook Reels** automatically.

> **Status:** early planning. This README describes what we intend to build. The stack below is a proposal and can change.

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

---

## Features

**MVP**
- [ ] CLI: `echo run <video>` produces clips in `output/`
- [ ] Transcription with word timestamps
- [ ] Highlight selection (LLM over the transcript)
- [ ] Cutting plus 9:16 reframe (center crop first, face tracking later)
- [ ] Burned-in captions
- [ ] Upload to YouTube Shorts (the easiest API to start with)

**Next**
- [ ] TikTok, Instagram and Facebook uploaders
- [ ] Face or speaker tracking for smart reframing
- [ ] Review step: approve or reject clips before they are published
- [ ] Scheduling and a posting queue
- [ ] Simple web dashboard
- [ ] Per-platform analytics fed back into clip scoring

---

## Proposed stack

| Layer            | Choice (proposed)                                   |
|------------------|-----------------------------------------------------|
| Language         | Python 3.12, managed with [uv](https://docs.astral.sh/uv/) |
| Lint / tests     | ruff + pytest, run in GitHub Actions CI             |
| Video processing | FFmpeg (via `ffmpeg-python` or subprocess)          |
| Transcription    | Whisper / faster-whisper (local) or a hosted STT API |
| Moment selection | LLM over the timestamped transcript, plus heuristics |
| Reframing        | MediaPipe / OpenCV face detection                   |
| Captions         | ASS subtitles burned in with FFmpeg                 |
| Queue / jobs     | Start synchronous; move to Celery/RQ + Redis later  |
| Storage          | Local disk first; S3-compatible later (needed for IG/FB, which pull video from a URL) |
| Config           | `.env` + `config.yaml`                              |

---

## Project layout (planned)

```
Echo/
├── src/echo/
│   ├── ingest/        # download + normalize input
│   ├── analyze/       # transcription, scene/audio signals
│   ├── select/        # pick highlight segments
│   ├── render/        # cut, reframe, captions
│   ├── publish/       # one module per platform
│   │   ├── youtube.py
│   │   ├── tiktok.py
│   │   ├── instagram.py
│   │   └── facebook.py
│   └── cli.py
├── tests/
├── .github/workflows/ # CI (lint, test, build) + release
├── config.example.yaml
├── .env.example
├── pyproject.toml
└── README.md
```

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

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/) and `ffmpeg` on your PATH.

```bash
git clone https://github.com/YassineMessioui/Echo.git && cd Echo
uv sync                 # creates .venv and installs deps (incl. dev)
uv run echo             # run the CLI
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

Planned once the pipeline exists:

```bash
cp .env.example .env              # add API keys / OAuth secrets
cp config.example.yaml config.yaml
uv run echo run path/to/video.mp4 --max-clips 5 --dry-run
```

`--dry-run` renders the clips without uploading anything.

## Contributing

`main` is protected: every change goes through a pull request, and the `lint` and `test` CI checks must pass before merging.

---

## Configuration (draft)

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
- Which LLM for moment selection, and what is the budget per video?

## License

TBD
