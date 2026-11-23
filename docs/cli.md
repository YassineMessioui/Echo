# `echo-clips` CLI

The command line is the V0 entry point (#13). It builds a `PipelineConfig` from the
options and calls `echo.pipeline.run_pipeline(job, build_stages(config), config, ...)`.

```
echo-clips run VIDEO [OPTIONS]
```

`VIDEO` is a local file or an `http(s)://` link (YouTube, Vimeo, …). For a link,
[yt-dlp](https://github.com/yt-dlp/yt-dlp) downloads **only the audio** into the job
folder, which is all transcription needs; the clip stage then reads just each clip's
section of the video (up to 4K, SDR) straight from the site. On a 68-min podcast that's
~67 MB before transcription can start instead of ~240 MB. Sites whose streams can't be
read that way fall back to downloading the full video. Without `--job-id`, the job is named after
the site and video id (e.g. `youtube-jNQXAC9IVRw`), so running the same link again
resumes that job instead of downloading it again. YouTube needs a JavaScript runtime
(deno, node or bun on `PATH`). Only clip content you have the rights to use.

| Option | Default | Maps to |
|---|---|---|
| `--job-id TEXT` | random 12-char hex | `Job.create(..., job_id=)`; reusing an id resumes that job |
| `--jobs-dir PATH` | `jobs` | `PipelineConfig.jobs_dir` |
| `--force` | off | `run_pipeline(force=True)`: re-run every stage |
| `--from-stage [ingest\|transcribe\|diarize\|llm\|score\|refine\|clip]` | none | `run_pipeline(from_stage=)`: re-run that stage and every later one |
| `--range START:END` | whole video | `IngestConfig.start` / `end`: only process that part (see [Focus range](#focus-range)) |
| `--whisper-model TEXT` | `small` | `TranscribeConfig.model_size` |
| `--language TEXT` | auto-detect | `TranscribeConfig.language` |
| `--llm-provider TEXT` | `ollama` | `LLMConfig.provider` (see [llm-providers.md](llm-providers.md)) |
| `--llm-model TEXT` | `llama3.2:3b` | `LLMConfig.model` |
| `--llm-concurrency INT` | 4 hosted, 1 Ollama | `LLMConfig.max_concurrency` (1–16): transcript windows sent to the LLM at once; use 1 if a free tier keeps rate-limiting you |
| `--min-duration FLOAT` | `15` | `RefineConfig.min_duration` (seconds) |
| `--max-duration FLOAT` | `60` | `RefineConfig.max_duration` (seconds; 60 is the platform cap) |
| `--max-clips INT` | all | `RefineConfig.max_clips`: keep the N best-scored clips (ties: closest to ~37 s), in time order; with `--no-score`, the N closest to ~37 s. Changing it on an existing job re-runs refine and clip |
| `--score / --no-score` | on | `ScoreConfig.enabled`: one extra LLM call rates every clip 0–10 for hook, payoff, standalone clarity and energy (`scores.json`, #21), so `--max-clips` keeps the best and the output lists clips best first |
| `--aspect [9:16\|source]` | `9:16` | `ClipConfig.aspect` |
| `--crop-x FLOAT` | `0.5` | `ClipConfig.crop_x` (0 = left, 1 = right) |
| `--reframe [centre\|face\|speaker]` | `centre` | `ClipConfig.reframe`: `face` puts the 9:16 window on the faces of each shot (camera cuts are detected; `--crop-x` where there's no face). Adds ~3–4 s per clip; the face model (230 KB, MIT) downloads once to `~/.cache/echo/models`. `speaker` frames like `face` until #109 |
| `--background [blur\|black]` | `blur` | `ClipConfig.background` |
| `--captions / --no-captions` | on | `CaptionConfig.enabled` (burned-in, word-highlighted) |
| `--speakers / --no-speakers` | off | `DiarizeConfig.enabled`: detect who speaks when (`speakers.json`); captions get a colour per speaker. Adds ~5 min of CPU per hour of audio; the models (~46 MB) download once to `~/.cache/echo/models` (`ECHO_MODELS_DIR`) |
| `--num-speakers N` | estimate | `DiarizeConfig.num_speakers`: exactly N speakers. When estimating, speakers with similar voices are merged (`merge_threshold`) and speakers with under 2 % of the speech (laughter, "mm-hm") are folded into the nearest speaker in time (`min_speaker_share`) |
| `-v, --verbose` | off | debug logging |

Option defaults come from `PipelineConfig`, so the two can't drift. Hosted LLM providers
read their API key from the environment (e.g. `GROQ_API_KEY`); see
[llm-providers.md](llm-providers.md).

## Focus range

`--range START:END` transcribes and searches only part of the video, e.g. to skip an
intro and outro. Each side is seconds (`120`, `90.5`) or a percentage of the full
duration (`15%`), and either may be left empty: `15%:85%`, `120:3300`, `:85%`, `10%:`.
Ingest extracts only that span; the transcript records where it starts
(`Transcript.offset`), so every timestamp, `clips.json` included, stays in
original-video time and clips are cut from the original. `job.json` records the
resolved span (`models.ingest_range`, in seconds), and changing `--range` on an existing
job re-runs every stage from ingest.

A malformed range (`abc`, `85%:15%`, `150%:`) exits with code 2. A range that only
turns out wrong once the video's duration is known (past the end, or a mix of seconds and
percent that ends up empty or inverted) fails the ingest stage with code 1.

## Output

One line per stage while it runs, e.g.

```
[1/6] ingest      done      2.1s
[2/6] transcribe  skipped
...
```

While a long stage runs (transcribe, llm, clip), a terminal shows one line redrawn in
place with its percentage, detail and elapsed time, e.g.
`[2/6] transcribe   42%  12:30 / 58:10  31.0s`, replaced by the stage's final line. When
the output isn't a terminal (piped, redirected), the same line is printed plainly each
time the stage passes another 10%, with no control characters. `--verbose` also uses the
plain lines, since its log lines would break a line redrawn in place.

At the end it prints the job directory and the path of every clip; with scores, best
first, each with its score and the LLM's reason:

```
clips (5, best first):
   9.0  jobs/goggins/clips/04-finding-true-potential-through-hard-work.mp4
        Incredible hook about being 297 pounds and terrified of wasting potential.
   8.4  jobs/goggins/clips/01-controlling-your-brain-in-hard-times.mp4
        ...
```

On failure it prints the stage and
error message (no traceback unless `--verbose`).

## `echo-clips posts JOB` (#101)

Writes a title (at most 70 characters), a 1–3 line description and 3–5 hashtags for every clip of a finished job, into `clips/publish.json` (see the README), all `pending` review. **Nothing is posted.** `JOB` is a job directory or a job id in `--jobs-dir`.

One LLM call covers as many clips as fit the model's window budget (usually the whole job on a hosted model). The LLM gets each clip's own transcript text and its topic, and invalid answers (too long, wrong hashtag count, missing clips) are sent back to it, like topic extraction.

**Frames (#129):** with a hosted provider, each clip's transcript goes with **one frame of the clip** (two moments side by side, as viewers see it, including a streamer's webcam), so a vision model (e.g. Groq `qwen/qwen3.8-27b`, Gemini) can tell a game's cutscene from the streamer. Those frames are sent to the provider; `--no-frames` keeps them local (off by default with Ollama). A model that refuses images falls back to text, with a note.

Running it again only writes text for clips that don't have any yet, so edits made in review are kept. `--regenerate` rewrites the text of clips still `pending`; approved and rejected clips are never touched. A post whose clip is gone after a re-run is dropped, unless it was already published.

| Option | Default | Meaning |
|---|---|---|
| `--jobs-dir PATH` | `jobs` | where to look for a job id |
| `--llm-provider`, `--llm-model` | as `run` | the LLM that writes the text |
| `--regenerate` | off | rewrite the text of clips still pending review |
| `--verbose`, `-v` | off | debug logging and full tracebacks |

```
posts: jobs/ep42/clips/publish.json
  [new] 01-solar-power-is-now-the-cheapest-electricity.mp4
        Solar power now the cheapest new electricity in Europe
        #solar #renewableenergy #Europe #cleanenergy #electricity
  [approved] 02-sleep-and-productivity.mp4
        How losing just one hour of sleep hurts focus
        #sleep #productivity #focus #health #wellbeing
```

## `echo-clips review JOB` (#102)

Lists the clips of a job with their post text (from `echo-clips posts`), duration and path, and records a human's decision. **Only approved clips are ever published**, and a clip can only be approved while its video exists. Without options it just lists. Clips are numbered as listed (1, 2, ...).

| Option | Meaning |
|---|---|
| `--approve 1,3-5` | approve these clips |
| `--reject 2` | reject these clips |
| `--reset 4` | put these clips back to pending |
| `--clip N --title T --description D --hashtags a,b` | edit clip N's text (`\n` in D for a new line; hashtags without `#`) |
| `-i`, `--interactive` | go through the pending clips one by one: approve, reject, edit, skip or quit. Each decision is saved at once |

A bad request (unknown clip, two decisions for one clip, a title over 100 characters) exits with 1 and changes nothing.

```
  1 [approved] 01-how-caffeine-keeps-you-awake.mp4  33.9s
      How caffeine blocks sleep signals
      Caffeine blocks adenosine, the molecule that makes you sleepy.
      #caffeine #sleep #biology #coffee
      jobs/podcast/clips/01-how-caffeine-keeps-you-awake.mp4
  2 [pending] 02-why-moving-bikes-stay-upright.mp4  37.5s
      ...
2 approved, 1 pending, 0 rejected
```

## Exit codes

| Code | Meaning |
|---|---|
| 0 | all stages done or skipped |
| 1 | a stage failed (`PipelineError`) or the input is invalid (`EchoError`) |
| 2 | bad usage (Typer/Click default) |
| 130 | interrupted with Ctrl+C (the stage is recorded as `failed`, `error="interrupted"`) |

## Examples

```bash
echo-clips run episode.mp4
echo-clips run "https://www.youtube.com/watch?v=VIDEO_ID" --llm-provider gemini
echo-clips run episode.mp4 --job-id ep42 --from-stage refine --max-duration 45
echo-clips run episode.mp4 --llm-provider groq --llm-model openai/gpt-oss-120b
echo-clips run episode.mp4 --aspect source
echo-clips run episode.mp4 --range 15%:85%
echo-clips run episode.mp4 --llm-provider gemini --max-clips 5   # the 5 best
echo-clips posts ep42 --llm-provider groq   # text for each clip, nothing is posted
echo-clips review ep42 -i                   # approve, reject or edit each clip
```
