# `echo-clips` CLI

The command line is the V0 entry point (#13). It builds a `PipelineConfig` from the
options and calls `echo.pipeline.run_pipeline(job, build_stages(config), config, ...)`.

```
echo-clips run VIDEO [OPTIONS]
```

| Option | Default | Maps to |
|---|---|---|
| `--job-id TEXT` | random 12-char hex | `Job.create(..., job_id=)`; reusing an id resumes that job |
| `--jobs-dir PATH` | `jobs` | `PipelineConfig.jobs_dir` |
| `--force` | off | `run_pipeline(force=True)`: re-run every stage |
| `--from-stage [ingest\|transcribe\|llm\|refine\|clip]` | none | `run_pipeline(from_stage=)`: re-run that stage and every later one |
| `--whisper-model TEXT` | `small` | `TranscribeConfig.model_size` |
| `--language TEXT` | auto-detect | `TranscribeConfig.language` |
| `--llm-provider TEXT` | `ollama` | `LLMConfig.provider` (see [llm-providers.md](llm-providers.md)) |
| `--llm-model TEXT` | `llama3.2:3b` | `LLMConfig.model` |
| `--min-duration FLOAT` | `15` | `RefineConfig.min_duration` (seconds) |
| `--max-duration FLOAT` | `60` | `RefineConfig.max_duration` (seconds; 60 is the platform cap) |
| `--aspect [9:16\|source]` | `9:16` | `ClipConfig.aspect` |
| `--crop-x FLOAT` | `0.5` | `ClipConfig.crop_x` (0 = left, 1 = right) |
| `--background [blur\|black]` | `blur` | `ClipConfig.background` |
| `-v, --verbose` | off | debug logging |

Option defaults come from `PipelineConfig`, so the two can't drift.

## Output

One line per stage while it runs, e.g.

```
[1/5] ingest      done      2.1s
[2/5] transcribe  skipped
...
```

then the job directory and the path of every clip. On failure it prints the stage and
error message (no traceback unless `--verbose`).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | all stages done or skipped |
| 1 | a stage failed (`PipelineError`) or the input is invalid (`EchoError`) |
| 2 | bad usage (Typer/Click default) |

## Examples

```bash
echo-clips run episode.mp4
echo-clips run episode.mp4 --job-id ep42 --from-stage refine --max-duration 45
echo-clips run episode.mp4 --llm-provider groq --llm-model openai/gpt-oss-120b
echo-clips run episode.mp4 --aspect source
```
