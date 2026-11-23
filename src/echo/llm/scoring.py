"""Clip scoring (#21): rate each topic as a short clip, so ``--max-clips`` keeps the best.

The score stage sits between ``llm`` and ``refine``. For every topic it takes the text
the clip would contain (``clip_text``) and asks the LLM to rate it 0–10 on hook,
payoff, standalone clarity and energy, with a one-line reason. ``overall`` is computed
here (``WEIGHTS``), not by the LLM, so the ranking rule is ours and stable.

As with topic extraction, the LLM never sees or returns times: clips are numbered
lines, and it answers with those numbers. As many clips go in one call as fit the
model's window budget (one call for an hour-long podcast on a long-context model), so
they're rated side by side on the same scale.

Link jobs with YouTube's "Most replayed" graph (``heatmap.json``, #123) also get a
``replay`` score per clip (``replay_score``), blended into ``overall`` with
``ScoreConfig.replay_weight``. Without a heatmap, nothing changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from echo import progress
from echo.config import LLMConfig, PipelineConfig
from echo.errors import ArtifactError, TopicExtractionError
from echo.interfaces import ClipScorer, LLMClient
from echo.job import Job
from echo.llm import prompts
from echo.llm.chunking import approx_tokens, resolve_max_tokens
from echo.models import ReplayHeatmap, Score, Segment, StageName, Topic

CRITERIA = ("hook", "payoff", "clarity", "energy")

WEIGHTS: dict[str, float] = {"hook": 0.3, "payoff": 0.3, "clarity": 0.25, "energy": 0.15}
"""``Score.overall`` = this weighted mean. Hook and payoff decide whether a short is
watched to the end; a clip that needs context fails on its own; energy matters least
because a calm, sharp insight still works."""

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def overall(values: Mapping[str, float]) -> float:
    return round(sum(WEIGHTS[c] * values[c] for c in CRITERIA), 2)


def _clip_segments(segments: list[Segment], topic: Topic, max_duration: float) -> list[Segment]:
    """``topic``'s segments from the start, up to ``max_duration`` seconds (refine trims
    long topics from the end). At least the first segment."""
    first = segments[topic.start_segment]
    kept = [first]
    for seg in segments[topic.start_segment + 1 : topic.end_segment + 1]:
        if seg.end - first.start > max_duration:
            break
        kept.append(seg)
    return kept


def clip_text(segments: list[Segment], topic: Topic, max_duration: float) -> str:
    """The transcript text of ``topic`` as a clip: its segments from the start, up to
    ``max_duration`` seconds (refine trims long topics from the end, so this is what
    the clip says). At least the first segment."""
    parts = (seg.text.strip() for seg in _clip_segments(segments, topic, max_duration))
    return " ".join(p for p in parts if p)


def clip_span(segments: list[Segment], topic: Topic, max_duration: float) -> tuple[float, float]:
    """The ``[start, end]`` (original-video time) of the text ``clip_text`` returns."""
    kept = _clip_segments(segments, topic, max_duration)
    return kept[0].start, kept[-1].end


def replay_score(heatmap: ReplayHeatmap, start: float, end: float) -> float | None:
    """How much viewers rewatch ``[start, end]``, 0–10: the time-weighted mean heatmap
    value over the span (points cut at its edges count for their overlap), mapped
    piecewise-linearly so 0 is 0, the video's median is 5 and its peak is 10. None
    when the span doesn't overlap the heatmap.

    Relative to the median, not absolute, because YouTube scales each video's graph to
    its own peak: a 0.4 is busy in one video and quiet in another.
    """
    covered = weighted = 0.0
    for p in heatmap.points:
        overlap = min(end, p.end) - max(start, p.start)
        if overlap > 0:
            covered += overlap
            weighted += overlap * p.value
    if covered <= 0:
        return None
    mean = weighted / covered
    values = [p.value for p in heatmap.points]
    median, peak = statistics.median(values), max(values)
    if mean >= median:
        replay = 5 + 5 * (mean - median) / (peak - median) if peak > median else 5.0
    else:
        replay = 5 * mean / median
    return round(min(max(replay, 0.0), 10.0), 2)


def blend(score: Score, replay: float | None, weight: float) -> Score:
    """``score`` with ``replay`` set and ``overall = (1 - weight) * llm + weight * replay``;
    unchanged when ``replay`` is None."""
    if replay is None:
        return score
    llm = overall({c: getattr(score, c) for c in CRITERIA})
    mixed = round((1 - weight) * llm + weight * replay, 2)
    return score.model_copy(update={"overall": mixed, "replay": replay})


class InvalidScoresError(ValueError):
    """The LLM answer can't be used; the message is sent back to the LLM on retry."""


def _number(item: dict[str, Any], key: str, i: int) -> float:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or math.isnan(value):
        raise InvalidScoresError(f'entry {i}: "{key}" must be a number from 0 to 10')
    if not 0 <= value <= 10:
        raise InvalidScoresError(f'entry {i}: "{key}" is {value}, not from 0 to 10')
    return float(value)


def parse_scores(raw: str, count: int) -> list[Score]:
    """Parse an LLM answer into one ``Score`` per clip id ``0..count-1``, in id order.

    Accepts ``{"scores": [...]}`` or a bare list, optionally in a Markdown code fence.
    Every id must appear exactly once. Raises InvalidScoresError with a message meant
    for the LLM.
    """
    text = raw.strip()
    if fenced := _FENCE.match(text):
        text = fenced.group(1)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidScoresError(f"the answer is not valid JSON ({e})") from e
    if isinstance(data, dict) and "scores" in data:
        data = data["scores"]
    if not isinstance(data, list):
        raise InvalidScoresError('expected a JSON object of the form {"scores": [...]}')

    scores: dict[int, Score] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise InvalidScoresError(f"entry {i} is not an object")
        clip = item.get("clip")
        if isinstance(clip, bool) or not isinstance(clip, int):
            raise InvalidScoresError(f'entry {i}: "clip" must be an integer clip id')
        if not 0 <= clip < count:
            raise InvalidScoresError(f"entry {i}: there is no clip {clip}; use 0 to {count - 1}")
        if clip in scores:
            raise InvalidScoresError(f"clip {clip} is scored more than once")
        values = {c: _number(item, c, i) for c in CRITERIA}
        reason = item.get("reason", "")
        if not isinstance(reason, str):
            raise InvalidScoresError(f'entry {i}: "reason" must be a string')
        scores[clip] = Score(overall=overall(values), reason=reason.strip(), **values)
    if missing := [i for i in range(count) if i not in scores]:
        raise InvalidScoresError(f"no score for clip(s) {', '.join(map(str, missing))}")
    return [scores[i] for i in range(count)]


class LLMClipScorer:
    """Implements ``echo.interfaces.ClipScorer`` on top of any ``LLMClient``.

    An invalid answer is retried up to ``config.max_retries`` times with the error and
    the previous answer appended, like ``LLMTopicExtractor``; ``LLMError`` from the
    client is not retried. ``close()`` closes the client (Ctrl+C).
    """

    prompt_version = prompts.SCORE_PROMPT_VERSION

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self.client = client
        self.config = config

    def score(self, clips: list[str]) -> list[Score]:
        if not clips:
            return []
        last = len(clips) - 1
        lines = "\n".join(f"[{i}] {text}" for i, text in enumerate(clips))
        original = prompts.SCORE_USER_PROMPT.format(first=0, last=last, clips=lines)
        prompt = original
        attempts = self.config.max_retries + 1
        for _ in range(attempts):
            raw = self.client.complete(
                prompts.SCORE_SYSTEM_PROMPT, prompt, json_schema=prompts.SCORES_SCHEMA
            )
            try:
                return parse_scores(raw, len(clips))
            except InvalidScoresError as e:
                error = e
                prompt = prompts.SCORE_RETRY_PROMPT.format(
                    original=original, error=e, answer=raw, first=0, last=last
                )
        raise TopicExtractionError(
            f"LLM scores for {len(clips)} clips were still invalid after {attempts} "
            f"attempts: {error}"
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


_PROMPT_TOKENS = 600
"""Budget kept for the system prompt and the answer (~40 tokens per clip on top)."""
_ANSWER_TOKENS_PER_CLIP = 40


def batches(texts: list[str], budget: int) -> list[range]:
    """Consecutive index ranges of ``texts`` whose prompt fits ``budget`` tokens. A clip
    too long for the budget alone gets a batch of its own."""
    room = max(budget - _PROMPT_TOKENS, 1)
    result: list[range] = []
    start, used = 0, 0
    for i, text in enumerate(texts):
        cost = approx_tokens(text) + _ANSWER_TOKENS_PER_CLIP
        if i > start and used + cost > room:
            result.append(range(start, i))
            start, used = i, 0
        used += cost
    if start < len(texts):
        result.append(range(start, len(texts)))
    return result


_POLL_S = 0.05


def _interruptible[T](fn: Callable[[], T], close: Callable[[], object] | None) -> T:
    """Run ``fn()`` on a daemon thread and wait in short steps, so Ctrl+C is handled at
    once even during an HTTP call (#83); on Ctrl+C, ``close()`` ends that call."""
    result: list[T] = []
    error: list[BaseException] = []

    def work() -> None:
        try:
            result.append(fn())
        except BaseException as e:  # re-raised on the stage thread
            error.append(e)

    thread = threading.Thread(target=work, name="llm-score", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(_POLL_S)
    except KeyboardInterrupt:
        if close is not None:
            close()
        raise
    if error:
        raise error[0]
    return result[0]


def settings_fingerprint(config: PipelineConfig, job: Job | None = None) -> str:
    """What ``scores.json`` depends on besides ``topics.json``: the prompt, the model,
    the clip length the text is cut to, and the batch budget; with a heatmap (#123),
    also that there is one and the blend weight. Without one it's the same as before
    #123, so jobs without a heatmap aren't re-scored."""
    data: dict[str, Any] = {
        "prompt": prompts.SCORE_PROMPT_VERSION,
        "provider": config.llm.provider,
        "model": config.llm.model,
        "host": config.llm.host if config.llm.provider != "ollama" else None,
        "max_duration": config.refine.max_duration,
        "budget": resolve_max_tokens(config.chunk, config.llm),
    }
    if job is not None and job.heatmap_path.is_file():
        data |= {"heatmap": True, "replay_weight": config.score.replay_weight}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
    return f"{prompts.SCORE_PROMPT_VERSION}|{config.llm.provider}/{config.llm.model}|{digest}"


@dataclass
class ScoreStage:
    """Stage adapter: ``transcript.json`` + ``topics.json`` → ``scores.json`` (#21).

    Optional: built only when ``ScoreConfig.enabled``. Its ``settings`` fingerprint
    (``settings_fingerprint``) re-scores an existing job when the prompt, the model,
    ``refine.max_duration`` or, with a heatmap, ``score.replay_weight`` change (or a
    heatmap appears), without re-extracting topics. ``models`` is recorded
    in ``job.json`` (``{"score_prompt": "scores-v1"}``).
    """

    scorer: ClipScorer
    models: Mapping[str, str] = field(
        default_factory=lambda: {"score_prompt": prompts.SCORE_PROMPT_VERSION}
    )

    name: ClassVar[StageName] = StageName.SCORE

    def outputs(self, job: Job) -> list[Path]:
        return [job.scores_path]

    def settings(self, config: PipelineConfig, job: Job) -> str:
        return settings_fingerprint(config, job)

    def run(self, job: Job, config: PipelineConfig) -> None:
        transcript = job.read_transcript()
        topics = job.read_topics()
        segments = transcript.segments
        for topic in topics:
            if topic.end_segment >= len(segments):
                raise ArtifactError(
                    f"topic {topic.title!r} ends at segment {topic.end_segment}, but the "
                    f"transcript has {len(segments)} segments"
                )
        texts = [clip_text(segments, t, config.refine.max_duration) for t in topics]
        groups = batches(texts, resolve_max_tokens(config.chunk, config.llm))
        scores: list[Score] = []
        for done, group in enumerate(groups, start=1):
            batch = [texts[i] for i in group]
            scores += _interruptible(
                lambda batch=batch: self.scorer.score(batch), getattr(self.scorer, "close", None)
            )
            progress.report(done / len(groups), f"batch {done}/{len(groups)}")
        if job.heatmap_path.is_file():  # a link with a "Most replayed" graph (#123)
            heatmap, weight = job.read_heatmap(), config.score.replay_weight
            spans = [clip_span(segments, t, config.refine.max_duration) for t in topics]
            scores = [
                blend(s, replay_score(heatmap, *span), weight)
                for s, span in zip(scores, spans, strict=True)
            ]
        job.write_scores(
            [t.model_copy(update={"score": s}) for t, s in zip(topics, scores, strict=True)]
        )
        if self.models:
            meta = job.read_meta()
            meta.models.update(self.models)
            job.write_meta(meta)
