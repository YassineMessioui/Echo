"""A title, description and hashtags for every clip (#101) → ``clips/publish.json``.

The LLM gets each clip's own transcript text (plus its topic's title and summary) and
refers to clips by number, as in topic extraction. As many clips go in one call as fit
the model's window budget: one call for a typical job on a hosted model.

Re-running never overwrites a human's work: clips that already have a post keep it
(text, approval, overrides and results); only new clips get text. ``regenerate``
rewrites the text of posts still ``pending`` review, and nothing else.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echo.config import ChunkConfig, LLMConfig
from echo.errors import ArtifactError, TopicExtractionError
from echo.interfaces import LLMClient
from echo.job import Job
from echo.llm.chunking import approx_tokens, resolve_max_tokens
from echo.models import Approval, ClipPost, ClipSpec, Transcript
from echo.publish import prompts

log = logging.getLogger(__name__)

TITLE_MAX = 70
"""Generated titles stay short enough for every platform's feed (the contract allows 100)."""
DESCRIPTION_MAX = 300
DESCRIPTION_LINES = 3
HASHTAGS = (3, 5)
"""Fewest and most hashtags per clip."""
CLIP_WORDS = 220
"""Words of a clip's transcript sent to the LLM: enough for a 60 s clip, so long clips
can't blow the budget."""

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_PROMPT_TOKENS = 700
"""Budget kept for the system prompt."""
_ANSWER_TOKENS_PER_CLIP = 120


@dataclass(frozen=True)
class ClipText:
    """What the LLM is told about one clip."""

    topic_title: str
    summary: str
    said: str


@dataclass(frozen=True)
class PostDraft:
    """Generated text for one clip, within the limits above."""

    title: str
    description: str
    hashtags: list[str]


class InvalidPostsError(ValueError):
    """The LLM answer can't be used; the message is sent back to the LLM on retry."""


def clip_text(transcript: Transcript, clip: ClipSpec, summary: str = "") -> ClipText:
    """The transcript text spoken during ``clip`` (segments overlapping it), capped at
    ``CLIP_WORDS`` words."""
    words: list[str] = []
    for seg in transcript.segments:
        if seg.end > clip.start and seg.start < clip.end:
            words += seg.text.split()
    said = " ".join(words[:CLIP_WORDS]) + (" …" if len(words) > CLIP_WORDS else "")
    return ClipText(topic_title=clip.topic_title, summary=summary, said=said)


def _string(item: dict[str, Any], key: str, i: int) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        raise InvalidPostsError(f'entry {i}: "{key}" must be a string')
    return value.strip()


def _hashtags(item: dict[str, Any], i: int) -> list[str]:
    value = item.get("hashtags")
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise InvalidPostsError(f'entry {i}: "hashtags" must be a list of strings')
    tags: list[str] = []
    for raw in value:
        tag = raw.strip().lstrip("#")
        if not tag or any(c.isspace() or c == "#" for c in tag):
            raise InvalidPostsError(
                f'entry {i}: hashtag "{raw}" must be one word without spaces or #'
            )
        if tag.lower() not in (t.lower() for t in tags):
            tags.append(tag)
    low, high = HASHTAGS
    if not low <= len(tags) <= high:
        raise InvalidPostsError(
            f"entry {i}: give {low} to {high} different hashtags, not {len(tags)}"
        )
    return tags


def _draft(item: dict[str, Any], i: int) -> PostDraft:
    title = " ".join(_string(item, "title", i).split())
    if not title:
        raise InvalidPostsError(f'entry {i}: "title" is empty')
    if len(title) > TITLE_MAX:
        raise InvalidPostsError(
            f'entry {i}: "title" has {len(title)} characters; the most is {TITLE_MAX}'
        )
    lines = [line.strip() for line in _string(item, "description", i).splitlines()]
    description = "\n".join(line for line in lines if line)
    if not description:
        raise InvalidPostsError(f'entry {i}: "description" is empty')
    if len(description.splitlines()) > DESCRIPTION_LINES:
        raise InvalidPostsError(f'entry {i}: "description" has more than {DESCRIPTION_LINES} lines')
    if len(description) > DESCRIPTION_MAX:
        raise InvalidPostsError(
            f'entry {i}: "description" has {len(description)} characters; '
            f"the most is {DESCRIPTION_MAX}"
        )
    return PostDraft(title=title, description=description, hashtags=_hashtags(item, i))


def parse_posts(raw: str, count: int) -> list[PostDraft]:
    """Parse an LLM answer into one ``PostDraft`` per clip number ``0..count-1``.

    Accepts ``{"posts": [...]}`` or a bare list, optionally in a Markdown code fence.
    Raises InvalidPostsError with a message meant for the LLM.
    """
    text = raw.strip()
    if fenced := _FENCE.match(text):
        text = fenced.group(1)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidPostsError(f"the answer is not valid JSON ({e})") from e
    if isinstance(data, dict) and "posts" in data:
        data = data["posts"]
    if not isinstance(data, list):
        raise InvalidPostsError('expected a JSON object of the form {"posts": [...]}')

    drafts: dict[int, PostDraft] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise InvalidPostsError(f"entry {i} is not an object")
        clip = item.get("clip")
        if isinstance(clip, bool) or not isinstance(clip, int):
            raise InvalidPostsError(f'entry {i}: "clip" must be an integer clip number')
        if not 0 <= clip < count:
            raise InvalidPostsError(f"entry {i}: there is no clip {clip}; use 0 to {count - 1}")
        if clip in drafts:
            raise InvalidPostsError(f"clip {clip} has more than one entry")
        drafts[clip] = _draft(item, i)
    if missing := [i for i in range(count) if i not in drafts]:
        raise InvalidPostsError(f"no entry for clip(s) {', '.join(map(str, missing))}")
    return [drafts[i] for i in range(count)]


def _format(i: int, clip: ClipText) -> str:
    topic = f"{clip.topic_title}. {clip.summary}".strip()
    return f"[{i}] Topic: {topic}\nSaid: {clip.said}"


class LLMPostWriter:
    """Writes ``PostDraft``s with any ``LLMClient``.

    An invalid answer is retried up to ``config.max_retries`` times with the error and
    the previous answer appended, like ``LLMTopicExtractor``; ``LLMError`` from the
    client is not retried.
    """

    prompt_version = prompts.POSTS_PROMPT_VERSION

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self.client = client
        self.config = config
        self.system = prompts.SYSTEM_PROMPT.format(
            title_max=TITLE_MAX,
            description_max=DESCRIPTION_MAX,
            hashtags_min=HASHTAGS[0],
            hashtags_max=HASHTAGS[1],
        )

    def write(self, clips: Sequence[ClipText], source: str = "") -> list[PostDraft]:
        """Text for ``clips``. ``source`` names the video they come from (e.g. its title
        and channel), so the text can say what it is (the game, show or creator)."""
        if not clips:
            return []
        last = len(clips) - 1
        lines = "\n\n".join(_format(i, c) for i, c in enumerate(clips))
        intro = f"Source video: {source}\n\n" if source else ""
        original = prompts.USER_PROMPT.format(source=intro, first=0, last=last, clips=lines)
        prompt = original
        attempts = self.config.max_retries + 1
        for _ in range(attempts):
            raw = self.client.complete(self.system, prompt, json_schema=prompts.POSTS_SCHEMA)
            try:
                return parse_posts(raw, len(clips))
            except InvalidPostsError as e:
                error = e
                prompt = prompts.RETRY_PROMPT.format(
                    original=original, error=e, answer=raw, first=0, last=last
                )
        raise TopicExtractionError(
            f"LLM post text for {len(clips)} clips was still invalid after {attempts} "
            f"attempts: {error}"
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def batches(clips: Sequence[ClipText], budget: int) -> list[range]:
    """Consecutive index ranges of ``clips`` whose prompt and answer fit ``budget``
    tokens. A clip too long for the budget alone gets a batch of its own."""
    room = max(budget - _PROMPT_TOKENS, 1)
    result: list[range] = []
    start, used = 0, 0
    for i, clip in enumerate(clips):
        cost = approx_tokens(_format(i, clip)) + _ANSWER_TOKENS_PER_CLIP
        if i > start and used + cost > room:
            result.append(range(start, i))
            start, used = i, 0
        used += cost
    if start < len(clips):
        result.append(range(start, len(clips)))
    return result


def source_of(job: Job) -> str:
    """What the job's video is, for the LLM: a link's title and channel, else the file
    name. Empty if unknown."""
    try:
        meta = job.read_meta()
    except ArtifactError:
        return ""
    info = meta.source_info
    title = str(info.get("title") or "").strip()
    uploader = str(info.get("uploader") or "").strip()
    if title:
        return f'"{title}" by {uploader}' if uploader else f'"{title}"'
    name = Path(meta.source).stem if meta.source else ""
    return f'a video file named "{name}"' if name else ""


@dataclass(frozen=True)
class PostsUpdate:
    posts: list[ClipPost]
    written: list[str]
    """File names of the clips whose text was (re)generated."""
    dropped: list[str]
    """File names of posts removed because their clip is gone (never published)."""


def update_posts(
    job: Job,
    writer: LLMPostWriter,
    chunk: ChunkConfig,
    *,
    regenerate: bool = False,
) -> PostsUpdate:
    """Give every clip in ``clips.json`` a post in ``clips/publish.json``.

    Existing posts are matched by clip file name and topic title and kept as they are,
    except that ``regenerate`` rewrites the text of those still pending review (their
    overrides and results stay). New ones start ``pending``. A post whose clip is gone
    is dropped, unless it was published somewhere: that record is kept, so the clip is
    never posted twice.
    """
    clips = job.read_clips()
    transcript = job.read_transcript()
    summaries = {t.title: t.summary for t in job.read_topics()} if job.topics_path.is_file() else {}
    old = job.read_posts() if job.publish_path.is_file() else []
    by_key = {(p.clip, p.topic_title): p for p in old}

    names = [job.clip_video_path(i, c.topic_title).name for i, c in enumerate(clips, start=1)]
    posts: list[ClipPost | None] = []
    todo: list[int] = []
    for i, (clip, name) in enumerate(zip(clips, names, strict=True)):
        post = by_key.pop((name, clip.topic_title), None)
        if post is None or (regenerate and post.approval is Approval.PENDING):
            todo.append(i)
        posts.append(post)

    if todo:
        texts = [
            clip_text(transcript, clips[i], summaries.get(clips[i].topic_title, "")) for i in todo
        ]
        drafts: list[PostDraft] = []
        for group in batches(texts, resolve_max_tokens(chunk, writer.config)):
            drafts += writer.write([texts[k] for k in group], source_of(job))
        for i, draft in zip(todo, drafts, strict=True):
            text = {
                "title": draft.title,
                "description": draft.description,
                "hashtags": draft.hashtags,
            }
            current = posts[i]
            if current is None:
                posts[i] = ClipPost(clip=names[i], topic_title=clips[i].topic_title, **text)
            else:
                posts[i] = current.model_copy(update=text)

    taken = set(names)
    kept = [
        p for p in by_key.values() if p.clip not in taken and any(r.ok for r in p.results.values())
    ]
    dropped = [p.clip for p in by_key.values() if p not in kept]
    for name in dropped:
        log.info("dropping the post for %s: its clip is gone", name)
    result = [p for p in posts if p is not None] + kept
    job.write_posts(result)
    if todo:
        meta = job.read_meta()
        meta.models["posts_prompt"] = writer.prompt_version
        job.write_meta(meta)
    return PostsUpdate(posts=result, written=[names[i] for i in todo], dropped=dropped)
