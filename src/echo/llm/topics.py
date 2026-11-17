"""Topic extraction: ask the LLM for topics in one transcript window and validate them.

The LLM never produces timestamps. It returns segment ids, which are checked against
the window here and mapped to times later by the refine stage.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from echo.config import LLMConfig
from echo.errors import TopicExtractionError
from echo.interfaces import LLMClient
from echo.llm import prompts
from echo.models import Topic, TranscriptWindow

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class InvalidTopicsError(ValueError):
    """The LLM answer can't be used; the message is sent back to the LLM on retry."""


def parse_topics(raw: str, window: TranscriptWindow) -> list[Topic]:
    """Parse an LLM answer into topics whose segment ids all lie inside ``window``.

    Accepts ``{"topics": [...]}`` or a bare list, optionally wrapped in a Markdown
    code fence. Raises InvalidTopicsError with a message meant for the LLM.
    """
    text = raw.strip()
    if fenced := _FENCE.match(text):
        text = fenced.group(1)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidTopicsError(f"the answer is not valid JSON ({e})") from e

    if isinstance(data, dict) and "topics" in data:
        data = data["topics"]
    if not isinstance(data, list):
        raise InvalidTopicsError('expected a JSON object of the form {"topics": [...]}')

    topics: list[Topic] = []
    for i, item in enumerate(data):
        try:
            topic = Topic.model_validate(item)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'topic'}: {err['msg']}" for err in e.errors()
            )
            raise InvalidTopicsError(f"topic {i} is invalid: {problems}") from e
        for field in ("start_segment", "end_segment"):
            segment_id = getattr(topic, field)
            if not window.contains(segment_id):
                raise InvalidTopicsError(
                    f"topic {i} ({topic.title!r}): {field} {segment_id} is not in this "
                    f"transcript; use ids from {window.start_segment} to {window.end_segment}"
                )
        topics.append(topic)
    return topics


class LLMTopicExtractor:
    """Implements ``echo.interfaces.TopicExtractor`` on top of any ``LLMClient``.

    An invalid answer is retried up to ``config.max_retries`` times, each time with the
    error and the previous answer appended to the prompt. ``LLMError`` from the client
    (provider down, model missing) is not retried.

    ``close()`` closes the client, if it has a ``close`` method, and may be called from
    another thread: ``extract_all`` does so on Ctrl+C, so calls in flight fail fast (#83).
    """

    prompt_version = prompts.PROMPT_VERSION

    def __init__(self, client: LLMClient, config: LLMConfig) -> None:
        self.client = client
        self.config = config

    def extract(self, window: TranscriptWindow) -> list[Topic]:
        original = prompts.USER_PROMPT.format(
            start_segment=window.start_segment,
            end_segment=window.end_segment,
            transcript=window.text,
        )
        prompt = original
        attempts = self.config.max_retries + 1
        for _ in range(attempts):
            raw = self.client.complete(
                prompts.SYSTEM_PROMPT, prompt, json_schema=prompts.TOPICS_SCHEMA
            )
            try:
                return parse_topics(raw, window)
            except InvalidTopicsError as e:
                error = e
                prompt = prompts.RETRY_PROMPT.format(
                    original=original,
                    error=e,
                    answer=raw,
                    start_segment=window.start_segment,
                    end_segment=window.end_segment,
                )
        raise TopicExtractionError(
            f"LLM output for segments {window.start_segment}–{window.end_segment} was still "
            f"invalid after {attempts} attempts: {error}"
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
