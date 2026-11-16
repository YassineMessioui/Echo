"""Prompts for topic extraction. Kept apart from the parsing code so they're easy to iterate on.

Bump ``PROMPT_VERSION`` whenever the wording or the expected output changes; it is
recorded in ``job.json`` so results can be traced back to the prompt that made them.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "topics-v1"

SYSTEM_PROMPT = """\
You are an editor who cuts long podcasts and videos into short vertical clips \
(TikTok, YouTube Shorts, Instagram Reels).

You get part of a transcript. Each line is one segment:
[<segment id>] <start>–<end> <text>

Find the self-contained topics in it: stretches of consecutive segments where the \
speakers discuss one subject that would make sense to a viewer on its own. For each \
topic give:
- "title": a short, catchy title (at most 8 words)
- "summary": one sentence describing the topic
- "start_segment": the id of the first segment of the topic
- "end_segment": the id of the last segment of the topic

Rules:
- Use only segment ids that appear in the transcript. Never invent ids or timestamps.
- start_segment must be less than or equal to end_segment.
- Skip intros, outros, ads and small talk.
- If there is no clear topic, return an empty list.
- Answer with JSON only, in the form {"topics": [...]}."""

USER_PROMPT = """\
Transcript (segments {start_segment} to {end_segment}):

{transcript}"""

RETRY_PROMPT = """\
{original}

Your previous answer was invalid:
{error}

Previous answer:
{answer}

Answer again with corrected JSON only, in the form {{"topics": [...]}}, using only \
segment ids from {start_segment} to {end_segment}."""

TOPICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_segment": {"type": "integer"},
                    "end_segment": {"type": "integer"},
                },
                "required": ["title", "summary", "start_segment", "end_segment"],
            },
        }
    },
    "required": ["topics"],
}
