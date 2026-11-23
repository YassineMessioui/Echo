"""Prompts for a clip's post text (#101): title, description and hashtags.

Bump ``POSTS_PROMPT_VERSION`` whenever the wording or the expected output changes; it
is recorded in ``job.json`` (``models["posts_prompt"]``).
"""

from __future__ import annotations

from typing import Any

POSTS_PROMPT_VERSION = "posts-v1"

SYSTEM_PROMPT = """\
You write the text that goes with short vertical clips (YouTube Shorts, TikTok, \
Instagram Reels) cut from a longer podcast or video.

You get numbered clips. Each starts with its number, then the topic it covers, then \
what is said in it:
[<clip number>] Topic: <topic title>. <topic summary>
Said: <transcript text>

For every clip write:
- "title": at most {title_max} characters. Make people want to watch, but promise only \
what the clip really delivers: no clickbait lies, no ALL CAPS, no emoji spam.
- "description": 1 to 3 short lines (at most {description_max} characters in all) \
saying what the viewer learns or sees.
- "hashtags": {hashtags_min} to {hashtags_max} relevant hashtags, without the # sign \
and without spaces (e.g. "sleep", "productivitytips").

Rules:
- Write in the language the clip is spoken in.
- Base everything on the clip's own words. Never invent facts, names or numbers.
- Answer with JSON only, in the form {{"posts": [{{"clip": <number>, "title": ..., \
"description": ..., "hashtags": [...]}}, ...]}}, with one entry for every clip."""

USER_PROMPT = """\
Clips {first} to {last}:

{clips}"""

RETRY_PROMPT = """\
{original}

Your previous answer was invalid:
{error}

Previous answer:
{answer}

Answer again with corrected JSON only, in the form {{"posts": [...]}}, with exactly one \
entry for each clip from {first} to {last}."""

POSTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "posts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["clip", "title", "description", "hashtags"],
            },
        }
    },
    "required": ["posts"],
}
