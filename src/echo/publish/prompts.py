"""Prompts for a clip's post text (#101): title, description and hashtags.

Bump ``POSTS_PROMPT_VERSION`` whenever the wording or the expected output changes; it
is recorded in ``job.json`` (``models["posts_prompt"]``).
"""

from __future__ import annotations

from typing import Any

POSTS_PROMPT_VERSION = "posts-v3"

SYSTEM_PROMPT = """\
You write the text that goes with short vertical clips (YouTube Shorts, TikTok, \
Instagram Reels) cut from a longer video: a game stream, a podcast, a show.

You may get the source video (its title and channel), then numbered clips. Each \
starts with its number, then the topic it covers, then what is said in it:
[<clip number>] Topic: <topic title>. <topic summary>
Said: <transcript text>

First decide the context of the whole video, once:
- "context_hashtags": 0 to {context_max} hashtags naming what the video is, so \
viewers searching for it find every clip: the game or show (e.g. "gta6") and the \
streamer or creator (e.g. "ishowspeed"). Use the source's title and channel. Empty \
only if neither names anything.

Then, for every clip:
- "title": at most {title_max} characters. Say who and what: the streamer, the \
characters, the game (e.g. "IShowSpeed reacts to Lucia confronting Jason in GTA 6"). \
Make people want to watch, but promise only what the clip delivers: no clickbait lies, \
no ALL CAPS, no emoji spam.
- "description": 1 to 3 short lines (at most {description_max} characters in all) \
saying what happens, with the same names.
- "hashtags": {hashtags_min} to {hashtags_max} hashtags without the # sign and without \
spaces. Every context hashtag first, then ones about this clip.

Rules:
- Name people. In a game, use the characters' names (e.g. Jason and Lucia in GTA 6) \
when the clip's words or well-known facts about the game make them clear. Never write \
"one person", "someone", "a man", "the speaker" or "a heated exchange between two \
people".
- In a stream, the streamer talks to chat and reacts, and the game's characters speak \
in cutscenes and missions. Tell them apart: a scene is the characters, the comments \
are the streamer's. When the clip is mostly the streamer reacting to a scene, say so \
("IShowSpeed reacts to ...").
- Dialogue in a game is fiction: describe it as the game's story, not real life.
- Write in the language the clip is spoken in.
- Never invent facts, names or numbers beyond the clip, its source and well-known \
facts about the game or show. If unsure of a character's name, describe their role \
in the scene instead ("Jason's partner").
- Answer with JSON only, in the form {{"context_hashtags": [...], "posts": \
[{{"clip": <number>, "title": ..., "description": ..., "hashtags": [...]}}, ...]}}, \
with one entry for every clip."""

USER_PROMPT = """\
{source}Clips {first} to {last}:

{clips}"""

RETRY_PROMPT = """\
{original}

Your previous answer was invalid:
{error}

Previous answer:
{answer}

Answer again with corrected JSON only, in the form {{"context_hashtags": [...], \
"posts": [...]}}, with exactly one entry for each clip from {first} to {last}."""

POSTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "context_hashtags": {"type": "array", "items": {"type": "string"}},
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
        },
    },
    "required": ["context_hashtags", "posts"],
}
