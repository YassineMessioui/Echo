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


# --- clip scoring (#21) -----------------------------------------------------------------

SCORE_PROMPT_VERSION = "scores-v1"

SCORE_SYSTEM_PROMPT = """\
You are an editor deciding which short vertical clips (TikTok, YouTube Shorts, \
Instagram Reels) from a long video are worth posting.

You get candidate clips. Each line is the full transcript of one clip:
[<clip id>] <text>

Rate every clip from 0 to 10 on:
- "hook": would the first sentence stop someone scrolling? A bold claim, a question \
or a surprise is high; greetings, filler ("so", "yeah", "and then") or starting \
mid-thought is low.
- "payoff": does it deliver something by the end: an insight, a punchline, a story's \
resolution or a useful takeaway?
- "clarity": does it make sense to someone who has not seen the rest of the video? \
Unexplained references ("that", "he", "like I said") or ending mid-thought are low.
- "energy": emotion, conviction, humour, stakes and pace.
- "reason": one short sentence explaining the scores.

Rules:
- Judge each clip on its own text only.
- Be critical and use the whole scale: a typical clip is 4 to 6; give 8 or more only \
to clips you would definitely post.
- Score every clip id exactly once, using only the ids given.
- Answer with JSON only, in the form \
{"scores": [{"clip": 0, "hook": 7, "payoff": 6, "clarity": 8, "energy": 5, \
"reason": "..."}]}."""

SCORE_USER_PROMPT = """\
Clips {first} to {last}:

{clips}"""

SCORE_RETRY_PROMPT = """\
{original}

Your previous answer was invalid:
{error}

Previous answer:
{answer}

Answer again with corrected JSON only, in the form {{"scores": [...]}}, with exactly one \
entry for each clip id from {first} to {last}."""

SCORES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip": {"type": "integer"},
                    "hook": {"type": "number"},
                    "payoff": {"type": "number"},
                    "clarity": {"type": "number"},
                    "energy": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["clip", "hook", "payoff", "clarity", "energy", "reason"],
            },
        }
    },
    "required": ["scores"],
}
