"""Follow the active speaker (#16, #109): ``ClipConfig.reframe = "speaker"``.

Builds on ``echo.framing``'s shots. Shots where the faces fit in the 9:16 window (or
with one face) are framed exactly as in ``"face"`` mode. In a shot where they don't
fit, like a wide two-shot, the window goes to whoever is speaking, from diarization
(``speakers.json``):

1. **Who is where:** each diarized speaker is matched to a seat, i.e. a face's rank
   from the left among the people in such shots. The match is the seat whose mouth moves
   most, relative to its usual motion, during that speaker's turns. Evidence is pooled
   over all of the clip's wide shots: a fixed camera keeps people in the same seats.
2. **When:** the active-speaker timeline is smoothed. A turn shorter than ``MIN_HOLD``
   (a "yeah", a laugh) doesn't move the window, so it never jitters.
3. **How:** the window pans over ``PAN`` seconds, ending before the shot does.

Without enough evidence to tell the speakers apart, a shot is framed like ``"face"``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from echo.framing import FIT_MARGIN, Shot, choose_x, main_people, plan

MIN_HOLD = 2.0
"""Seconds a speaker must talk before the window moves to them."""
PAN = 0.4
"""Seconds a pan from one speaker to the other takes."""
MIN_EVIDENCE = 1.15
"""A seat's mouth motion during a speaker's turns must be this much above its own
average (and above the other seats') for the speaker to be placed there."""

Turn = tuple[float, float, str]
"""(start, end, speaker) in seconds from the clip start."""
Keyframe = tuple[float, float, float]
"""(time, window centre, pan duration): the window reaches ``centre`` ``pan`` seconds
after ``time`` (0 = a cut)."""


def _speaker_at(turns: Sequence[Turn], t: float) -> str | None:
    """Who speaks at ``t``: the latest-starting turn that covers it."""
    label = None
    for start, end, speaker in turns:
        if start <= t < end:
            label = speaker
    return label


def timeline(turns: Sequence[Turn], start: float, end: float) -> list[tuple[float, str]]:
    """(time, speaker) changes within ``[start, end)``, smoothed: runs shorter than
    ``MIN_HOLD`` are given to the speaker before them (or, at the start, after them);
    silences keep the last speaker."""
    changes: list[tuple[float, str]] = []
    first = _speaker_at(turns, start)
    if first is not None:
        changes.append((start, first))
    for t0, t1, speaker in sorted(turns):
        if start < t0 < end and t1 > t0:
            changes.append((t0, speaker))
    if not changes:
        return []
    changes[0] = (start, changes[0][1])

    def compress(items: list[tuple[float, str]]) -> list[tuple[float, str]]:
        out: list[tuple[float, str]] = []
        for t, speaker in items:
            if not out or out[-1][1] != speaker:
                out.append((t, speaker))
        return out

    def talking(speaker: str, t0: float, t1: float) -> float:
        """Seconds ``speaker`` talks within ``[t0, t1)`` (pauses don't count)."""
        return sum(
            max(0.0, min(b, t1) - max(a, t0)) for a, b, s in turns if s == speaker
        )  # fmt: skip

    changes = compress(changes)
    while True:
        ends = [t for t, _ in changes[1:]] + [end]
        held = [talking(s, t, t_end) for (t, s), t_end in zip(changes, ends, strict=True)]
        short = [i for i, h in enumerate(held) if h < MIN_HOLD]
        if not short or len(changes) == 1:
            return changes
        # Drop the shortest run (the earliest of equals), then merge and look again.
        i = min(short, key=lambda j: (held[j], j))
        if i == 0:
            changes = compress([(start, changes[1][1]), *changes[2:]])
        else:
            changes = compress(changes[:i] + changes[i + 1 :])


def _wide(shot: Shot, window: float) -> list[float] | None:
    """Seat positions (cx of the main people, left to right) if ``shot`` has two or more
    people who don't fit in the window, else None."""
    main = main_people(shot.people)
    if len(main) < 2:
        return None
    if max(p.right for p in main) - min(p.left for p in main) <= FIT_MARGIN * window:
        return None
    return sorted(p.cx for p in main)


def _seat(cx: float, seats: Sequence[float]) -> int:
    return int(np.argmin([abs(cx - s) for s in seats]))


def map_speakers(
    shots: Sequence[Shot], turns: Sequence[Turn], window: float
) -> dict[str, tuple[int, int]]:
    """Speaker -> seat ``(number of people, rank from the left)`` in the clip's wide shots,
    for the speakers whose mouth-motion evidence is clear."""
    # motion[(n, rank)][speaker] = motions of that seat while that speaker talks
    motion: dict[tuple[int, int], dict[str | None, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for shot in shots:
        seats = _wide(shot, window)
        if seats is None:
            continue
        for t, faces in shot.mouths:
            speaker = _speaker_at(turns, t)
            for cx, value in faces:
                motion[(len(seats), _seat(cx, seats))][speaker].append(value)
    # Each seat's motion while a speaker talks, relative to its overall average.
    score: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    for seat, by_speaker in motion.items():
        overall = np.mean([v for values in by_speaker.values() for v in values])
        if overall <= 0:
            continue
        for speaker, values in by_speaker.items():
            if speaker is not None and len(values) >= 2:
                score[speaker][seat] = float(np.mean(values) / overall)
    # Greedy one-to-one: the clearest (speaker, seat) pairs first.
    pairs = sorted(
        ((v, speaker, seat) for speaker, seats in score.items() for seat, v in seats.items()),
        reverse=True,
    )
    mapping: dict[str, tuple[int, int]] = {}
    for value, speaker, seat in pairs:
        if value < MIN_EVIDENCE or speaker in mapping or seat in mapping.values():
            continue
        others = [v for s, v in score[speaker].items() if s != seat and s[0] == seat[0]]
        if others and value < MIN_EVIDENCE * max(others):
            continue  # this speaker's mouth evidence isn't clearly on one seat
        mapping[speaker] = seat
    return mapping


def plan_speakers(
    shots: Sequence[Shot], turns: Sequence[Turn], window: float, default: float
) -> list[Keyframe]:
    """Keyframes for ``"speaker"`` framing: ``framing.plan`` per shot, except that wide
    shots follow the active speaker, where the speaker's seat is known."""
    mapping = map_speakers(shots, turns, window)
    half = window / 2
    keyframes: list[Keyframe] = []

    def add(t: float, x: float, pan: float = 0.0) -> None:
        x = min(max(x, half), 1 - half)
        if keyframes and abs(x - keyframes[-1][1]) < 0.02:
            return
        keyframes.append((t, x, pan))

    for shot in shots:
        seats = _wide(shot, window)
        path = timeline(turns, shot.start, shot.end) if seats is not None else []
        placed = [(t, s) for t, s in path if s in mapping and mapping[s][0] == len(seats or [])]
        if seats is None or not placed:
            for t, x in plan([shot], window, default):
                add(t, x)
            continue
        fallback = choose_x(shot.people, window)
        for i, (t, speaker) in enumerate(path):
            if speaker in mapping and mapping[speaker][0] == len(seats):
                x = seats[mapping[speaker][1]]
            elif keyframes and i > 0:
                continue  # unknown speaker: stay where we are
            else:
                x = fallback if fallback is not None else default
            if i == 0:
                add(shot.start, x)  # a new shot: cut straight to it
            elif t + PAN <= shot.end:
                add(t, x, PAN)
    return keyframes or [(0.0, default, 0.0)]
