"""Review (#102): a human approves, rejects or edits each clip's post before anything
is published.

The rules live here, apart from the CLI, so the publish command (#103) uses the same
gate: ``ready_to_publish`` only ever returns approved posts, and refuses outright when
asked for a clip that isn't approved.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from echo.errors import PublishError
from echo.models import Approval, ClipPost, Platform


class ReviewError(ValueError):
    """A review request that can't be applied (bad clip number, invalid text)."""


def parse_numbers(spec: str, count: int) -> list[int]:
    """Clip numbers as shown by ``review`` (1-based: ``"1,3-5"``) → 0-based indexes,
    sorted and without duplicates. Raises ReviewError for anything outside 1..count."""
    indexes: set[int] = set()
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        low, _, high = part.partition("-")
        try:
            first, last = int(low), int(high or low)
        except ValueError:
            raise ReviewError(f"{part!r} is not a clip number or range (e.g. 1,3-5)") from None
        if not 1 <= first <= last <= count:
            raise ReviewError(f"no clip {part}: clips are numbered 1 to {count}")
        indexes.update(range(first - 1, last))
    if not indexes:
        raise ReviewError("no clip numbers given")
    return sorted(indexes)


def set_approval(post: ClipPost, approval: Approval, clips_dir: Path) -> ClipPost:
    """``post`` with ``approval``. A clip can only be approved while its video exists."""
    if approval is Approval.APPROVED and not (clips_dir / post.clip).is_file():
        raise ReviewError(f"can't approve {post.clip}: the video is missing from {clips_dir}")
    return post.model_copy(update={"approval": approval})


def edit_text(
    post: ClipPost,
    *,
    title: str | None = None,
    description: str | None = None,
    hashtags: Sequence[str] | None = None,
) -> ClipPost:
    """``post`` with new text, validated against the ``ClipPost`` contract (title length,
    hashtags without ``#`` or spaces; a leading ``#`` is dropped). ``None`` keeps a field."""
    changes: dict[str, object] = {}
    if title is not None:
        changes["title"] = " ".join(title.split())
    if description is not None:
        changes["description"] = description.strip()
    if hashtags is not None:
        changes["hashtags"] = [t.strip().lstrip("#") for t in hashtags if t.strip()]
    try:
        return ClipPost.model_validate(post.model_dump() | changes)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )
        raise ReviewError(f"invalid text for {post.clip}: {problems}") from e


def ready_to_publish(
    posts: Sequence[ClipPost], platform: Platform, only: Sequence[int] | None = None
) -> list[ClipPost]:
    """The posts to publish on ``platform``: approved and not yet published there.

    ``only`` (0-based indexes) limits it to those clips, and then every one of them must
    be approved: asking to publish a pending or rejected clip raises PublishError rather
    than skipping it quietly. Nothing unapproved is ever returned.
    """
    chosen = list(posts) if only is None else [posts[i] for i in only]
    if only is not None:
        if refused := [p.clip for p in chosen if p.approval is not Approval.APPROVED]:
            raise PublishError(
                f"not approved, so not published: {', '.join(refused)} "
                "(approve them with echo-clips review first)"
            )
    return [p for p in chosen if p.approval is Approval.APPROVED and not p.published_on(platform)]
