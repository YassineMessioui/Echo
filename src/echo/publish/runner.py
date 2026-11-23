"""Publish a job's approved clips to one platform (#67, #103).

Clips come only from ``echo.publish.review.ready_to_publish``, so nothing unapproved is
ever sent. Each result is written to ``clips/publish.json`` as soon as it comes back:
an interrupted or failed run never loses track of what was already posted, and a re-run
never posts a clip twice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from echo.config import PublishConfig
from echo.interfaces import Publisher
from echo.job import Job
from echo.models import ClipPost, PublishResult
from echo.publish.review import ready_to_publish


def publish_job(
    job: Job,
    publisher: Publisher,
    config: PublishConfig,
    *,
    only: Sequence[int] | None = None,
    on_result: Callable[[ClipPost, PublishResult], None] | None = None,
) -> list[tuple[ClipPost, PublishResult]]:
    """Post the approved, not yet published clips of ``job`` with ``publisher``.

    ``only`` limits it to those clips (0-based, as listed by ``review``); any of them
    that isn't approved raises PublishError before anything is sent. PublishError from
    the publisher (sign-in, quota) stops the run; the results so far are already saved.
    """
    posts = job.read_posts()
    platform = publisher.platform
    todo = ready_to_publish(posts, platform, only)
    index = {p.clip: i for i, p in enumerate(posts)}
    done: list[tuple[ClipPost, PublishResult]] = []
    for post in todo:
        result = publisher.publish(job.clips_dir / post.clip, post, config)
        i = index[post.clip]
        posts[i] = posts[i].model_copy(update={"results": posts[i].results | {platform: result}})
        job.write_posts(posts)
        done.append((posts[i], result))
        if on_result is not None:
            on_result(posts[i], result)
    return done
