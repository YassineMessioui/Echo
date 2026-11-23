"""Vertical 9:16 reframing: build the ffmpeg video filter for ``ClipConfig.aspect``.

The clip stage appends the filter to its cut so reframing happens in the same re-encode.
How a source is framed depends on its aspect ratio (width / height):

- **≈ 9:16** (within ``ASPECT_TOLERANCE``): scale only.
- **≥ 4:3** (16:9, 16:10, 4:3, ...): crop a full-height 9:16 window, placed by
  ``crop_x``, then scale. Wide sources lose less to a crop than they would to padding.
- **Anything else** (1:1, 5:4, or narrower than 9:16): scale to fit inside the frame
  without distortion and fill the rest with ``background``: black bars, or a blurred,
  zoomed copy of the same frame. A centre crop of 1:1 would cut almost half the picture,
  which is why it is padded (#26).

The result is a simple filtergraph (one video input, one video output): pass it as
``-vf``. The blur background uses internal ``split``/``overlay`` labels, which ``-vf``
accepts; to use it in ``-filter_complex``, prefix the input pad (``[0:v]``) and append an
output label.
"""

from __future__ import annotations

from pathlib import Path

from echo.config import ClipConfig
from echo.errors import InvalidInputError
from echo.ffmpeg import probe

TARGET = (1080, 1920)
VERTICAL = TARGET[0] / TARGET[1]

ASPECT_TOLERANCE = 0.01
"""Relative difference from 9:16 still treated as 9:16 (e.g. 1080x1918)."""

CROP_MIN_ASPECT = 4 / 3
"""Sources at least this wide are cropped to 9:16; narrower ones are padded."""

BLUR_SIZE = (270, 480)
"""The blurred background is built at quarter size, then scaled up: same look, far cheaper."""


def _even(n: float) -> int:
    return max(2, int(n) // 2 * 2)


def _window(width: int, height: int) -> tuple[int, int]:
    h = _even(height)
    return min(_even(h * VERTICAL), _even(width)), h


def _crop(width: int, height: int, crop_x: float, x: str | None = None) -> str:
    w, h = _window(width, height)
    if x is None:
        x = str(round((width - w) * crop_x))
    return f"crop={w}:{h}:{x}:0"


def crop_window(width: int, height: int, *, sar: float = 1.0) -> tuple[int, int, int] | None:
    """(displayed frame width, height, crop window width) in pixels if ``video_filter``
    crops a ``width`` x ``height`` source with pixel shape ``sar`` to 9:16, else None
    (already 9:16, or padded). A ``crop_x_expr`` positions the window in this width."""
    if width <= 0 or height <= 0 or sar <= 0:
        return None
    if abs(sar - 1) > ASPECT_TOLERANCE:
        width, height = _even(width * sar), _even(height)
    aspect = width / height
    if abs(aspect - VERTICAL) <= ASPECT_TOLERANCE * VERTICAL:
        return None
    if aspect >= CROP_MIN_ASPECT * (1 - ASPECT_TOLERANCE):
        return width, _even(height), _window(width, height)[0]
    return None


def _fit(width: int, height: int) -> tuple[int, int]:
    """Largest even size with the source's aspect ratio that fits inside ``TARGET``."""
    scale = min(TARGET[0] / width, TARGET[1] / height)
    return _even(width * scale), _even(height * scale)


def video_filter(
    width: int,
    height: int,
    config: ClipConfig,
    *,
    sar: float = 1.0,
    crop_x_expr: str | None = None,
) -> str | None:
    """ffmpeg ``-vf`` filter that frames a ``width`` x ``height`` source as 1080x1920,
    or None when ``config.aspect == "source"`` (keep the original framing).

    ``width``/``height``/``sar`` describe the frames the filter receives, i.e. after
    ffmpeg's auto-rotation (``MediaInfo.frame_size`` / ``frame_sar``). Non-square pixels
    are first resampled to square ones so the framing uses the displayed shape.

    ``crop_x_expr`` replaces ``config.crop_x`` when the source is cropped: the window's
    left edge in pixels, as a number or an ffmpeg expression of ``t`` (``echo.framing``).
    """
    if config.aspect == "source":
        return None
    if width <= 0 or height <= 0 or sar <= 0:
        raise ValueError(f"invalid video size {width}x{height} (sar {sar})")
    prefix = ""
    if abs(sar - 1) > ASPECT_TOLERANCE:
        width, height = _even(width * sar), _even(height)
        prefix = f"scale={width}:{height},setsar=1,"
    out_w, out_h = TARGET
    # lanczos: sharper than the default bicubic when a 9:16 crop is enlarged (#16).
    scale = f"scale={out_w}:{out_h}:flags=lanczos"
    aspect = width / height

    if abs(aspect - VERTICAL) <= ASPECT_TOLERANCE * VERTICAL:
        return f"{prefix}{scale},setsar=1"
    if aspect >= CROP_MIN_ASPECT * (1 - ASPECT_TOLERANCE):
        crop = _crop(width, height, config.crop_x, crop_x_expr)
        return f"{prefix}{crop},{scale},setsar=1"

    fit_w, fit_h = _fit(width, height)
    x, y = (out_w - fit_w) // 2, (out_h - fit_h) // 2
    if config.background == "black":
        return (
            f"{prefix}scale={fit_w}:{fit_h}:flags=lanczos,"
            f"pad={out_w}:{out_h}:{x}:{y}:black,setsar=1"
        )
    blur_w, blur_h = BLUR_SIZE
    return (
        f"{prefix}split=2[bg][fg];"
        f"[bg]scale={blur_w}:{blur_h}:force_original_aspect_ratio=increase,"
        f"crop={blur_w}:{blur_h},boxblur=10:2,{scale},setsar=1[back];"
        f"[fg]scale={fit_w}:{fit_h}:flags=lanczos,setsar=1[front];"
        f"[back][front]overlay={x}:{y},setsar=1"
    )


def frame_geometry(path: Path) -> tuple[int, int, float]:
    """(width, height, SAR) of the frames a ``-vf`` filter receives from ``path``'s video
    stream (probed with ffprobe). Raises InvalidInputError if it has none."""
    info = probe(path)
    size = info.frame_size
    if not info.has_video or not size or not all(size):
        raise InvalidInputError(f"no video stream: {path}")
    return size[0], size[1], info.frame_sar


def video_filter_for(path: Path, config: ClipConfig) -> str | None:
    """``video_filter`` for the video stream of ``path`` (probed with ffprobe)."""
    if config.aspect == "source":
        return None
    width, height, sar = frame_geometry(path)
    return video_filter(width, height, config, sar=sar)
