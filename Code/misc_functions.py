import numpy as np
import cv2 as cv

# BGR colour of the dark-blue pixel at the top-left edge of the game UI
# (RGB 24,30,72). Every other pixel coordinate in the project is relative to it.
CORNER_BGR       = np.array([72, 30, 24], dtype=np.int16)
CORNER_TOLERANCE = 5

# Largest believable elixir increase between two reads, if the caller does not
# pass one. env.py passes a time-based value instead (see ELIXIR_MAX_RATE).
DEFAULT_MAX_RISE = 0.15

# Elixir bar geometry (corner-relative)
_ELIXIR_Y        = 978
_ELIXIR_FULL_X   = 538
_ELIXIR_X_BARS   = [154, 194, 233, 272, 311, 350, 389, 428, 468, 507]
_ELIXIR_SEG_W    = 39

# HSV ranges
_LOWER_ELIXIR     = np.array([143, 190, 150])
_UPPER_ELIXIR     = np.array([153, 240, 230])
_LOWER_LIGHT_BLUE = np.array([104, 150, 119])
_UPPER_LIGHT_BLUE = np.array([117, 218, 175])


def _to_bgr(img):
    if img.ndim == 3 and img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return img


def get_top_corner(img):
    """
    Return (x, y) of the first pixel (row-major order) matching CORNER_BGR
    within ±CORNER_TOLERANCE, or None if the game is not visible.

    Vectorised with NumPy — the original per-pixel Python loop took seconds
    on a full capture; this takes milliseconds.
    """
    bgr  = _to_bgr(img).astype(np.int16)
    hits = np.all(np.abs(bgr - CORNER_BGR) <= CORNER_TOLERANCE, axis=2)
    idx  = np.flatnonzero(hits)
    if idx.size == 0:
        return None
    y, x = divmod(int(idx[0]), hits.shape[1])
    return (x, y)


def _in_range(hsv_pixel, lower, upper) -> bool:
    return bool((lower <= hsv_pixel).all() and (hsv_pixel <= upper).all())


def get_elixir(img, corner, last_elixir: float = 0.0,
               max_rise: float = DEFAULT_MAX_RISE) -> float:
    """
    Read the elixir bar from the game frame.

    Parameters
    ----------
    img          : BGR or BGRA frame from mss/cv2.
    corner       : (x, y) top-left of the game window, from get_top_corner().
    last_elixir  : elixir value from the previous read. Elixir can only rise
                   slowly, so any reading more than `max_rise` above it is
                   treated as a colour-noise spike and clamped. Pass 0.0 when
                   the previous value is unknown — no clamping is applied then.
    max_rise     : largest believable increase since the last read.

    Returns
    -------
    float in [0.0, 10.0], rounded to 2 decimal places.
    """
    bgr  = _to_bgr(img)
    h, w = bgr.shape[:2]
    cx, cy = corner
    py = _ELIXIR_Y + cy

    if py >= h or _ELIXIR_FULL_X + cx >= w or max(_ELIXIR_X_BARS) + cx >= w:
        return last_elixir

    # Convert the single row we need to HSV once instead of pixel-by-pixel.
    hsv_row = cv.cvtColor(bgr[py:py + 1], cv.COLOR_BGR2HSV)[0]

    if _in_range(hsv_row[_ELIXIR_FULL_X + cx], _LOWER_ELIXIR, _UPPER_ELIXIR):
        raw = 10.0
    else:
        elixir = 0.0
        for x_bar in _ELIXIR_X_BARS:
            px = x_bar + cx
            if _in_range(hsv_row[px], _LOWER_ELIXIR, _UPPER_ELIXIR):
                elixir += 1
                continue
            # Partially filled segment: count light-blue pixels
            for px2 in range(px, min(px + _ELIXIR_SEG_W, w)):
                if _in_range(hsv_row[px2], _LOWER_LIGHT_BLUE, _UPPER_LIGHT_BLUE):
                    elixir += 1 / _ELIXIR_SEG_W
                else:
                    break
            break
        raw = round(elixir, 2)

    # Spike suppression — only clamp upward jumps (playing a card is a valid
    # drop) and only when we actually know the previous value.
    if last_elixir > 0.0 and raw > last_elixir + max_rise:
        return round(last_elixir + max_rise, 2)
    return raw
