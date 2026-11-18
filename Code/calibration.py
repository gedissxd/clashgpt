"""
calibration.py — shared start-up helpers for train.py, play.py and debug_cv.py.
"""

import argparse
import time


def _countdown(msg: str, seconds: int):
    for i in range(seconds, 0, -1):
        print(f"\r{msg} — {i}s …", end="", flush=True)
        time.sleep(1)
    print()


def calibrate() -> tuple[int, int, int, int]:
    """Ask the user to hover the game window's corners; return (l, t, w, h)."""
    import pyautogui

    print("=== Calibration ===")
    _countdown("Move mouse to TOP-LEFT corner of game", 3)
    tl = pyautogui.position()
    _countdown("Move mouse to BOTTOM-RIGHT corner of game", 3)
    br = pyautogui.position()
    left, top     = tl.x, tl.y
    width, height = br.x - tl.x, br.y - tl.y
    if width <= 0 or height <= 0:
        raise SystemExit("Calibration failed: bottom-right must be below and to "
                         "the right of top-left. Please run again.")
    print(f"Capture region: {left},{top}  {width}×{height}px")
    print(f"Tip: skip this next time with  --region {left} {top} {width} {height}")
    return left, top, width, height


def add_region_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="Screen capture region of the emulator. Skips mouse calibration "
             "(the values are printed after each calibration).")


def get_region(args) -> tuple[int, int, int, int]:
    return tuple(args.region) if args.region else calibrate()


def mask_fn(env):
    """ActionMasker hook: unwrap to the CREnv and return its current mask."""
    return unwrap(env).action_masks()


def unwrap(env):
    """Return the innermost CREnv from a stack of gym wrappers."""
    from env import CREnv
    base = env
    while not isinstance(base, CREnv) and hasattr(base, "env"):
        base = base.env
    return base
