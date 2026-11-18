"""
debug_cv.py — live view of what the agent "sees", without playing or training.

Opens a window ("CR Agent") showing YOLO + SORT boxes (green = ally,
red = enemy), track IDs, elixir, hand, next card and tower HP. Nothing is
clicked, so you can play the game yourself and check the vision pipeline.

Usage
-----
  python debug_cv.py                         # calibrate with the mouse
  python debug_cv.py --region 0 40 568 1010  # reuse a previous calibration

Press Q in the window (or Ctrl+C in the terminal) to quit.
"""
import argparse

from calibration import add_region_arg, get_region
from constants import WAIT_ACTION
from env import CREnv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_region_arg(parser)
    args = parser.parse_args()

    env = CREnv(calibration_region=get_region(args), render_mode="human", dry_run=True)
    try:
        env.reset()
        print("Vision debug running — press Q in the 'CR Agent' window to quit.")
        while True:
            _, _, terminated, _, info = env.step(WAIT_ACTION)
            if env.last_key in (ord("q"), ord("Q")):
                break
            if terminated:
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
