"""
play.py — let a trained agent play, without training it.

Usage
-----
  # Play with the final model (Models/cr_agent_final.zip)
  python play.py

  # Play with a specific checkpoint
  python play.py --model ../Models/checkpoints/cr_model_1024_steps.zip

  # Watch what the agent WOULD do, without it clicking anything
  python play.py --dry-run

  # Show the live annotated vision window, play 5 games, skip calibration
  python play.py --render --games 5 --region 0 40 568 1010

When a game ends (banner detected) you are asked whether you won, so the
script can print a running win rate. Start the next game, then press Enter.
Ctrl+C stops at any time.
"""
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO

from calibration import add_region_arg, get_region
from env import CREnv
from paths import FINAL_PATH, BANNER_TEMPLATE, check_assets
from train import BannerDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=str(FINAL_PATH) + ".zip",
                        help="Path to a saved .zip model (default: Models/cr_agent_final.zip)")
    parser.add_argument("--games", type=int, default=0,
                        help="Number of games to play (0 = until Ctrl+C)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of always taking the most likely one")
    parser.add_argument("--dry-run", action="store_true",
                        help="Decide actions but never click")
    parser.add_argument("--render", action="store_true",
                        help="Show the live annotated vision window")
    add_region_arg(parser)
    args = parser.parse_args()

    model_path = Path(args.model)
    check_assets(model_path, BANNER_TEMPLATE)

    region = get_region(args)
    banner = BannerDetector(region)
    env    = CREnv(calibration_region=region,
                   render_mode="human" if args.render else None,
                   dry_run=args.dry_run)

    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(str(model_path), device="cpu")

    wins = games = 0
    try:
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, action_masks=env.action_masks(),
                                      deterministic=not args.stochastic)
            obs, _, _, _, info = env.step(int(action))
            if info["action_str"] != "WAIT":
                print(f"[play] step {info['step']:>5}  elixir {info['elixir']:.1f}  "
                      f"→ {info['action_str']}")

            if banner.detected():
                games += 1
                won = input("\nGame over. Did you win? (y/n): ").strip().lower() == "y"
                wins += int(won)
                print(f"Record: {wins}W / {games - wins}L  ({100 * wins / games:.0f}% win rate)")
                if args.games and games >= args.games:
                    break
                input("Start a new game, then press Enter when ready… ")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
