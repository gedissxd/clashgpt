"""
train.py — MaskablePPO training script for the CR agent.

Usage
-----
  # Start fresh
  python train.py

  # Resume from latest checkpoint
  python train.py --resume

  # Resume from specific checkpoint
  python train.py --resume --checkpoint path/to/model.zip

  # Skip the mouse calibration (values are printed after each calibration)
  python train.py --region 0 40 568 1010

Checkpoints are saved every CHECKPOINT_FREQ steps under  Models/checkpoints/
TensorBoard logs are written to                         Models/tensorboard/

Launch TensorBoard with (from the repo root):
  tensorboard --logdir Models/tensorboard
"""
import argparse
import time
import glob
import os
from pathlib import Path
import threading

try:
    import msvcrt          # Windows-only: non-blocking keyboard for hotkeys
except ImportError:        # other OSes: training still works, hotkeys don't
    msvcrt = None

import cv2 as cv
import mss
import numpy as np

from sb3_contrib                     import MaskablePPO
from sb3_contrib.common.wrappers     import ActionMasker
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils   import get_linear_fn

from env import CREnv
from calibration import add_region_arg, get_region, mask_fn, unwrap
from paths import (MODEL_DIR, CKPT_DIR, TB_DIR, FINAL_PATH,
                   BANNER_TEMPLATE as BANNER_TEMPLATE_PATH, check_assets)

BANNER_MATCH_THRESHOLD = 0.80


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# ── Hyperparameters ───────────────────────────────────────────────────────────
#
# Tuned for SHORT training windows (~10-15 hours total, 80-150 games).
#
#   n_steps   128     Gradient update every ~45-60s of real play instead of
#                     2-4 minutes. Multiple updates per game rather than
#                     spanning games. Critical when total episode count is low.
#
#   batch_size 64     2 minibatches per update with n_steps=128. Keeps
#                     gradient noise reasonable without needing more data.
#
#   learning_rate 5e-4  Slightly higher than default to make each update
#                       count more. Decays to 1e-5 via linear schedule so
#                       it doesn't destabilise later.
#
#   gamma     0.97    Effective horizon ~33 steps (~10-15s real time).
#                     Matches card→tower-damage delay in CR well.
#
#   ent_coef  0.02→0.005  Lower starting entropy than before — with only
#                         80-150 games we can't afford long exploration.
#                         Still enough to prevent early collapse.
#
#   n_epochs  10      Keep at 10 — squeezes more signal out of each rollout
#                     without overfitting since clip_range guards it.

HYPERPARAMS = dict(
    learning_rate   = 5e-4,
    n_steps         = 128,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.97,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.02,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    device          = _default_device(),
    tensorboard_log = str(TB_DIR),
)

TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 128          # save every rollout

# Entropy: 0.02 → 0.005 over full run.
# Low enough to exploit reward signal quickly, still prevents total collapse.
USE_ENT_SCHEDULE = True
ENT_START        = 0.02
ENT_END          = 0.005

# LR: 5e-4 → 1e-5 linear decay over full run.
USE_LR_SCHEDULE  = True
LR_START         = 5e-4
LR_END           = 1e-5


# ── Banner detector ───────────────────────────────────────────────────────────

class BannerDetector:
    """Detects the end-of-game banner via normalised cross-correlation."""

    def __init__(self, region: tuple[int, int, int, int]):
        self._mon = {"left": region[0], "top": region[1],
                     "width": region[2], "height": region[3]}
        self._sct = mss.mss()

        template_bgr = cv.imread(str(BANNER_TEMPLATE_PATH))
        if template_bgr is None:
            raise FileNotFoundError(f"Banner template not found: {BANNER_TEMPLATE_PATH}")
        self._template = cv.cvtColor(template_bgr, cv.COLOR_BGR2GRAY)

    def detected(self) -> bool:
        raw    = self._sct.grab(self._mon)
        img    = np.array(raw, dtype=np.uint8)[:, :, :3]
        gray   = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(gray, self._template, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(result)
        return max_val >= BANNER_MATCH_THRESHOLD


# ── Entropy decay callback ────────────────────────────────────────────────────

class EntropyCoefficientCallback(BaseCallback):
    """Linearly decays model.ent_coef from start to end over training."""

    def __init__(self, start: float, end: float, total_timesteps: int):
        super().__init__()
        self._start = start
        self._end   = end
        self._total = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self._total, 1.0)
        self.model.ent_coef = self._end + (self._start - self._end) * (1.0 - progress)
        return True


# ── Episode stats + banner handler ───────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """
    Logs per-episode stats to TensorBoard and handles end-of-game banner
    detection — pauses training, prompts for win/loss, then resumes.
    """

    def __init__(self, banner: BannerDetector, pause_cb_ref: "PauseAndSaveCallback", env):
        super().__init__()
        self._env      = env
        self._banner   = banner
        self._pause_cb = pause_cb_ref

        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int]   = []
        self._wins:       list[int]   = []

        self._current_ep_reward = 0.0
        self._current_ep_len    = 0
        self._banner_cooldown   = 0

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos",   [{}])
        dones   = self.locals.get("dones",   [False])
        rewards = self.locals.get("rewards", [0.0])

        for info, done, reward in zip(infos, dones, rewards):
            self._current_ep_reward += reward
            self._current_ep_len    += 1

            if done:
                self._ep_rewards.append(self._current_ep_reward)
                self._ep_lengths.append(self._current_ep_len)

                health = info.get("health", {})
                # The env reports the outcome the user typed at the banner.
                # Fall back to a tower-based guess if the episode ended
                # some other way.
                won = info.get("won")
                if won is None:
                    won = (health.get("enemy_left",  100) <= 0 or
                           health.get("enemy_right", 100) <= 0)
                self._wins.append(int(won))

                self.logger.record("episode/reward",         self._current_ep_reward)
                self.logger.record("episode/length",         self._current_ep_len)
                self.logger.record("episode/ally_left_hp",   health.get("ally_left",   0))
                self.logger.record("episode/ally_right_hp",  health.get("ally_right",  0))
                self.logger.record("episode/enemy_left_hp",  health.get("enemy_left",  0))
                self.logger.record("episode/enemy_right_hp", health.get("enemy_right", 0))

                window = self._wins[-20:]   # smaller window — only ~150 games total
                self.logger.record("episode/win_rate_20", sum(window) / len(window))
                self.logger.record("episode/won", int(won))

                self._current_ep_reward = 0.0
                self._current_ep_len    = 0

        # ── Banner detection ──────────────────────────────────────────────────
        if self._banner_cooldown > 0:
            self._banner_cooldown -= 1
        elif self._banner.detected():
            self._handle_game_end()
            self._banner_cooldown = 300

        # Returning False tells SB3 to stop training cleanly.
        return not self._pause_cb.stop_requested

    def _handle_game_end(self):
        self._pause_cb._pause      = True
        self._pause_cb._block_keys = True
        print("\n\n[BANNER DETECTED] Game has ended.")

        while True:
            answer = input("Did you win? (y/n): ").strip().lower()
            if answer in ("y", "n"):
                break
            print("  Please enter y or n.")

        won = (answer == "y")
        # The win is recorded once, on the next step, when the env reports
        # terminated=True with info["won"] (previously it was counted twice).
        unwrap(self._env).force_done(won=won)
        print(f"  Logged as {'WIN' if won else 'LOSS'}.")

        while True:
            answer = input("Continue training or end? (continue/end): ").strip().lower()
            if answer in ("continue", "end"):
                break
            print("  Please type 'continue' or 'end'.")

        self._pause_cb._block_keys = False

        if answer == "end":
            self._pause_cb._save_now()
            self._pause_cb.stop_requested = True
            self._pause_cb._pause = False
            return

        # Do NOT call reset() manually here — PPO calls it automatically
        # after it sees terminated=True from the force_done step.
        # Just unpause so the rollout loop can consume force_done.
        input("\nStart a new game, then press Enter when ready… ")
        for i in range(3, 0, -1):
            print(f"\rResuming in {i}…", end="", flush=True)
            time.sleep(1)
        print("\rResuming training…   \n")
        self._pause_cb._pause = False


# ── Pause / save / quit callback ─────────────────────────────────────────────

class PauseAndSaveCallback(BaseCallback):
    """Hotkeys: S = save  P = pause/resume  Q = quit cleanly."""

    def __init__(self, save_path: Path):
        super().__init__()
        self._save_path  = save_path
        self._pause      = False
        self._block_keys = False
        self.stop_requested = False
        if msvcrt is None:
            print("Hotkeys unavailable on this OS — use Ctrl+C to stop (model is saved).")
            return
        self._thread     = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        print("Hotkeys: [S] save  [P] pause/resume  [Q] quit cleanly")
        while not self.stop_requested:
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if self._block_keys:
                    continue
                if key == 's':
                    self._save_now()
                elif key == 'p':
                    self._pause = not self._pause
                    print(f"\n{'Paused' if self._pause else 'Resumed'}")
                elif key == 'q':
                    # Raising here would only kill this background thread, so
                    # instead flag the main loop to stop (see _on_step).
                    print("\nQuitting after the current step …")
                    self.stop_requested = True
                    self._pause = False
            else:
                time.sleep(0.05)

    def _save_now(self):
        path = str(self._save_path / f"manual_save_{self.num_timesteps}steps")
        self.model.save(path)
        print(f"\n✓ Saved → {path}.zip")

    def _on_step(self) -> bool:
        while self._pause and not self.stop_requested:
            time.sleep(0.1)
        return not self.stop_requested


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _latest_checkpoint() -> Path | None:
    zips = sorted(glob.glob(str(CKPT_DIR / "*.zip")), key=os.path.getmtime)
    return Path(zips[-1]) if zips else None


def _steps_from_checkpoint(path: Path) -> int:
    try:
        parts = path.stem.split("_")
        for i, part in enumerate(parts):
            if part == "steps" and i > 0:
                return int(parts[i - 1])
    except Exception:
        pass
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--render", action="store_true",
                        help="Show the live annotated vision window while training")
    add_region_arg(parser)
    args = parser.parse_args()

    check_assets(BANNER_TEMPLATE_PATH)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    region = get_region(args)

    print("Loading banner detector …")
    banner = BannerDetector(region)

    print("Building environment …")
    env = CREnv(calibration_region=region,
                render_mode="human" if args.render else None)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, filename=str(MODEL_DIR / "monitor.csv"))

    lr = get_linear_fn(LR_START, LR_END, 1.0) if USE_LR_SCHEDULE else LR_START

    pause_cb      = PauseAndSaveCallback(save_path=CKPT_DIR)
    stats_cb      = EpisodeStatsCallback(banner=banner, pause_cb_ref=pause_cb, env=env)
    ent_cb        = EntropyCoefficientCallback(ENT_START, ENT_END, TOTAL_TIMESTEPS) \
                    if USE_ENT_SCHEDULE else None
    checkpoint_cb = CheckpointCallback(
        save_freq          = CHECKPOINT_FREQ,
        save_path          = str(CKPT_DIR),
        name_prefix        = "cr_model",
        save_replay_buffer = False,
        save_vecnormalize  = False,
        verbose            = 1,
    )
    cb_list = [checkpoint_cb, stats_cb, pause_cb]
    if ent_cb is not None:
        cb_list.append(ent_cb)
    callbacks = CallbackList(cb_list)

    # ── Load or create model ──────────────────────────────────────────────────
    steps_done = 0
    resume     = False

    if args.resume or args.checkpoint:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
        if ckpt_path is None or not ckpt_path.exists():
            print("No checkpoint found — starting fresh.")
        else:
            print(f"Resuming from: {ckpt_path}")
            model = MaskablePPO.load(
                str(ckpt_path),
                env             = env,
                device          = HYPERPARAMS["device"],
                verbose         = HYPERPARAMS["verbose"],
                tensorboard_log = HYPERPARAMS["tensorboard_log"],
            )
            model.learning_rate = lr
            model.ent_coef      = ENT_START
            # The model stores its own step count — more reliable than parsing
            # the filename (manual saves are named differently).
            steps_done = model.num_timesteps or _steps_from_checkpoint(ckpt_path)
            resume = True

    if not resume:
        model = MaskablePPO(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = lr,
            n_steps         = HYPERPARAMS["n_steps"],
            batch_size      = HYPERPARAMS["batch_size"],
            n_epochs        = HYPERPARAMS["n_epochs"],
            gamma           = HYPERPARAMS["gamma"],
            gae_lambda      = HYPERPARAMS["gae_lambda"],
            clip_range      = HYPERPARAMS["clip_range"],
            ent_coef        = ENT_START,
            vf_coef         = HYPERPARAMS["vf_coef"],
            max_grad_norm   = HYPERPARAMS["max_grad_norm"],
            verbose         = HYPERPARAMS["verbose"],
            device          = HYPERPARAMS["device"],
            tensorboard_log = HYPERPARAMS["tensorboard_log"],
        )

    remaining_steps = max(TOTAL_TIMESTEPS - steps_done, 0)
    if remaining_steps == 0:
        print(f"Already at {steps_done} steps. Increase TOTAL_TIMESTEPS to continue.")
        env.close()
        return

    print(f"\nTraining for {remaining_steps:,} steps "
          f"({steps_done:,} done / {TOTAL_TIMESTEPS:,} total)")
    print(f"Checkpoints → {CKPT_DIR}")
    print(f"TensorBoard → tensorboard --logdir \"{TB_DIR}\"")
    print(f"\nHyperparams: n_steps={HYPERPARAMS['n_steps']}  "
          f"batch={HYPERPARAMS['batch_size']}  "
          f"lr={LR_START}→{LR_END}  "
          f"ent={ENT_START}→{ENT_END}  "
          f"gamma={HYPERPARAMS['gamma']}\n")

    try:
        model.learn(
            total_timesteps     = remaining_steps,
            callback            = callbacks,
            reset_num_timesteps = not resume,
            tb_log_name         = "MaskablePPO",
            progress_bar        = False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving …")

    model.save(str(FINAL_PATH))
    print(f"\nFinal model saved → {FINAL_PATH}.zip")
    env.close()


if __name__ == "__main__":
    main()