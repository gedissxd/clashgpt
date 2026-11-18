# Clash Royale Autonomous Agent

A reinforcement learning agent that plays Clash Royale in real time on a Windows PC via an Android emulator. The agent uses computer vision (dual YOLOv11 models + SORT tracking) to perceive the game state, and a Maskable PPO policy to decide which cards to play and where to place them.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Project Documents](#project-documents)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Large Assets (Google Drive)](#large-assets-google-drive)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Module Reference](#module-reference)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [Training Details](#training-details)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [Known Limitations](#known-limitations)

---

## Quick Start

1. **Get the code and assets.** Clone this repo, then download `Images/`, `YOLO/` and `Models/` from the [Google Drive folder](https://drive.google.com/drive/folders/1zatBii3zCUWsibtnpse99O7PVHbQvDBs?usp=sharing) and put them in the repo root, next to `Code/`.
2. **Install** (Windows, Python 3.10+):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. **Open Clash Royale in an emulator** (BlueStacks / LDPlayer) in portrait mode at the same window size used during development (game area ≈ 561×998 px) and start a battle.
4. **Check the vision first:**
   ```bash
   cd Code
   python debug_cv.py
   ```
   Hover the mouse over the emulator's top-left corner, then bottom-right, when prompted. The script prints a `--region …` you can reuse to skip this step next time. Boxes, elixir, hand and tower HP in the `CR Agent` window should match the game. If they don't, the window size is wrong.
5. **Let the trained agent play:** `python play.py` (or `python play.py --dry-run --render` to watch its decisions without it clicking).
6. **Train / keep training:** `python train.py` (fresh) or `python train.py --resume`.

Move the mouse into a screen corner at any time to trigger PyAutoGUI's fail-safe and stop all clicking.

---

## Overview

The agent runs a fixed 8-card deck against a standard opponent deck, playing entirely through screen capture and simulated mouse clicks — no game memory reading or API access. Every decision is made from pixels alone.

**Player deck:** Giant, Bowler, Witch, Graveyard, Guards, Arrows, Snowball, Minions

**Opponent deck (tracked):** Goblin Gang, Princess, Ice Spirit, Rocket, Goblin Barrel, Knight, Log, Cannon

The system has three entry points:
- `train.py` — live training loop with MaskablePPO, checkpoint saving, TensorBoard logging, and a banner detector that pauses training when a game ends to collect win/loss labels.
- `play.py` — runs a trained model against real games without training it (optionally `--dry-run` so it never clicks).
- `debug_cv.py` — a live screen mirror with YOLO + SORT overlay for debugging the vision pipeline. Never clicks, so you can play yourself.

---

## Project Documents

| Document | Description |
|----------|-------------|
| [Report](Report.pdf) | Full technical write-up |
| [Poster](Poster.pdf) | Research poster |
| [Presentation](Presentation.pdf) | Slide deck |

---

## Architecture

```
Screen (mss)
    │
    ▼
get_top_corner()          ← anchors all pixel coords to the game window
    │
    ├──► LBGG YOLO         ← detects troops & buildings
    ├──► SARG YOLO         ← detects spells & projectiles
    │         │
    │         ▼
    │     CRTracker (SORT) ← assigns stable IDs, separate ally/enemy pools
    │
    ├──► get_hand()        ← template matching → 4-card hand + next card
    ├──► get_elixir()      ← HSV color scan of elixir bar → float [0,10]
    └──► get_tower_health()← pixel scan of HP bars → 6 tower HP values
              │
              ▼
        build_observation() → 202-float vector
              │
              ▼
        MaskablePPO policy
              │
              ▼
        get_action_mask()   ← masks unaffordable / out-of-hand cards
              │
              ▼
        execute_action()    ← PyAutoGUI clicks (hand slot → placement)
```

---

## Repository Structure

```
clashgpt/
│
├── Code/
│   ├── actions.py          # Action masking and PyAutoGUI execution
│   ├── calibration.py      # Mouse calibration, --region flag, wrapper helpers
│   ├── constants.py        # Card costs, placement coords, action-space layout
│   ├── debug_cv.py         # Live vision debug window (never clicks)
│   ├── env.py              # Gymnasium CREnv — wraps the full pipeline
│   ├── hand_detection.py   # Template-matching hand reader
│   ├── misc_functions.py   # get_top_corner(), get_elixir()
│   ├── observation.py      # build_observation() → 202-float vector
│   ├── paths.py            # Where Images/, YOLO/, Models/ live (CLASHGPT_ROOT)
│   ├── play.py             # Run a trained model without training
│   ├── reward.py           # Hybrid dense/sparse reward function
│   ├── tracker.py          # SORT-based multi-object tracker (CRTracker)
│   ├── tower_health.py     # Pixel-scan HP reader for all 6 towers
│   └── train.py            # MaskablePPO training script
│
├── tests/                  # Offline unit tests (no emulator/GPU needed)
├── requirements.txt
│
├── Poster.pdf
├── Report.pdf
├── Presentation.pdf
│
├── Images/             # Reference images used by CV modules (on Drive)
│   ├── cards_in_hand/  # 9 card template PNGs for get_hand()
│   ├── mini_cards/     # 8 mini-card templates for next-card detection
│   ├── test_images/    # Static game screenshots for offline testing
│   └── banner.png      # End-of-game banner template for BannerDetector
│
├── YOLO/               # YOLO training data and weights (on Drive)
│   ├── LBGG/           # "Troops & Buildings" model
│   │   └── LBGG_model/train/weights/best.pt
│   └── SARG/           # "Spells & Ranged" model
│       └── SARG_model/train/weights/best.pt
│
└── Models/             # RL training outputs (on Drive)
    ├── checkpoints/    # Auto-saved .zip checkpoints every 512 steps
    ├── tensorboard/    # TensorBoard event files
    ├── monitor.csv     # SB3 Monitor wrapper episode log
    └── cr_agent_final.zip  # Final saved policy
```

---

## Large Assets (Google Drive)

The `Images/`, `YOLO/`, and `Models/` folders are too large for GitHub and are hosted on Google Drive:

**[Download Images, YOLO Weights & Models](https://drive.google.com/drive/folders/1zatBii3zCUWsibtnpse99O7PVHbQvDBs?usp=sharing)**

After downloading, put the three folders in the repository root (next to `Code/`). All paths are resolved in `Code/paths.py`, so nothing needs editing.

To keep them somewhere else, point `CLASHGPT_ROOT` at the folder that contains them:
```bash
set CLASHGPT_ROOT=D:\ClashGPT
```
(PowerShell: `$env:CLASHGPT_ROOT = "D:\ClashGPT"`). If a file is missing, the scripts stop with a message naming it.

---

## Setup & Installation

**Requirements:** Windows 10/11, Python 3.10+, an Android emulator (e.g. BlueStacks or LDPlayer) running Clash Royale. A CUDA GPU is strongly recommended; without one everything falls back to CPU, which works but YOLO will be slow.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Optional, to run the offline tests (works on any OS):
```bash
pip install pytest
python -m pytest tests
```

---

## Running the Project

### Training

```bash
cd Code

# Start a fresh training run
python train.py

# Resume from the most recent checkpoint
python train.py --resume

# Resume from a specific checkpoint
python train.py --resume --checkpoint "../Models/checkpoints/cr_model_1024_steps.zip"

# Skip mouse calibration / show the live vision window while training
python train.py --resume --region 0 40 568 1010 --render
```

**Calibration prompt** — unless you pass `--region`, at startup, the script asks you to mouse to the top-left corner of the game window (hold for 3 s), then the bottom-right corner (hold for 3 s). This sets the capture region. `get_top_corner()` then scans the captured frame for the characteristic dark-blue pixel at the game's top-left corner (BGR `[72, 30, 24]`, tolerance ±5) to anchor all subsequent coordinates.

**During training:**
- `S` — manually save a checkpoint
- `P` — pause / resume
- `Q` — save and quit cleanly

Hotkeys need Windows (`msvcrt`); elsewhere use Ctrl+C, which also saves the model.

When the end-of-game banner is detected (template match score ≥ 0.80), training pauses and the terminal prompts for `y` (win) or `n` (loss), then `continue` or `end`, then asks you to start a new game and press Enter.

**TensorBoard** (from the repo root):
```bash
tensorboard --logdir Models/tensorboard
```
Useful charts: `episode/win_rate_20`, `episode/reward`, `episode/length`.

### Playing with a trained model

```bash
cd Code
python play.py                                   # uses Models/cr_agent_final.zip
python play.py --model ../Models/checkpoints/cr_model_1024_steps.zip
python play.py --dry-run --render                # watch decisions, never click
python play.py --games 5 --stochastic            # 5 games, sample actions
```

After each game you're asked whether you won and a running win rate is printed.

### Debug Vision

```bash
cd Code
python debug_cv.py
```

Opens a live OpenCV window (`CR Agent`) showing YOLO bounding boxes color-coded by team (green = ally, red = enemy), track IDs, elixir, hand, next card, and tower HPs. Nothing is clicked. Press `Q` in the window to quit.

---

## Module Reference

### `constants.py`
Single source of truth for all card data and action-space geometry. Everything else imports from here — do not hardcode card names or placement coords elsewhere.

Key exports:
- `PLAYER_DECK` — ordered list of 8 cards the agent runs
- `CARD_COSTS` — elixir cost per card
- `CARD_PLACEMENTS` — dict mapping each card to its list of valid `(x, y)` drop coordinates, all relative to the `corner` anchor
- `HAND_SLOT_COORDS` — pixel centres of hand slots 0–3, corner-relative
- `ACTION_OFFSETS` — first action index for each card's placement block
- `WAIT_ACTION` — index 48 (the last action)
- `N_ACTIONS` — 49 total
- `ACTION_INDEX_MAP` — reverse map: `int → (card, placement_idx)` or `None` for WAIT

### `actions.py`
- `get_action_mask(hand, elixir) → np.ndarray(49, bool)` — masks any card not in the current 4-card hand or whose cost exceeds current elixir. WAIT is always unmasked.
- `execute_action(action_idx, hand, corner, capture_offset, dry_run) → bool` — translates a flat action index into two PyAutoGUI clicks: first the hand slot, then the placement coordinate. Returns `False` for WAIT or unresolvable actions. With `dry_run=True` nothing is clicked.
- `action_to_str(action_idx) → str` — human-readable label for logging (e.g. `giant@(242,756)`).

### `misc_functions.py`
- `get_top_corner(img) → (x, y)` — finds the first pixel (row-major) matching BGR `[72, 30, 24]` (±5 tolerance), a fixed dark-blue pixel at the game's top-left edge. Vectorised with NumPy. This anchor is used as `corner` everywhere.
- `get_elixir(img, corner, last_elixir, max_rise) → float` — reads the elixir bar by scanning 10 fixed x-positions at `y=978` (corner-relative) in HSV. Full bar detected via a single pixel check at `x=538`. Partial bars use fractional pixel counting. Upward jumps larger than `max_rise` are clamped to suppress color noise (no clamping when the previous value is unknown). `env.py` sets `max_rise` from the elapsed time so the reading keeps up in double/triple elixir.

### `hand_detection.py`
- `get_hand(img, corner) → (hand: list[str], next_card: str)` — templates are loaded once at import. Converts the frame to grayscale, crops the hand region (`y+820:y+951, x+124:x+541`), and runs `cv2.matchTemplate` (TM_CCOEFF_NORMED) against each of the 9 card templates. The top 4 spatially-separated matches (min 50px apart) are sorted left-to-right to give the 4-card hand. Next card is detected similarly against 8 mini-card templates in a separate crop (`y+930:y+986, x+29:x+75`).

### `tower_health.py`
- `get_tower_health(img, corner, last_health, tower_state) → (results, segments)` — scans 4 princess tower HP bars and 2 king tower HP bars by walking pixel columns and matching against known health bar BGR palettes (±18 tolerance). King bars are only scanned once a princess tower on that side reaches 0. Towers confirmed dead over `KING_CONFIRM_FRAMES=3` consecutive frames are permanently locked at 0 in `tower_state["dead"]` to prevent resurrection artifacts.
- `make_tower_state() → dict` — creates a fresh per-episode state dict with `dead: set` and `king_confirm` counters.
- `debug_draw(image, corner, last_health, tower_state, out_path)` — saves an annotated PNG showing every scanned pixel color-coded (green = health, orange = bar end, yellow = unexpected/king polled).

### `tracker.py`
- `CRTracker(max_age, min_hits, iou_threshold)` — maintains two independent SORT track pools (ally and enemy) so cross-team ID collisions are impossible. Class-consistency enforcement: a detection is forbidden from matching a track whose majority-voted class differs.
- `tracker.update(detections) → list[TrackResult]` — runs Kalman predict → Hungarian assignment → update. Tracks returned only after `min_hits=2` confirmed frames. Tracks survive up to `max_age=8` frames without a match.
- `TrackResult` fields: `track_id, x1, y1, x2, y2, cls, conf, team, age, hits`

### `observation.py`
- `build_observation(tracks, hand, next_card, elixir, tower_health) → np.ndarray(202,)` — assembles the flat observation vector. See [Observation Space](#observation-space) below.

### `reward.py`
- `evaluate_reward(...) → (float, list[str])` — the reward and the reasons it fired. See [Reward Function](#reward-function) below.
- `compute_reward(...)` / `reward_summary(...)` — thin wrappers returning just the number / just the reason string.

### `env.py`
- `CREnv(calibration_region, render_mode, dry_run)` — the Gymnasium environment. Loads both YOLO models on init (GPU if available, else CPU). Runs YOLO every `YOLO_SKIP_FRAMES=3` steps (the other frames let the tracker coast on Kalman predictions). HP noise filter suppresses tower deltas < 0.5% between steps. Spell-miss detection waits `SPELL_EVAL_STEPS=6` steps after Arrows/Snowball and fires the miss penalty if no enemy track disappeared. A goblin-barrel intercept routine can override the policy's action to defend a tower. The episode ends when `force_done()` is called by the banner handler; `info["won"]` carries the outcome.
- `render_mode="human"` — opens a `CR Agent` OpenCV window with bounding boxes, HUD overlay, and per-step reward.

### `train.py`
- `BannerDetector` — polls the capture region for `banner.png` via template matching every step (with a 300-step cooldown after a detection) to identify game-end screens.
- `EpisodeStatsCallback` — logs per-episode reward, length, and tower HPs to TensorBoard. Handles the banner-detected pause flow.
- `EntropyCoefficientCallback` — linearly decays `model.ent_coef` from 0.02 → 0.005 over the full training run (done manually per step because MaskablePPO does not accept a callable for `ent_coef`).
- `PauseAndSaveCallback` — background keyboard listener thread for S/P/Q hotkeys (Windows). `Q` sets a flag that stops `model.learn()` cleanly; the final model is then saved.

---

## Observation Space

`Box(low=0.0, high=1.0, shape=(202,), dtype=float32)`

| Indices | Content | Size |
|---------|---------|------|
| 0 – 149 | Up to 30 tracked troops × 5 features | 150 |
| 150 – 185 | 4 hand slots × 9-class one-hot (CARD_NAMES) | 36 |
| 186 – 194 | Next card one-hot (9 classes) | 9 |
| 195 | Elixir ÷ 10 | 1 |
| 196 – 199 | Princess tower HPs ÷ 100 (ally_left, ally_right, enemy_left, enemy_right) | 4 |
| 200 | Ally king HP ÷ 100 | 1 |
| 201 | Enemy king HP ÷ 100 | 1 |

**Troop feature vector (5 floats):**

| Index | Feature | Range |
|-------|---------|-------|
| 0 | Bounding box centre x ÷ game width | [0, 1] |
| 1 | Bounding box centre y ÷ game height | [0, 1] |
| 2 | Team (1.0 = ally, 0.0 = enemy) | {0, 1} |
| 3 | Class index ÷ (n\_classes − 1) | [0, 1] |
| 4 | min(track age, 60) ÷ 60 | [0, 1] |

If more than 30 troops are tracked, skeletons are dropped first (oldest first), then non-skeletons are trimmed by age.

**Troop classes (18 total):** giant, bowler, witch, graveyard, guard, arrows, snowball, minion, princess, ice\_spirit, rocket, goblin\_barrel, knight, log, cannon, skeleton, goblin, spear\_goblin

**Card classes (9, includes "empty"):** arrows, bowler, giant, graveyard, guards, minions, snowball, witch, empty

Team assignment uses fixed class sets in `env.py` (`_ALLY_CLASSES` / `_ENEMY_CLASSES`), since each deck's units are known.

---

## Action Space

`Discrete(49)` — one action per valid placement coordinate per card, plus one WAIT action.

| Card | # Placements | Action Indices |
|------|-------------|----------------|
| giant | 6 | 0 – 5 |
| bowler | 6 | 6 – 11 |
| witch | 5 | 12 – 16 |
| graveyard | 2 | 17 – 18 |
| guards | 6 | 19 – 24 |
| arrows | 8 | 25 – 32 |
| snowball | 8 | 33 – 40 |
| minions | 6 | 41 – 46 |
| WAIT | — | 48 |

Lane assignment (used by reward): placement `x < 280` = left lane, `x ≥ 280` = right lane.

The action mask zeros out any card not currently in the 4-card hand or whose elixir cost exceeds current elixir. WAIT (index 48) is always unmasked. The mask is fed to MaskablePPO via the `ActionMasker` wrapper so the policy never samples an illegal action.

---

## Reward Function

Hybrid dense + sparse design. All weights live in `RewardWeights` in `reward.py` and can be tuned without touching logic. Every step's reasons are printed as `[reward] …` and shown in the render overlay.

### Dense (every step)

| Signal | Weight |
|--------|--------|
| Enemy princess tower HP lost (per 1%) | +0.02 |
| Ally princess tower HP lost (per 1%) | −0.04 |
| Enemy troop deep in our half (`norm_y > 0.75`) | −0.01 × depth past 0.75 |
| WAIT while a card in hand is affordable | −0.03 |
| WAIT while nothing is affordable and elixir < 4 | +0.01 |

### Sparse (event-triggered)

| Event | Reward |
|-------|--------|
| Win / loss (from banner prompt) | +5.0 / −5.0 |
| Enemy / ally princess tower destroyed | +3.0 / −3.0 |
| Enemy Princess track disappears | +1.5 |
| Card played with elixir < cost + 1.5 | −0.2 |
| Non-giant card played onto an empty board | −0.5 |
| Non-giant played into an empty lane while the other lane has allies | −0.5 |
| Defensive unit (bowler/witch/guards/minions) played with enemies approaching in that lane | +0.5 per enemy |
| Defensive unit placed at y 450–560 / y ≥ 700 | +0.5 / −0.5 |
| Graveyard with a giant over the bridge in that lane | +10.0 |
| Graveyard with a giant in that lane (not yet over bridge) | +5.0 |
| Graveyard with any ally in the enemy half of that lane | +2.0 |
| Witch played in a lane with a friendly giant | +2.0 |
| Other card played in a lane with a friendly giant | +0.5 + 0.1 per supporting troop |
| Spell (Arrows/Snowball) missed | −5.0 |
| Spell killed goblins / Arrows killed Princess | +0.4 each / +2.0 |
| Spell intercepted a goblin barrel near our tower | +5.0 |
| Spell at tower after a barrel landed (scaled by speed) | up to +6.0 |

The push bonuses teach the core Giant–Graveyard combo: place the Giant first, let it advance, then drop the Graveyard when the Giant is over the bridge.

## Training Details

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Algorithm | MaskablePPO (sb3-contrib) | Handles discrete action masking natively |
| `n_steps` | 128 | Several updates per game — critical when only ~100 games are played |
| `batch_size` | 64 | 2 minibatches per rollout |
| `n_epochs` | 10 | |
| `gamma` | 0.97 | ~33-step effective horizon; matches card→tower-damage delay |
| `gae_lambda` | 0.95 | |
| `clip_range` | 0.2 | |
| `ent_coef` | 0.02 → 0.005 (linear) | Enough early entropy to prevent collapse, low enough to exploit quickly |
| `learning_rate` | 5e-4 → 1e-5 (linear) | |
| `vf_coef` | 0.5 | |
| `max_grad_norm` | 0.5 | |
| `device` | cuda (cpu fallback) | |
| Total timesteps | 1,000,000 | |
| Checkpoint frequency | every 128 steps | Saved to `Models/checkpoints/` |

Checkpoints are named `cr_model_<N>_steps.zip` (manual `S` saves: `manual_save_<N>steps.zip`). `--resume` picks the most recently modified checkpoint and continues from the step count stored inside the model, so the LR/entropy schedules pick up where they left off.

---

## Computer Vision Pipeline

### Game Window Anchoring
All pixel coordinates throughout the codebase are stored as offsets from a `corner = (x, y)` anchor. `get_top_corner()` finds this anchor by scanning for the fixed dark-blue border pixel at the top-left of the game UI (BGR `[72, 30, 24]`, ±5). This means the code works regardless of where the emulator window is positioned on screen.

### YOLO Models
Two YOLOv11 models run on the cropped game region (561×998 px), with results merged before the tracker:
- **LBGG** (`best.pt` in `YOLO/LBGG/`) — detects troops and buildings
- **SARG** (`best.pt` in `YOLO/SARG/`) — detects spells and projectiles

Inference threshold: `conf=0.35`, `iou=0.45`. YOLO runs every 3rd frame (`YOLO_SKIP_FRAMES=3`); on skipped frames the tracker coasts on its Kalman predictions.

### SORT Tracker
Two independent SORT pools (ally/enemy) with a class-consistency check: detections can only match tracks of the same class. Parameters (as used by `CREnv`): `max_age=4`, `min_hits=2`, `iou_threshold=0.25`. The 7-state Kalman filter tracks `[x_c, y_c, area, aspect_ratio, vx, vy, v_area]`.

### Hand Detection
Template matching (TM_CCOEFF_NORMED) against 9 reference PNGs in `Images/cards_in_hand/`. The top 4 matches with centres at least 50 px apart (left-to-right order) identify the current hand. Next card uses 8 mini-card templates in `Images/mini_cards/`.

### Elixir Reading
10 fixed pixel positions along the elixir bar are sampled in HSV. Each full-bar pixel scores +1 elixir; partial-bar pixels use sub-pixel interpolation over a 39-pixel scan window to give fractional elixir values. Upward spikes faster than 1.2 elixir/second (just above triple-elixir regen) are clamped.

### Tower Health Reading
HP bars are scanned column-by-column in BGR, matching against known palette colors (±18 tolerance). Princess bars stop scanning on a "terminate" color (the empty bar edge). King bars are only activated once a princess tower on that side falls. A 3-frame confirmation window prevents single noisy frames from declaring towers dead prematurely.

---

## Known Limitations

- **Windows-focused** — developed on Windows. Training hotkeys use `msvcrt` (Windows only; elsewhere use Ctrl+C).
- **Fixed emulator resolution** — all pixel coordinates assume the emulator is running at the specific resolution captured during development. Rescaling the window will break hand detection, elixir reading, and tower health scanning.
- **Manual game reset** — the agent does not navigate Clash Royale menus. A human must start each game and press Enter in the terminal when the game is ready.
- **No opponent modeling** — the opponent's deck is tracked by YOLO but not explicitly used in strategy beyond positional awareness.
- **Single environment** — training runs in a single non-vectorized environment, so sample efficiency is limited by real-time game speed (~2–3 s/step).
- **Scripted barrel defense is off-policy** — when the goblin-barrel intercept overrides the policy's action, PPO still records the action it originally chose, so those steps are slightly mislabeled in the rollout.
