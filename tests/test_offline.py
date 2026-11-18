"""Offline tests — no emulator, GPU, YOLO weights or display needed.

Run from the repo root:  python -m pytest tests
"""
import numpy as np
import pytest

from actions import get_action_mask, action_to_str
from constants import ACTION_OFFSETS, N_ACTIONS, WAIT_ACTION
from misc_functions import get_top_corner, get_elixir
from observation import build_observation, OBS_SIZE
from reward import compute_reward, reward_summary, WEIGHTS
from tracker import CRTracker, Detection, TrackResult

FULL_HP = {k: 100.0 for k in ("ally_left", "ally_right", "enemy_left",
                              "enemy_right", "ally_king", "enemy_king")}


def _track(x, y, team="ally", cls="giant", tid=1):
    return TrackResult(tid, x - 10, y - 10, x + 10, y + 10, cls, 0.9, team, 5, 5)


# ── constants / actions ─────────────────────────────────────────────────────

def test_action_space_layout():
    assert N_ACTIONS == 49 and WAIT_ACTION == 48


def test_mask_respects_hand_and_elixir():
    mask = get_action_mask(["giant", "snowball", "empty", "arrows"], elixir=2.5)
    assert mask[WAIT_ACTION]
    assert mask[ACTION_OFFSETS["snowball"]]           # cost 2, in hand
    assert not mask[ACTION_OFFSETS["giant"]]          # cost 5, can't afford
    assert not mask[ACTION_OFFSETS["arrows"]]         # cost 3, can't afford
    assert not mask[ACTION_OFFSETS["witch"]]          # not in hand


def test_action_to_str():
    assert action_to_str(WAIT_ACTION) == "WAIT"
    assert action_to_str(0).startswith("giant@")


# ── vision helpers ──────────────────────────────────────────────────────────

def test_get_top_corner_finds_first_pixel_row_major():
    img = np.zeros((200, 300, 3), np.uint8)
    img[50, 120] = (72, 30, 24)
    img[50, 200] = (74, 31, 22)        # within tolerance, but later in row
    img[90, 10]  = (72, 30, 24)        # later row
    assert get_top_corner(img) == (120, 50)


def test_get_top_corner_none_and_bgra():
    assert get_top_corner(np.zeros((10, 10, 4), np.uint8)) is None


def _elixir_frame(n_full):
    """Synthetic frame with `n_full` full elixir segments, corner at (0,0)."""
    import cv2 as cv
    img = np.zeros((1000, 600, 3), np.uint8)
    purple = cv.cvtColor(np.uint8([[[148, 215, 190]]]), cv.COLOR_HSV2BGR)[0, 0]
    xs = [154, 194, 233, 272, 311, 350, 389, 428, 468, 507]
    for x in xs[:n_full]:
        img[978, x] = purple
    return img


def test_elixir_not_clamped_when_previous_unknown():
    # Before the fix this returned 0.15 at the start of every game.
    assert get_elixir(_elixir_frame(5), (0, 0), last_elixir=0.0) == 5.0


def test_elixir_spike_clamped():
    assert get_elixir(_elixir_frame(5), (0, 0), last_elixir=2.0, max_rise=0.5) == 2.5
    assert get_elixir(_elixir_frame(2), (0, 0), last_elixir=6.0) == 2.0  # drops OK


# ── observation ─────────────────────────────────────────────────────────────

def test_observation_shape_and_range():
    tracks = [_track(100, 100 + i, tid=i) for i in range(40)]
    obs = build_observation(tracks, ["giant", "witch", "empty", "arrows"],
                            "bowler", 7.0, FULL_HP)
    assert obs.shape == (OBS_SIZE,)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


# ── reward ──────────────────────────────────────────────────────────────────

def test_idle_with_playable_card_is_penalised():
    r = compute_reward(FULL_HP, FULL_HP, done=False, won=False,
                       elixir=8.0, hand=["giant", "witch", "arrows", "guards"],
                       verbose=False)
    assert r < 0


def test_win_and_loss_are_rewarded():
    win  = compute_reward(FULL_HP, FULL_HP, True, True,  elixir=0, verbose=False)
    loss = compute_reward(FULL_HP, FULL_HP, True, False, elixir=0, verbose=False)
    assert win > 0 > loss


def test_lane_split_consistent_at_x_250():
    # A giant at x=250 is in the LEFT lane (split is x=280). With the old
    # GAME_W=420 some helpers put it in the right lane.
    giant = _track(250, 300, cls="giant")
    s = reward_summary(FULL_HP, FULL_HP, False, False, played_card="graveyard",
                       played_lane="left", played_y=240, elixir=10,
                       tracks=[giant])
    assert "GY_BRIDGE" in s
    assert "wrong_lane" not in s


def test_summary_matches_compute():
    kw = dict(prev_health=FULL_HP,
              curr_health={**FULL_HP, "enemy_left": 90.0},
              done=False, won=False, elixir=9.0, hand=["giant"])
    assert "dealt" in reward_summary(**kw)
    assert compute_reward(**kw, verbose=False) == pytest.approx(
        10 * WEIGHTS.hp_dealt_per_pct + WEIGHTS.idle_penalty)


# ── tracker ─────────────────────────────────────────────────────────────────

def test_tracker_keeps_ids_and_separates_teams():
    tr = CRTracker(max_age=4, min_hits=2)
    for step in range(4):
        out = tr.update([
            Detection(10 + step, 10, 40 + step, 40, 0.9, "giant", "ally"),
            Detection(10 + step, 10, 40 + step, 40, 0.9, "knight", "enemy"),
        ])
    ids = {(t.team, t.track_id) for t in out}
    assert len(ids) == 2
    assert {t.team for t in out} == {"ally", "enemy"}
