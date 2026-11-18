"""
actions.py — action masking and execution for the CR agent.

Public API
----------
get_action_mask(hand, elixir)  ->  np.ndarray shape (N_ACTIONS,) dtype bool
    True  = action is legal this step.
    False = action is masked out (MaskablePPO will never sample it).

    Masking rules:
      • A card's entire placement block is masked if:
          (a) the card is not currently in hand[0:4], OR
          (b) the player has less elixir than the card's cost.
      • WAIT_ACTION is always True.

execute_action(action_idx, hand, corner, capture_offset, dry_run)  ->  bool
    Translates a flat action index into PyAutoGUI clicks and executes it.
    Returns True if an actual card was played, False for wait.
    `hand`   : list[str] of length 4, left-to-right (from get_hand()).
    `corner` : (cx, cy) pixel offset returned by get_top_corner().
"""

import time
import numpy as np

from constants import (
    CARD_COSTS,
    CARD_PLACEMENTS,
    HAND_SLOT_COORDS,
    ACTION_OFFSETS,
    ACTION_INDEX_MAP,
    WAIT_ACTION,
    N_ACTIONS,
    PLAYER_DECK,
)

_pyautogui = None


def _gui():
    """Import PyAutoGUI lazily so the vision/reward code (and tests) can be
    imported on machines without a display."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.PAUSE = 0.05     # lower pause reduces latency during play
        pyautogui.FAILSAFE = True  # slam the mouse into a screen corner to abort
        _pyautogui = pyautogui
    return _pyautogui


def play_card_at(
    slot_idx: int,
    target: tuple[int, int],
    corner: tuple[int, int],
    capture_offset: tuple[int, int] = (0, 0),
    dry_run: bool = False,
) -> None:
    """Click hand slot `slot_idx`, then the corner-relative `target` point."""
    if dry_run:
        return
    cx, cy = corner
    ox, oy = capture_offset
    sx, sy = HAND_SLOT_COORDS[slot_idx]
    gui = _gui()
    gui.click(ox + cx + sx, oy + cy + sy)
    time.sleep(0.05)
    gui.click(ox + cx + target[0], oy + cy + target[1])


# ── Action mask ───────────────────────────────────────────────────────────────

def get_action_mask(hand: list[str], elixir: float) -> np.ndarray:
    """
    Parameters
    ----------
    hand   : list of exactly 4 card name strings (may include "empty").
    elixir : current elixir as a float in [0, 10].

    Returns
    -------
    Boolean mask of shape (N_ACTIONS,).
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    hand_set = set(hand) - {"empty"}

    for card in PLAYER_DECK:
        block_start = ACTION_OFFSETS[card]
        block_end   = block_start + len(CARD_PLACEMENTS[card])

        in_hand      = card in hand_set
        can_afford   = elixir >= CARD_COSTS[card]

        if in_hand and can_afford:
            mask[block_start:block_end] = True
        # else: entire placement block stays False

    # Wait is always legal
    mask[WAIT_ACTION] = True

    return mask


# ── Action execution ──────────────────────────────────────────────────────────

def execute_action(
    action_idx: int,
    hand: list[str],
    corner: tuple[int, int],
    capture_offset: tuple[int, int] = (0, 0),  # (left, top) from calibrate()
    dry_run: bool = False,                     # True = decide but never click
) -> bool:
    if action_idx == WAIT_ACTION:
        return False

    card_placement = ACTION_INDEX_MAP.get(action_idx)
    if card_placement is None:
        return False

    card, placement_idx = card_placement

    slot_idx = next((i for i, c in enumerate(hand[:4]) if c == card), None)
    if slot_idx is None:
        return False

    play_card_at(slot_idx, CARD_PLACEMENTS[card][placement_idx],
                 corner, capture_offset, dry_run)
    return True


# ── Utility ───────────────────────────────────────────────────────────────────

def action_to_str(action_idx: int) -> str:
    """Human-readable label for logging / dashboard."""
    if action_idx == WAIT_ACTION:
        return "WAIT"
    info = ACTION_INDEX_MAP.get(action_idx)
    if info is None:
        return f"UNKNOWN({action_idx})"
    card, pi = info
    x, y = CARD_PLACEMENTS[card][pi]
    return f"{card}@({x},{y})"
