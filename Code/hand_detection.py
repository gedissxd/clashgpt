import cv2 as cv

from constants import CARD_NAMES
from paths import CARDS_IN_HAND, MINI_CARDS

_METHOD = cv.TM_CCOEFF_NORMED

# Slot x-positions (inside the hand crop) used to decide where an "empty" goes.
_SLOT_X = [16, 122, 227, 332]


def _load_templates():
    """Load all card templates once at import instead of from disk every frame."""
    cards, minis = [], []
    for name in CARD_NAMES:
        path = CARDS_IN_HAND / f"{name}.png"
        img  = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not load card image: {path}")
        else:
            cards.append((name, img))
    for name in CARD_NAMES[:-1]:                      # no mini "empty"
        path = MINI_CARDS / f"mini_{name}.png"
        img  = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not load card image: {path}")
        else:
            minis.append((name, img))
    return cards, minis


_CARD_TEMPLATES, _MINI_TEMPLATES = _load_templates()


def _select_spread_matches(matches, n=4, min_x_dist=50):
    selected = []
    for match in matches:
        _, (x, _), _ = match
        if all(abs(x - sx) >= min_x_dist for _, (sx, _), _ in selected):
            selected.append(match)
        if len(selected) == n:
            break
    return selected


def get_hand(img, corner):
    """Return (hand: list[str] of length 4, next_card: str | None)."""
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _x, _y = corner
    region = img[_y + 820 : _y + 951, _x + 124 : _x + 541]

    best_matches = []
    for name, card in _CARD_TEMPLATES:
        if region.shape[0] < card.shape[0] or region.shape[1] < card.shape[1]:
            return ["empty"] * 4, None
        result = cv.matchTemplate(region, card, _METHOD)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        best_matches.append((max_val, max_loc, name))

    best_matches.sort(key=lambda m: m[0], reverse=True)
    top_four = _select_spread_matches(best_matches)
    top_four.sort(key=lambda m: m[1][0])
    hand = [name for _, _, name in top_four]

    if len(hand) < 4:
        for i in range(len(hand)):
            if abs(top_four[i][1][0] - _SLOT_X[i]) > 5:
                hand.insert(i, "empty")
                break
    # Always hand back exactly 4 slots so slot indices stay valid.
    hand = (hand + ["empty"] * 4)[:4]

    # ── Next card ─────────────────────────────────────────────────────────────
    next_region = img[_y + 930 : _y + 986, _x + 29 : _x + 75]
    best_val, next_card = 0.0, None
    for name, card in _MINI_TEMPLATES:
        if next_region.shape[0] < card.shape[0] or next_region.shape[1] < card.shape[1]:
            return ["empty"] * 4, None
        result = cv.matchTemplate(next_region, card, _METHOD)
        _, max_val, _, _ = cv.minMaxLoc(result)
        if max_val > best_val:
            best_val, next_card = max_val, name

    return hand, next_card
