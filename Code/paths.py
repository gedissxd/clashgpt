"""
paths.py — single place that knows where the large assets live.

By default the project root is the folder that contains `Code/` (i.e. the
repository root), so after downloading the Google Drive assets you just drop
`Images/`, `YOLO/` and `Models/` next to `Code/`.

To keep the assets somewhere else, set the CLASHGPT_ROOT environment variable:

    Windows (PowerShell):  $env:CLASHGPT_ROOT = "D:\\ClashGPT"
    Windows (cmd):         set CLASHGPT_ROOT=D:\\ClashGPT
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CLASHGPT_ROOT", Path(__file__).resolve().parent.parent))

IMAGES_DIR      = ROOT / "Images"
CARDS_IN_HAND   = IMAGES_DIR / "cards_in_hand"
MINI_CARDS      = IMAGES_DIR / "mini_cards"
BANNER_TEMPLATE = IMAGES_DIR / "banner.png"

YOLO_DIR     = ROOT / "YOLO"
LBGG_WEIGHTS = YOLO_DIR / "LBGG" / "LBGG_model" / "train" / "weights" / "best.pt"
SARG_WEIGHTS = YOLO_DIR / "SARG" / "SARG_model" / "train" / "weights" / "best.pt"

MODEL_DIR  = ROOT / "Models"
CKPT_DIR   = MODEL_DIR / "checkpoints"
TB_DIR     = MODEL_DIR / "tensorboard"
FINAL_PATH = MODEL_DIR / "cr_agent_final"


def check_assets(*required: Path) -> None:
    """Fail early with a helpful message if downloaded assets are missing."""
    missing = [p for p in required if not p.exists()]
    if missing:
        lines = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing required asset(s):\n  {lines}\n\n"
            f"Download Images/, YOLO/ and Models/ from the Google Drive link in "
            f"README.md and place them in:\n  {ROOT}\n"
            f"or set CLASHGPT_ROOT to the folder that contains them."
        )
