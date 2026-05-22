"""Shared constants and workspace paths."""

from __future__ import annotations

from pathlib import Path

WORK_DIR = Path("./growz_eval_workspace")
DATA_DIR = WORK_DIR / "data"
GOLDEN_DIR = WORK_DIR / "golden_set"

DEFAULT_INPUT_CSV = GOLDEN_DIR / "golden_set.csv"
DEFAULT_VERIFIED_CSV = GOLDEN_DIR / "golden_set_verified.csv"

SEED = 42

IMG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

CROP_DISTRIBUTION: dict[str, float] = {
    "cotton": 0.27,
    "grape": 0.27,
    "tomato": 0.20,
    "potato": 0.10,
    "wheat": 0.08,
    "other": 0.08,
}

CLASS_BALANCE: dict[str, float] = {
    "healthy": 0.20,
    "common": 0.60,
    "rare": 0.15,
    "ambiguous": 0.05,
}

HEALTHY_KEYWORDS: tuple[str, ...] = ("healthy", "health", "normal", "ok")

COMMON_DISEASES: dict[str, tuple[str, ...]] = {
    "cotton": ("bacterial_blight", "leaf_curl", "fusarium", "verticillium",
               "powdery_mildew", "aphid", "target_spot"),
    "grape": ("esca", "black_rot", "leaf_blight", "downy_mildew",
              "powdery_mildew", "dead_arm"),
    "tomato": ("late_blight", "early_blight", "leaf_mold", "bacterial_spot",
               "septoria", "yellow_leaf_curl"),
    "potato": ("late_blight", "early_blight"),
    "wheat": ("rust", "septoria", "powdery_mildew"),
}

KNOWN_CROPS: tuple[str, ...] = (
    "cotton", "grape", "tomato", "potato", "wheat",
    "maize", "corn", "apple", "pepper", "onion",
)

GEMINI_MODEL = "gemini-3.1-pro-preview"
CLAUDE_MODEL = "claude-opus-4-6"
