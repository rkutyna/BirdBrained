"""Shared constants and utilities for NABirds bird species classification.

Centralises values and helpers used by training_engine.py, prepare.py, and
bird_pipeline.py so they stay in sync.  Has no torch dependency — safe to
import from lightweight scripts.

Note: train.py intentionally inlines its own copies so that the Codex agent
can modify a single self-contained file without touching imports.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Image / model constants
# ---------------------------------------------------------------------------
TARGET_SIZE = 240
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("NABirds Dataset/nabirds")
IMAGES_DIR = DATA_ROOT / "images"
ARTIFACTS_DIR = Path("artifacts")
LABELS_DIR = ARTIFACTS_DIR / "labels"
DEFAULT_LABEL_NAMES_CSV = LABELS_DIR / "label_names.csv"
ALL_SPECIFIC_LABEL_NAMES_CSV = LABELS_DIR / "label_names_nabirds_all_specific.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def canonicalize_name(name: str) -> str:
    """Normalise a species name for fuzzy matching against NABirds classes."""
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower().replace("grey", "gray").replace("orioles", "oriole")
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def crop_resize_pad_bbox(
    img: Image.Image,
    bbox: tuple,
    size: int = TARGET_SIZE,
    pad_rgb: tuple = IMAGENET_PAD_RGB,
) -> Image.Image:
    """Crop to bounding box, resize preserving aspect ratio, pad to square."""
    x, y, w, h = bbox
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(img.width, int(np.ceil(x + w)))
    y2 = min(img.height, int(np.ceil(y + h)))
    cropped = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img
    scale = min(size / cropped.width, size / cropped.height)
    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((new_w, new_h), resample=Image.BILINEAR)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top
    return ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=pad_rgb,
    )
