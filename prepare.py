#!/usr/bin/env python3
"""prepare.py — One-time setup for autoresearch experiments.

Verifies the NABirds dataset exists, builds train/val/test split DataFrames,
confirms label files, and serialises DataLoaders to artifacts/ so that
train.py can load them instantly without re-parsing metadata every run.

Usage:
    python prepare.py          # run once before the first experiment
    python prepare.py --force  # re-build even if cache exists

This file is NEVER modified by the research agent.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from nabirds_common import (
    TARGET_SIZE,
    IMAGENET_PAD_RGB,
    SEED,
    DATA_ROOT,
    IMAGES_DIR,
    ARTIFACTS_DIR,
    canonicalize_name as _canonicalize_name,
    crop_resize_pad_bbox,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABEL_NAMES_CSV = ARTIFACTS_DIR / "label_names.csv"
SPLIT_FILE = DATA_ROOT / "train_test_split_8020_target_species.txt"

CACHE_PKL = ARTIFACTS_DIR / "autoresearch_splits.pkl"

VAL_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Build splits
# ---------------------------------------------------------------------------
def build_splits() -> dict:
    """Parse NABirds metadata and return {train_df, val_df, test_df, label_names}."""
    # --- Load metadata ---
    images = pd.read_csv(DATA_ROOT / "images.txt", sep=" ", names=["image_id", "image_rel_path"])
    labels = pd.read_csv(DATA_ROOT / "image_class_labels.txt", sep=" ", names=["image_id", "class_id"])
    splits = pd.read_csv(SPLIT_FILE, sep=" ", names=["image_id", "is_train"])
    bboxes = pd.read_csv(DATA_ROOT / "bounding_boxes.txt", sep=" ", names=["image_id", "x", "y", "w", "h"])

    class_rows = []
    with open(DATA_ROOT / "classes.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cid, cname = line.split(maxsplit=1)
                class_rows.append((int(cid), cname))
    classes = pd.DataFrame(class_rows, columns=["class_id", "class_name"])

    # --- Load label names (98 target species) ---
    label_df = pd.read_csv(LABEL_NAMES_CSV)
    label_names = label_df["species"].dropna().astype(str).tolist()
    num_classes = len(label_names)
    print(f"  Label names: {num_classes} species from {LABEL_NAMES_CSV}")

    # --- Map NABirds class_ids to our 0..97 targets ---
    valid_class_ids = set(labels["class_id"].unique())
    classes = classes[classes["class_id"].isin(valid_class_ids)].copy()
    classes["canon"] = classes["class_name"].map(_canonicalize_name)
    species_to_idx = {s: i for i, s in enumerate(label_names)}
    class_id_to_idx = {}
    for species in label_names:
        canon = _canonicalize_name(species)
        matched = classes.loc[classes["canon"] == canon, "class_id"].tolist()
        y = species_to_idx[species]
        for cid in matched:
            class_id_to_idx[cid] = y

    # --- Join and filter ---
    df = images.merge(labels, on="image_id").merge(splits, on="image_id").merge(bboxes, on="image_id")
    df = df[df["class_id"].isin(class_id_to_idx)].copy()
    df["target"] = df["class_id"].map(class_id_to_idx)
    df["image_path"] = df["image_rel_path"].map(lambda p: str(IMAGES_DIR / p))
    df["is_train"] = pd.to_numeric(df["is_train"], errors="coerce").fillna(-1).astype(int)

    test_df = df[df["is_train"] == 0].copy().reset_index(drop=True)
    trainval_df = df[df["is_train"] == 1].copy().reset_index(drop=True)

    # --- Train / val split ---
    n = len(trainval_df)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    n_train = n - n_val
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    train_df = trainval_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df = trainval_df.iloc[idx[n_train:]].reset_index(drop=True)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "label_names": label_names,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_dataset() -> bool:
    """Check that the NABirds dataset and required files are present."""
    ok = True
    checks = [
        (DATA_ROOT, "NABirds root"),
        (IMAGES_DIR, "Images directory"),
        (DATA_ROOT / "images.txt", "images.txt"),
        (DATA_ROOT / "image_class_labels.txt", "image_class_labels.txt"),
        (DATA_ROOT / "bounding_boxes.txt", "bounding_boxes.txt"),
        (DATA_ROOT / "classes.txt", "classes.txt"),
        (SPLIT_FILE, "Train/test split file"),
        (LABEL_NAMES_CSV, "Label names CSV"),
    ]
    for path, desc in checks:
        if path.exists():
            print(f"  [OK] {desc}: {path}")
        else:
            print(f"  [MISSING] {desc}: {path}")
            ok = False
    return ok


def verify_sample_images(train_df: pd.DataFrame, n: int = 5) -> bool:
    """Spot-check that a few training images can be loaded and cropped."""
    ok = True
    sample = train_df.sample(n=min(n, len(train_df)), random_state=SEED)
    for _, row in sample.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            crop_resize_pad_bbox(img, (row["x"], row["y"], row["w"], row["h"]))
        except Exception as e:
            print(f"  [FAIL] Could not load {row['image_path']}: {e}")
            ok = False
    if ok:
        print(f"  [OK] Spot-checked {n} sample images — all load and crop correctly")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="One-time setup for autoresearch.")
    parser.add_argument("--force", action="store_true", help="Re-build cache even if it exists")
    args = parser.parse_args()

    print("=" * 60)
    print("AUTORESEARCH — prepare.py")
    print("=" * 60)

    # Step 1: Verify dataset
    print("\n[1/4] Verifying NABirds dataset...")
    if not verify_dataset():
        print("\nERROR: Missing required files. Cannot continue.")
        sys.exit(1)

    # Step 2: Check if cache already exists
    if CACHE_PKL.exists() and not args.force:
        print(f"\n[2/4] Cache already exists: {CACHE_PKL}")
        print("  Use --force to rebuild. Loading to verify...")
        with open(CACHE_PKL, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train_df']):,} | Val: {len(data['val_df']):,} | Test: {len(data['test_df']):,}")
        print(f"  Classes: {len(data['label_names'])}")
    else:
        # Step 2: Build splits
        print("\n[2/4] Building train/val/test splits...")
        data = build_splits()
        print(f"  Train: {len(data['train_df']):,} | Val: {len(data['val_df']):,} | Test: {len(data['test_df']):,}")
        print(f"  Classes: {len(data['label_names'])}")

        # Save cache
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PKL, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved to {CACHE_PKL}")

    # Step 3: Verify sample images
    print("\n[3/4] Spot-checking sample images...")
    verify_sample_images(data["train_df"])

    # Step 4: Confirm label file
    print("\n[4/4] Confirming label file...")
    label_df = pd.read_csv(LABEL_NAMES_CSV)
    species = label_df["species"].dropna().tolist()
    print(f"  [OK] {LABEL_NAMES_CSV}: {len(species)} species")
    print(f"  First 5: {species[:5]}")
    print(f"  Last 5:  {species[-5:]}")

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE — ready for autoresearch experiments")
    print(f"  Cache: {CACHE_PKL}")
    print(f"  Run:   python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
