#!/usr/bin/env python3
"""prepare_birdsnap.py — Download Birdsnap from HuggingFace and prepare for combined training.

Downloads the Birdsnap dataset (500 North American bird species, ~50K images),
maps species to NABirds base_species targets via canonicalize_name(), saves images
locally, and creates a pickle with the same DataFrame format as prepare.py.

Only images for species matching NABirds base_species are kept (~335 species).
All Birdsnap images go to training only — val/test stay NABirds for fair comparison.

Prerequisites:
    pip install datasets   # one-time, script will auto-install if missing

Usage:
    python dataprep/prepare_birdsnap.py
    python dataprep/prepare_birdsnap.py --force  # re-download even if cache exists
"""
from __future__ import annotations

import argparse
import gc
import importlib
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Allow loading truncated images instead of raising OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
BIRDSNAP_DIR = Path("Birdsnap_Dataset")
BIRDSNAP_IMAGES_DIR = BIRDSNAP_DIR / "images"
BIRDSNAP_IMAGES_TXT = BIRDSNAP_DIR / "images.txt"
EXTERNAL_DIR = ARTIFACTS_DIR / "external"
LABELS_DIR = ARTIFACTS_DIR / "labels"

BIRDSNAP_PKL = EXTERNAL_DIR / "birdsnap_splits.pkl"
BASE_SPECIES_CSV = LABELS_DIR / "label_names_nabirds_base_species.csv"

HF_DATASET_NAME = "sasha/birdsnap"
STREAM_INDEX_RE = re.compile(r"train_(\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def canonicalize_name(name: str) -> str:
    """Normalise a species name for fuzzy matching (same as nabirds_common.py)."""
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower().replace("grey", "gray").replace("orioles", "oriole")
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def ensure_datasets_library():
    """Install HuggingFace datasets library if not present."""
    try:
        importlib.import_module("datasets")
        return
    except ImportError:
        print("  Installing HuggingFace 'datasets' library...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets"],
            stdout=subprocess.DEVNULL,
        )
        print("  Installed successfully.")


def load_base_species_labels() -> list[str]:
    """Load the 404 base species names from the label CSV."""
    if not BASE_SPECIES_CSV.exists():
        raise FileNotFoundError(
            f"Base species label file not found: {BASE_SPECIES_CSV}\n"
            "Run: python dataprep/prepare.py --species base_species --force"
        )
    df = pd.read_csv(BASE_SPECIES_CSV)
    return df["species"].dropna().astype(str).tolist()


def build_species_mapping(
    base_species_labels: list[str],
) -> dict[str, int]:
    """Build canonical name -> NABirds target index mapping.

    Returns dict mapping canonicalized species name to target index.
    """
    canon_to_idx = {}
    for idx, name in enumerate(base_species_labels):
        canon = canonicalize_name(name)
        canon_to_idx[canon] = idx
    return canon_to_idx


def save_image(img: Image.Image, dest: Path) -> None:
    """Save a PIL image to disk as JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    img.save(dest, "JPEG", quality=95)


def load_birdsnap_bbox_metadata() -> tuple[list[dict], dict[str, dict[str, int]]]:
    """Load Birdsnap bbox metadata from the original images.txt table if present."""
    if not BIRDSNAP_IMAGES_TXT.exists():
        print(f"  WARNING: {BIRDSNAP_IMAGES_TXT} not found; falling back to full-image boxes.")
        return [], {}

    df = pd.read_csv(BIRDSNAP_IMAGES_TXT, sep="\t")
    required_cols = {"path", "bb_x1", "bb_y1", "bb_x2", "bb_y2"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            f"  WARNING: {BIRDSNAP_IMAGES_TXT} is missing columns {sorted(missing_cols)}; "
            "falling back to full-image boxes."
        )
        return [], {}

    bbox_cols = ["bb_x1", "bb_y1", "bb_x2", "bb_y2"]
    for col in bbox_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    records = df.to_dict("records")

    lookups: dict[str, dict[str, int]] = {}
    for key in ["path", "url", "md5"]:
        if key in df.columns:
            series = df[key].dropna().astype(str)
            lookups[key] = {value: int(idx) for idx, value in series.items()}

    valid_bbox_count = 0
    for record in records:
        if bbox_from_metadata_record(record) is not None:
            valid_bbox_count += 1

    print(
        f"  Loaded Birdsnap metadata: {len(records):,} rows | "
        f"valid bbox rows: {valid_bbox_count:,}"
    )
    return records, lookups


def lookup_birdsnap_metadata_record(
    example: dict, stream_idx: int, metadata_records: list[dict], metadata_lookups: dict[str, dict[str, int]]
) -> dict | None:
    """Match a streamed HF example to its Birdsnap metadata row."""
    candidate_fields = [
        ("path", "path"),
        ("image_path", "path"),
        ("url", "url"),
        ("image_url", "url"),
        ("md5", "md5"),
        ("image_md5", "md5"),
    ]

    for field_name, lookup_name in candidate_fields:
        lookup = metadata_lookups.get(lookup_name)
        if not lookup:
            continue
        value = example.get(field_name)
        if value is None:
            continue
        idx = lookup.get(str(value))
        if idx is not None:
            return metadata_records[idx]

    if 0 <= stream_idx < len(metadata_records):
        return metadata_records[stream_idx]
    return None


def bbox_from_metadata_record(record: dict | None) -> tuple[int, int, int, int] | None:
    """Convert Birdsnap x1/y1/x2/y2 metadata into x/y/w/h."""
    if not record:
        return None

    coords = [record.get("bb_x1"), record.get("bb_y1"), record.get("bb_x2"), record.get("bb_y2")]
    if any(pd.isna(value) for value in coords):
        return None

    x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h


def bbox_from_saved_birdsnap_path(img_path: Path, metadata_records: list[dict]) -> tuple[int, int, int, int] | None:
    """Recover a bbox from a saved Birdsnap filename like train_013705.jpg."""
    match = STREAM_INDEX_RE.match(img_path.stem)
    if not match:
        return None
    stream_idx = int(match.group(1))
    if not (0 <= stream_idx < len(metadata_records)):
        return None
    return bbox_from_metadata_record(metadata_records[stream_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download and prepare Birdsnap dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    parser.add_argument("--build-from-disk", action="store_true",
                        help="Skip HuggingFace download, build pickle from images already on disk")
    args = parser.parse_args()

    print("=" * 60)
    print("BIRDSNAP — prepare_birdsnap.py")
    print("=" * 60)

    # Check cache
    if BIRDSNAP_PKL.exists() and not args.force:
        print(f"\nCache already exists: {BIRDSNAP_PKL}")
        print("Use --force to re-download.")
        with open(BIRDSNAP_PKL, "rb") as f:
            data = pickle.load(f)
        print(f"  Train images: {len(data['train_df']):,}")
        print(f"  Matched species: {data.get('num_matched_species', '?')}")
        return

    # Step 1: Load NABirds base species labels
    print("\n[1] Loading NABirds base species labels...")
    base_labels = load_base_species_labels()
    print(f"  {len(base_labels)} base species loaded")

    print("\n[1b] Loading Birdsnap bbox metadata...")
    birdsnap_metadata_records, birdsnap_metadata_lookups = load_birdsnap_bbox_metadata()

    # -----------------------------------------------------------------------
    # --build-from-disk: skip HuggingFace, build pickle from existing images
    # -----------------------------------------------------------------------
    if args.build_from_disk:
        print("\n[2] Building pickle from images already on disk...")
        rows = []
        species_with_images: set[int] = set()
        bbox_hits = 0
        bbox_fallbacks = 0

        for class_dir in sorted(BIRDSNAP_IMAGES_DIR.glob("class_*")):
            target = int(class_dir.name.split("_")[1])
            if target >= len(base_labels):
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    bbox = bbox_from_saved_birdsnap_path(img_path, birdsnap_metadata_records)
                    if bbox is None:
                        bbox = (0, 0, w, h)
                        bbox_fallbacks += 1
                    else:
                        bbox_hits += 1
                    rows.append({
                        "image_path": str(img_path),
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                        "target": target,
                    })
                    species_with_images.add(target)
                except Exception:
                    pass  # Skip corrupt images

            if len(rows) % 5000 < 200:
                print(f"    Scanned {len(rows):,} images...")

        print(f"  Found {len(rows):,} images across {len(species_with_images)} species")
        print(f"  Bounding boxes recovered: {bbox_hits:,} | Full-image fallbacks: {bbox_fallbacks:,}")

        train_df = pd.DataFrame(rows)
        rng = np.random.default_rng(42)
        train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)

        matched = [base_labels[t] for t in sorted(species_with_images)]
        data = {
            "train_df": train_df,
            "label_names": base_labels,
            "num_matched_species": len(matched),
            "matched_species": matched,
            "unmatched_species": [],
        }
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(BIRDSNAP_PKL, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        per_class = train_df.groupby("target").size()
        print("\n" + "=" * 60)
        print("BIRDSNAP PREPARATION COMPLETE (from disk)")
        print(f"  Images: {len(train_df):,}")
        print(f"  Matched species: {len(matched)}")
        print(f"  Target classes: {len(base_labels)} (base_species)")
        print(f"  Images/class: min={per_class.min()}, median={int(per_class.median())}, max={per_class.max()}")
        print(f"  Cache: {BIRDSNAP_PKL}")
        print("=" * 60)
        return

    # -----------------------------------------------------------------------
    # Normal path: stream from HuggingFace
    # -----------------------------------------------------------------------

    # Step 2: Install and import datasets library
    print("\n[2] Ensuring HuggingFace datasets library...")
    ensure_datasets_library()
    from datasets import load_dataset

    # Step 3: Build species mapping (canonical name -> target index)
    print(f"\n[3] Building species name mapping...")
    canon_to_idx = build_species_mapping(base_labels)
    print(f"  {len(canon_to_idx)} canonical base species names indexed")

    # Step 4: Stream images and save to disk (no full dataset cache needed)
    print(f"\n[4] Streaming Birdsnap from HuggingFace ({HF_DATASET_NAME})...")
    print("  Using streaming mode to avoid caching the full dataset (~64 GB).")
    BIRDSNAP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    saved = 0
    bbox_hits = 0
    bbox_fallbacks = 0
    matched_species: set[str] = set()
    unmatched_species: set[str] = set()

    for split_name in ["train"]:
        print(f"  Streaming {split_name} split...")
        stream = load_dataset(HF_DATASET_NAME, split=split_name, streaming=True)

        corrupt = 0
        for i, example in enumerate(stream):
            try:
                species_name = example["label"]
                canon = canonicalize_name(species_name)

                if canon not in canon_to_idx:
                    skipped += 1
                    unmatched_species.add(species_name)
                    del example
                    continue

                target = canon_to_idx[canon]
                matched_species.add(species_name)
                img = example["image"]

                img_filename = f"{split_name}_{i:06d}.jpg"
                species_dir = BIRDSNAP_IMAGES_DIR / f"class_{target:04d}"
                img_path = species_dir / img_filename

                if not img_path.exists():
                    save_image(img, img_path)

                w, h = img.size
                bbox = bbox_from_metadata_record(
                    lookup_birdsnap_metadata_record(
                        example, i, birdsnap_metadata_records, birdsnap_metadata_lookups
                    )
                )
                if bbox is None:
                    bbox = (0, 0, w, h)
                    bbox_fallbacks += 1
                else:
                    bbox_hits += 1
                rows.append({
                    "image_path": str(img_path),
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                    "target": target,
                })

                saved += 1
                if saved % 5000 == 0:
                    print(f"    Saved {saved:,} images...")
            except Exception as e:
                corrupt += 1
                if corrupt <= 5:
                    print(f"    Skipping corrupt image {split_name}#{i}: {e}")
            finally:
                # Free PIL image and HF example to prevent memory leak
                if "img" in dir():
                    img.close()
                    del img
                try:
                    del example
                except NameError:
                    pass
                if i % 500 == 0:
                    gc.collect()
        if corrupt:
            print(f"    Skipped {corrupt} corrupt images in {split_name}")

    matched = sorted(matched_species)
    unmatched = sorted(unmatched_species)
    print(f"  Total saved: {saved:,} | Skipped (unmatched species): {skipped:,}")
    print(f"  Matched: {len(matched)} / {len(matched) + len(unmatched)} Birdsnap species")
    print(f"  Bounding boxes recovered: {bbox_hits:,} | Full-image fallbacks: {bbox_fallbacks:,}")
    if unmatched[:5]:
        print(f"    Unmatched examples: {unmatched[:5]}")

    train_df = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)

    # Step 5: Save pickle
    print("\n[5] Saving pickle...")
    data = {
        "train_df": train_df,
        "label_names": base_labels,
        "num_matched_species": len(matched),
        "matched_species": matched,
        "unmatched_species": unmatched,
    }
    with open(BIRDSNAP_PKL, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved to {BIRDSNAP_PKL}")

    # Summary
    print("\n" + "=" * 60)
    print("BIRDSNAP PREPARATION COMPLETE")
    print(f"  Images: {saved:,}")
    print(f"  Matched species: {len(matched)}")
    print(f"  Target classes: {len(base_labels)} (base_species)")
    print(f"  Cache: {BIRDSNAP_PKL}")
    per_class = train_df.groupby("target").size()
    print(f"  Images/class: min={per_class.min()}, median={int(per_class.median())}, max={per_class.max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
