#!/usr/bin/env python3
"""prepare_combined.py — Merge NABirds + external datasets for combined training.

Loads NABirds splits and appends Birdsnap and iNaturalist training data.
Validation and test sets stay NABirds-only for fair comparison.

Supports two modes via --species:
  base_combined    — 404 base_species classes + all external data
  subset98_combined — 98-species subset + external data filtered to those 98

Prerequisites:
    python dataprep/prepare.py --species base_species   # build NABirds base_species splits
    python dataprep/prepare_birdsnap.py         # download and prepare Birdsnap
    python dataprep/prepare_inat.py             # download and prepare iNaturalist

Usage:
    python dataprep/prepare_combined.py                            # base_combined (404 classes)
    python dataprep/prepare_combined.py --species subset98_combined  # subset98 + external
    python dataprep/prepare_combined.py --force                    # rebuild even if cache exists
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")

EXTERNAL_DIR = ARTIFACTS_DIR / "external"
LABELS_DIR = ARTIFACTS_DIR / "labels"
SPLITS_DIR = ARTIFACTS_DIR / "splits"

BIRDSNAP_PKL = EXTERNAL_DIR / "birdsnap_splits.pkl"
INAT_PKL = EXTERNAL_DIR / "inat_splits.pkl"
BASE_SPECIES_CSV = LABELS_DIR / "label_names_nabirds_base_species.csv"
SUBSET98_CSV = LABELS_DIR / "label_names.csv"

SEED = 42

SPECIES_CONFIGS = {
    "base_combined": {
        "nabirds_pkl": SPLITS_DIR / "base_species.pkl",
        "combined_pkl": SPLITS_DIR / "base_combined.pkl",
        "label_csv": BASE_SPECIES_CSV,
    },
    "subset98_combined": {
        "nabirds_pkl": SPLITS_DIR / "subset98.pkl",
        "combined_pkl": SPLITS_DIR / "subset98_combined.pkl",
        "label_csv": SUBSET98_CSV,
    },
}


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


def build_target_remap(
    base_labels: list[str], subset_labels: list[str]
) -> dict[int, int]:
    """Map base_species target indices to subset target indices.

    External datasets use base_species targets (0-403). This maps those to
    subset98 targets (0-97) for species that appear in both label sets.
    """
    base_canon = {canonicalize_name(n): i for i, n in enumerate(base_labels)}
    remap = {}
    for sub_idx, name in enumerate(subset_labels):
        canon = canonicalize_name(name)
        if canon in base_canon:
            remap[base_canon[canon]] = sub_idx
    return remap


def load_external_dataset(
    pkl_path: Path, name: str, label_names: list[str],
    target_remap: dict[int, int] | None = None,
) -> tuple[str, pd.DataFrame] | None:
    """Load an external dataset pickle and optionally remap targets."""
    required_cols = {"image_path", "x", "y", "w", "h", "target"}

    if not pkl_path.exists():
        print(f"  WARNING: {pkl_path} not found, skipping {name}.")
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    train_df = data["train_df"]

    assert required_cols.issubset(set(train_df.columns)), (
        f"{name} DataFrame missing columns: {required_cols - set(train_df.columns)}"
    )

    if target_remap is not None:
        # Filter to only species in the subset, then remap target indices
        mask = train_df["target"].isin(target_remap.keys())
        train_df = train_df[mask].copy()
        train_df["target"] = train_df["target"].map(target_remap)
    else:
        max_target = train_df["target"].max()
        assert max_target < len(label_names), (
            f"{name} max target {max_target} >= num classes {len(label_names)}"
        )

    print(f"  {name} train: {len(train_df):,} images")
    print(f"  Matched species: {data.get('num_matched_species', '?')}")
    return name, train_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge NABirds + external datasets.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    parser.add_argument("--species", choices=list(SPECIES_CONFIGS.keys()),
                        default="base_combined",
                        help="Which combined mode to build (default: base_combined)")
    args = parser.parse_args()

    cfg = SPECIES_CONFIGS[args.species]
    combined_pkl = cfg["combined_pkl"]

    print("=" * 60)
    print(f"COMBINED DATASET — prepare_combined.py ({args.species})")
    print("=" * 60)

    # Check cache
    if combined_pkl.exists() and not args.force:
        print(f"\nCache already exists: {combined_pkl}")
        print("Use --force to rebuild.")
        with open(combined_pkl, "rb") as f:
            data = pickle.load(f)
        print(f"  Train: {len(data['train_df']):,} | Val: {len(data['val_df']):,} | Test: {len(data['test_df']):,}")
        print(f"  Classes: {len(data['label_names'])}")
        return

    # Step 1: Load NABirds splits
    print(f"\n[1/4] Loading NABirds splits ({args.species})...")
    nabirds_pkl = cfg["nabirds_pkl"]
    if not nabirds_pkl.exists():
        print(f"  ERROR: {nabirds_pkl} not found.")
        sys.exit(1)
    with open(nabirds_pkl, "rb") as f:
        nabirds = pickle.load(f)
    nabirds_train = nabirds["train_df"]
    nabirds_val = nabirds["val_df"]
    nabirds_test = nabirds["test_df"]
    label_names = nabirds["label_names"]
    print(f"  NABirds train: {len(nabirds_train):,} | val: {len(nabirds_val):,} | test: {len(nabirds_test):,}")
    print(f"  Classes: {len(label_names)}")

    # Build target remap if using subset mode
    target_remap = None
    if args.species == "subset98_combined":
        base_labels = pd.read_csv(BASE_SPECIES_CSV)["species"].dropna().astype(str).tolist()
        target_remap = build_target_remap(base_labels, label_names)
        print(f"  Mapped {len(target_remap)} base_species targets to subset98 targets")

    # Step 2: Load external datasets
    datasets_loaded = []

    print("\n[2/4] Loading Birdsnap data...")
    result = load_external_dataset(BIRDSNAP_PKL, "Birdsnap", label_names, target_remap)
    if result:
        datasets_loaded.append(result)

    print("\n[2b/4] Loading iNaturalist data...")
    result = load_external_dataset(INAT_PKL, "iNaturalist", label_names, target_remap)
    if result:
        datasets_loaded.append(result)

    if not datasets_loaded:
        print("\nERROR: No external datasets found. Nothing to combine.")
        sys.exit(1)

    # Step 3: Combine training data
    print("\n[3/4] Combining training data...")
    keep_cols = ["image_path", "x", "y", "w", "h", "target"]
    all_train_dfs = [nabirds_train[keep_cols]]
    for name, df in datasets_loaded:
        all_train_dfs.append(df[keep_cols])

    combined_train = pd.concat(all_train_dfs, ignore_index=True)

    # Shuffle combined training data
    rng = np.random.default_rng(SEED)
    combined_train = combined_train.iloc[rng.permutation(len(combined_train))].reset_index(drop=True)

    print(f"  NABirds train: {len(nabirds_train):,}")
    for name, df in datasets_loaded:
        print(f"  {name} train: {len(df):,}")
    print(f"  Combined train: {len(combined_train):,}")
    print(f"  Val (NABirds only): {len(nabirds_val):,}")
    print(f"  Test (NABirds only): {len(nabirds_test):,}")

    # Per-class distribution
    per_class = combined_train.groupby("target").size()
    print(f"  Images/class: min={per_class.min()}, median={int(per_class.median())}, max={per_class.max()}")

    # Step 4: Save combined pickle
    print("\n[4/4] Saving combined pickle...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "train_df": combined_train,
        "val_df": nabirds_val,
        "test_df": nabirds_test,
        "label_names": label_names,
    }
    with open(combined_pkl, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    print("\n" + "=" * 60)
    print("COMBINED DATASET READY")
    print(f"  Mode: {args.species}")
    print(f"  Train: {len(combined_train):,} images ({len(label_names)} classes)")
    print(f"  Val:   {len(nabirds_val):,} (NABirds only)")
    print(f"  Test:  {len(nabirds_test):,} (NABirds only)")
    print(f"  Cache: {combined_pkl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
