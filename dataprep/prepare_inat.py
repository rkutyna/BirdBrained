#!/usr/bin/env python3
"""prepare_inat.py — Download iNaturalist research-grade photos for NABirds species.

Queries the iNaturalist API for research-grade observations of each NABirds
base_species, downloads up to --max-per-species photos per species, and creates
a pickle with the same DataFrame format as prepare_birdsnap.py.

All iNat images go to training only — val/test stay NABirds for fair comparison.

Prerequisites:
    python dataprep/prepare.py --species base_species   # build NABirds base_species splits

Usage:
    python dataprep/prepare_inat.py                        # full download (~112K images)
    python dataprep/prepare_inat.py --force                # rebuild from scratch
    python dataprep/prepare_inat.py --resume               # resume interrupted download
    python dataprep/prepare_inat.py --max-per-species 100  # fewer images per species
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
INAT_DIR = Path("iNaturalist_Dataset")
INAT_IMAGES_DIR = INAT_DIR / "images"
EXTERNAL_DIR = ARTIFACTS_DIR / "external"
LABELS_DIR = ARTIFACTS_DIR / "labels"

INAT_PKL = EXTERNAL_DIR / "inat_splits.pkl"
INAT_MANIFEST = EXTERNAL_DIR / "inat_manifest.json"
BASE_SPECIES_CSV = LABELS_DIR / "label_names_nabirds_base_species.csv"

# iNaturalist API
INAT_API_BASE = "https://api.inaturalist.org/v1"
DEFAULT_MAX_PER_SPECIES = 280
DOWNLOAD_WORKERS = 10
API_DELAY = 1.0  # seconds between API requests (rate limit)
SEED = 42

# Known name differences: NABirds name -> alternate iNat search terms
# These species have different preferred common names on iNaturalist
ALTERNATE_NAMES: dict[str, list[str]] = {
    "Harris's Hawk": ["Harris's Hawk", "Harris Hawk"],
    "White-winged Scoter": ["American White-winged Scoter", "White-winged Scoter"],
    "Swallow-tailed Kite": ["American Swallow-tailed Kite", "Swallow-tailed Kite"],
    "Green-winged Teal": ["Green-winged Teal", "Common Teal"],
    "Purple Gallinule": ["American Purple Gallinule", "Purple Gallinule"],
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


def load_base_species_labels() -> list[str]:
    """Load the 404 base species names from the label CSV."""
    if not BASE_SPECIES_CSV.exists():
        raise FileNotFoundError(
            f"Base species label file not found: {BASE_SPECIES_CSV}\n"
            "Run: python dataprep/prepare.py --species base_species --force"
        )
    df = pd.read_csv(BASE_SPECIES_CSV)
    return df["species"].dropna().astype(str).tolist()


def api_get(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    """GET with retry and backoff."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:  # Rate limited
                wait = 2 ** (attempt + 2)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"    API error after {retries} retries: {e}")
                return {}
    return {}


def search_taxon(species_name: str, session: requests.Session) -> dict | None:
    """Search iNaturalist for a bird taxon by common name.

    Tries exact canonical match first, then alternate names from ALTERNATE_NAMES,
    then falls back to first Aves result.
    """
    search_names = ALTERNATE_NAMES.get(species_name, [species_name])
    canon_query = canonicalize_name(species_name)

    for search_name in search_names:
        data = api_get(session, f"{INAT_API_BASE}/taxa", {
            "q": search_name,
            "rank": "species",
            "per_page": 10,
            "is_active": "true",
        })
        time.sleep(API_DELAY)

        results = data.get("results", [])
        if not results:
            continue

        # Try exact canonical match on preferred common name
        for taxon in results:
            preferred = taxon.get("preferred_common_name", "")
            if preferred and canonicalize_name(preferred) == canon_query:
                return taxon

        # Try canonical match on the alternate search name itself
        alt_canon = canonicalize_name(search_name)
        for taxon in results:
            preferred = taxon.get("preferred_common_name", "")
            if preferred and canonicalize_name(preferred) == alt_canon:
                return taxon

    # Last resort: search original name and take first bird result
    data = api_get(session, f"{INAT_API_BASE}/taxa", {
        "q": species_name,
        "rank": "species",
        "per_page": 5,
        "is_active": "true",
    })
    time.sleep(API_DELAY)
    for taxon in data.get("results", []):
        if taxon.get("iconic_taxon_name") == "Aves":
            return taxon

    return None


def get_photo_urls(
    taxon_id: int, max_photos: int, session: requests.Session
) -> list[dict]:
    """Get photo URLs for research-grade observations of a taxon.

    Returns list of dicts with 'url' and 'photo_id' keys.
    """
    photos: list[dict] = []
    page = 1
    per_page = 200  # API max

    while len(photos) < max_photos and page <= 50:
        data = api_get(session, f"{INAT_API_BASE}/observations", {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": per_page,
            "page": page,
        })
        time.sleep(API_DELAY)

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                if url:
                    # Replace 'square' thumbnail with 'medium' (500px)
                    url = url.replace("/square.", "/medium.")
                    photos.append({
                        "url": url,
                        "photo_id": photo["id"],
                    })
                    if len(photos) >= max_photos:
                        break
            if len(photos) >= max_photos:
                break

        if len(results) < per_page:
            break
        page += 1

    return photos[:max_photos]


def download_image(
    url: str, dest: Path, session: requests.Session
) -> tuple[bool, int, int]:
    """Download image from URL and save as JPEG. Returns (success, width, height)."""
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")
        w, h = img.size
        if w < 50 or h < 50:  # Skip tiny/corrupt images
            return False, 0, 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=95)
        return True, w, h
    except Exception:
        return False, 0, 0


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_resolve_species(
    base_labels: list[str],
    manifest: dict,
    session: requests.Session,
    max_per_species: int,
    force: bool,
) -> dict:
    """Phase 1+2: Resolve species to taxon IDs and collect photo URLs."""
    total = len(base_labels)
    matched = 0
    unmatched_names = []

    for idx, species_name in enumerate(base_labels):
        key = str(idx)

        # Skip already-resolved species (resume)
        if key in manifest and manifest[key].get("taxon_id") and not force:
            matched += 1
            continue

        # Search for taxon
        taxon = search_taxon(species_name, session)

        if taxon is None:
            print(f"  [{idx+1:3d}/{total}] X {species_name} -- not found")
            manifest[key] = {
                "species_name": species_name,
                "taxon_id": None,
                "inat_name": None,
                "photos": [],
            }
            unmatched_names.append(species_name)
            continue

        taxon_id = taxon["id"]
        inat_name = taxon.get("preferred_common_name", taxon.get("name", "?"))

        # Get photo URLs
        photos = get_photo_urls(taxon_id, max_per_species, session)

        manifest[key] = {
            "species_name": species_name,
            "inat_name": inat_name,
            "taxon_id": taxon_id,
            "photos": photos,
        }
        matched += 1

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  [{idx+1:3d}/{total}] {species_name} -> {inat_name} ({len(photos)} photos)")

        # Save manifest periodically for resume
        if (idx + 1) % 25 == 0:
            EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
            with open(INAT_MANIFEST, "w") as f:
                json.dump(manifest, f)

    # Final manifest save
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(INAT_MANIFEST, "w") as f:
        json.dump(manifest, f)

    total_photos = sum(len(m.get("photos", [])) for m in manifest.values())
    print(f"\n  Matched: {matched} / {total} species")
    print(f"  Total photo URLs: {total_photos:,}")
    if unmatched_names:
        print(f"  Unmatched: {unmatched_names}")

    return manifest


def phase_download_images(manifest: dict) -> int:
    """Phase 3: Download images using parallel workers. Returns count downloaded."""
    INAT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Build download tasks (skip already-downloaded images)
    tasks = []
    skipped = 0
    for idx_str, entry in manifest.items():
        target = int(idx_str)
        for photo in entry.get("photos", []):
            dest = INAT_IMAGES_DIR / f"class_{target:04d}" / f"inat_{photo['photo_id']}.jpg"
            if dest.exists():
                skipped += 1
            else:
                tasks.append((photo["url"], dest))

    print(f"  To download: {len(tasks):,} images ({skipped:,} already on disk)")

    if not tasks:
        return 0

    downloaded = 0
    failed = 0
    dl_session = requests.Session()
    dl_session.headers.update({"User-Agent": "NABirds-Classifier-Research/1.0"})

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(download_image, url, dest, dl_session): dest
            for url, dest in tasks
        }

        for future in as_completed(futures):
            success, _, _ = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1

            total_done = downloaded + failed
            if total_done % 2000 == 0:
                print(f"    Progress: {total_done:,}/{len(tasks):,} "
                      f"(ok: {downloaded:,}, fail: {failed:,})")

    print(f"  New downloads: {downloaded:,} | Failed: {failed:,} | Already on disk: {skipped:,}")
    return downloaded


def phase_yolo_detect(manifest: dict, batch_size: int = 16) -> dict[str, tuple[int, int, int, int]]:
    """Phase 4: Run YOLO bird detection on all downloaded images.

    Returns dict mapping image_path -> (x, y, w, h) bounding box.
    Images with no bird detected are excluded (will be skipped in DataFrame).
    """
    from ultralytics import YOLO

    BIRD_CLASS = 14  # COCO class index for 'bird'
    YOLO_MODEL = "yolo11n.pt"

    model = YOLO(YOLO_MODEL)

    # Use MPS (Apple GPU) if available, else CPU
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"  YOLO device: {device}")

    # Collect all image paths
    all_paths = []
    for idx_str, entry in manifest.items():
        target = int(idx_str)
        species_dir = INAT_IMAGES_DIR / f"class_{target:04d}"
        if not species_dir.exists():
            continue
        all_paths.extend(sorted(species_dir.glob("inat_*.jpg")))

    print(f"  Running YOLO on {len(all_paths):,} images (batch_size={batch_size})...")

    bboxes: dict[str, tuple[int, int, int, int]] = {}
    no_bird = 0

    for batch_start in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[batch_start:batch_start + batch_size]
        results = model(batch_paths, device=device, verbose=False)

        for img_path, result in zip(batch_paths, results):
            boxes = result.boxes
            # Filter to bird detections only
            bird_mask = boxes.cls == BIRD_CLASS
            bird_boxes = boxes[bird_mask]

            if len(bird_boxes) == 0:
                no_bird += 1
                continue

            # Take the highest-confidence bird detection
            best_idx = bird_boxes.conf.argmax()
            x1, y1, x2, y2 = bird_boxes.xyxy[best_idx].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            # Skip tiny detections (likely false positives)
            if w < 20 or h < 20:
                no_bird += 1
                continue

            bboxes[str(img_path)] = (x, y, w, h)

        processed = min(batch_start + batch_size, len(all_paths))
        if processed % 5000 < batch_size or processed == len(all_paths):
            print(f"    Processed {processed:,}/{len(all_paths):,} "
                  f"(detected: {len(bboxes):,}, no bird: {no_bird:,})")

    print(f"  YOLO complete: {len(bboxes):,} bird detections, {no_bird:,} images skipped")
    return bboxes


def phase_build_dataframe(
    manifest: dict, base_labels: list[str], bboxes: dict[str, tuple[int, int, int, int]]
) -> pd.DataFrame:
    """Phase 5: Build DataFrame using YOLO bounding boxes."""
    rows = []

    for idx_str, entry in manifest.items():
        target = int(idx_str)
        species_dir = INAT_IMAGES_DIR / f"class_{target:04d}"
        if not species_dir.exists():
            continue

        for img_path in sorted(species_dir.glob("inat_*.jpg")):
            path_str = str(img_path)
            # Only include images where YOLO detected a bird
            if path_str not in bboxes:
                continue
            x, y, w, h = bboxes[path_str]
            rows.append({
                "image_path": path_str,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "target": target,
            })

    train_df = pd.DataFrame(rows)

    # Shuffle
    rng = np.random.default_rng(SEED)
    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(drop=True)

    return train_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download iNaturalist research-grade photos for NABirds species."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted download (reuse existing manifest)")
    parser.add_argument("--max-per-species", type=int, default=DEFAULT_MAX_PER_SPECIES,
                        help=f"Max photos per species (default: {DEFAULT_MAX_PER_SPECIES})")
    args = parser.parse_args()

    print("=" * 60)
    print("iNATURALIST — prepare_inat.py")
    print("=" * 60)

    # Check cache
    if INAT_PKL.exists() and not args.force and not args.resume:
        print(f"\nCache already exists: {INAT_PKL}")
        print("Use --force to rebuild or --resume to continue interrupted download.")
        with open(INAT_PKL, "rb") as f:
            data = pickle.load(f)
        print(f"  Train images: {len(data['train_df']):,}")
        print(f"  Matched species: {data.get('num_matched_species', '?')}")
        return

    # Step 1: Load NABirds base species labels
    print("\n[1/4] Loading NABirds base species labels...")
    base_labels = load_base_species_labels()
    print(f"  {len(base_labels)} base species loaded")

    session = requests.Session()
    session.headers.update({"User-Agent": "NABirds-Classifier-Research/1.0"})

    # Step 2: Resolve species and fetch photo URLs
    manifest: dict = {}
    if INAT_MANIFEST.exists() and not args.force:
        print(f"\n[2/4] Loading existing manifest from {INAT_MANIFEST}...")
        with open(INAT_MANIFEST) as f:
            manifest = json.load(f)
        resolved = sum(1 for m in manifest.values() if m.get("taxon_id"))
        print(f"  {resolved} species already resolved in manifest")

    needs_api = args.force or len(manifest) < len(base_labels)
    if needs_api:
        print(f"\n[2/4] Resolving species via iNaturalist API...")
        print(f"  Target: {args.max_per_species} photos/species, {len(base_labels)} species")
        print(f"  Rate limit: ~{API_DELAY}s/request — API phase takes ~20-30 min")
        manifest = phase_resolve_species(
            base_labels, manifest, session, args.max_per_species, args.force
        )
    else:
        total_photos = sum(len(m.get("photos", [])) for m in manifest.values())
        matched = sum(1 for m in manifest.values() if m.get("taxon_id"))
        print(f"  Manifest complete: {matched} species, {total_photos:,} photo URLs")

    # Step 3: Download images
    print(f"\n[3/5] Downloading images ({DOWNLOAD_WORKERS} parallel workers)...")
    phase_download_images(manifest)

    # Step 4: YOLO bird detection for bounding boxes
    print("\n[4/5] Running YOLO bird detection...")
    bboxes = phase_yolo_detect(manifest)

    # Step 5: Build DataFrame and save pickle
    print("\n[5/5] Building DataFrame and saving pickle...")
    train_df = phase_build_dataframe(manifest, base_labels, bboxes)

    if train_df.empty:
        print("\nERROR: No images downloaded. Check network connection and try --resume.")
        sys.exit(1)

    matched_species = [
        entry["species_name"]
        for entry in manifest.values()
        if entry.get("taxon_id")
    ]
    unmatched_species = [
        entry["species_name"]
        for entry in manifest.values()
        if not entry.get("taxon_id")
    ]

    data = {
        "train_df": train_df,
        "label_names": base_labels,
        "num_matched_species": len(matched_species),
        "matched_species": matched_species,
        "unmatched_species": unmatched_species,
    }
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(INAT_PKL, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    per_class = train_df.groupby("target").size()
    print("\n" + "=" * 60)
    print("iNATURALIST PREPARATION COMPLETE")
    print(f"  Images: {len(train_df):,}")
    print(f"  Matched species: {len(matched_species)}")
    if unmatched_species:
        print(f"  Unmatched: {unmatched_species}")
    print(f"  Target classes: {len(base_labels)} (base_species)")
    print(f"  Images/class: min={per_class.min()}, median={int(per_class.median())}, max={per_class.max()}")
    print(f"  Cache: {INAT_PKL}")
    print(f"  Manifest: {INAT_MANIFEST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
