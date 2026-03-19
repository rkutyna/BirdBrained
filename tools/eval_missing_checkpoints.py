"""
Evaluate checkpoints that are missing from artifacts/logs/run_summary.csv
and append results in the same format.

Run with:  python eval_missing_checkpoints.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
SUMMARY_CSV = Path("artifacts/logs/run_summary.csv")
DATA_ROOT = Path("NABirds Dataset/nabirds")
IMAGES_DIR = DATA_ROOT / "images"

LABEL_NAMES_98 = ARTIFACTS_DIR / "label_names.csv"
LABEL_NAMES_555 = ARTIFACTS_DIR / "label_names_nabirds_all_specific.csv"

SPLIT_80_20_TARGET = DATA_ROOT / "train_test_split_8020_target_species.txt"
SPLIT_80_20_ALL = DATA_ROOT / "train_test_split_8020_all_specific.txt"

BATCH_SIZE = 64
TARGET_SIZE = 240
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)

SUMMARY_COLUMNS = [
    "run_group_id", "run_label", "run_started_at", "stage", "seed",
    "split_file", "batch_size", "num_epochs", "lr", "weight_decay",
    "label_smoothing", "best_val_acc", "test_loss", "test_acc",
    "test_time_s", "run_time_s", "checkpoint_path", "config_json", "timestamp",
]

# ---------------------------------------------------------------------------
# Dataset helpers (matches training notebook exactly)
# ---------------------------------------------------------------------------

def canonicalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower().replace("grey", "gray").replace("orioles", "oriole")
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def crop_resize_pad(img, bbox, size=240, pad_rgb=(124, 116, 104)):
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
    return ImageOps.expand(resized, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_rgb)


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = crop_resize_pad(img, (row["x"], row["y"], row["w"], row["h"]),
                              size=TARGET_SIZE, pad_rgb=IMAGENET_PAD_RGB)
        return self.transform(img), int(row["target"])


def build_test_df(split_file: Path, label_names_csv: Path) -> pd.DataFrame:
    """Build test split dataframe matching the training notebook logic."""
    images = pd.read_csv(DATA_ROOT / "images.txt", sep=" ", names=["image_id", "image_rel_path"])
    labels = pd.read_csv(DATA_ROOT / "image_class_labels.txt", sep=" ", names=["image_id", "class_id"])
    splits = pd.read_csv(split_file, sep=" ", names=["image_id", "is_train"])
    bboxes = pd.read_csv(DATA_ROOT / "bounding_boxes.txt", sep=" ", names=["image_id", "x", "y", "w", "h"])

    class_rows = []
    with open(DATA_ROOT / "classes.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cid, cname = line.split(maxsplit=1)
                class_rows.append((int(cid), cname))
    classes = pd.DataFrame(class_rows, columns=["class_id", "class_name"])

    label_df = pd.read_csv(label_names_csv)
    label_names = label_df["species"].dropna().astype(str).tolist()
    valid_class_ids = set(labels["class_id"].unique())
    classes = classes[classes["class_id"].isin(valid_class_ids)].copy()

    is_all_specific = "all_specific" in label_names_csv.name

    if is_all_specific:
        classes = classes.sort_values("class_id").reset_index(drop=True)
        class_id_to_idx = {int(cid): idx for idx, cid in enumerate(classes["class_id"])}
    else:
        TARGET_SPECIES = label_names
        classes["canon"] = classes["class_name"].map(canonicalize_name)
        species_to_idx = {s: i for i, s in enumerate(TARGET_SPECIES)}
        class_id_to_idx = {}
        for species in TARGET_SPECIES:
            canon = canonicalize_name(species)
            matched = classes.loc[classes["canon"] == canon, "class_id"].tolist()
            y = species_to_idx[species]
            for cid in matched:
                class_id_to_idx[cid] = y

    df = images.merge(labels, on="image_id").merge(splits, on="image_id").merge(bboxes, on="image_id")
    df = df[df["class_id"].isin(class_id_to_idx)].copy()
    df["target"] = df["class_id"].map(class_id_to_idx)
    df["image_path"] = df["image_rel_path"].map(lambda p: str(IMAGES_DIR / p))
    df["is_train"] = pd.to_numeric(df["is_train"], errors="coerce").fillna(-1).astype(int)
    return df[df["is_train"] == 0].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, test_df: pd.DataFrame, device: torch.device) -> dict:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = TestDataset(test_df, transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total = correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, (images, targets) in enumerate(dl):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            correct += int(preds.eq(targets).sum())
            total += targets.size(0)
            running_loss += loss.item() * targets.size(0)
            print(f"  batch {i+1}/{len(dl)} | running acc={correct/total:.4f}", end="\r")

    elapsed = time.perf_counter() - t0
    print()
    return {
        "test_loss": running_loss / max(1, total),
        "test_acc": correct / max(1, total),
        "test_time_s": elapsed,
        "n_samples": total,
    }


# ---------------------------------------------------------------------------
# Checkpoint metadata inference from filename
# ---------------------------------------------------------------------------

def infer_checkpoint_meta(ckpt_path: Path) -> dict | None:
    """Infer split file, label CSV, and stage from the checkpoint filename."""
    name = ckpt_path.name
    is_all_specific = "all_specific" in name

    if "tt_50-50" in name:
        print(f"  SKIP: no 50-50 split file available for {name}")
        return None

    if "tt_80-20" in name or not re.search(r"tt_\d+-\d+", name):
        split_file = SPLIT_80_20_ALL if is_all_specific else SPLIT_80_20_TARGET
    else:
        print(f"  SKIP: unrecognised split tag in {name}")
        return None

    label_names_csv = LABEL_NAMES_555 if is_all_specific else LABEL_NAMES_98

    if "layer3_layer4" in name:
        stage = "stage3"
    elif "layer4_finetuned" in name:
        stage = "stage2"
    else:
        stage = "stage1"

    return {
        "split_file": split_file,
        "label_names_csv": label_names_csv,
        "stage": stage,
        "is_all_specific": is_all_specific,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load existing summary to find already-evaluated checkpoint paths.
    if not SUMMARY_CSV.exists():
        print(f"ERROR: {SUMMARY_CSV} not found.")
        sys.exit(1)

    summary_df = pd.read_csv(SUMMARY_CSV)
    logged_paths: set[str] = set()
    for p in summary_df["checkpoint_path"].dropna():
        logged_paths.add(str(p))
        logged_paths.add(Path(p).name)

    # Find all .pt files in artifacts/
    all_ckpts = sorted(ARTIFACTS_DIR.glob("*.pt"))
    missing = [c for c in all_ckpts if str(c) not in logged_paths and c.name not in logged_paths]

    if not missing:
        print("All checkpoints are already in run_summary.csv. Nothing to do.")
        return

    print(f"Found {len(missing)} checkpoint(s) missing from run_summary.csv:")
    for c in missing:
        print(f"  {c.name}")

    new_rows: list[dict] = []

    for ckpt_path in missing:
        print(f"\nEvaluating: {ckpt_path.name}")
        meta = infer_checkpoint_meta(ckpt_path)
        if meta is None:
            continue

        split_file: Path = meta["split_file"]
        label_names_csv: Path = meta["label_names_csv"]

        if not split_file.exists():
            print(f"  SKIP: split file not found: {split_file}")
            continue
        if not label_names_csv.exists():
            print(f"  SKIP: label names CSV not found: {label_names_csv}")
            continue

        print(f"  Split: {split_file.name} | Labels: {label_names_csv.name} | Stage: {meta['stage']}")

        try:
            test_df = build_test_df(split_file, label_names_csv)
            print(f"  Test samples: {len(test_df)}")
        except Exception as e:
            print(f"  SKIP: failed to build test dataset: {e}")
            continue

        label_names = pd.read_csv(label_names_csv)["species"].dropna().tolist()
        num_classes = len(label_names)
        print(f"  Classes: {num_classes}")

        try:
            model = load_model(ckpt_path, num_classes, device)
        except Exception as e:
            print(f"  SKIP: failed to load model: {e}")
            continue

        try:
            results = evaluate(model, test_df, device)
        except Exception as e:
            print(f"  SKIP: evaluation failed: {e}")
            continue

        print(f"  test_acc={results['test_acc']:.4f}  test_loss={results['test_loss']:.4f}  time={results['test_time_s']:.1f}s")

        row = {
            "run_group_id": "manual_eval",
            "run_label": "manual_eval",
            "run_started_at": "",
            "stage": meta["stage"],
            "seed": "",
            "split_file": split_file.name,
            "batch_size": BATCH_SIZE,
            "num_epochs": "",
            "lr": "",
            "weight_decay": "",
            "label_smoothing": "",
            "best_val_acc": "",
            "test_loss": results["test_loss"],
            "test_acc": results["test_acc"],
            "test_time_s": results["test_time_s"],
            "run_time_s": "",
            "checkpoint_path": str(ckpt_path),
            "config_json": json.dumps({"manual_eval": True, "n_test_samples": results["n_samples"]}),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        new_rows.append(row)

    if not new_rows:
        print("\nNo new rows to append (all skipped).")
        return

    # Append to CSV
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        for row in new_rows:
            writer.writerow(row)

    print(f"\nAppended {len(new_rows)} row(s) to {SUMMARY_CSV}")
    for row in new_rows:
        print(f"  {Path(row['checkpoint_path']).name}  test_acc={row['test_acc']:.4f}")


if __name__ == "__main__":
    main()
