"""
Evaluate checkpoints that are missing from artifacts/logs/run_summary.csv
and append results in the same format.

Run with:
    python tools/eval_missing_checkpoints.py
    python tools/eval_missing_checkpoints.py --checkpoint artifacts/resnet50/base_combined/best_kaggle.pt
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("NABirds_Dataset/nabirds")
IMAGES_DIR = DATA_ROOT / "images"

LABELS_DIR = ARTIFACTS_DIR / "labels"
LABEL_NAMES_98 = LABELS_DIR / "label_names.csv"
LABEL_NAMES_555 = LABELS_DIR / "label_names_nabirds_all_specific.csv"
LABEL_NAMES_404 = LABELS_DIR / "label_names_nabirds_base_species.csv"

SPLITS_DIR = ARTIFACTS_DIR / "splits"
SUBSET98_PKL = SPLITS_DIR / "subset98.pkl"
FULL555_PKL = SPLITS_DIR / "full555.pkl"
BASE_SPECIES_PKL = SPLITS_DIR / "base_species.pkl"
BASE_COMBINED_PKL = SPLITS_DIR / "base_combined.pkl"
SUBSET98_COMBINED_PKL = SPLITS_DIR / "subset98_combined.pkl"

SPLIT_80_20_TARGET = DATA_ROOT / "train_test_split_8020_target_species.txt"
SPLIT_80_20_ALL = DATA_ROOT / "train_test_split_8020_all_specific.txt"

DEFAULT_BATCH_SIZE = 64
HIGH_RES_BATCH_SIZE = 32
DEFAULT_TARGET_SIZE = 240
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)
USE_TTA = True

SUMMARY_COLUMNS = [
    "run_group_id", "run_label", "run_started_at", "stage", "seed",
    "split_file", "batch_size", "num_epochs", "lr", "weight_decay",
    "label_smoothing", "best_val_acc", "test_loss", "test_acc",
    "test_time_s", "run_time_s", "checkpoint_path", "config_json", "timestamp",
]

MODE_CONFIGS = {
    "subset98": {
        "split_pkl": SUBSET98_PKL,
        "label_names_csv": LABEL_NAMES_98,
    },
    "full555": {
        "split_pkl": FULL555_PKL,
        "label_names_csv": LABEL_NAMES_555,
    },
    "base_species": {
        "split_pkl": BASE_SPECIES_PKL,
        "label_names_csv": LABEL_NAMES_404,
    },
    "base_combined": {
        "split_pkl": BASE_COMBINED_PKL,
        "label_names_csv": LABEL_NAMES_404,
    },
    "subset98_combined": {
        "split_pkl": SUBSET98_COMBINED_PKL,
        "label_names_csv": LABEL_NAMES_98,
    },
}

CHECKPOINT_OVERRIDES = {
    "artifacts/resnet50/base_combined/best_kaggle.pt": {
        "species_mode": "base_combined",
        "target_size": 360,
        "stage": "best",
        "run_label": "manual_eval_360px",
        "use_tta": True,
    },
}

PATH_PREFIX_REMAP = {
    "NABirds Dataset/nabirds": PROJECT_ROOT / "NABirds_Dataset" / "nabirds",
    "NABirds_Dataset/nabirds": PROJECT_ROOT / "NABirds_Dataset" / "nabirds",
    "Birdsnap Dataset": PROJECT_ROOT / "Birdsnap_Dataset",
    "Birdsnap_Dataset": PROJECT_ROOT / "Birdsnap_Dataset",
    "iNaturalist Dataset": PROJECT_ROOT / "iNaturalist_Dataset",
    "iNaturalist_Dataset": PROJECT_ROOT / "iNaturalist_Dataset",
}

# ---------------------------------------------------------------------------
# Dataset helpers (matches training notebook exactly)
# ---------------------------------------------------------------------------

def canonicalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower().replace("grey", "gray").replace("orioles", "oriole")
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def crop_resize_pad(img, bbox, size=DEFAULT_TARGET_SIZE, pad_rgb=(124, 116, 104)):
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
    def __init__(self, df: pd.DataFrame, transform, target_size: int):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = crop_resize_pad(img, (row["x"], row["y"], row["w"], row["h"]),
                              size=self.target_size, pad_rgb=IMAGENET_PAD_RGB)
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


def load_test_bundle_from_split_pkl(split_pkl: Path, label_names_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    required_cols = {"image_path", "x", "y", "w", "h", "target"}

    with open(split_pkl, "rb") as f:
        data = pickle.load(f)

    if "test_df" not in data:
        raise KeyError(f"{split_pkl} is missing test_df")

    test_df = data["test_df"].copy().reset_index(drop=True)
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        raise KeyError(f"{split_pkl} test_df missing columns: {sorted(missing_cols)}")

    test_df["image_path"] = test_df["image_path"].map(resolve_image_path)
    unresolved_mask = ~test_df["image_path"].map(lambda p: Path(p).exists())
    if unresolved_mask.any():
        sample_missing = test_df.loc[unresolved_mask, "image_path"].head(3).tolist()
        raise FileNotFoundError(
            f"{unresolved_mask.sum()} cached image paths could not be resolved. Sample: {sample_missing}"
        )

    artifact_label_names = [str(x) for x in data.get("label_names", [])]
    csv_label_names = pd.read_csv(label_names_csv)["species"].dropna().astype(str).tolist()

    if artifact_label_names and len(artifact_label_names) != len(csv_label_names):
        print(
            "  WARNING: label count mismatch between "
            f"{split_pkl.name} ({len(artifact_label_names)}) and {label_names_csv.name} "
            f"({len(csv_label_names)}). Using labels from the split artifact."
        )
        label_names = artifact_label_names
    else:
        label_names = csv_label_names or artifact_label_names

    return test_df, label_names


@lru_cache(maxsize=200000)
def resolve_image_path(raw_path: str | Path) -> str:
    raw = str(raw_path)
    p = Path(raw)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(PROJECT_ROOT / p)
        candidates.append(Path(raw))

    norm = raw.replace("\\", "/").lstrip("./")
    for prefix, target_root in PATH_PREFIX_REMAP.items():
        if norm == prefix or norm.startswith(prefix + "/"):
            suffix = norm[len(prefix):].lstrip("/")
            candidates.append(target_root / suffix)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return raw


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _unwrap_checkpoint_state(state: Any) -> Any:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                return nested
    return state


def _checkpoint_num_classes_from_state(state: Any) -> int | None:
    state_dict = _unwrap_checkpoint_state(state)
    if not isinstance(state_dict, dict):
        return None

    for key in ("fc.weight", "fc.1.weight", "module.fc.weight", "module.fc.1.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return int(weight.shape[0])
    return None


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


def _build_fc_head(state_dict: dict[str, Any], in_features: int, num_classes: int) -> nn.Module:
    if "fc.weight" in state_dict:
        return nn.Linear(in_features, num_classes)
    return nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if any(key.startswith("module.") for key in state_dict):
        return {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_unwrap_checkpoint_state(state))
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state_dict)}")

    expected_num_classes = _checkpoint_num_classes_from_state(state_dict)
    if expected_num_classes is not None and expected_num_classes != num_classes:
        raise RuntimeError(
            f"Checkpoint expects {expected_num_classes} classes, but evaluation dataset has {num_classes}."
        )

    if "avgpool.p" in state_dict:
        model.avgpool = GeM(p=float(state_dict["avgpool.p"].item()))
    model.fc = _build_fc_head(state_dict, model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    *,
    target_size: int,
    batch_size: int,
    use_tta: bool,
) -> dict:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = TestDataset(test_df, transform, target_size=target_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total = correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    use_amp = device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    with torch.no_grad():
        for i, (images, targets) in enumerate(dl):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                if use_tta:
                    logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
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
    """Infer split source, label CSV, stage, and eval settings for a checkpoint."""
    name = ckpt_path.name
    norm_path = ckpt_path.as_posix()
    override = CHECKPOINT_OVERRIDES.get(norm_path) or CHECKPOINT_OVERRIDES.get(str(ckpt_path))

    species_mode = None
    path_text = norm_path.lower()
    for candidate in ("subset98_combined", "base_combined", "base_species", "full555", "subset98"):
        if candidate in path_text:
            species_mode = candidate
            break

    if override and override.get("species_mode"):
        species_mode = str(override["species_mode"])

    if species_mode in MODE_CONFIGS:
        stage = infer_stage_name(name)
        if override and override.get("stage"):
            stage = str(override["stage"])
        target_size = int(override["target_size"]) if override and override.get("target_size") else DEFAULT_TARGET_SIZE
        eval_batch_size = (
            int(override["eval_batch_size"])
            if override and override.get("eval_batch_size")
            else (HIGH_RES_BATCH_SIZE if target_size > DEFAULT_TARGET_SIZE else DEFAULT_BATCH_SIZE)
        )
        return {
            "split_source": "pickle",
            "split_pkl": MODE_CONFIGS[species_mode]["split_pkl"],
            "label_names_csv": MODE_CONFIGS[species_mode]["label_names_csv"],
            "stage": stage,
            "species_mode": species_mode,
            "target_size": target_size,
            "eval_batch_size": eval_batch_size,
            "run_label": str(override["run_label"]) if override and override.get("run_label") else "manual_eval",
            "use_tta": bool(override["use_tta"]) if override and "use_tta" in override else USE_TTA,
        }

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

    stage = infer_stage_name(name)

    return {
        "split_source": "legacy",
        "split_file": split_file,
        "label_names_csv": label_names_csv,
        "stage": stage,
        "is_all_specific": is_all_specific,
        "target_size": int(override["target_size"]) if override and override.get("target_size") else DEFAULT_TARGET_SIZE,
        "eval_batch_size": (
            int(override["eval_batch_size"])
            if override and override.get("eval_batch_size")
            else DEFAULT_BATCH_SIZE
        ),
        "run_label": str(override["run_label"]) if override and override.get("run_label") else "manual_eval",
        "use_tta": bool(override["use_tta"]) if override and "use_tta" in override else USE_TTA,
    }


def infer_stage_name(name: str) -> str:
    lower_name = name.lower()
    if "best" in lower_name:
        return "best"
    if "layer3_layer4" in lower_name:
        return "stage3"
    if "layer4_finetuned" in lower_name:
        return "stage2"
    return "stage1"


def resolve_requested_checkpoints(requested: list[str], all_ckpts: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()

    for item in requested:
        path = Path(item)
        matches: list[Path] = []
        if path.exists():
            matches.append(path)
        else:
            matches.extend(c for c in all_ckpts if str(c) == item or c.name == item or item in str(c))

        if not matches:
            print(f"WARNING: no checkpoint matched {item!r}")
            continue

        for match in matches:
            key = str(match)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(match)

    return sorted(resolved)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate missing checkpoints and append run_summary rows.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Specific checkpoint path, basename, or substring to evaluate. Repeatable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate requested checkpoints even if they already exist in run_summary.csv.",
    )
    args = parser.parse_args()

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

    all_ckpts = sorted((ARTIFACTS_DIR / "resnet50").rglob("*.pt"))

    if args.checkpoint:
        selected = resolve_requested_checkpoints(args.checkpoint, all_ckpts)
        if not selected:
            print("No checkpoints selected. Nothing to do.")
            return
        to_eval = selected if args.force else [
            c for c in selected if str(c) not in logged_paths and c.name not in logged_paths
        ]
    else:
        to_eval = [c for c in all_ckpts if str(c) not in logged_paths and c.name not in logged_paths]

    if not to_eval:
        print("All checkpoints are already in run_summary.csv. Nothing to do.")
        return

    print(f"Found {len(to_eval)} checkpoint(s) to evaluate:")
    for c in to_eval:
        print(f"  {c.name}")

    new_rows: list[dict] = []

    for ckpt_path in to_eval:
        print(f"\nEvaluating: {ckpt_path.name}")
        meta = infer_checkpoint_meta(ckpt_path)
        if meta is None:
            continue

        label_names_csv: Path = meta["label_names_csv"]
        target_size = int(meta["target_size"])
        eval_batch_size = int(meta["eval_batch_size"])
        use_tta = bool(meta["use_tta"])

        if not label_names_csv.exists():
            print(f"  SKIP: label names CSV not found: {label_names_csv}")
            continue

        if meta["split_source"] == "pickle":
            split_source = Path(meta["split_pkl"])
        else:
            split_source = Path(meta["split_file"])

        if not split_source.exists():
            print(f"  SKIP: split source not found: {split_source}")
            continue

        print(
            f"  Split: {split_source.name} | Labels: {label_names_csv.name} | Stage: {meta['stage']} "
            f"| Size: {target_size}px | Batch: {eval_batch_size} | TTA: {'ON' if use_tta else 'OFF'}"
        )

        try:
            if meta["split_source"] == "pickle":
                test_df, label_names = load_test_bundle_from_split_pkl(split_source, label_names_csv)
            else:
                test_df = build_test_df(split_source, label_names_csv)
                label_names = pd.read_csv(label_names_csv)["species"].dropna().astype(str).tolist()
            print(f"  Test samples: {len(test_df)}")
        except Exception as e:
            print(f"  SKIP: failed to build test dataset: {e}")
            continue

        num_classes = len(label_names)
        print(f"  Classes: {num_classes}")

        try:
            model = load_model(ckpt_path, num_classes, device)
        except Exception as e:
            print(f"  SKIP: failed to load model: {e}")
            continue

        try:
            results = evaluate(
                model,
                test_df,
                device,
                target_size=target_size,
                batch_size=eval_batch_size,
                use_tta=use_tta,
            )
        except Exception as e:
            print(f"  SKIP: evaluation failed: {e}")
            continue

        print(f"  test_acc={results['test_acc']:.4f}  test_loss={results['test_loss']:.4f}  time={results['test_time_s']:.1f}s")

        row = {
            "run_group_id": "manual_eval",
            "run_label": meta["run_label"],
            "run_started_at": "",
            "stage": meta["stage"],
            "seed": "",
            "split_file": split_source.name,
            "batch_size": eval_batch_size,
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
            "config_json": json.dumps(
                {
                    "manual_eval": True,
                    "n_test_samples": results["n_samples"],
                    "target_size": target_size,
                    "species_mode": meta.get("species_mode"),
                    "use_tta": use_tta,
                }
            ),
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
