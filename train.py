#!/usr/bin/env python3
"""train.py — Self-contained autoresearch training script.

Runs a ResNet-50 bird species classifier on the NABirds dataset (98 target
species) with a fixed wall-clock time budget.  The research agent iterates on
this file; prepare.py and the logging/checkpointing block at the end are
never modified.

Usage:
    python prepare.py          # once, to build the cached splits
    python train.py            # run one experiment (default 30-min budget)
"""
from __future__ import annotations

import csv
import os
import pickle
import random
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ===================================================================
# CONFIGURATION — the agent modifies this section
# ===================================================================

# Wall-clock time budget for the entire run (training only, excludes eval)
TIME_BUDGET_SEC = 1800  # 30 minutes

# Notes: the agent fills this in to describe what changed this run
NOTES = "cap stage2 at 6 to reach stage3"

# --- Model ---
BACKBONE = "resnet50"  # options: resnet50, efficientnet_b0, mobilenet_v3_large
DROPOUT = 0.4

# --- Data ---
BATCH_SIZE = 32
NUM_WORKERS = 2

# --- Training stages ---
# Each stage: (prefixes_to_unfreeze, lr, max_epochs)
# Training will proceed through stages in order until TIME_BUDGET_SEC is hit.
STAGES = [
    {
        "name": "head_only",
        "unfreeze": ("fc.",),
        "lr": 1e-2,
        "max_epochs": 6,
    },
    {
        "name": "layer4+head",
        "unfreeze": ("layer4.", "fc."),
        "lr": 1e-4,
        "max_epochs": 6,
    },
    {
        "name": "layer3+layer4+head",
        "unfreeze": ("layer3.", "layer4.", "fc."),
        "lr": 1e-4,
        "max_epochs": 8,
    },
]

# --- Optimizer ---
OPTIMIZER = "adam"  # options: adam, adamw, sgd
WEIGHT_DECAY = 2e-4
MOMENTUM = 0.9  # only used for SGD

# --- Regularization ---
LABEL_SMOOTHING = 0.05

# --- Augmentation ---
CROP_SCALE_MIN = 0.6
CROP_SCALE_MAX = 1.0
JITTER_BRIGHTNESS = 0.3
JITTER_CONTRAST = 0.3
JITTER_SATURATION = 0.3
JITTER_HUE = 0.1
RANDOM_ERASING_P = 0.3
RANDOM_ERASING_SCALE_MAX = 0.2

# --- Scheduler ---
SCHEDULER = "none"  # options: none, cosine, step
SCHEDULER_T_MAX = 30  # for cosine: total epochs across all stages
SCHEDULER_STEP_SIZE = 5  # for step
SCHEDULER_GAMMA = 0.5  # for step

# --- Mixed precision ---
USE_AMP = True  # float16 on CUDA/MPS — roughly 30% faster per epoch

# ===================================================================
# CONSTANTS — do not modify
# ===================================================================
SEED = 42
TARGET_SIZE = 240
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)

DATA_ROOT = Path("NABirds Dataset/nabirds")
IMAGES_DIR = DATA_ROOT / "images"
ARTIFACTS_DIR = Path("artifacts")
CACHE_PKL = ARTIFACTS_DIR / "autoresearch_splits.pkl"
LABEL_NAMES_CSV = ARTIFACTS_DIR / "label_names.csv"
LOG_CSV = ARTIFACTS_DIR / "autoresearch_log.csv"
BEST_CKPT = ARTIFACTS_DIR / "autoresearch_best.pt"

LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "top1_val_acc",
    "top1_test_acc",
    "peak_memory_mb",
    "total_epochs",
    "training_seconds",
    "time_budget_sec",
    "status",
    "notes",
]


# ===================================================================
# DATASET — inlined from training_engine.py (zero external imports)
# ===================================================================
def _canonicalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower().replace("grey", "gray").replace("orioles", "oriole")
    name = name.replace("-", " ").replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _crop_resize_pad_bbox(
    img: Image.Image,
    bbox: tuple,
    size: int = TARGET_SIZE,
    pad_rgb: tuple = IMAGENET_PAD_RGB,
) -> Image.Image:
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


class NABirdsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = _crop_resize_pad_bbox(
            img,
            (row["x"], row["y"], row["w"], row["h"]),
            size=TARGET_SIZE,
            pad_rgb=IMAGENET_PAD_RGB,
        )
        return self.transform(img), int(row["target"])


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached splits from prepare.py. Returns (train_df, val_df, test_df, label_names)."""
    if not CACHE_PKL.exists():
        raise FileNotFoundError(
            f"Cache not found at {CACHE_PKL}. Run `python prepare.py` first."
        )
    with open(CACHE_PKL, "rb") as f:
        data = pickle.load(f)
    return data["train_df"], data["val_df"], data["test_df"], data["label_names"]


def append_log_row(row: dict[str, object]) -> None:
    """Append one run to the log, migrating older CSVs to the current schema."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
        return

    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_columns = reader.fieldnames or []
        existing_rows = list(reader)

    if existing_columns != LOG_COLUMNS:
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            for existing in existing_rows:
                writer.writerow({k: existing.get(k, "") for k in LOG_COLUMNS})
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})
        return

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


# ===================================================================
# TRANSFORMS
# ===================================================================
def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            TARGET_SIZE,
            scale=(CROP_SCALE_MIN, CROP_SCALE_MAX),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST,
            saturation=JITTER_SATURATION,
            hue=JITTER_HUE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=RANDOM_ERASING_P,
            scale=(0.02, RANDOM_ERASING_SCALE_MAX),
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ===================================================================
# MODEL
# ===================================================================
def build_model(num_classes: int) -> nn.Module:
    if BACKBONE == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, num_classes),
        )
    elif BACKBONE == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, num_classes),
        )
    elif BACKBONE == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=DROPOUT),
            nn.Linear(1280, num_classes),
        )
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE}")
    return model


def build_optimizer(params, lr: float):
    if OPTIMIZER == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")


def build_scheduler(optimizer):
    if SCHEDULER == "none":
        return None
    elif SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=SCHEDULER_T_MAX
        )
    elif SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA
        )
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")


# ===================================================================
# TRAINING
# ===================================================================
def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def non_blocking_transfer(device: torch.device) -> bool:
    """Use non-blocking transfers only on CUDA.

    On this setup, `non_blocking=True` on MPS corrupts labels during training
    and evaluation, which can make accuracy appear near-perfect while the saved
    checkpoint collapses to a single class.
    """
    return device.type == "cuda"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scheduler=None,
    deadline: float = float("inf"),
    scaler=None,
) -> tuple[float, float, bool]:
    """Train for one epoch. Returns (avg_loss, avg_acc, timed_out)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    can_non_block = non_blocking_transfer(device)
    use_amp = USE_AMP and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    for images, targets in loader:
        if time.time() >= deadline:
            return running_loss / max(1, total), correct / max(1, total), True

        images = images.to(device, non_blocking=can_non_block)
        targets = targets.to(device, non_blocking=can_non_block)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * targets.size(0)
        correct += int(preds.eq(targets).sum())
        total += targets.size(0)

    if scheduler is not None:
        scheduler.step()

    return running_loss / max(1, total), correct / max(1, total), False


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, top1_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    can_non_block = non_blocking_transfer(device)
    use_amp = USE_AMP and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    for images, targets in loader:
        images = images.to(device, non_blocking=can_non_block)
        targets = targets.to(device, non_blocking=can_non_block)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * targets.size(0)
        correct += int(preds.eq(targets).sum())
        total += targets.size(0)

    return running_loss / max(1, total), correct / max(1, total)


def get_current_best() -> float:
    """Read autoresearch_log.csv and return the best top1_val_acc so far."""
    if not LOG_CSV.exists():
        return 0.0
    try:
        df = pd.read_csv(LOG_CSV)
        if "top1_val_acc" in df.columns and len(df) > 0:
            return float(df["top1_val_acc"].max())
    except Exception:
        pass
    return 0.0


# ===================================================================
# MAIN
# ===================================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    print(f"{'='*60}")
    print(f"AUTORESEARCH — train.py")
    print(f"Run ID: {run_id}")
    print(f"Notes:  {NOTES}")
    print(f"Budget: {TIME_BUDGET_SEC}s ({TIME_BUDGET_SEC/60:.1f} min)")
    print(f"{'='*60}")

    # --- Setup ---
    set_seeds()
    device = select_device()
    print(f"Device: {device}")

    train_df, val_df, test_df, label_names = load_splits()
    num_classes = len(label_names)
    print(
        f"Classes: {num_classes} | Train: {len(train_df):,} | "
        f"Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    train_transform, eval_transform = build_transforms()
    pin = device.type not in ("mps", "cpu")
    pw = NUM_WORKERS > 0  # persistent workers avoid respawn overhead each epoch
    train_loader = DataLoader(
        NABirdsDataset(train_df, train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw,
    )
    val_loader = DataLoader(
        NABirdsDataset(val_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw,
    )
    test_loader = DataLoader(
        NABirdsDataset(test_df, eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw,
    )

    # --- Build model ---
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    use_amp = USE_AMP and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    print(f"Model: {BACKBONE} | Dropout: {DROPOUT} | Label smoothing: {LABEL_SMOOTHING}")
    print(f"AMP: {'ON' if use_amp else 'OFF'} | Workers: {NUM_WORKERS}")

    # --- Training loop with wall-clock budget ---
    wall_start = time.time()
    deadline = wall_start + TIME_BUDGET_SEC
    best_val_acc = -1.0
    best_state = None
    timed_out = False
    total_epochs = 0

    for stage_idx, stage in enumerate(STAGES):
        if timed_out:
            break

        stage_name = stage["name"]
        prefixes = stage["unfreeze"]
        lr = stage["lr"]
        max_epochs = stage["max_epochs"]

        # Freeze / unfreeze
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in prefixes)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- Stage {stage_idx+1}: {stage_name} ---")
        print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
        print(f"  LR: {lr} | Max epochs: {max_epochs}")

        optimizer = build_optimizer(
            filter(lambda p: p.requires_grad, model.parameters()), lr
        )
        scheduler = build_scheduler(optimizer)

        for epoch in range(1, max_epochs + 1):
            if time.time() >= deadline:
                timed_out = True
                break

            t0 = time.time()
            train_loss, train_acc, epoch_timed_out = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scheduler, deadline, scaler
            )
            elapsed = time.time() - wall_start

            if epoch_timed_out:
                timed_out = True
                print(f"  Epoch {epoch}: TIMED OUT at {elapsed:.0f}s")
                break

            # Quick val check to track best
            val_loss, val_acc = evaluate(model, val_loader, device)
            total_epochs += 1

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            remaining = deadline - time.time()
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"best={best_val_acc:.4f} [{elapsed:.0f}s elapsed, {remaining:.0f}s left]"
            )

    # --- Restore best weights ---
    if best_state is not None:
        model.load_state_dict(best_state)

    # === EVALUATION & LOGGING (do not modify below this line) =========
    wall_train_time = time.time() - wall_start

    # Peak memory
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"Training complete: {total_epochs} epochs in {wall_train_time:.1f}s")

    # Final val evaluation with best weights
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Final val_loss={val_loss:.4f}  top1_val_acc={val_acc:.4f}")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Final test_loss={test_loss:.4f} top1_test_acc={test_acc:.4f}")

    # Check against current best
    prev_best = get_current_best()
    is_new_best = val_acc > prev_best
    status = "keep" if is_new_best else "discard"
    if is_new_best:
        print(f"NEW BEST! {val_acc:.4f} > {prev_best:.4f}")
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"Checkpoint saved: {BEST_CKPT}")
    else:
        print(f"Not a new best ({val_acc:.4f} <= {prev_best:.4f}), checkpoint not saved.")

    # Append to log
    append_log_row({
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top1_val_acc": round(val_acc, 6),
        "top1_test_acc": round(test_acc, 6),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "total_epochs": total_epochs,
        "training_seconds": round(wall_train_time, 1),
        "time_budget_sec": TIME_BUDGET_SEC,
        "status": status,
        "notes": NOTES,
    })
    print(f"Logged to {LOG_CSV}")

    # Structured summary (machine-readable, like Karpathy autoresearch)
    print("---")
    print(f"top1_val_acc:     {val_acc:.6f}")
    print(f"top1_test_acc:    {test_acc:.6f}")
    print(f"training_seconds: {wall_train_time:.1f}")
    print(f"total_epochs:     {total_epochs}")
    print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
    print(f"time_budget_sec:  {TIME_BUDGET_SEC}")
    print(f"status:           {status}")
    print(f"notes:            {NOTES}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
