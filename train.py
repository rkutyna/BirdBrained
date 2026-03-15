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

import copy
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ===================================================================
# CONFIGURATION — the agent modifies this section
# ===================================================================

# Wall-clock time budget for the entire run (training only, excludes eval)
TIME_BUDGET_SEC = 3600  # set by autorun.py --time-budget

# Notes: the agent fills this in to describe what changed this run
NOTES = "enable EMA"

# --- Model ---
BACKBONE = "resnet50"  # options: resnet50, efficientnet_b0, mobilenet_v3_large
DROPOUT = 0.4

# --- Species mode ---
SPECIES_MODE = "full555"  # set by autorun.py --species

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
        "max_epochs": 1,
    },
    {
        "name": "layer4+head",
        "unfreeze": ("layer4.", "fc."),
        "lr": 1e-4,
        "max_epochs": 4,
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

# --- CutMix / Mixup ---
CUTMIX_ALPHA = 0.0   # 0 = off; try 1.0 for standard CutMix
MIXUP_ALPHA = 0.0    # 0 = off; try 0.2 for standard Mixup
# When both > 0, each batch randomly gets one or the other (50/50).

# --- TrivialAugmentWide ---
USE_TRIVIAL_AUGMENT = False  # replaces ColorJitter with TrivialAugmentWide

# --- Generalized Mean (GeM) pooling ---
USE_GEM_POOLING = True  # replaces avgpool in ResNet-50; learns to focus on discriminative regions
GEM_P_INIT = 3.0         # initial p (learnable parameter; 1=avg, inf=max)

# --- Exponential Moving Average (EMA) ---
USE_EMA = True     # smoothed model weights for evaluation/checkpoint
EMA_DECAY = 0.999  # higher = more smoothing; 0.999 standard for ~10k samples

# --- Gradient clipping ---
GRAD_CLIP_NORM = 0.0  # 0 = off; try 1.0 to stabilize unfreezing stages

# --- Test-time augmentation (TTA) ---
USE_TTA = True  # horizontal flip TTA at final evaluation only (free accuracy)

# --- Learning rate warmup ---
WARMUP_EPOCHS = 0  # 0 = off; try 2. Linear warmup per stage before scheduler

# --- Layer-wise learning rate decay (LLRD) ---
LLRD_DECAY = 0.8  # 0 = off; try 0.8. Earlier ResNet layers get lower LR (ResNet-50 only)

# --- Focal loss ---
USE_FOCAL_LOSS = False  # down-weights easy examples, focuses on confusing species pairs
FOCAL_GAMMA = 2.0       # higher = more focus on hard examples (0 = standard CE)


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

# Paths derived from SPECIES_MODE
if SPECIES_MODE == "full555":
    CACHE_PKL = ARTIFACTS_DIR / "autoresearch_splits_full555.pkl"
    LABEL_NAMES_CSV = ARTIFACTS_DIR / "label_names_nabirds_all_specific.csv"
    LOG_CSV = ARTIFACTS_DIR / "autoresearch_log_full555.csv"
    BEST_CKPT = ARTIFACTS_DIR / "autoresearch_best_full555.pt"
else:
    CACHE_PKL = ARTIFACTS_DIR / "autoresearch_splits.pkl"
    LABEL_NAMES_CSV = ARTIFACTS_DIR / "label_names.csv"
    LOG_CSV = ARTIFACTS_DIR / "autoresearch_log.csv"
    BEST_CKPT = ARTIFACTS_DIR / "autoresearch_best.pt"
PROGRESS_FILE = ARTIFACTS_DIR / "autoresearch_progress.json"

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
    "analysis",
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


def write_progress(
    stage: str, epoch: int, elapsed_sec: float,
    remaining_sec: float, budget_sec: float, best_val_acc: float,
) -> None:
    """Write a progress file that the outer autorun loop can poll."""
    import json as _json
    PROGRESS_FILE.write_text(
        _json.dumps({
            "stage": stage,
            "epoch": epoch,
            "elapsed_sec": round(elapsed_sec),
            "remaining_sec": round(remaining_sec),
            "budget_sec": round(budget_sec),
            "best_val_acc": round(best_val_acc, 6),
        }),
        encoding="utf-8",
    )


# ===================================================================
# TRAINING ANALYSIS — automatic diagnostics after each experiment
# ===================================================================
def analyze_training(
    epoch_history: list[dict],
    timed_out: bool,
    best_val_acc: float,
    prev_best: float,
) -> str:
    """Analyze epoch history and return diagnostic string.

    Detects: overfitting, underfitting, convergence issues, time pressure,
    val_acc plateau, and regression vs previous best.
    """
    findings = []

    if not epoch_history:
        return "no_epochs_completed"

    last = epoch_history[-1]
    first = epoch_history[0]
    train_accs = [e["train_acc"] for e in epoch_history]
    val_accs = [e["val_acc"] for e in epoch_history]
    best_epoch = max(range(len(val_accs)), key=lambda i: val_accs[i])

    # --- Overfitting: large train/val gap ---
    gap = last["train_acc"] - last["val_acc"]
    if gap > 0.10:
        findings.append(f"OVERFITTING(gap={gap:.3f})")
    elif gap > 0.05:
        findings.append(f"mild_overfit(gap={gap:.3f})")

    # --- Underfitting: both accuracies low ---
    if last["val_acc"] < 0.5 and last["train_acc"] < 0.6:
        findings.append("UNDERFITTING")

    # --- Val accuracy still improving at end (could use more epochs) ---
    if len(val_accs) >= 3:
        recent = val_accs[-3:]
        if recent[-1] >= max(recent[:-1]):
            findings.append("val_still_improving")

    # --- Val accuracy peaked early then declined ---
    if best_epoch < len(val_accs) - 1:
        decline = val_accs[best_epoch] - last["val_acc"]
        epochs_since = len(val_accs) - 1 - best_epoch
        if decline > 0.01 and epochs_since >= 2:
            findings.append(f"val_peaked_epoch_{best_epoch+1}(decline={decline:.3f})")

    # --- Plateau: val accuracy barely changed over last N epochs ---
    if len(val_accs) >= 4:
        recent_range = max(val_accs[-4:]) - min(val_accs[-4:])
        if recent_range < 0.005:
            findings.append(f"plateau(range={recent_range:.4f})")

    # --- Time pressure ---
    if timed_out:
        findings.append("hit_time_budget")

    # --- Regression vs previous best ---
    if prev_best > 0:
        delta = best_val_acc - prev_best
        if delta < -0.01:
            findings.append(f"REGRESSED(delta={delta:+.4f})")
        elif delta > 0.005:
            findings.append(f"improved(delta={delta:+.4f})")
        else:
            findings.append(f"flat(delta={delta:+.4f})")

    # --- Summary stats ---
    findings.append(
        f"train_acc={last['train_acc']:.4f},val_acc={last['val_acc']:.4f},"
        f"best_at_epoch_{best_epoch+1}/{len(val_accs)}"
    )

    return "; ".join(findings)


# ===================================================================
# ADVANCED COMPONENTS — GeM pooling, EMA, Focal loss, CutMix/Mixup
# ===================================================================
class GeM(nn.Module):
    """Generalized Mean pooling — learns to focus on discriminative regions.

    p=1 is average pooling, p->inf is max pooling. Initialized at p=3, the
    network learns the optimal trade-off. Consistently +1-2% on fine-grained
    recognition vs standard average pooling.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


class ModelEMA:
    """Exponential Moving Average of model weights for smoother evaluation.

    Maintains a shadow copy whose weights are a running average of the training
    model. Use ema.ema_model for evaluation — it generalizes better than any
    single training snapshot, especially on small datasets.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


class FocalLoss(nn.Module):
    """Focal loss — down-weights well-classified examples.

    For fine-grained classification, many species are easy to tell apart while
    a few confusing pairs dominate errors. Focal loss spends more capacity on
    those hard pairs. gamma=0 reduces to standard cross-entropy.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def cutmix_data(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix: paste a random patch from a shuffled image onto each image.

    Returns (mixed_images, targets_a, targets_b, lam) where lam is the
    proportion of the original image remaining.
    """
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    H, W = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_ratio))
    cut_h = max(1, int(H * cut_ratio))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return images, targets, targets[index], lam


def mixup_data(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup: blend two images and their labels proportionally."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, targets, targets[index], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Weighted loss for CutMix/Mixup — blends loss for both target sets."""
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)


# ===================================================================
# TRANSFORMS
# ===================================================================
def build_transforms():
    if USE_TRIVIAL_AUGMENT:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                TARGET_SIZE, scale=(CROP_SCALE_MIN, CROP_SCALE_MAX),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=RANDOM_ERASING_P, scale=(0.02, RANDOM_ERASING_SCALE_MAX),
            ),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                TARGET_SIZE, scale=(CROP_SCALE_MIN, CROP_SCALE_MAX),
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
                p=RANDOM_ERASING_P, scale=(0.02, RANDOM_ERASING_SCALE_MAX),
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
        if USE_GEM_POOLING:
            model.avgpool = GeM(p=GEM_P_INIT)
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


def build_optimizer_llrd(model: nn.Module, lr: float, decay_factor: float):
    """Build optimizer with layer-wise learning rate decay for ResNet-50.

    Earlier layers get exponentially lower LR. With decay=0.8 and 6 layer
    groups, the stem gets ~0.33x the head LR. This preserves transferable
    low-level features while letting later layers specialize.
    """
    layer_groups = [
        ("conv1.", "bn1."),
        ("layer1.",),
        ("layer2.",),
        ("layer3.",),
        ("layer4.",),
        ("fc.",),
    ]
    num_groups = len(layer_groups)
    param_groups = []
    for idx, prefixes in enumerate(layer_groups):
        group_lr = lr * (decay_factor ** (num_groups - 1 - idx))
        params = [
            p for n, p in model.named_parameters()
            if any(n.startswith(pf) for pf in prefixes) and p.requires_grad
        ]
        if params:
            param_groups.append({"params": params, "lr": group_lr})
    # Include GeM p parameter if present
    if USE_GEM_POOLING and hasattr(model, "avgpool") and isinstance(model.avgpool, GeM):
        gem_params = list(model.avgpool.parameters())
        if gem_params:
            param_groups.append({"params": gem_params, "lr": lr})
    if not param_groups:
        param_groups = [{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": lr}]
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adam":
        return torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        return torch.optim.SGD(param_groups, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")


def build_scheduler(optimizer, max_epochs: int | None = None):
    if SCHEDULER == "none":
        if WARMUP_EPOCHS > 0:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS,
            )
        return None
    if SCHEDULER == "cosine":
        effective_epochs = max(1, (max_epochs or SCHEDULER_T_MAX) - WARMUP_EPOCHS)
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=effective_epochs, eta_min=1e-6,
        )
    elif SCHEDULER == "step":
        main_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA,
        )
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")
    if WARMUP_EPOCHS > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main_sched], milestones=[WARMUP_EPOCHS],
        )
    return main_sched


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
    ema=None,
) -> tuple[float, float, bool]:
    """Train for one epoch. Returns (avg_loss, avg_acc, timed_out)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    can_non_block = non_blocking_transfer(device)
    use_amp = USE_AMP and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    do_cutmix = CUTMIX_ALPHA > 0
    do_mixup = MIXUP_ALPHA > 0

    for images, targets in loader:
        if time.time() >= deadline:
            return running_loss / max(1, total), correct / max(1, total), True

        images = images.to(device, non_blocking=can_non_block)
        targets = targets.to(device, non_blocking=can_non_block)

        # CutMix / Mixup augmentation (applied per-batch on GPU)
        mixed = False
        if do_cutmix or do_mixup:
            if do_cutmix and do_mixup:
                use_cutmix = random.random() < 0.5
            else:
                use_cutmix = do_cutmix
            if use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(images, targets, CUTMIX_ALPHA)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, targets, MIXUP_ALPHA)
            mixed = True

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            if mixed:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * batch_size
        if mixed:
            correct += (
                lam * preds.eq(targets_a).float().sum().item()
                + (1 - lam) * preds.eq(targets_b).float().sum().item()
            )
        else:
            correct += int(preds.eq(targets).sum())
        total += batch_size

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


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate with horizontal-flip TTA. Returns (avg_loss, top1_accuracy).

    Averages logits from the original image and its horizontal flip.
    Typically +0.5-1% accuracy for free (only costs 2x inference time).
    """
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
            logits_orig = model(images)
            logits_flip = model(torch.flip(images, dims=[3]))

        avg_logits = (logits_orig + logits_flip) / 2.0
        loss = criterion(avg_logits, targets)
        preds = avg_logits.argmax(dim=1)
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
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    use_amp = USE_AMP and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    print(f"Model: {BACKBONE} | Dropout: {DROPOUT} | Label smoothing: {LABEL_SMOOTHING}")
    print(f"AMP: {'ON' if use_amp else 'OFF'} | Workers: {NUM_WORKERS}")
    extras = []
    if USE_GEM_POOLING:
        extras.append(f"GeM(p={GEM_P_INIT})")
    if USE_EMA:
        extras.append(f"EMA({EMA_DECAY})")
    if CUTMIX_ALPHA > 0:
        extras.append(f"CutMix(α={CUTMIX_ALPHA})")
    if MIXUP_ALPHA > 0:
        extras.append(f"Mixup(α={MIXUP_ALPHA})")
    if USE_TRIVIAL_AUGMENT:
        extras.append("TrivialAugment")
    if GRAD_CLIP_NORM > 0:
        extras.append(f"GradClip({GRAD_CLIP_NORM})")
    if USE_TTA:
        extras.append("TTA")
    if WARMUP_EPOCHS > 0:
        extras.append(f"Warmup({WARMUP_EPOCHS}ep)")
    if LLRD_DECAY > 0:
        extras.append(f"LLRD({LLRD_DECAY})")
    if USE_FOCAL_LOSS:
        extras.append(f"Focal(γ={FOCAL_GAMMA})")
    if extras:
        print(f"Extras: {', '.join(extras)}")

    # --- EMA ---
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    # --- Training loop with wall-clock budget ---
    wall_start = time.time()
    deadline = wall_start + TIME_BUDGET_SEC
    best_val_acc = -1.0
    best_state = None
    timed_out = False
    total_epochs = 0
    epoch_history: list[dict] = []

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
        # GeM p parameter should always be trainable when GeM is active
        if USE_GEM_POOLING and hasattr(model, "avgpool") and isinstance(model.avgpool, GeM):
            for p in model.avgpool.parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- Stage {stage_idx+1}: {stage_name} ---")
        print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
        print(f"  LR: {lr} | Max epochs: {max_epochs}")

        if LLRD_DECAY > 0 and BACKBONE == "resnet50":
            optimizer = build_optimizer_llrd(model, lr, LLRD_DECAY)
        else:
            optimizer = build_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()), lr
            )
        scheduler = build_scheduler(optimizer, max_epochs)

        for epoch in range(1, max_epochs + 1):
            if time.time() >= deadline:
                timed_out = True
                break

            t0 = time.time()
            train_loss, train_acc, epoch_timed_out = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scheduler, deadline, scaler, ema,
            )
            elapsed = time.time() - wall_start

            if epoch_timed_out:
                timed_out = True
                print(f"  Epoch {epoch}: TIMED OUT at {elapsed:.0f}s")
                break

            # Quick val check to track best
            val_loss, val_acc = evaluate(model, val_loader, device)
            total_epochs += 1
            epoch_history.append({
                "stage": stage_name, "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if ema is not None:
                    best_state = {k: v.cpu().clone() for k, v in ema.ema_model.state_dict().items()}
                else:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            remaining = deadline - time.time()
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"best={best_val_acc:.4f} [{elapsed:.0f}s elapsed, {remaining:.0f}s left]"
            )
            write_progress(stage_name, epoch, elapsed, remaining, TIME_BUDGET_SEC, best_val_acc)

    # --- Restore best weights ---
    eval_model = model
    if best_state is not None:
        model.load_state_dict(best_state)
        eval_model = model
    if ema is not None and best_state is None:
        eval_model = ema.ema_model

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
    eval_fn = evaluate_with_tta if USE_TTA else evaluate
    val_loss, val_acc = eval_fn(eval_model, val_loader, device)
    print(f"Final val_loss={val_loss:.4f}  top1_val_acc={val_acc:.4f}")
    test_loss, test_acc = eval_fn(eval_model, test_loader, device)
    print(f"Final test_loss={test_loss:.4f} top1_test_acc={test_acc:.4f}")

    # Check against current best
    prev_best = get_current_best()
    is_new_best = val_acc > prev_best
    status = "keep" if is_new_best else "discard"
    if is_new_best:
        print(f"NEW BEST! {val_acc:.4f} > {prev_best:.4f}")
        torch.save(eval_model.state_dict(), BEST_CKPT)
        print(f"Checkpoint saved: {BEST_CKPT}")
    else:
        print(f"Not a new best ({val_acc:.4f} <= {prev_best:.4f}), checkpoint not saved.")

    # Auto-analyze training trajectory
    analysis = analyze_training(epoch_history, timed_out, best_val_acc, prev_best)
    print(f"Analysis: {analysis}")

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
        "analysis": analysis,
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
    print(f"analysis:         {analysis}")
    print(f"{'='*60}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == "__main__":
    main()
