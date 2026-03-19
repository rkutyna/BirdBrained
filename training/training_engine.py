#!/usr/bin/env python3
"""training_engine.py — Bird species classifier training module + job queue manager.

Run as a subprocess to execute one queued training job:
    python training/training_engine.py --job-id <id>

Import queue helpers (no torch required) from other modules:
    from training.training_engine import TrainingConfig, add_to_queue, load_queue, ...
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is importable regardless of how this script is launched.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from nabirds_common import (
    TARGET_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENET_PAD_RGB,
    DATA_ROOT,
    IMAGES_DIR,
    ARTIFACTS_DIR,
    DEFAULT_LABEL_NAMES_CSV,
    ALL_SPECIFIC_LABEL_NAMES_CSV,
    canonicalize_name as _canonicalize_name,
    crop_resize_pad_bbox as _crop_resize_pad_bbox,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS_DIR = Path("artifacts/logs")

SPLIT_80_20_TARGET = DATA_ROOT / "train_test_split_8020_target_species.txt"
SPLIT_80_20_ALL = DATA_ROOT / "train_test_split_8020_all_specific.txt"

SUMMARY_CSV = LOGS_DIR / "run_summary.csv"
QUEUE_JSON = LOGS_DIR / "run_queue.json"

SUMMARY_COLUMNS = [
    "run_group_id", "run_label", "run_started_at", "stage", "seed",
    "split_file", "batch_size", "num_epochs", "lr", "weight_decay",
    "label_smoothing", "best_val_acc", "test_loss", "test_acc",
    "test_time_s", "run_time_s", "checkpoint_path", "config_json", "timestamp",
]


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Identity
    run_label: str = "manual"
    run_group_id: str = ""  # auto-set from job_id if empty

    # Species
    species_mode: str = "98"  # "98" = 98 target species, "555" = all NABirds-specific

    # Data
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    val_fraction: float = 0.10

    # Stage selection — which stages to run, in order
    stages_to_run: list = field(default_factory=lambda: [1, 2, 3])

    # Stage 1: head-only (fc frozen backbone)
    stage1_epochs: int = 20
    stage1_lr: float = 1e-3

    # Stage 2: layer4 + fc unfrozen
    stage2_epochs: int = 10
    stage2_lr: float = 2e-4

    # Stage 3: layer3 + layer4 + fc unfrozen
    stage3_epochs: int = 8
    stage3_lr: float = 1e-4

    # Shared training params
    weight_decay: float = 2e-4
    label_smoothing: float = 0.0

    # Augmentation
    crop_scale_min: float = 0.6
    crop_scale_max: float = 1.0
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    jitter_hue: float = 0.1
    random_erasing_p: float = 0.3
    random_erasing_scale_max: float = 0.2


# ---------------------------------------------------------------------------
# Queue management  (no torch dependency — safe to import anywhere)
# ---------------------------------------------------------------------------
def load_queue() -> list[dict]:
    if not QUEUE_JSON.exists():
        return []
    try:
        with open(QUEUE_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_queue(queue: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_JSON.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)
    tmp.replace(QUEUE_JSON)


def add_to_queue(config: TrainingConfig) -> str:
    queue = load_queue()
    job_id = str(uuid.uuid4())[:8]
    if not config.run_group_id:
        config.run_group_id = job_id
    queue.append({
        "id": job_id,
        "status": "pending",
        "config": asdict(config),
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "completed_at": None,
        "error": None,
    })
    save_queue(queue)
    return job_id


def remove_from_queue(job_id: str) -> bool:
    """Remove a pending job. Returns True if removed."""
    queue = load_queue()
    new_queue = [j for j in queue if not (j["id"] == job_id and j["status"] == "pending")]
    if len(new_queue) == len(queue):
        return False
    save_queue(new_queue)
    return True


def clear_finished_jobs() -> int:
    """Remove done/failed/cancelled jobs. Returns count removed."""
    queue = load_queue()
    new_queue = [j for j in queue if j["status"] not in ("done", "failed", "cancelled")]
    removed = len(queue) - len(new_queue)
    if removed:
        save_queue(new_queue)
    return removed


def cancel_running_job(job_id: str) -> None:
    """Write a cancel sentinel; training loop picks it up after the current epoch."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"cancel_{job_id}.txt").write_text("cancel", encoding="utf-8")


def get_progress(job_id: str) -> dict | None:
    progress_file = LOGS_DIR / f"progress_{job_id}.json"
    if not progress_file.exists():
        return None
    try:
        with open(progress_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_training_log(job_id: str) -> str:
    log_path = LOGS_DIR / f"training_{job_id}.log"
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Internal queue helpers (used by training subprocess)
# ---------------------------------------------------------------------------
def _write_progress(job_id: str, data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    progress_file = LOGS_DIR / f"progress_{job_id}.json"
    tmp = progress_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(progress_file)


def _check_cancel(job_id: str) -> bool:
    sentinel = LOGS_DIR / f"cancel_{job_id}.txt"
    if sentinel.exists():
        try:
            sentinel.unlink()
        except Exception:
            pass
        return True
    return False


def update_queue_status(job_id: str, **kwargs) -> None:
    queue = load_queue()
    for job in queue:
        if job["id"] == job_id:
            job.update(kwargs)
            break
    save_queue(queue)


# ---------------------------------------------------------------------------
# Data helpers (canonicalize_name and crop_resize_pad_bbox imported from nabirds_common)
# ---------------------------------------------------------------------------
class _NABirdsDataset(torch.utils.data.Dataset):
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


def _build_dataset_dfs(config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    """Load and split the NABirds dataset. Returns train_df, val_df, test_df, label_names, split_file."""
    is_all_specific = config.species_mode == "555"
    split_file = SPLIT_80_20_ALL if is_all_specific else SPLIT_80_20_TARGET
    label_names_csv = ALL_SPECIFIC_LABEL_NAMES_CSV if is_all_specific else DEFAULT_LABEL_NAMES_CSV

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

    if is_all_specific:
        classes = classes.sort_values("class_id").reset_index(drop=True)
        class_id_to_idx = {int(cid): idx for idx, cid in enumerate(classes["class_id"])}
    else:
        classes["canon"] = classes["class_name"].map(_canonicalize_name)
        species_to_idx = {s: i for i, s in enumerate(label_names)}
        class_id_to_idx = {}
        for species in label_names:
            canon = _canonicalize_name(species)
            matched = classes.loc[classes["canon"] == canon, "class_id"].tolist()
            y = species_to_idx[species]
            for cid in matched:
                class_id_to_idx[cid] = y

    df = images.merge(labels, on="image_id").merge(splits, on="image_id").merge(bboxes, on="image_id")
    df = df[df["class_id"].isin(class_id_to_idx)].copy()
    df["target"] = df["class_id"].map(class_id_to_idx)
    df["image_path"] = df["image_rel_path"].map(lambda p: str(IMAGES_DIR / p))
    df["is_train"] = pd.to_numeric(df["is_train"], errors="coerce").fillna(-1).astype(int)

    test_df = df[df["is_train"] == 0].copy().reset_index(drop=True)
    trainval_df = df[df["is_train"] == 1].copy().reset_index(drop=True)

    n = len(trainval_df)
    n_val = max(1, int(round(n * config.val_fraction)))
    n_train = n - n_val
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(n)
    train_df = trainval_df.iloc[idx[:n_train]].reset_index(drop=True)
    val_df = trainval_df.iloc[idx[n_train:]].reset_index(drop=True)

    return train_df, val_df, test_df, label_names, split_file


def _build_transforms(config: TrainingConfig):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            TARGET_SIZE,
            scale=(config.crop_scale_min, config.crop_scale_max),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.jitter_brightness,
            contrast=config.jitter_contrast,
            saturation=config.jitter_saturation,
            hue=config.jitter_hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=config.random_erasing_p,
            scale=(0.02, config.random_erasing_scale_max),
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _non_blocking_transfer(device: torch.device) -> bool:
    """Use non-blocking transfers only on CUDA.

    On this setup, `non_blocking=True` on MPS can corrupt labels and produce
    misleadingly high accuracy with unusable checkpoints.
    """
    return device.type == "cuda"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    progress_fn: Any = None,
    scaler=None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    t0 = time.perf_counter()
    total_batches = len(loader)
    report_every = max(1, total_batches // 20)  # ~20 updates per epoch
    can_non_block = _non_blocking_transfer(device)
    use_amp = device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=can_non_block)
            targets = targets.to(device, non_blocking=can_non_block)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            preds = logits.argmax(dim=1)
            running_loss += loss.item() * targets.size(0)
            running_correct += int(preds.eq(targets).sum())
            running_total += targets.size(0)

            if progress_fn and (batch_idx % report_every == 0 or batch_idx == total_batches):
                progress_fn(
                    batch=batch_idx,
                    total_batches=total_batches,
                    running_loss=running_loss / max(1, running_total),
                    running_acc=running_correct / max(1, running_total),
                )

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return avg_loss, avg_acc, time.perf_counter() - t0


def _evaluate_test(
    model: nn.Module,
    test_df: pd.DataFrame,
    eval_transform,
    device: torch.device,
    batch_size: int,
) -> dict:
    ds = _NABirdsDataset(test_df, eval_transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = correct = 0
    running_loss = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, targets in dl:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            correct += int(preds.eq(targets).sum())
            total += targets.size(0)
            running_loss += loss.item() * targets.size(0)
    return {
        "test_loss": running_loss / max(1, total),
        "test_acc": correct / max(1, total),
        "test_time_s": time.perf_counter() - t0,
    }


def _train_stage(
    model: nn.Module,
    optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    job_id: str,
    stage_num: int,
    progress_base: dict,
    scaler=None,
) -> tuple[nn.Module, float, bool]:
    """Train one stage. Returns (model_with_best_weights, best_val_acc, was_cancelled)."""
    best_val_acc = -1.0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):

        def _batch_progress(phase: str, *, batch, total_batches, running_loss, running_acc):
            data = {
                **progress_base,
                "stage": stage_num,
                "epoch": epoch,
                "total_epochs": epochs,
                "phase": phase,
                "batch": batch,
                "total_batches": total_batches,
                "best_val_acc": round(best_val_acc, 6) if best_val_acc >= 0 else None,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            # Use phase-specific keys so the frontend can show the right metric
            data[f"{phase}_acc"] = round(running_acc, 6)
            data[f"{phase}_loss"] = round(running_loss, 6)
            _write_progress(job_id, data)

        train_loss, train_acc, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer,
            progress_fn=lambda **kw: _batch_progress("train", **kw),
            scaler=scaler,
        )
        val_loss, val_acc, _ = _run_epoch(
            model, val_loader, criterion, device,
            progress_fn=lambda **kw: _batch_progress("val", **kw),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        _write_progress(job_id, {
            **progress_base,
            "stage": stage_num,
            "epoch": epoch,
            "total_epochs": epochs,
            "phase": None,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "best_val_acc": round(best_val_acc, 6),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

        if _check_cancel(job_id):
            if best_state is not None:
                model.load_state_dict(best_state)
            return model, best_val_acc, True

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, False


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------
def _append_run_summary(row: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


_STAGE_TO_FILENAME = {
    1: "head_only",
    2: "layer4_finetuned",
    3: "layer3_layer4_finetuned",
}

_STAGE_UNFREEZE_PREFIXES: dict[int, tuple[str, ...]] = {
    1: ("fc.",),
    2: ("layer4.", "fc."),
    3: ("layer3.", "layer4.", "fc."),
}

_STAGE_DESCRIPTIONS = {
    1: "head only",
    2: "layer4 + fc",
    3: "layer3 + layer4 + fc",
}


def _stage_hyperparams(config: TrainingConfig, stage_num: int) -> tuple[int, float]:
    """Return (epochs, lr) for the given stage number."""
    return {
        1: (config.stage1_epochs, config.stage1_lr),
        2: (config.stage2_epochs, config.stage2_lr),
        3: (config.stage3_epochs, config.stage3_lr),
    }[stage_num]


def _ckpt_name(config: TrainingConfig, stage_num: int, run_id: str) -> str:
    suffix = "_all_specific" if config.species_mode == "555" else ""
    stage_str = _STAGE_TO_FILENAME[stage_num]
    return f"resnet50_nabirds_{stage_str}{suffix}_tt_80-20_{config.run_label}_{run_id}.pt"


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train_model(config: TrainingConfig, job_id: str) -> None:
    """Run the full training pipeline for one job. Called from __main__."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started_at = datetime.now().isoformat(timespec="seconds")

    if not config.run_group_id:
        config.run_group_id = job_id

    update_queue_status(job_id, status="running", started_at=run_started_at)
    _write_progress(job_id, {
        "job_id": job_id, "status": "starting",
        "stage": 0, "epoch": 0, "total_epochs": 0,
        "updated_at": run_started_at,
    })

    try:
        # Device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Device: {device}")

        _set_seeds(config.seed)

        _write_progress(job_id, {
            "job_id": job_id, "status": "loading_data",
            "stage": 0, "epoch": 0, "total_epochs": 0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

        train_df, val_df, test_df, label_names, split_file = _build_dataset_dfs(config)
        num_classes = len(label_names)
        print(f"Classes: {num_classes} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        train_transform, eval_transform = _build_transforms(config)
        pin = device.type not in ("mps", "cpu")
        pw = config.num_workers > 0
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        train_loader = DataLoader(
            _NABirdsDataset(train_df, train_transform),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=pin, persistent_workers=pw,
        )
        val_loader = DataLoader(
            _NABirdsDataset(val_df, eval_transform),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=pin, persistent_workers=pw,
        )

        config_json = json.dumps({
            **asdict(config),
            "num_classes": num_classes,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        })

        progress_base = {"job_id": job_id, "status": "running"}
        stages_to_run = sorted(config.stages_to_run)
        stage_model: nn.Module | None = None

        for stage_num in stages_to_run:
            epochs, lr = _stage_hyperparams(config, stage_num)
            prefixes = _STAGE_UNFREEZE_PREFIXES[stage_num]
            print(f"--- Stage {stage_num}: {_STAGE_DESCRIPTIONS[stage_num]} ---")

            stage_t0 = time.perf_counter()

            if stage_num == 1:
                stage_model = _build_model(num_classes).to(device)
            elif stage_model is None:
                raise RuntimeError(
                    f"Stage {stage_num} requires preceding stages to have run."
                )

            for name, param in stage_model.named_parameters():
                param.requires_grad = any(name.startswith(p) for p in prefixes)

            criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, stage_model.parameters()),
                lr=lr, weight_decay=config.weight_decay,
            )
            _write_progress(job_id, {
                **progress_base, "stage": stage_num, "epoch": 0,
                "total_epochs": epochs,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })

            stage_model, best_val, cancelled = _train_stage(
                stage_model, optimizer, criterion, train_loader, val_loader,
                epochs, device, job_id, stage_num, progress_base, scaler=scaler,
            )

            ckpt_path = ARTIFACTS_DIR / "resnet50" / "runs" / _ckpt_name(config, stage_num, run_id)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stage_model.state_dict(), ckpt_path)
            print(f"Stage {stage_num} saved: {ckpt_path.name} | best_val_acc={best_val:.4f}")

            test_res = _evaluate_test(stage_model, test_df, eval_transform, device, config.batch_size)
            print(f"Stage {stage_num} test_acc={test_res['test_acc']:.4f}")
            _append_run_summary({
                "run_group_id": config.run_group_id, "run_label": config.run_label,
                "run_started_at": run_started_at, "stage": f"stage{stage_num}",
                "seed": config.seed, "split_file": split_file.name,
                "batch_size": config.batch_size, "num_epochs": epochs, "lr": lr,
                "weight_decay": config.weight_decay, "label_smoothing": config.label_smoothing,
                "best_val_acc": best_val, "test_loss": test_res["test_loss"],
                "test_acc": test_res["test_acc"], "test_time_s": test_res["test_time_s"],
                "run_time_s": time.perf_counter() - stage_t0,
                "checkpoint_path": str(ckpt_path), "config_json": config_json,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

            if cancelled:
                update_queue_status(job_id, status="cancelled",
                                     completed_at=datetime.now().isoformat(timespec="seconds"))
                _write_progress(job_id, {
                    "job_id": job_id, "status": "cancelled",
                    "stage": stage_num, "epoch": 0, "total_epochs": 0,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })
                return

        # All stages complete
        update_queue_status(job_id, status="done",
                             completed_at=datetime.now().isoformat(timespec="seconds"))
        _write_progress(job_id, {
            "job_id": job_id, "status": "done",
            "stage": max(stages_to_run), "epoch": 0, "total_epochs": 0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"Job {job_id} complete.")

    except KeyboardInterrupt:
        # Process was killed (SIGINT / Streamlit Stop button). KeyboardInterrupt
        # is a BaseException, not Exception, so it would bypass the handler below
        # and leave the job stuck as "running" in the queue forever.
        update_queue_status(job_id, status="cancelled",
                             completed_at=datetime.now().isoformat(timespec="seconds"))
        _write_progress(job_id, {
            "job_id": job_id, "status": "cancelled",
            "stage": 0, "epoch": 0, "total_epochs": 0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        raise
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(err)
        update_queue_status(job_id, status="failed", error=str(e),
                             completed_at=datetime.now().isoformat(timespec="seconds"))
        _write_progress(job_id, {
            "job_id": job_id, "status": "failed",
            "stage": 0, "epoch": 0, "total_epochs": 0,
            "error": str(e),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        raise


# ---------------------------------------------------------------------------
# CLI entry point (called as subprocess)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one queued training job.")
    parser.add_argument("--job-id", required=True, help="Job ID from run_queue.json")
    args = parser.parse_args()

    queue = load_queue()
    job = next((j for j in queue if j["id"] == args.job_id), None)
    if job is None:
        print(f"ERROR: Job {args.job_id} not found in {QUEUE_JSON}")
        sys.exit(1)

    cfg = TrainingConfig(**job["config"])
    train_model(cfg, args.job_id)
