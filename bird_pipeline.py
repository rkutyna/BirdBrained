from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import gc
import json
import logging
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)
TARGET_SIZE = 240
DEFAULT_LABEL_NAMES_CSV = "artifacts/label_names.csv"
ALL_SPECIFIC_LABEL_NAMES_CSV = "artifacts/label_names_nabirds_all_specific.csv"


@dataclass
class PipelineConfig:
    classifier_checkpoint: str
    label_names_csv: str = DEFAULT_LABEL_NAMES_CSV
    yolo_weights: str = "yolo11n.pt"
    yolo_conf: float = 0.25
    device: str = "auto"
    output_root: str = "artifacts/pipeline_runs"
    write_metadata_to_originals: bool = True
    yolo_batch_size: int = 2
    classifier_batch_size: int = 16


@dataclass
class InputItem:
    source_type: str  # 'upload' or 'folder'
    source_path: str
    image_name: str
    pil_image: Image.Image | None
    uploaded_file: Any | None = None


@dataclass
class BirdDetection:
    conf: float
    cls_id: int
    cls_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class ResultRow:
    run_id: str
    source_type: str
    source_path: str
    image_name: str
    yolo_detected: bool
    yolo_conf: float | None
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None
    crop_path: str | None
    sharpness_score_100: float | None
    sharpness_color: str | None
    sharpness_level: str | None
    pred_species: str | None
    pred_confidence: float | None
    pred_confidence_color: str | None
    pred_top5: list | None
    original_path: str | None
    classifier_checkpoint: str
    yolo_weights: str
    timestamp: str
    metadata_written: bool | None = None
    metadata_method: str | None = None
    metadata_error: str | None = None


@dataclass
class ErrorRow:
    run_id: str
    source_type: str
    source_path: str
    image_name: str
    error_type: str
    error_message: str
    timestamp: str


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    if device == "auto":
        return get_default_device()
    return device


def list_classifier_checkpoints(artifacts_dir: str = "artifacts") -> list[str]:
    return sorted(str(p) for p in Path(artifacts_dir).glob("*.pt"))


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


def checkpoint_num_classes(checkpoint_path: str) -> int | None:
    state = torch.load(checkpoint_path, map_location="cpu")
    return _checkpoint_num_classes_from_state(state)


def read_label_names(label_names_path: str | Path) -> list[str]:
    path = Path(label_names_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing label names CSV: {path}")

    df = pd.read_csv(path)
    if "species" not in df.columns:
        raise ValueError(f"Label names CSV must contain a 'species' column: {path}")
    return df["species"].dropna().astype(str).tolist()


def _candidate_label_name_paths(requested_path: str | Path) -> list[Path]:
    requested = Path(requested_path)
    candidates = [
        requested,
        Path(DEFAULT_LABEL_NAMES_CSV),
        Path(ALL_SPECIFIC_LABEL_NAMES_CSV),
    ]
    if requested.parent.exists():
        candidates.extend(sorted(requested.parent.glob("label_names*.csv")))
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        candidates.extend(sorted(artifacts_dir.glob("label_names*.csv")))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_label_names_for_checkpoint(
    checkpoint_path: str,
    label_names_path: str,
    expected_num_classes: int | None = None,
) -> tuple[str, list[str], bool]:
    requested_path = Path(label_names_path)
    label_names = read_label_names(requested_path)

    if expected_num_classes is None:
        expected_num_classes = checkpoint_num_classes(checkpoint_path)

    if expected_num_classes is None or len(label_names) == expected_num_classes:
        return str(requested_path), label_names, False

    matches: list[tuple[Path, list[str]]] = []
    available_counts: list[str] = []
    for candidate in _candidate_label_name_paths(requested_path):
        if not candidate.exists():
            continue
        try:
            candidate_labels = read_label_names(candidate)
        except Exception:
            continue
        available_counts.append(f"{candidate} ({len(candidate_labels)})")
        if len(candidate_labels) == expected_num_classes:
            matches.append((candidate, candidate_labels))

    if len(matches) == 1:
        resolved_path, resolved_labels = matches[0]
        return str(resolved_path), resolved_labels, True

    if len(matches) > 1:
        local_matches = [match for match in matches if match[0].parent == requested_path.parent]
        if len(local_matches) == 1:
            resolved_path, resolved_labels = local_matches[0]
            return str(resolved_path), resolved_labels, True

    raise ValueError(
        f"Checkpoint expects {expected_num_classes} classes, but label CSV {requested_path} has "
        f"{len(label_names)}. Available label CSVs: {', '.join(available_counts) or 'none found'}"
    )


def list_jpeg_files_from_folder(folder: str, recursive: bool) -> list[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Invalid folder path: {folder}")

    exts = {".jpg", ".jpeg"}
    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def confidence_color(conf: float | None) -> str | None:
    if conf is None:
        return None
    if conf > 0.75:
        return "green"
    if conf >= 0.40:
        return "yellow"
    return "red"


def confidence_level(conf: float | None) -> str | None:
    color = confidence_color(conf)
    if color == "green":
        return "high"
    if color == "yellow":
        return "medium"
    if color == "red":
        return "low"
    return None


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def crop_resize_pad(
    img: Image.Image,
    size: int = TARGET_SIZE,
    pad_rgb: tuple[int, int, int] = IMAGENET_PAD_RGB,
) -> Image.Image:
    scale = min(size / img.width, size / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), pad_rgb)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _tenengrad_energy_map(gray: np.ndarray) -> np.ndarray:
    # Sobel gradients on a reflect-padded grayscale image.
    p = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2]
        + 2.0 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    )
    gy = (
        p[:-2, :-2]
        + 2.0 * p[:-2, 1:-1]
        + p[:-2, 2:]
        - p[2:, :-2]
        - 2.0 * p[2:, 1:-1]
        - p[2:, 2:]
    )
    return gx * gx + gy * gy


def _sliding_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def tenengrad_sharpness(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    energy = _tenengrad_energy_map(gray)
    h, w = energy.shape

    # Patch setup: window is 10% of crop width, stride is 50% of window.
    window = max(3, int(round(0.10 * w)))
    window = min(window, w, h)
    stride = max(1, int(round(0.50 * window)))

    xs = _sliding_starts(w, window, stride)
    ys = _sliding_starts(h, window, stride)

    patch_scores: list[float] = []
    for y in ys:
        for x in xs:
            patch = energy[y : y + window, x : x + window]
            patch_scores.append(float(np.mean(patch)))

    if not patch_scores:
        return float(np.mean(energy))

    k = min(5, len(patch_scores))
    top_k = sorted(patch_scores, reverse=True)[:k]
    return float(sum(top_k) / k)


def sharpness_percentile_bounds(
    raw_scores: list[float],
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> tuple[float, float]:
    arr = np.asarray(raw_scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0

    low = float(np.percentile(arr, low_pct))
    high = float(np.percentile(arr, high_pct))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = low + 1.0
    if high <= low:
        high = low + 1e-6
    return low, high


def normalize_tenengrad_score(raw_score: float, low_ref: float, high_ref: float) -> float:
    if not np.isfinite(raw_score):
        return 0.0
    if raw_score <= low_ref:
        return 0.0
    if raw_score >= high_ref:
        return 100.0
    normalized = 100.0 * (raw_score - low_ref) / max(1e-12, (high_ref - low_ref))
    return float(clamp(normalized, 0.0, 100.0))


def classifier_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _build_fc_head(state_dict: dict, in_features: int, num_classes: int) -> nn.Module:
    """Build the fc head matching the format the checkpoint was saved with."""
    if "fc.weight" in state_dict:
        # Old format: bare nn.Linear — keys fc.weight / fc.bias
        return nn.Linear(in_features, num_classes)
    # New format: nn.Sequential(Dropout, Linear) — keys fc.1.weight / fc.1.bias
    return nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )


def load_classifier(
    checkpoint_path: str,
    label_names_path: str,
    device: str,
) -> tuple[torch.nn.Module, list[str]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_checkpoint_state(state)
    expected_num_classes = _checkpoint_num_classes_from_state(state_dict)

    _, label_names, _ = resolve_label_names_for_checkpoint(
        checkpoint_path=checkpoint_path,
        label_names_path=label_names_path,
        expected_num_classes=expected_num_classes,
    )
    num_classes = len(label_names)

    if expected_num_classes is not None and expected_num_classes != num_classes:
        raise RuntimeError(
            f"Checkpoint expects {expected_num_classes} classes, but resolved label CSV has {num_classes}."
        )

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = _build_fc_head(state_dict, model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, label_names


def load_yolo(yolo_weights: str) -> YOLO:
    return YOLO(yolo_weights)


def _best_bird_from_yolo_result(result: Any) -> BirdDetection | None:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or boxes.cls is None or boxes.conf is None:
        return None

    names = result.names if isinstance(result.names, dict) else {}
    best: BirdDetection | None = None
    for xyxy, cls_id, det_conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        cid = int(cls_id)
        cls_name = str(names.get(cid, "")).lower()
        if cls_name != "bird":
            continue
        det = BirdDetection(
            conf=float(det_conf),
            cls_id=cid,
            cls_name=cls_name,
            bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
        )
        if best is None or det.conf > best.conf:
            best = det
    return best


def detect_best_bird(
    image: Image.Image,
    yolo_model: YOLO,
    conf: float,
    device: str,
) -> BirdDetection | None:
    arr = np.array(image)
    results = yolo_model.predict(source=arr, conf=conf, device=device, verbose=False)
    if not results:
        return None
    return _best_bird_from_yolo_result(results[0])


def detect_best_birds_batch(
    images: list[Image.Image],
    yolo_model: YOLO,
    conf: float,
    device: str,
) -> list[BirdDetection | None]:
    if not images:
        return []
    arrs = [np.array(image) for image in images]
    results = yolo_model.predict(source=arrs, conf=conf, device=device, verbose=False)
    if not results:
        return [None] * len(images)
    return [_best_bird_from_yolo_result(r) for r in results]



def crop_detection_fullres(image: Image.Image, xyxy: tuple[float, float, float, float]) -> Image.Image | None:
    w, h = image.size
    x1, y1, x2, y2 = xyxy

    x1 = clamp(x1, 0, w)
    y1 = clamp(y1, 0, h)
    x2 = clamp(x2, 0, w)
    y2 = clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def classify_crop(
    crop_img: Image.Image,
    classifier_model: torch.nn.Module,
    label_names: list[str],
    device: str,
) -> dict[str, Any]:
    x = crop_resize_pad(crop_img)
    t = classifier_transform()(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier_model(t)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        topk = min(5, probs.numel())
        confs, idxs = torch.topk(probs, k=topk)

    top5 = []
    for rank, (conf, idx) in enumerate(zip(confs.tolist(), idxs.tolist()), start=1):
        top5.append(
            {
                "rank": rank,
                "species": label_names[idx],
                "confidence": float(conf),
            }
        )

    return {
        "species": top5[0]["species"],
        "confidence": top5[0]["confidence"],
        "top5": top5,
    }


def classify_crops_batch(
    crop_imgs: list[Image.Image],
    classifier_model: torch.nn.Module,
    label_names: list[str],
    device: str,
) -> list[dict[str, Any]]:
    if not crop_imgs:
        return []

    transform = classifier_transform()
    tensors = []
    for crop_img in crop_imgs:
        x = crop_resize_pad(crop_img)
        tensors.append(transform(x))

    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        logits = classifier_model(batch)
        probs = torch.softmax(logits, dim=1)
        topk = min(5, probs.shape[1])
        confs, idxs = torch.topk(probs, k=topk, dim=1)

    out: list[dict[str, Any]] = []
    for conf_row, idx_row in zip(confs.tolist(), idxs.tolist()):
        top5 = []
        for rank, (conf, idx) in enumerate(zip(conf_row, idx_row), start=1):
            top5.append(
                {
                    "rank": rank,
                    "species": label_names[idx],
                    "confidence": float(conf),
                }
            )
        out.append(
            {
                "species": top5[0]["species"],
                "confidence": top5[0]["confidence"],
                "top5": top5,
            }
        )
    return out


def _safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _iter_chunks(seq: list[Any], size: int) -> Any:
    chunk_size = max(1, int(size))
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def _new_run_dir(output_root: str) -> tuple[str, Path, Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / run_id
    crops_dir = run_dir / "crops"
    originals_dir = run_dir / "originals"
    crops_dir.mkdir(parents=True, exist_ok=True)
    originals_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir, crops_dir, originals_dir


def _build_run_logger(run_id: str, run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"bird_pipeline.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fh = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _prediction_comment(species: str, confidence: float, checkpoint_path: str, run_id: str) -> str:
    level = confidence_level(confidence) or "unknown"
    return (
        f"bird_prediction species={species}; confidence_level={level}; "
        f"checkpoint={Path(checkpoint_path).name}; run_id={run_id}"
    )


def _keyword_component(value: str) -> str:
    # Lightroom hierarchy levels are separated by '|'; commas/semicolons are keyword separators.
    clean = value.replace("|", "/").replace(",", " ").replace(";", " ")
    return " ".join(clean.split())


def _hierarchical_keyword(path_parts: list[str]) -> str:
    return "|".join(_keyword_component(part) for part in path_parts if part.strip())


def write_prediction_metadata(
    image_path: str | Path,
    species: str,
    confidence: float,
    checkpoint_path: str,
    run_id: str,
    sharpness_score_100: float | None = None,
    sharpness_level_name: str | None = None,
) -> tuple[bool, str | None, str | None]:
    target = Path(image_path)
    if not target.exists():
        return False, None, f"missing_file:{target}"

    exiftool = shutil.which("exiftool")
    if exiftool is None:
        return False, None, "exiftool_not_found"

    conf_level = confidence_level(confidence) or "unknown"
    comment = _prediction_comment(species, confidence, checkpoint_path, run_id)
    hierarchical_keywords = [
        _hierarchical_keyword(["Species", species]),
        _hierarchical_keyword(["Species Confidence", conf_level]),
    ]
    if sharpness_score_100 is not None:
        hierarchical_keywords.append(
            _hierarchical_keyword(["Sharpness Score", f"{sharpness_score_100:.1f}"])
        )
        if sharpness_level_name:
            hierarchical_keywords.append(_hierarchical_keyword(["Sharpness", sharpness_level_name]))
        comment = f"{comment}; sharpness_score={sharpness_score_100:.1f}/100"
        if sharpness_level_name:
            comment = f"{comment}; sharpness_level={sharpness_level_name}"

    cmd = [
        exiftool,
        "-overwrite_original",
        "-m",
        "-api",
        "NoDups=1",
        f"-EXIF:UserComment={comment}",
    ]
    for keyword in hierarchical_keywords:
        cmd.append(f"-XMP-lr:HierarchicalSubject+={keyword}")
    cmd.append(str(target))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, "exiftool", None
    except subprocess.CalledProcessError as e:
        message = (e.stderr or e.stdout or str(e)).strip()
        return False, "exiftool", message[:500]
    except Exception as e:
        return False, "exiftool", str(e)


def run_inference_batch(
    inputs: list[InputItem],
    config: PipelineConfig,
    progress_callback: Any | None = None,
) -> tuple[list[ResultRow], list[ErrorRow], Path]:
    if not inputs:
        raise ValueError("No input images provided.")

    device = resolve_device(config.device)
    run_id, run_dir, crops_dir, originals_dir = _new_run_dir(config.output_root)
    logger = _build_run_logger(run_id, run_dir)
    logger.info("Run started | run_id=%s | total_inputs=%d | device=%s", run_id, len(inputs), device)
    logger.info("Config | %s", json.dumps(asdict(config), sort_keys=True))

    classifier_model, label_names = load_classifier(
        checkpoint_path=config.classifier_checkpoint,
        label_names_path=config.label_names_csv,
        device=device,
    )
    logger.info("Classifier loaded | labels=%d", len(label_names))
    yolo_model = load_yolo(config.yolo_weights)
    logger.info("YOLO loaded | weights=%s | conf=%.3f", config.yolo_weights, config.yolo_conf)

    results: list[ResultRow] = []
    errors: list[ErrorRow] = []
    sharpness_raw_by_result_idx: dict[int, float] = {}
    total = len(inputs)
    yolo_batch_size = max(1, int(config.yolo_batch_size))
    classifier_batch_size = max(1, int(config.classifier_batch_size))
    logger.info(
        "Batching config | yolo_batch_size=%d | classifier_batch_size=%d",
        yolo_batch_size,
        classifier_batch_size,
    )

    processed_count = 0
    for batch_items in _iter_chunks(inputs, yolo_batch_size):
        loaded: list[dict[str, Any]] = []

        for item in batch_items:
            item_index = processed_count + len(loaded) + 1
            ts = datetime.now().isoformat(timespec="seconds")
            image = None
            try:
                if progress_callback is not None:
                    progress_callback(processed_count, total, f"Loading {item.image_name}")

                if item.pil_image is not None:
                    image = item.pil_image.convert("RGB")
                elif item.uploaded_file is not None:
                    item.uploaded_file.seek(0)
                    with Image.open(item.uploaded_file) as im:
                        image = im.convert("RGB")
                else:
                    with Image.open(item.source_path) as im:
                        image = im.convert("RGB")

                original_name = f"{item_index:05d}_{_safe_stem(Path(item.image_name).name)}"
                original_path = originals_dir / original_name
                image.save(original_path, "JPEG", quality=95, optimize=True)
                loaded.append(
                    {
                        "item": item,
                        "idx": item_index,
                        "ts": ts,
                        "image": image,
                        "crop": None,
                        "detection": None,
                        "original_path": str(original_path),
                    }
                )
            except Exception as e:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                logger.exception("Error loading image=%s", item.image_name)
                errors.append(
                    ErrorRow(
                        run_id=run_id,
                        source_type=item.source_type,
                        source_path=item.source_path,
                        image_name=item.image_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        timestamp=ts,
                    )
                )
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total, f"Error on {item.image_name}: {type(e).__name__}")

        if not loaded:
            continue

        images = [r["image"] for r in loaded]
        if progress_callback is not None:
            progress_callback(processed_count, total, f"Detecting birds in batch ({len(images)} images)")

        try:
            detections = detect_best_birds_batch(
                images=images,
                yolo_model=yolo_model,
                conf=config.yolo_conf,
                device=device,
            )
            if len(detections) != len(loaded):
                detections = (detections + [None] * len(loaded))[: len(loaded)]
        except Exception:
            logger.exception("Batch detection failed, falling back to single-image detection")
            detections = []
            for record in loaded:
                item = record["item"]
                try:
                    detections.append(
                        detect_best_bird(
                            image=record["image"],
                            yolo_model=yolo_model,
                            conf=config.yolo_conf,
                            device=device,
                        )
                    )
                except Exception as e:
                    logger.exception("Error detecting bird image=%s", item.image_name)
                    errors.append(
                        ErrorRow(
                            run_id=run_id,
                            source_type=item.source_type,
                            source_path=item.source_path,
                            image_name=item.image_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            timestamp=record["ts"],
                        )
                    )
                    detections.append(None)
                    record["skip"] = True
                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total, f"Error on {item.image_name}: {type(e).__name__}")

        classify_queue: list[dict[str, Any]] = []
        for record, detection in zip(loaded, detections):
            if record.get("skip"):
                continue

            item = record["item"]
            ts = record["ts"]
            record["detection"] = detection
            if detection is None:
                logger.info("No bird detected | image=%s", item.image_name)
                results.append(
                    ResultRow(
                        run_id=run_id,
                        source_type=item.source_type,
                        source_path=item.source_path,
                        image_name=item.image_name,
                        yolo_detected=False,
                        yolo_conf=None,
                        bbox_x1=None,
                        bbox_y1=None,
                        bbox_x2=None,
                        bbox_y2=None,
                        crop_path=None,
                        pred_species=None,
                        pred_confidence=None,
                        pred_confidence_color=None,
                        pred_top5=None,
                        sharpness_score_100=None,
                        sharpness_color=None,
                        sharpness_level=None,
                        original_path=record["original_path"],
                        classifier_checkpoint=config.classifier_checkpoint,
                        yolo_weights=config.yolo_weights,
                        timestamp=ts,
                    )
                )
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total, f"Processed {processed_count}/{total}: {item.image_name}")
                continue

            crop = crop_detection_fullres(record["image"], detection.bbox_xyxy)
            if crop is None:
                logger.warning("Degenerate crop | image=%s | bbox=%s", item.image_name, detection.bbox_xyxy)
                results.append(
                    ResultRow(
                        run_id=run_id,
                        source_type=item.source_type,
                        source_path=item.source_path,
                        image_name=item.image_name,
                        yolo_detected=False,
                        yolo_conf=None,
                        bbox_x1=None,
                        bbox_y1=None,
                        bbox_x2=None,
                        bbox_y2=None,
                        crop_path=None,
                        pred_species=None,
                        pred_confidence=None,
                        pred_confidence_color=None,
                        pred_top5=None,
                        sharpness_score_100=None,
                        sharpness_color=None,
                        sharpness_level=None,
                        original_path=record["original_path"],
                        classifier_checkpoint=config.classifier_checkpoint,
                        yolo_weights=config.yolo_weights,
                        timestamp=ts,
                    )
                )
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total, f"Processed {processed_count}/{total}: {item.image_name}")
                continue

            record["crop"] = crop
            try:
                record["sharpness_raw"] = tenengrad_sharpness(crop)
            except Exception:
                logger.exception("Failed to compute sharpness | image=%s", item.image_name)
                record["sharpness_raw"] = None
            classify_queue.append(record)

        for cls_chunk in _iter_chunks(classify_queue, classifier_batch_size):
            if progress_callback is not None:
                progress_callback(processed_count, total, f"Classifying crops batch ({len(cls_chunk)} images)")

            try:
                preds = classify_crops_batch(
                    crop_imgs=[r["crop"] for r in cls_chunk],
                    classifier_model=classifier_model,
                    label_names=label_names,
                    device=device,
                )
            except Exception as e:
                logger.exception("Batch classification failed")
                for record in cls_chunk:
                    item = record["item"]
                    errors.append(
                        ErrorRow(
                            run_id=run_id,
                            source_type=item.source_type,
                            source_path=item.source_path,
                            image_name=item.image_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            timestamp=record["ts"],
                        )
                    )
                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total, f"Error on {item.image_name}: {type(e).__name__}")
                continue

            for record, pred in zip(cls_chunk, preds):
                item = record["item"]
                detection = record["detection"]
                crop = record["crop"]
                ts = record["ts"]
                sharpness_raw = record.get("sharpness_raw")
                try:
                    metadata_written: bool | None = None
                    metadata_method: str | None = None
                    metadata_error: str | None = None
                    if config.write_metadata_to_originals:
                        if item.source_type != "folder":
                            metadata_written = False
                            metadata_error = "source_is_upload_not_local_file"

                    crop_name = f"{_safe_stem(Path(item.image_name).stem)}_bird_{detection.conf:.3f}.jpg"
                    crop_path = crops_dir / crop_name
                    crop.save(crop_path, "JPEG", quality=95, optimize=True)

                    x1, y1, x2, y2 = detection.bbox_xyxy
                    results.append(
                        ResultRow(
                            run_id=run_id,
                            source_type=item.source_type,
                            source_path=item.source_path,
                            image_name=item.image_name,
                            yolo_detected=True,
                            yolo_conf=detection.conf,
                            bbox_x1=x1,
                            bbox_y1=y1,
                            bbox_x2=x2,
                            bbox_y2=y2,
                            crop_path=str(crop_path),
                            sharpness_score_100=None,
                            sharpness_color=None,
                            sharpness_level=None,
                            pred_species=pred["species"],
                            pred_confidence=pred["confidence"],
                            pred_confidence_color=confidence_color(pred["confidence"]),
                            pred_top5=pred["top5"],
                            original_path=record["original_path"],
                            classifier_checkpoint=config.classifier_checkpoint,
                            yolo_weights=config.yolo_weights,
                            timestamp=ts,
                            metadata_written=metadata_written,
                            metadata_method=metadata_method,
                            metadata_error=metadata_error,
                        )
                    )
                    result_idx = len(results) - 1
                    if sharpness_raw is not None and np.isfinite(sharpness_raw):
                        sharpness_raw_by_result_idx[result_idx] = float(sharpness_raw)
                    logger.info(
                        "Classified | image=%s | species=%s | conf=%.4f",
                        item.image_name,
                        pred["species"],
                        pred["confidence"],
                    )
                except Exception as e:
                    logger.exception("Error post-processing image=%s", item.image_name)
                    errors.append(
                        ErrorRow(
                            run_id=run_id,
                            source_type=item.source_type,
                            source_path=item.source_path,
                            image_name=item.image_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            timestamp=ts,
                        )
                    )
                finally:
                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total, f"Processed {processed_count}/{total}: {item.image_name}")

        for record in loaded:
            crop = record.get("crop")
            image = record.get("image")
            if crop is not None:
                try:
                    crop.close()
                except Exception:
                    pass
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass
            record["crop"] = None
            record["image"] = None

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        logger.info("Memory cleanup checkpoint | processed=%d/%d", processed_count, total)

    if sharpness_raw_by_result_idx:
        raw_scores = list(sharpness_raw_by_result_idx.values())
        p5, p95 = sharpness_percentile_bounds(raw_scores, low_pct=5.0, high_pct=95.0)
        logger.info(
            "Sharpness normalization | mode=percentile | p5=%.6f | p95=%.6f | n=%d",
            p5,
            p95,
            len(raw_scores),
        )
        for idx, raw in sharpness_raw_by_result_idx.items():
            score_100 = normalize_tenengrad_score(raw, low_ref=p5, high_ref=p95)
            score_01 = score_100 / 100.0
            results[idx].sharpness_score_100 = score_100
            results[idx].sharpness_color = confidence_color(score_01)
            results[idx].sharpness_level = confidence_level(score_01)

    if config.write_metadata_to_originals:
        for row in results:
            if not row.yolo_detected or row.source_type != "folder":
                continue
            if row.pred_species is None or row.pred_confidence is None:
                continue
            metadata_written, metadata_method, metadata_error = write_prediction_metadata(
                image_path=row.source_path,
                species=row.pred_species,
                confidence=row.pred_confidence,
                checkpoint_path=config.classifier_checkpoint,
                run_id=run_id,
                sharpness_score_100=row.sharpness_score_100,
                sharpness_level_name=row.sharpness_level,
            )
            row.metadata_written = metadata_written
            row.metadata_method = metadata_method
            row.metadata_error = metadata_error
            if metadata_written:
                logger.info("Metadata tagged | image=%s | method=%s", row.image_name, metadata_method)
            else:
                logger.warning(
                    "Metadata tag failed | image=%s | method=%s | err=%s",
                    row.image_name,
                    metadata_method,
                    metadata_error,
                )

    results_df = pd.DataFrame([asdict(r) for r in results])
    errors_df = pd.DataFrame([asdict(e) for e in errors])

    results_csv = run_dir / "results.csv"
    results_json = run_dir / "results.json"
    errors_csv = run_dir / "errors.csv"

    results_df.to_csv(results_csv, index=False)
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    if not errors_df.empty:
        errors_df.to_csv(errors_csv, index=False)

    logger.info(
        "Run completed | results=%d | errors=%d | output_dir=%s",
        len(results),
        len(errors),
        str(run_dir),
    )
    return results, errors, run_dir


def input_items_from_uploaded_files(uploaded_files: list[Any]) -> list[InputItem]:
    items: list[InputItem] = []
    for uf in uploaded_files:
        name = uf.name
        if Path(name).suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        items.append(
            InputItem(
                source_type="upload",
                source_path=name,
                image_name=name,
                pil_image=None,
                uploaded_file=uf,
            )
        )
    return items


def input_items_from_folder(folder: str, recursive: bool) -> list[InputItem]:
    files = list_jpeg_files_from_folder(folder, recursive=recursive)
    items: list[InputItem] = []
    for p in files:
        items.append(
            InputItem(
                source_type="folder",
                source_path=str(p),
                image_name=p.name,
                pil_image=None,
            )
        )
    return items
