"""Export BirdBrained PyTorch models to CoreML for the iOS app.

Usage:
    python tools/export_coreml.py                # export everything
    python tools/export_coreml.py --classifier 98
    python tools/export_coreml.py --detector
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.bird_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
    _GeM,
    _build_fc_head,
    _unwrap_checkpoint_state,
    read_label_names,
)

CLASSIFIER_VARIANTS = {
    "98": {
        "checkpoint": "artifacts/resnet50/subset98_combined/best.pt",
        "labels": "artifacts/labels/label_names.csv",
        "output_name": "BirdClassifier98",
    },
    "404": {
        "checkpoint": "artifacts/resnet50/base_combined/best.pt",
        "labels": "artifacts/labels/label_names_nabirds_base_species.csv",
        "output_name": "BirdClassifier404",
    },
}

DETECTOR_WEIGHTS = "yolo11n.pt"
DETECTOR_OUTPUT_NAME = "BirdDetector"
DEFAULT_OUTPUT_DIR = "artifacts/coreml"


class NormalizedClassifier(nn.Module):
    """Wraps the ResNet classifier with baked-in ImageNet normalization + softmax.

    CoreML will deliver inputs in [0, 1] (via ImageType scale=1/255). This wrapper
    handles the per-channel mean/std subtraction exactly, then returns probabilities
    so CoreML's ClassifierConfig can attach class labels directly.
    """

    def __init__(self, backbone: nn.Module, mean: tuple, std: tuple) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        logits = self.backbone(x)
        return torch.softmax(logits, dim=1)


def build_classifier_from_checkpoint(checkpoint_path: Path, num_classes: int) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state(state)

    model = models.resnet50(weights=None)
    if "avgpool.p" in state_dict:
        model.avgpool = _GeM(p=float(state_dict["avgpool.p"].item()))
    model.fc = _build_fc_head(state_dict, model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_classifier(variant: str, output_dir: Path) -> Path:
    import coremltools as ct

    config = CLASSIFIER_VARIANTS[variant]
    checkpoint_path = REPO_ROOT / config["checkpoint"]
    labels_path = REPO_ROOT / config["labels"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    labels = read_label_names(labels_path)
    print(f"[{variant}] {len(labels)} labels from {labels_path.name}")

    backbone = build_classifier_from_checkpoint(checkpoint_path, len(labels))
    wrapped = NormalizedClassifier(backbone, IMAGENET_MEAN, IMAGENET_STD).eval()

    example_input = torch.rand(1, 3, TARGET_SIZE, TARGET_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)

    image_input = ct.ImageType(
        name="image",
        shape=example_input.shape,
        scale=1.0 / 255.0,
        color_layout=ct.colorlayout.RGB,
    )

    print(f"[{variant}] converting to CoreML (FP16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(class_labels=labels),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    mlmodel.short_description = f"BirdBrained species classifier ({len(labels)} classes, ResNet-50 + GeM)"
    mlmodel.author = "BirdBrained"
    mlmodel.license = "See project repository"
    mlmodel.version = "1.0"

    output_path = output_dir / f"{config['output_name']}.mlpackage"
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    print(f"[{variant}] saved -> {output_path}")
    return output_path


def export_detector(output_dir: Path) -> Path:
    from ultralytics import YOLO

    weights_path = REPO_ROOT / DETECTOR_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    print(f"[detector] exporting {weights_path.name} via Ultralytics...")
    model = YOLO(str(weights_path))
    exported = model.export(format="coreml", nms=True, half=True, imgsz=640)
    exported_path = Path(exported)

    output_path = output_dir / f"{DETECTOR_OUTPUT_NAME}.mlpackage"
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    if exported_path.is_dir():
        shutil.move(str(exported_path), str(output_path))
    else:
        target = output_dir / exported_path.name
        shutil.move(str(exported_path), str(target))
        output_path = target

    print(f"[detector] saved -> {output_path}")
    return output_path


def copy_labels(output_dir: Path) -> None:
    for variant, config in CLASSIFIER_VARIANTS.items():
        src = REPO_ROOT / config["labels"]
        if not src.exists():
            continue
        dest = output_dir / f"label_names_{variant}.csv"
        shutil.copy2(src, dest)
        print(f"[labels] copied -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BirdBrained models to CoreML.")
    parser.add_argument(
        "--classifier",
        choices=["98", "404", "both", "none"],
        default="both",
        help="Which classifier variant(s) to export.",
    )
    parser.add_argument(
        "--detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export the YOLO detector.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write .mlpackage outputs (relative to repo root).",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    variants_to_export: list[str] = []
    if args.classifier == "both":
        variants_to_export = ["98", "404"]
    elif args.classifier in ("98", "404"):
        variants_to_export = [args.classifier]

    for variant in variants_to_export:
        export_classifier(variant, output_dir)

    if args.detector:
        export_detector(output_dir)

    copy_labels(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
