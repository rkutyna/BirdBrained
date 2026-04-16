"""Verify CoreML classifier exports match PyTorch predictions on sample photos.

Runs the same 240x240 padded crops through both the PyTorch checkpoint and the
exported CoreML model, compares top-1 species, and reports match rate.

Usage:
    .venv-coreml/bin/python tools/verify_coreml.py --variant 98 --photos personal_library_test --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.bird_pipeline import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENET_PAD_RGB,
    TARGET_SIZE,
    crop_resize_pad,
    read_label_names,
)
from tools.export_coreml import CLASSIFIER_VARIANTS, build_classifier_from_checkpoint


def pytorch_predict(model: torch.nn.Module, image: Image.Image) -> tuple[str, float, list[str]]:
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return labels[idx], float(probs[idx].item()), [labels[i] for i in torch.topk(probs, 5).indices.tolist()]


def coreml_predict(ml_model, image: Image.Image) -> tuple[str, float, list[str]]:
    result = ml_model.predict({"image": image})
    # ClassifierConfig outputs: classLabel (top-1) + probabilities dict {label: prob}
    top1 = result["classLabel"]
    probs = result.get("classLabel_probs") or result.get("classLabelProbs") or result.get("var_827")
    if isinstance(probs, dict):
        top5 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return top1, float(probs[top1]), [k for k, _ in top5]
    return top1, 1.0, [top1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["98", "404"], default="98")
    parser.add_argument("--photos", default="personal_library_test")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    import coremltools as ct

    config = CLASSIFIER_VARIANTS[args.variant]
    checkpoint = REPO_ROOT / config["checkpoint"]
    labels_path = REPO_ROOT / config["labels"]
    mlpackage = REPO_ROOT / "artifacts/coreml" / f"{config['output_name']}.mlpackage"

    global labels
    labels = read_label_names(labels_path)

    print(f"Loading PyTorch classifier from {checkpoint.name}...")
    torch_model = build_classifier_from_checkpoint(checkpoint, len(labels))

    print(f"Loading CoreML model from {mlpackage.name}...")
    ml_model = ct.models.MLModel(str(mlpackage))

    photos_dir = REPO_ROOT / args.photos
    photo_files = sorted(p for p in photos_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})[: args.limit]
    print(f"Verifying {len(photo_files)} photos from {photos_dir.name}/...\n")

    matches = 0
    for photo in photo_files:
        img = Image.open(photo).convert("RGB")
        padded = crop_resize_pad(img, size=TARGET_SIZE, pad_rgb=IMAGENET_PAD_RGB)

        pt_label, pt_conf, pt_top5 = pytorch_predict(torch_model, padded)
        cm_label, cm_conf, cm_top5 = coreml_predict(ml_model, padded)

        match = "✓" if pt_label == cm_label else "✗"
        if pt_label == cm_label:
            matches += 1
        print(f"{match} {photo.name:<25} pt={pt_label:<30} ({pt_conf:.3f})  cm={cm_label:<30} ({cm_conf:.3f})")

    pct = 100.0 * matches / max(1, len(photo_files))
    print(f"\nTop-1 match: {matches}/{len(photo_files)} ({pct:.1f}%)  — pass threshold: 95%")


if __name__ == "__main__":
    main()
