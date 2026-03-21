"""Download bird classifier checkpoints from HuggingFace Hub.

Usage:
    python download_models.py              # Download all checkpoints
    python download_models.py --model 98   # Download 98-species only
    python download_models.py --model 404  # Download 404-species only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

# TODO: Replace with your HuggingFace repo ID after uploading checkpoints.
# Upload checkpoints with:
#   huggingface-cli upload YOUR_USERNAME/bird-classifier \
#       artifacts/resnet50/subset98_combined/best.pt subset98_combined/best.pt
#   huggingface-cli upload YOUR_USERNAME/bird-classifier \
#       artifacts/resnet50/base_combined/best.pt base_combined/best.pt
HF_REPO_ID = "YOUR_USERNAME/bird-classifier"

MODELS = {
    "98": {
        "hf_filename": "subset98_combined/best.pt",
        "local_path": Path("artifacts/resnet50/subset98_combined/best.pt"),
        "description": "98 species (combined) — 97.4% test accuracy",
    },
    "404": {
        "hf_filename": "base_combined/best.pt",
        "local_path": Path("artifacts/resnet50/base_combined/best.pt"),
        "description": "404 base species (combined) — 93.6% test accuracy",
    },
}


def download_model(key: str) -> None:
    info = MODELS[key]
    local_path = info["local_path"]
    if local_path.exists():
        print(f"  Already exists: {local_path}")
        return
    print(f"  Downloading {info['description']}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=info["hf_filename"],
        local_dir=str(local_path.parent.parent),
    )
    print(f"  Saved to {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bird classifier checkpoints from HuggingFace Hub")
    parser.add_argument(
        "--model",
        choices=["98", "404", "all"],
        default="all",
        help="Which model to download (default: all)",
    )
    args = parser.parse_args()

    print("Downloading bird classifier checkpoints...")
    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    for key in keys:
        download_model(key)
    print("Done.")


if __name__ == "__main__":
    main()
