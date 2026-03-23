# Bird Photography Processing Pipeline

An end-to-end system for detecting, classifying, and cataloging bird species in personal photography. The pipeline combines YOLO object detection with a fine-tuned ResNet-50 classifier, a patch-based sharpness scorer, and optional Lightroom metadata tagging — all driven from a Streamlit application.

## Quick Start

```bash
# 1. Clone and install
git clone --branch MVP https://github.com/rkutyna/BirdBrained
cd BirdBrained
python -m venv .venv          # Use "python3" if your system requires it (common on macOS/Linux)

# Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows (Command Prompt):
# .venv\Scripts\activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 2. Download model checkpoints from HuggingFace Hub
python download_models.py

# 3. (Optional) Install exiftool for Lightroom metadata tagging
# macOS:
brew install exiftool
# Ubuntu / Debian:
sudo apt install libimage-exiftool-perl
# Windows (Chocolatey):
# choco install exiftool
# Windows (manual): download from https://exiftool.org and add to PATH

# 4. Launch the app
streamlit run frontend/bird_gallery_frontend.py
```

## What It Does

Given a folder (or upload) of bird photos, the pipeline:

1. **Detects** birds using YOLO, keeping the highest-confidence detection per image.
2. **Crops** the detected bird at full resolution.
3. **Classifies** the crop into one of the target species using a fine-tuned ResNet-50.
4. **Scores sharpness** of each bird crop using patch-based Tenengrad (Sobel gradient energy).
5. **Tags original JPEGs** with species, confidence, and sharpness metadata for Lightroom search and filtering.
6. **Evaluates accuracy** against manually labeled ground truth when available.

## Architecture

```
                    ┌──────────────┐
    Input JPEGs ──▸ │  YOLO 11n    │──▸ Best bird bbox per image
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Full-res    │──▸ Bird crop (original resolution)
                    │  Crop        │
                    └──────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌──────────────┐ ┌──────────────┐
            │  ResNet-50   │ │  Tenengrad   │
            │  Classifier  │ │  Sharpness   │
            └──────────────┘ └──────────────┘
                    │             │
                    ▼             ▼
            Species top-5    Sharpness 0-100
            + confidence     + level/color
                    │             │
                    └──────┬──────┘
                           ▼
                    ┌──────────────┐
                    │  exiftool    │──▸ Tagged JPEG (XMP + EXIF + IPTC)
                    │  Metadata    │
                    └──────────────┘
```

## Available Models

| Model | Species | Test Accuracy | Size |
|-------|---------|---------------|------|
| 98 species (combined) | 98 target species | 97.4% | ~91 MB |
| 404 base species (combined) | 404 NABirds base species | 93.6% | ~98 MB |

Both models use ResNet-50 pretrained on ImageNet V2, fine-tuned on [NABirds](https://dl.allawnmilner.com/nabirds) augmented with Birdsnap and iNaturalist data. Training used three-stage progressive unfreezing (head only → layer4 + head → layer3 + layer4 + head).

Download checkpoints with:

```bash
python download_models.py              # Download all
python download_models.py --model 98   # 98-species only
python download_models.py --model 404  # 404-species only
```

## Running Inference

```bash
streamlit run frontend/bird_gallery_frontend.py
```

Configure in the sidebar:

- **Species mode** — choose between 98 target species or 404 base species. Checkpoints are filtered to match.
- **Classifier checkpoint** — select from downloaded checkpoints. The sidebar displays the recorded test accuracy from `experiment_log.csv` when available.
- **YOLO settings** — weights path, confidence threshold, batch size.
- **Input mode**:
  - *Upload files* — drag-and-drop JPEGs (metadata tagging unavailable).
  - *Folder path* — process JPEGs from a local directory, optionally recursive.
- **Metadata tagging** — toggle writing predictions to original JPEGs (folder mode only).

**Results include:**

- Summary metrics: total images, birds detected, mean confidence, mean sharpness, mean crop megapixels.
- Paginated photo gallery with original + crop side-by-side, species prediction, confidence badge, sharpness score, top-5 list.
- CSV and JSON export downloads.

### Ground-Truth Evaluation

If the input folder contains a `labels.csv` (created by the labeling tool), the app automatically computes:

- Top-1 and Top-5 accuracy against your labels.
- Per-photo verdict badges: Correct, Wrong, no bird detected, not in model subset.
- Photos with no detection or out-of-scope species are excluded from the accuracy denominator.
- Predictions are compared at the base-species level (sex/morph qualifiers stripped).
- Gallery filtering by verdict.

### Confidence and Sharpness Thresholds

| Level  | Color  | Confidence range | Sharpness range |
|--------|--------|------------------|-----------------|
| High   | Green  | > 75%            | > 75            |
| Medium | Yellow | 40–75%           | 40–75           |
| Low    | Red    | < 40%            | < 40            |

## Photo Labeling Tool

```bash
streamlit run frontend/label_photos.py
```

Create ground-truth labels for accuracy evaluation:

- Enter a folder path to load all images.
- Navigate with Prev/Next or jump to the next unlabeled photo.
- Search and select from 404 top-level NABirds species.
- Labels save to `labels.csv` in the folder, picked up automatically by the inference app.

## Metadata for Lightroom

When metadata tagging is enabled (folder mode + `exiftool` available), the pipeline writes to each original JPEG:

| Field | Content |
|-------|---------|
| `XMP-lr:HierarchicalSubject` | Lightroom hierarchical keywords (pipe-separated) |
| `XMP-dc:Subject` | Flat keywords |
| `IPTC:Keywords` | Flat keywords (legacy compatibility) |
| `EXIF:UserComment` | Human-readable prediction summary |

**Hierarchical keyword structure** (visible in Lightroom's keyword tree):

```
Species|American Robin
Species Confidence|High
Sharpness Score|82
Sharpness|High
```

Tags are written in a two-pass approach: old prediction tags are cleared first, then fresh tags are written. This makes re-running inference on the same folder safe — predictions are always current.

## Project Structure

```
capstone/
├── frontend/
│   ├── bird_gallery_frontend.py     Main inference app
│   └── label_photos.py              Manual photo labeling tool
│
├── inference/
│   └── bird_pipeline.py             YOLO detection + ResNet classification + sharpness + metadata
│
├── artifacts/
│   ├── labels/                      Species label CSVs
│   │   ├── label_names.csv          98 target species
│   │   ├── label_names_nabirds_all_specific.csv   555 NABirds-specific classes
│   │   └── label_names_nabirds_base_species.csv   404 base species
│   └── resnet50/                    Model checkpoints (downloaded from HuggingFace)
│       ├── subset98_combined/
│       │   ├── best.pt              98-species checkpoint
│       │   └── experiment_log.csv   Training history
│       └── base_combined/
│           ├── best.pt              404-species checkpoint
│           └── experiment_log.csv   Training history
│
├── .streamlit/config.toml           Streamlit theme
├── download_models.py               Download checkpoints from HuggingFace Hub
├── requirements.txt                 Python dependencies
├── yolo11n.pt                       YOLO 11n detection weights
└── README.md
```

## Requirements

### System

- Python 3.10+
- `exiftool` — required only for metadata tagging to original JPEGs
  - macOS: `brew install exiftool`
  - Ubuntu / Debian: `sudo apt install libimage-exiftool-perl`
  - Windows: `choco install exiftool` or download from [exiftool.org](https://exiftool.org)

### Python Packages

```bash
pip install -r requirements.txt
```

## Species Label Files

| File | Classes | Description |
|------|---------|-------------|
| `artifacts/labels/label_names.csv` | 98 | Target species subset |
| `artifacts/labels/label_names_nabirds_all_specific.csv` | 555 | All NABirds-specific classes including sex/morph variants |
| `artifacts/labels/label_names_nabirds_base_species.csv` | 404 | Top-level species (variants collapsed) — used by the labeling tool |

## Notes

- Classifier checkpoints with either a bare `nn.Linear` head or `nn.Sequential(Dropout, Linear)` head are detected and loaded automatically.
- The checkpoint auto-selector picks the checkpoint with the highest test accuracy from `experiment_log.csv`.
- YOLO detection filters by class name `"bird"` — non-bird detections are discarded.
