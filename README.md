# Bird Photography Processing Pipeline

An end-to-end system for detecting, classifying, and cataloging bird species in personal photography. The pipeline combines YOLO object detection with a fine-tuned ResNet-50 classifier, a patch-based sharpness scorer, and optional Lightroom metadata tagging — all driven from Streamlit applications for inference, training, evaluation, and photo labeling.

## What It Does

Given a folder (or upload) of bird photos, the pipeline:

1. **Detects** birds using YOLO, keeping the highest-confidence detection per image.
2. **Crops** the detected bird at full resolution.
3. **Classifies** the crop into one of 98 target species (or 555 NABirds-specific classes) using a fine-tuned ResNet-50.
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

### Model Details

- **Detection**: YOLO 11n (`yolo11n.pt`), filtering for `"bird"` class detections.
- **Classification**: ResNet-50 pretrained on ImageNet V2, fine-tuned on [NABirds](https://dl.allawnmilner.com/nabirds) with a `Sequential(Dropout(0.4), Linear)` head.
- **Training strategy**: Three-stage progressive unfreezing:
  1. **Stage 1** — Head only (fc layer), backbone frozen.
  2. **Stage 2** — `layer4` + fc unfrozen, lower learning rate.
  3. **Stage 3** — `layer3` + `layer4` + fc unfrozen, lowest learning rate.
- **Sharpness**: Patch-based Tenengrad scoring — Sobel gradient energy computed over sliding windows (10% of crop width, 50% stride), final score = mean of top-5 patches. Normalized per-run to 0–100 using p5/p95 percentile bounds.

### Confidence and Sharpness Thresholds

| Level  | Color  | Confidence range | Sharpness range |
|--------|--------|------------------|-----------------|
| High   | Green  | > 75%            | > 75            |
| Medium | Yellow | 40–75%           | 40–75           |
| Low    | Red    | < 40%            | < 40            |

## Project Structure

```
capstone/
├── frontend/                                  Streamlit applications
│   ├── bird_gallery_frontend.py               Main app (Classify + Training tabs)
│   ├── nabirds_frontend.py                    Checkpoint comparison app
│   └── label_photos.py                        Manual photo labeling tool
│
├── training/                                  Training orchestration
│   ├── autorun.py                             Codex-driven autoresearch loop
│   └── training_engine.py                     Training pipeline + job queue (subprocess)
│
├── dataprep/                                  Dataset preparation
│   ├── prepare.py                             NABirds train/val/test splits
│   ├── prepare_birdsnap.py                    Birdsnap download from HuggingFace
│   ├── prepare_inat.py                        iNaturalist API download
│   └── prepare_combined.py                    Merge NABirds + external datasets
│
├── inference/                                 Core inference pipeline
│   └── bird_pipeline.py                       Detection, classification, sharpness, metadata
│
├── tools/                                     Standalone utilities
│   ├── yolo_test.py                           YOLO bird detection + crop extraction
│   ├── eval_missing_checkpoints.py            Retroactive checkpoint evaluation
│   ├── monitor.py                             Live terminal monitor for autoresearch
│   └── replay_train_variants.py               Replay saved train.py variants
│
├── notebooks/                                 Jupyter notebooks (historical/exploratory)
│   ├── resnet.ipynb                           ResNet training development
│   ├── overnight.ipynb                        Extended training experiments
│   ├── bird_infer_pipeline.ipynb              Inference pipeline development
│   ├── visualize_runs.ipynb                   Training run visualization
│   └── vit_test.ipynb                         Vision Transformer experiments
│
├── prompts/                                   Codex prompt templates
│   └── autoresearch_codex_prompt.txt
│
├── artifacts/
│   ├── labels/                                Species label CSVs
│   │   ├── label_names.csv                    98 target species
│   │   ├── label_names_nabirds_all_specific.csv  555 NABirds-specific classes
│   │   └── label_names_nabirds_base_species.csv  404 base species
│   ├── splits/                                Cached train/val/test DataFrames
│   │   ├── subset98.pkl, full555.pkl, etc.
│   │   └── *_combined.pkl                     Combined dataset splits
│   ├── resnet50/                              Model checkpoints (gitignored)
│   │   ├── subset98/                          98-species autoresearch outputs
│   │   │   ├── best.pt                        Best checkpoint
│   │   │   └── experiment_log.csv             Experiment history
│   │   ├── full555/                           555-species outputs
│   │   └── runs/                              Individual training run checkpoints
│   ├── external/                              External dataset caches
│   │   ├── birdsnap_splits.pkl
│   │   ├── inat_splits.pkl
│   │   └── inat_manifest.json
│   ├── autoresearch_runner/                   Per-iteration artifacts
│   ├── logs/                                  Training logs (gitignored)
│   │   ├── run_summary.csv                    Consolidated run results
│   │   ├── run_queue.json                     Job queue state
│   │   ├── progress_*.json                    Real-time job progress
│   │   └── training_*.log                     Per-job subprocess logs
│   └── pipeline_runs/<run_id>/                Inference outputs (gitignored)
│       ├── originals/, crops/
│       ├── results.csv / .json
│       └── pipeline.log
│
├── .streamlit/config.toml                     Streamlit theme configuration
├── NABirds_Dataset/nabirds/                   NABirds metadata and images (not in git)
│
├── train.py                                   Self-contained training script (Codex-managed)
├── nabirds_common.py                          Shared constants and utilities
├── program.md                                 Codex agent instructions
├── REPO_MAP.md                                Comprehensive repository map
└── README.md
```

See [REPO_MAP.md](REPO_MAP.md) for a detailed file-by-file map with dependency graphs and data flows.

## Requirements

### System

- Python 3.10+
- `exiftool` — required only for metadata tagging to original JPEGs
  - macOS: `brew install exiftool`
  - Ubuntu: `sudo apt install libimage-exiftool-perl`

### Python Packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pillow streamlit ultralytics torch torchvision
```

Optional (RAW `.nef` handling in `tools/yolo_test.py` only):

```bash
pip install rawpy
```

### Data

Training and evaluation require the [NABirds dataset](https://dl.allawnmilner.com/nabirds) extracted to `NABirds_Dataset/nabirds/`. Inference on personal photos does not require this dataset — only a trained checkpoint (`.pt` file) in `artifacts/resnet50/`.

## Running Inference (Primary App)

```bash
streamlit run frontend/bird_gallery_frontend.py
```

### Classify Tab

Configure in the sidebar:

- **Classifier checkpoint** — select from `artifacts/resnet50/`. The sidebar displays the recorded test accuracy from `run_summary.csv` when available.
- **Species mode** — toggle between 98 target species or 555 all-specific NABirds classes. Checkpoints are filtered to match.
- **YOLO settings** — weights path, confidence threshold, batch size.
- **Input mode**:
  - *Upload files* — drag-and-drop JPEGs (metadata tagging unavailable).
  - *Folder path* — process JPEGs from a local directory, optionally recursive.
- **Metadata tagging** — toggle writing predictions to original JPEGs (folder mode only).

**Results include:**

- Summary metrics: total images, birds detected, mean confidence, mean sharpness, mean crop megapixels.
- Paginated photo gallery with original + crop side-by-side, species prediction, confidence badge, sharpness score, top-5 list.
- CSV and JSON export downloads.

**Ground-truth evaluation** — if the input folder contains a `labels.csv` (created by the labeling tool), the app automatically computes:

- Top-1 and Top-5 accuracy against your labels.
- Per-photo verdict badges: ✓ Correct, ✗ Wrong, ⊘ no bird detected, ⊘ not in model subset.
- Photos with no detection or out-of-scope species are excluded from the accuracy denominator.
- Predictions are compared at the base-species level (sex/morph qualifiers like "Breeding male" are stripped).
- Gallery filtering by verdict: Correct, Wrong, No bird detected, Not in model subset, Unlabeled.

### Training Tab

Manage ResNet-50 fine-tuning runs from the browser:

- **Single runs** — configure all hyperparameters (per-stage epochs/LR, augmentation, weight decay, label smoothing, batch size, seed).
- **Parameter sweeps** — vary one parameter multiplicatively across a range (e.g., stage1_lr from 1e-2 to 1e-4 at 0.1× steps). All other parameters come from a base config.
- **Job queue** — jobs run sequentially as background subprocesses. The browser stays responsive with 3-second auto-refresh.
- **Live progress** — batch-level progress bar (updates ~20 times per epoch), showing current phase (Training/Validating), batch count, and running accuracy as it climbs.
- **Controls** — cancel a running job (stops after current epoch) or remove pending jobs individually. Clear finished jobs in bulk.
- **Run history** — table from `run_summary.csv` with stage filtering, sorted by test accuracy.

**Three-stage training:**

| Stage | Unfrozen layers          | Typical LR | Purpose                           |
|-------|--------------------------|------------|-----------------------------------|
| 1     | FC head only             | 1e-3       | Learn species mapping on frozen features |
| 2     | `layer4` + FC            | 2e-4       | Adapt high-level features          |
| 3     | `layer3` + `layer4` + FC | 1e-4       | Fine-tune mid-level features       |

Each stage saves a checkpoint, evaluates on the test split, and records results to `run_summary.csv`. The best checkpoint is auto-selected by test accuracy when starting inference.

## Photo Labeling Tool

```bash
streamlit run frontend/label_photos.py
```

Create ground-truth labels for accuracy evaluation:

- Enter a folder path to load all images.
- Navigate with Prev/Next or jump to the next unlabeled photo.
- Search and select from 404 top-level NABirds species (sex/morph variants collapsed).
- Labels save to `labels.csv` in the folder, picked up automatically by the inference app.

## Checkpoint Comparison

```bash
streamlit run frontend/nabirds_frontend.py
```

Compare checkpoints side-by-side on the NABirds test split:

- Browse test images with top-5 predictions from two checkpoints (A vs B).
- Run full test evaluation (top-1 and top-5 accuracy).
- Per-species accuracy breakdown with filtering (e.g., show lowest-performing species).

To retroactively evaluate checkpoints not yet in `run_summary.csv`:

```bash
python tools/eval_missing_checkpoints.py
```

## Overnight Autoresearch

`training/autorun.py` is an outer loop for unattended Codex-driven experimentation. It
launches a fresh `codex exec` session each iteration, asks it to do exactly one
`train.py` experiment, then keeps or restores `train.py` based on the newest
row in the experiment log CSV.

Basic usage:

```bash
python training/autorun.py --iterations 20 --danger-full-access
```

Time-based stop condition:

```bash
python training/autorun.py --hours 8 --danger-full-access
```

Useful flags:

- `--patience 8` stops after 8 consecutive non-improving iterations.
- `--cooldown-seconds 5` waits between Codex sessions.
- `--extra-instructions path/to/file.txt` appends custom guidance to every
  Codex prompt.
- `--dry-run` builds the iteration prompt and state directory without invoking
  Codex.

Runner artifacts are written to `artifacts/autoresearch_runner/`.

Note: in this repo, PyTorch only sees Apple MPS when Codex runs unsandboxed, so
`--danger-full-access` is the practical mode for overnight training on macOS.

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

## Species Label Files

| File | Classes | Description |
|------|---------|-------------|
| `artifacts/labels/label_names.csv` | 98 | Target species subset — used for most training runs |
| `artifacts/labels/label_names_nabirds_all_specific.csv` | 555 | All NABirds-specific classes including sex/morph variants |
| `artifacts/labels/label_names_nabirds_base_species.csv` | 404 | Top-level species (variants collapsed) — used by the labeling tool |

## Standalone Utilities

### YOLO Detection (`tools/yolo_test.py`)

```bash
python tools/yolo_test.py <source_folder> [--imgsz 1280] [--device mps] [--recursive]
```

Standalone bird detection with crop extraction. Supports JPEG, PNG, BMP, TIFF, and RAW `.nef` files (with `rawpy`). Saves top-3 bird crops per image.

## Notes

- Classifier checkpoints with either a bare `nn.Linear` head or `nn.Sequential(Dropout, Linear)` head are detected and loaded automatically.
- The checkpoint auto-selector picks the checkpoint with the highest test accuracy from `run_summary.csv`, falling back to stage-rank heuristic (stage 3 > stage 2 > stage 1) if no summary data exists.
- YOLO detection filters by class name `"bird"` — non-bird detections are discarded.
- Training runs as a subprocess (`python training/training_engine.py --job-id <id>`), communicating progress via JSON files on disk. This keeps the Streamlit frontend responsive during long training runs.
