# Bird Photography Processing Pipeline

This repository contains a bird-photo workflow that:

1. Detects birds in photos with YOLO.
2. Crops the best bird detection per image.
3. Classifies bird species with a fine-tuned ResNet-50 model.
4. Scores bird-crop sharpness using patch-based Tenengrad (Sobel energy).
5. Writes prediction metadata to original JPEGs (for Lightroom search/filtering) when processing local folders.
6. Evaluates prediction accuracy against manually labeled ground truth.

## Main Components

- `bird_pipeline.py`: Core batch pipeline (load models, detect, crop, classify, metadata tagging, exports).
- `bird_gallery_frontend.py`: Streamlit app for running inference, browsing run outputs, training management, and accuracy evaluation.
- `nabirds_frontend.py`: Streamlit app for checkpoint comparison/evaluation on NABirds test data.
- `training_engine.py`: Training pipeline with job queue, per-epoch progress tracking, and cancel support.
- `label_photos.py`: Streamlit app for manually labeling personal photos with species names.
- `eval_missing_checkpoints.py`: Script to retroactively evaluate `.pt` checkpoints not yet in `run_summary.csv`.
- `yolo_test.py`: Standalone YOLO utility for detection and crop extraction.
- `artifacts/`: Model checkpoints, label maps, logs, and pipeline run outputs.
- `NABirds Dataset/nabirds/`: NABirds metadata and images used for training/evaluation.

## Pipeline Flow

For each input image:

1. Load image and save a copied original into the run directory.
2. Run YOLO and keep the highest-confidence detection with class name `bird`.
3. Crop bounding box from the full-resolution image.
4. Resize+pad crop to `240x240`, normalize with ImageNet stats.
5. Compute patch-based Tenengrad on the crop:
   - Window size = 10% of crop width (square window)
   - Stride = 50% of window size
   - Final score = mean of top-5 patch scores
   - Normalize to a `0-100` sharpness score using run-level percentile calibration (`p5 -> 0`, `p95 -> 100`)
   - Assign level/color using same thresholds as prediction confidence:
     - `low` / red: `< 40`
     - `medium` / yellow: `40-75`
     - `high` / green: `> 75`
6. Run ResNet classifier and return top-5 species + confidences (top-1 used for display).
7. Optionally tag original local JPEG metadata via `exiftool` (including sharpness).
8. Save run outputs (`results.csv`, `results.json`, crops, logs, errors if any).

## Requirements

### System

- Python 3.10+ recommended
- `exiftool` (required only for metadata tagging to original files)

### Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pillow streamlit ultralytics torch torchvision
```

Optional (needed only for RAW `.nef` handling in `yolo_test.py`):

```bash
pip install rawpy
```

## Running Inference (Primary App)

Launch the Streamlit gallery app:

```bash
streamlit run bird_gallery_frontend.py
```

The app has two tabs:

### Classify tab

In the sidebar:

- Select a classifier checkpoint (`artifacts/*.pt`). The sidebar shows the recorded test accuracy for the selected checkpoint.
- Set YOLO weights and confidence threshold.
- Choose input mode:
  - `Upload files`: process uploaded JPEGs.
  - `Folder path`: process JPEGs from a local folder (supports recursive scan).
- Toggle metadata tagging:
  - Tagging is applied only in `Folder path` mode.
  - Upload mode cannot tag original files by design.

If the input folder contains a `labels.csv` (created by `label_photos.py`), the run summary automatically shows ground-truth accuracy metrics:

- **Top-1 accuracy** and **Top-5 accuracy** computed against your labels.
- Per-photo verdict badges in the gallery: ✓ Correct, ✗ Wrong, ⊘ no bird detected, ⊘ not in model subset.
- Photos where no bird was detected, or whose labeled species is not in the model's training set, are excluded from the accuracy denominator.
- Predictions are compared at the base-species level — sex/morph qualifiers (e.g. "Breeding male") are stripped before matching.

### Training tab

Manage and monitor ResNet-50 fine-tuning runs directly from the browser:

- Define single runs or multiplicative parameter sweeps.
- Jobs run sequentially in a background subprocess; the browser stays responsive.
- Live per-epoch progress (train/val accuracy) is shown while a job runs.
- Cancel a running job or remove pending jobs from the queue individually.
- Run history is pulled from `artifacts/logs/run_summary.csv` with filtering and sorting.

## Photo Labeling Tool

To manually label a folder of personal photos for accuracy testing:

```bash
streamlit run label_photos.py
```

- Enter a folder path to load all images.
- Navigate photos one at a time with Prev/Next buttons or jump to the next unlabeled image.
- Search and select a species from the 404 top-level NABirds species (sex/morph variants collapsed).
- Labels are saved to `labels.csv` in the same folder and are picked up automatically by the inference app.

## Metadata for Lightroom

When enabled and supported (`Folder path` + `exiftool` available), the pipeline writes:

- `XMP-lr:HierarchicalSubject` keywords
- `EXIF:UserComment`

Keyword paths are hierarchical (using Lightroom's `|` separators):

- `Species|<predicted species>`
- `Species Confidence|<high|medium|low>`
- `Sharpness Score|<0-100>`
- `Sharpness|<high|medium|low>`

## Run Outputs

Each run creates a timestamped folder under `artifacts/pipeline_runs/<run_id>/`:

- `originals/`: copied originals used for the run
- `crops/`: detected bird crops
- `results.csv`: row-level structured outputs (includes top-5 predictions)
- `results.json`: JSON export of results
- `errors.csv`: written when errors occur
- `pipeline.log`: detailed execution log

## Checkpoint Evaluation / Comparison

To compare checkpoints on the NABirds test split:

```bash
streamlit run nabirds_frontend.py
```

This app provides:

- Top-5 prediction comparison on selected test images
- Full test evaluation (top-1 / top-5)
- Per-species top-1 accuracy tables

To evaluate any `.pt` checkpoint files not yet recorded in `run_summary.csv`:

```bash
python eval_missing_checkpoints.py
```

## Training

### Via the browser (recommended)

Use the Training tab in `bird_gallery_frontend.py` (see above).

### Via notebook

- `resnet.ipynb`: ResNet training/fine-tuning workflow

Training metrics and configs are tracked in:

- `artifacts/logs/run_summary.csv`
- `artifacts/logs/epoch_metrics.csv`
- `artifacts/logs/run_configs.jsonl`

## Species Label Files

| File | Classes | Description |
|---|---|---|
| `artifacts/label_names.csv` | 98 | Target species subset used for most training runs |
| `artifacts/label_names_nabirds_all_specific.csv` | 555 | All NABirds-specific classes including sex/morph variants |
| `artifacts/label_names_nabirds_base_species.csv` | 404 | Top-level species names (variants collapsed) — used by the labeling tool |

## Notes

- Classifier checkpoints saved with either a bare `nn.Linear` head or an `nn.Sequential(Dropout, Linear)` head are both handled automatically.
- YOLO bird detection relies on class name matching to `"bird"` in YOLO outputs.
