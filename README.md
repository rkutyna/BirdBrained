# Bird Photography Processing Pipeline

This repository contains a bird-photo workflow that:

1. Detects birds in photos with YOLO.
2. Crops the best bird detection per image.
3. Classifies bird species with a fine-tuned ResNet-50 model.
4. Scores bird-crop sharpness using patch-based Tenengrad (Sobel energy).
5. Writes prediction metadata to original JPEGs (for Lightroom search/filtering) when processing local folders.

## Main Components

- `bird_pipeline.py`: Core batch pipeline (load models, detect, crop, classify, metadata tagging, exports).
- `bird_gallery_frontend.py`: Streamlit app for running inference and browsing run outputs.
- `nabirds_frontend.py`: Streamlit app for checkpoint comparison/evaluation on NABirds test data.
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
6. Run ResNet classifier and keep top-1 species + confidence (top-5 also computed).
7. Optionally tag original local JPEG metadata via `exiftool` (including sharpness).
8. Save run outputs (`results.csv`, `results.json`, crops, logs, errors if any).

## Requirements

### System

- Python 3.10+ recommended
- `exiftool` (required only for metadata tagging to original files)

### Python packages

Install the core packages used by the codebase:

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

In the sidebar:

- Select a classifier checkpoint (`artifacts/*.pt`).
- Set YOLO weights and confidence threshold.
- Choose input mode:
  - `Upload files`: process uploaded JPEGs.
  - `Folder path`: process JPEGs from a local folder (supports recursive scan).
- Toggle metadata tagging:
  - Tagging is applied only in `Folder path` mode.
  - Upload mode cannot tag original files by design.

## Metadata for Lightroom

When enabled and supported (`Folder path` + `exiftool` available), the pipeline writes:

- `XMP-lr:HierarchicalSubject` keywords
- `EXIF:UserComment`

Keyword paths are hierarchical (using Lightroom's `|` separators), which makes the fields easier to browse/filter in Lightroom's Keyword List:

- `Species|<predicted species>`
- `Species Confidence|<high|medium|low>`
- `Sharpness Score|<0-100>`
- `Sharpness|<high|medium|low>`

The plain top-level species keyword is intentionally not written.

## Run Outputs

Each run creates a timestamped folder under:

`artifacts/pipeline_runs/<run_id>/`

Typical contents:

- `originals/`: copied originals used for the run
- `crops/`: detected bird crops
- `results.csv`: row-level structured outputs
- `results.json`: JSON export of results
- `errors.csv`: written when errors occur
- `pipeline.log`: detailed execution log

## Checkpoint Evaluation / Comparison

To compare two checkpoints on NABirds test split:

```bash
streamlit run nabirds_frontend.py
```

This app provides:

- Top-5 prediction comparison on selected test images
- Full test evaluation (top-1 / top-5)
- Per-species top-1 accuracy tables

## Training Artifacts and Notebooks

Training and experimentation are primarily notebook-driven:

- `resnet.ipynb`: ResNet training/fine-tuning workflow
- `overnight.ipynb`: multi-run experiment runner
- `bird_infer_pipeline.ipynb`: inference walkthrough

Saved training metrics/configs are tracked in:

- `artifacts/logs/run_summary.csv`
- `artifacts/logs/epoch_metrics.csv`
- `artifacts/logs/run_configs.jsonl`

## Notes

- Current default species label file is `artifacts/label_names.csv` (98 classes).
- An alternate fine-grained label map also exists: `artifacts/label_names_nabirds_all_specific.csv` (555 classes).
- YOLO bird detection currently relies on class name matching to `"bird"` in YOLO outputs.
