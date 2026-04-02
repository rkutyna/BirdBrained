# Repository Map

Comprehensive map of the Bird Photography Processing Pipeline repository.

## Directory Tree

```
capstone/
├── frontend/                        Streamlit applications
│   ├── bird_gallery_frontend.py     Main app (Classify + Training tabs)
│   ├── nabirds_frontend.py          Checkpoint comparison app
│   └── label_photos.py              Manual photo labeling tool
│
├── training/                        Training orchestration
│   ├── autorun.py                   Codex-driven autoresearch loop
│   └── training_engine.py           Background training subprocess + job queue
│
├── dataprep/                        Dataset preparation
│   ├── prepare.py                   NABirds train/val/test splits
│   ├── prepare_birdsnap.py          Birdsnap download from HuggingFace
│   ├── prepare_inat.py              iNaturalist API download
│   └── prepare_combined.py          Merge NABirds + external datasets
│
├── inference/                       Core inference pipeline
│   └── bird_pipeline.py             YOLO detection + ResNet classification + sharpness + metadata
│
├── tools/                           Standalone utilities
│   ├── yolo_test.py                 YOLO bird detection + crop extraction
│   ├── eval_missing_checkpoints.py  Retroactive checkpoint evaluation
│   ├── monitor.py                   Live terminal monitor for autoresearch
│   └── replay_train_variants.py     Replay saved train.py variants
│
├── notebooks/                       Jupyter notebooks (historical/exploratory)
│   ├── resnet.ipynb                 ResNet training development
│   ├── overnight.ipynb              Extended training experiments
│   ├── bird_infer_pipeline.ipynb    Inference pipeline development
│   ├── visualize_runs.ipynb         Training run visualization
│   └── vit_test.ipynb               Vision Transformer experiments
│
├── prompts/                         Codex prompt templates
│   └── autoresearch_codex_prompt.txt
│
├── artifacts/                       Generated outputs (mostly gitignored)
│   ├── labels/                      Species label CSVs
│   │   ├── label_names.csv          98 target species
│   │   ├── label_names_nabirds_all_specific.csv   555 NABirds-specific classes
│   │   └── label_names_nabirds_base_species.csv   404 base species
│   ├── splits/                      Cached train/val/test DataFrames
│   │   ├── subset98.pkl, full555.pkl, base_species.pkl
│   │   └── base_combined.pkl, subset98_combined.pkl
│   ├── resnet50/                    Model checkpoints (gitignored)
│   │   ├── subset98/               98-species autoresearch outputs
│   │   │   ├── best.pt             Best checkpoint
│   │   │   └── experiment_log.csv  Experiment history
│   │   ├── full555/                555-species outputs
│   │   ├── base_combined/          Base combined outputs
│   │   ├── subset98_combined/      Subset98 combined outputs
│   │   └── runs/                   Individual training run checkpoints
│   ├── external/                    External dataset caches
│   │   ├── birdsnap_splits.pkl     Birdsnap training data
│   │   ├── inat_splits.pkl         iNaturalist training data
│   │   └── inat_manifest.json      iNat API manifest
│   ├── autoresearch_status.json     Current autoresearch run status
│   ├── autoresearch_progress.json   Live training progress
│   ├── autoresearch_runner/         Per-iteration artifacts (train.py snapshots, decisions)
│   │   └── history.jsonl            Iteration-by-iteration metadata
│   ├── logs/                        Training logs (gitignored)
│   │   ├── run_summary.csv          Consolidated run results
│   │   ├── run_queue.json           Job queue state
│   │   ├── progress_*.json          Real-time progress (frontend reads)
│   │   └── training_*.log           Subprocess logs
│   └── pipeline_runs/<run_id>/      Inference outputs (gitignored)
│       ├── originals/               Input image copies
│       ├── crops/                   Detected bird crops
│       ├── results.csv / .json      Per-image results
│       ├── errors.csv               Error log
│       └── pipeline.log             Execution log
│
├── .streamlit/config.toml           Streamlit theme
│
├── train.py                         Self-contained training script (Codex-managed, stays at root)
├── nabirds_common.py                Shared constants and utilities
├── program.md                       Codex agent instructions
├── yolo11n.pt                       YOLO 11n detection weights
├── README.md                        Project documentation
├── REPO_MAP.md                      This file
└── .gitignore
```

## File Details

### Root (stays at root for architectural reasons)

| File | Purpose | Why at root |
|------|---------|-------------|
| `train.py` | Self-contained ResNet-50 training script. Configurable stages, LR, augmentation, regularization. Logs to `artifacts/resnet50/<species>/experiment_log.csv`. | Codex autoresearch agent modifies this file directly; `autorun.py` references it by path. |
| `nabirds_common.py` | Shared constants (`TARGET_SIZE`, `IMAGENET_MEAN/STD`, `SEED`, dataset paths) and helpers (`canonicalize_name`, `crop_resize_pad_bbox`). | Imported by files in multiple subdirectories; root placement keeps it universally importable. |
| `program.md` | Instructions for the Codex autoresearch agent — what it can/cannot modify, config sections, output format. | Referenced by `training/autorun.py` as `ROOT / "program.md"`. |
| `yolo11n.pt` | YOLO 11n model weights for bird detection. | Used by `inference/bird_pipeline.py` at runtime. |

### frontend/

| File | Lines | Purpose | Local imports |
|------|-------|---------|---------------|
| `bird_gallery_frontend.py` | ~1,600 | Main Streamlit app with Classify tab (batch inference, ground-truth eval, metadata tagging) and Training tab (single runs, sweeps, job queue, live progress). | `inference.bird_pipeline`, `training.training_engine` |
| `nabirds_frontend.py` | ~800 | Streamlit app for side-by-side checkpoint comparison on NABirds test split. Browse images, compare top-5 predictions, per-species accuracy. | `inference.bird_pipeline` |
| `label_photos.py` | ~150 | Streamlit tool for manually labeling bird photos. Navigate folder, search/select from 404 species, saves `labels.csv`. | None |

### training/

| File | Lines | Purpose | Local imports |
|------|-------|---------|---------------|
| `autorun.py` | ~700 | Outer-loop Codex-driven autoresearch runner. Launches `codex exec` sessions, snapshots `train.py`, enforces keep/restore based on log improvement. Git integration for commits. | None (orchestrator) |
| `training_engine.py` | ~750 | Dual role: (1) importable job queue helpers (`TrainingConfig`, `add_to_queue`, `load_queue`, `get_progress`, etc.) and (2) subprocess training runner (`python training/training_engine.py --job-id <id>`). | `nabirds_common` |

### dataprep/

| File | Lines | Purpose | Local imports |
|------|-------|---------|---------------|
| `prepare.py` | ~240 | One-time setup: parses NABirds metadata, builds train/val/test split DataFrames, caches as pickles. Supports subset98, full555, base_species modes. | `nabirds_common` |
| `prepare_birdsnap.py` | ~320 | Downloads Birdsnap (~50K images, 500 species) from HuggingFace, maps to NABirds base_species (~335 match), saves training-only pickle. | None |
| `prepare_inat.py` | ~570 | Queries iNaturalist API for research-grade bird photos, downloads up to 280/species, multi-threaded, creates training-only pickle. | None |
| `prepare_combined.py` | ~240 | Merges NABirds + Birdsnap + iNaturalist splits. Modes: `base_combined` (404 classes) or `subset98_combined` (98 classes). Val/test stay NABirds-only. | None |

### inference/

| File | Lines | Purpose | Local imports |
|------|-------|---------|---------------|
| `bird_pipeline.py` | ~1,160 | Core pipeline: YOLO detection -> full-res crop -> ResNet classification -> Tenengrad sharpness -> exiftool metadata tagging. Exports `PipelineConfig`, `load_classifier`, `run_inference_batch`, `classify_crops_batch`, etc. | None |

### tools/

| File | Lines | Purpose | Local imports |
|------|-------|---------|---------------|
| `yolo_test.py` | ~150 | Standalone YOLO bird detection. CLI: specify folder, saves top-3 crops per image. Supports JPEG/PNG/BMP/TIFF/RAW. | None |
| `eval_missing_checkpoints.py` | ~350 | Scans `artifacts/resnet50/` for checkpoints not in `run_summary.csv`, evaluates on NABirds test split, appends results. | None |
| `monitor.py` | ~230 | Live terminal display of autoresearch progress (ANSI colors, auto-refresh). Reads status JSON, log CSV, progress JSON. | None |
| `replay_train_variants.py` | ~350 | Replay historical `train.py` variants or manifest-defined config deltas for controlled re-evaluation. | None |

## Dependency Graph

```
frontend/bird_gallery_frontend.py
├── imports: inference/bird_pipeline.py
├── imports: training/training_engine.py
└── subprocess: training/training_engine.py --job-id <id>

frontend/nabirds_frontend.py
└── imports: inference/bird_pipeline.py

training/training_engine.py
└── imports: nabirds_common.py

training/autorun.py
├── reads/modifies: train.py
├── reads: program.md
├── reads: artifacts/resnet50/<species>/experiment_log.csv
├── reads: prompts/autoresearch_codex_prompt.txt
└── subprocess: codex exec (which runs train.py)

dataprep/prepare.py
└── imports: nabirds_common.py

inference/bird_pipeline.py
└── (standalone — inlines its own constants)

train.py
└── (standalone — intentionally self-contained for Codex)
```

## Data Flow

### Training Pipeline (Autoresearch)
```
dataprep/prepare.py ──> artifacts/splits/*.pkl
                                    │
training/autorun.py ──> modifies train.py config
                                    │
                              train.py ──> artifacts/resnet50/<species>/experiment_log.csv
                                    │         artifacts/resnet50/<species>/best.pt
                                    │
training/autorun.py ──> reads log, decides keep/restore train.py
                              (loop)
```

### Training Pipeline (Frontend)
```
frontend/bird_gallery_frontend.py
    │ Training Tab: configure + submit
    ▼
training/training_engine.py (queue helpers)
    │ add_to_queue() → artifacts/logs/run_queue.json
    ▼
training/training_engine.py (subprocess)
    │ python training/training_engine.py --job-id <id>
    │ → progress_<id>.json (live updates)
    │ → artifacts/logs/run_summary.csv
    ▼
frontend/bird_gallery_frontend.py
    │ reads progress, displays live bars
```

### Inference Pipeline
```
frontend/bird_gallery_frontend.py
    │ Classify Tab: upload/folder input
    ▼
inference/bird_pipeline.py
    │
    ├── YOLO 11n ──> bird bounding boxes
    ├── Full-res crop
    ├── ResNet-50 ──> species top-5 + confidence
    ├── Tenengrad ──> sharpness 0-100
    └── exiftool ──> XMP/EXIF/IPTC metadata on originals
    │
    ▼
artifacts/pipeline_runs/<run_id>/
    results.csv, results.json, crops/, pipeline.log
```

### External Data Preparation
```
dataprep/prepare.py
    └── artifacts/splits/base_species.pkl

dataprep/prepare_birdsnap.py ──> artifacts/external/birdsnap_splits.pkl
dataprep/prepare_inat.py     ──> artifacts/external/inat_splits.pkl

dataprep/prepare_combined.py
    ├── reads: artifacts/splits/*.pkl + external/*.pkl
    └── writes: artifacts/splits/*_combined.pkl
```

## Entry Points

All commands should be run from the project root directory.

| Command | Purpose |
|---------|---------|
| `streamlit run frontend/bird_gallery_frontend.py` | Main app (inference + training) |
| `streamlit run frontend/nabirds_frontend.py` | Checkpoint comparison |
| `streamlit run frontend/label_photos.py` | Photo labeling |
| `python train.py` | Run one training experiment |
| `python training/autorun.py --iterations 20 --danger-full-access` | Autoresearch loop |
| `python tools/monitor.py` | Live autoresearch monitor |
| `python dataprep/prepare.py` | Build NABirds splits (one-time) |
| `python dataprep/prepare_birdsnap.py` | Download Birdsnap |
| `python dataprep/prepare_inat.py` | Download iNaturalist photos |
| `python dataprep/prepare_combined.py` | Merge datasets |
| `python tools/eval_missing_checkpoints.py` | Evaluate unlogged checkpoints |
| `python tools/yolo_test.py <folder>` | Standalone YOLO detection |
| `python tools/replay_train_variants.py <manifest.json>` | Replay experiments |

## Key Architectural Decisions

1. **`train.py` is intentionally self-contained** — it duplicates constants from `nabirds_common.py` so the Codex autoresearch agent can modify a single file without touching imports.

2. **`nabirds_common.py` centralises shared values** for the human-maintained code (`training_engine.py`, `prepare.py`, `bird_pipeline.py`), keeping them in sync.

3. **All scripts use CWD-relative paths** (`Path("artifacts")`, `Path("NABirds_Dataset/nabirds")`) — always run from the project root.

4. **Entry points in subdirectories include a `sys.path` fix** to ensure the project root is importable regardless of how the script is launched.
