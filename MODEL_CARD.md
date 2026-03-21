---
license: mit
tags:
  - image-classification
  - birds
  - resnet
  - pytorch
  - wildlife
datasets:
  - nabirds
  - birdsnap
  - inaturalist
pipeline_tag: image-classification
---

# Bird Species Classifier (ResNet-50)

Fine-tuned ResNet-50 models for classifying North American bird species from cropped bird photographs.

## Model Description

These models are ResNet-50 backbones pretrained on ImageNet V2, fine-tuned on the [NABirds](https://dl.allawnmilner.com/nabirds) dataset augmented with [Birdsnap](https://thomasberg.org/) and [iNaturalist](https://www.inaturalist.org/) data. They are designed for use in a photography processing pipeline that first detects birds with YOLO, crops them at full resolution, then classifies the crop.

### Architecture

- **Backbone**: ResNet-50 (ImageNet V2 pretrained)
- **Pooling**: Generalized Mean (GeM) pooling
- **Head**: `Sequential(Dropout(0.4), Linear(2048, num_classes))`
- **Input size**: 240x240 pixels, normalized with ImageNet mean/std
- **Preprocessing**: `ToTensor()` + `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`

### Training Strategy

Three-stage progressive unfreezing:

| Stage | Unfrozen Layers | Purpose |
|-------|-----------------|---------|
| 1 | FC head only | Learn species mapping on frozen backbone features |
| 2 | `layer4` + FC | Adapt high-level features |
| 3 | `layer3` + `layer4` + FC | Fine-tune mid-level features |

Training was conducted using an automated research loop (Codex-driven) with 2-hour time budgets per experiment for the 98-species model and 4-10 hour budgets for the 404-species model.

## Available Checkpoints

### `subset98_combined/best.pt` — 98 Target Species

| Metric | Value |
|--------|-------|
| Top-1 Test Accuracy | **97.4%** |
| Top-1 Val Accuracy | 97.6% |
| Classes | 98 target species |
| Training Data | NABirds + Birdsnap + iNaturalist (~38K training images) |
| Total Epochs | 12 |
| Training Time | 2 hours |
| Peak Memory | 589 MB |
| File Size | ~91 MB |

Best run: `20260319_074647_c9dbe6` — stage3 cap=6 + layer2 lr=1.5e-5

### `base_combined/best.pt` — 404 Base Species

| Metric | Value |
|--------|-------|
| Top-1 Test Accuracy | **93.6%** |
| Top-1 Val Accuracy | 93.6% |
| Classes | 404 NABirds base species (sex/morph variants collapsed) |
| Training Data | NABirds + Birdsnap + iNaturalist (~166K training images) |
| Total Epochs | 20 |
| Training Time | ~9.6 hours |
| Peak Memory | 898 MB |
| Batch Size | 128 |
| File Size | ~98 MB |

Best run: `20260319_234135_b8fe6e` — bs=128 + stage lrs 3e-4/6e-5

## Usage

### With the Bird Photography Pipeline

```bash
git clone <repo-url>
cd capstone
pip install -r requirements.txt
python download_models.py
streamlit run frontend/bird_gallery_frontend.py
```

### Standalone Inference (PyTorch)

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load checkpoint
state_dict = torch.load("subset98_combined/best.pt", map_location="cpu")

# Build model
model = models.resnet50()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(model.fc.in_features, 98),  # or 404 for base_combined
)
model.load_state_dict(state_dict)
model.eval()

# Preprocess a cropped bird image
transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

img = Image.open("bird_crop.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    top5_probs, top5_indices = probs.topk(5)
```

Label names are provided in the repository as CSV files:
- `label_names.csv` — 98 target species
- `label_names_nabirds_base_species.csv` — 404 base species

## Training Data

| Dataset | Images | Species | Role |
|---------|--------|---------|------|
| [NABirds](https://dl.allawnmilner.com/nabirds) | ~48K | 555 specific / 404 base | Train + Val + Test |
| [Birdsnap](https://thomasberg.org/) | ~50K | ~335 matched | Train only |
| [iNaturalist](https://www.inaturalist.org/) | ~70K | up to 280/species | Train only |

Validation and test splits use NABirds data only (no external data leakage).

## Limitations

- Trained on North American bird species only (NABirds taxonomy).
- Expects **cropped bird images** as input — not full scene photos. Use a bird detector (e.g., YOLO) to crop first.
- The 98-species model covers only a curated subset; out-of-distribution species will be misclassified into the nearest known class.
- Performance may degrade on heavily backlit, motion-blurred, or partially occluded subjects.

## Citation

If you use these models, please cite the NABirds dataset:

```bibtex
@inproceedings{van2015building,
  title={Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection},
  author={Van Horn, Grant and Branson, Steve and Farrell, Ryan and Haber, Scott and Barry, Jessie and Ipeirotis, Panos and Perona, Pietro and Belongie, Serge},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={595--604},
  year={2015}
}
```
