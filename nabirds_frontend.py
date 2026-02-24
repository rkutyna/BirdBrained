from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


TARGET_SIZE = 240
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_PAD_RGB = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)


@dataclass
class FrontendData:
    label_names: List[str]
    test_df: pd.DataFrame
    species_to_idx: Dict[str, int]


class NABirdsEvalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.df = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = crop_resize_pad(
            img,
            bbox=(row["x"], row["y"], row["w"], row["h"]),
            size=TARGET_SIZE,
            pad_rgb=IMAGENET_PAD_RGB,
        )
        return self.transform(img), int(row["target"])


def canonicalize_name(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)\s*", " ", name)
    name = name.lower()
    name = name.replace("grey", "gray")
    name = name.replace("orioles", "oriole")
    name = name.replace("-", " ")
    name = name.replace("'", "")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def parse_classes_file(classes_path: Path) -> pd.DataFrame:
    rows = []
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, cname = line.split(maxsplit=1)
            rows.append((int(cid), cname))
    return pd.DataFrame(rows, columns=["class_id", "class_name"])


def crop_resize_pad(
    img: Image.Image,
    bbox: Tuple[float, float, float, float],
    size: int = 240,
    pad_rgb: Tuple[int, int, int] = (124, 116, 104),
) -> Image.Image:
    x, y, w, h = bbox
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(img.width, int(np.ceil(x + w)))
    y2 = min(img.height, int(np.ceil(y + h)))

    if x2 <= x1 or y2 <= y1:
        cropped = img
    else:
        cropped = img.crop((x1, y1, x2, y2))

    scale = min(size / cropped.width, size / cropped.height)
    new_w = max(1, int(round(cropped.width * scale)))
    new_h = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    return ImageOps.expand(
        resized,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=pad_rgb,
    )


@st.cache_data(show_spinner=False)
def load_frontend_data(dataset_root: str, artifacts_dir: str) -> FrontendData:
    data_root = Path(dataset_root)
    artifacts = Path(artifacts_dir)

    labels_path = artifacts / "label_names.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")

    label_names = pd.read_csv(labels_path)["species"].tolist()
    species_to_idx = {s: i for i, s in enumerate(label_names)}

    images = pd.read_csv(data_root / "images.txt", sep=" ", names=["image_id", "image_rel_path"])
    labels = pd.read_csv(data_root / "image_class_labels.txt", sep=" ", names=["image_id", "class_id"])
    splits = pd.read_csv(data_root / "train_test_split.txt", sep=" ", names=["image_id", "is_train"])
    bboxes = pd.read_csv(data_root / "bounding_boxes.txt", sep=" ", names=["image_id", "x", "y", "w", "h"])
    classes = parse_classes_file(data_root / "classes.txt")

    valid_class_ids = set(labels["class_id"].unique())
    classes = classes[classes["class_id"].isin(valid_class_ids)].copy()
    classes["canon"] = classes["class_name"].map(canonicalize_name)

    canon_to_class_ids: Dict[str, List[int]] = {}
    for canon, grp in classes.groupby("canon"):
        canon_to_class_ids[canon] = sorted(set(grp["class_id"].tolist()))

    class_id_to_species_idx: Dict[int, int] = {}
    missing_species = []
    for species_name in label_names:
        canon = canonicalize_name(species_name)
        matched_class_ids = canon_to_class_ids.get(canon, [])
        if not matched_class_ids:
            missing_species.append(species_name)
            continue
        y = species_to_idx[species_name]
        for class_id in matched_class_ids:
            class_id_to_species_idx[class_id] = y

    if missing_species:
        raise RuntimeError(
            "Could not map these trained species back to NABirds classes: "
            + ", ".join(missing_species)
        )

    df = images.merge(labels, on="image_id", how="inner")
    df = df.merge(splits, on="image_id", how="inner")
    df = df.merge(bboxes, on="image_id", how="inner")

    df = df[df["class_id"].isin(class_id_to_species_idx)].copy()
    df["target"] = df["class_id"].map(class_id_to_species_idx)
    idx_to_species = {v: k for k, v in species_to_idx.items()}
    df["species"] = df["target"].map(idx_to_species)
    df["image_path"] = df["image_rel_path"].map(lambda p: str(data_root / "images" / p))

    df["is_train"] = pd.to_numeric(df["is_train"], errors="coerce").fillna(-1).astype(int)
    test_df = df[df["is_train"] == 0].copy().reset_index(drop=True)

    return FrontendData(
        label_names=label_names,
        test_df=test_df,
        species_to_idx=species_to_idx,
    )


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, num_classes: int, device_name: str):
    device = torch.device(device_name)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def get_device_choice() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def predict_topk(
    model: torch.nn.Module,
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    labels: List[str],
    device: torch.device,
    k: int = 5,
):
    x = crop_resize_pad(image, bbox=bbox, size=TARGET_SIZE, pad_rgb=IMAGENET_PAD_RGB)
    x = eval_transform()(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.topk(probs, k=min(k, probs.numel()))

    rows = []
    for rank, (c, i) in enumerate(zip(conf.tolist(), idx.tolist()), start=1):
        rows.append(
            {
                "rank": rank,
                "species": labels[i],
                "confidence": c,
                "confidence_%": 100.0 * c,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    show_progress: bool = True,
):
    ds = NABirdsEvalDataset(test_df, transform=eval_transform())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0
    top1_correct = 0
    top5_correct = 0

    per_species_total = {}
    per_species_top1_correct = {}

    progress = st.progress(0.0, text="Running evaluation...") if show_progress else None

    processed = 0
    with torch.no_grad():
        for images, targets in dl:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top1 = probs.argmax(dim=1)
            k = min(5, probs.shape[1])
            topk_idx = torch.topk(probs, k=k, dim=1).indices

            top1_hits = top1.eq(targets)
            top5_hits = topk_idx.eq(targets.unsqueeze(1)).any(dim=1)

            bs = targets.size(0)
            total += bs
            top1_correct += int(top1_hits.sum().item())
            top5_correct += int(top5_hits.sum().item())

            for y, hit in zip(targets.cpu().tolist(), top1_hits.cpu().tolist()):
                per_species_total[y] = per_species_total.get(y, 0) + 1
                per_species_top1_correct[y] = per_species_top1_correct.get(y, 0) + int(hit)

            processed += bs
            if progress is not None:
                progress.progress(min(processed / max(1, len(ds)), 1.0), text=f"Running evaluation... {processed}/{len(ds)}")

    if progress is not None:
        progress.empty()

    return {
        "n_samples": total,
        "top1_acc": top1_correct / max(1, total),
        "top5_acc": top5_correct / max(1, total),
        "per_species_total": per_species_total,
        "per_species_top1_correct": per_species_top1_correct,
    }


def per_species_stats_table(
    results: dict,
    label_names: List[str],
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for idx, species in enumerate(label_names):
        n = results["per_species_total"].get(idx, 0)
        c = results["per_species_top1_correct"].get(idx, 0)
        acc = (c / n) if n > 0 else np.nan
        rows.append(
            {
                "model": model_name,
                "species": species,
                "n_test": n,
                "top1_correct": c,
                "top1_acc": acc,
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="NABirds ResNet Viewer", layout="wide")
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap');
:root {
  --bg: #eef5f1;
  --surface: #f8fcfa;
  --surface-2: #ffffff;
  --ink: #153126;
  --muted: #4f6f62;
  --accent: #1f8f61;
  --accent-dark: #126845;
  --accent-soft: #d9efe4;
  --border: #c7e2d4;
}
html, body, [class*="css"]  {
  font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
  color: var(--ink) !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"] {
  color: var(--ink) !important;
}
.stApp {
  background:
    radial-gradient(1200px 540px at 8% -8%, #d4ecdf 0%, transparent 55%),
    radial-gradient(900px 460px at 100% 0%, #e1f3ea 0%, transparent 50%),
    var(--bg);
}
h1, h2, h3, h4, h5, h6, p, label, small {
  color: var(--ink) !important;
}

a {
  color: var(--accent-dark) !important;
}

h1, h2, h3 {
  letter-spacing: 0.01em;
  color: var(--ink);
}
h1 {
  font-weight: 700;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f3fbf7 0%, #e9f5ef 100%);
  border-right: 1px solid var(--border);
  color: var(--ink) !important;
}

section[data-testid="stSidebar"] * {
  color: var(--ink) !important;
}

div[data-testid="stForm"],
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 0.6rem 0.75rem;
}

div[data-testid="stMetric"] {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.4rem 0.5rem;
}

div.stButton > button {
  background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dark) 100%);
  color: #ffffff;
  border: 0;
  border-radius: 10px;
  font-weight: 600;
}

div.stButton > button:hover {
  filter: brightness(1.05);
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}

[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid var(--border);
}

[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stDataFrame"] *,
[data-testid="stMetric"] * {
  color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stCaptionContainer"] {
  color: var(--ink) !important;
  background: transparent !important;
}

/* Ensure inline code / code blocks (e.g., image paths) are readable */
code, kbd, samp {
  background: #eaf5ef !important;
  color: #123d2c !important;
  border: 1px solid #c7e2d4 !important;
  border-radius: 6px !important;
  padding: 0.08rem 0.35rem !important;
}

pre, [data-testid="stCodeBlock"] {
  background: #edf7f2 !important;
  border: 1px solid #c7e2d4 !important;
  border-radius: 10px !important;
}

pre code, [data-testid="stCodeBlock"] code {
  background: transparent !important;
  color: #123d2c !important;
  border: 0 !important;
  padding: 0 !important;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
input, textarea {
  background: #ffffff !important;
  color: var(--ink) !important;
}

/* Dropdown menus: dark surface + light text for readability */
[data-baseweb="select"] > div {
  background: #184f3a !important;
  border-color: #0f3a2a !important;
  color: #f3fff8 !important;
}

[data-baseweb="select"] *,
[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="select"] svg,
[data-baseweb="select"] div {
  color: #f3fff8 !important;
  fill: #f3fff8 !important;
}

div[role="listbox"] {
  background: #184f3a !important;
  border: 1px solid #0f3a2a !important;
}

div[role="listbox"] *,
ul[role="listbox"] *,
div[role="option"],
li[role="option"] {
  color: #f3fff8 !important;
}

div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
  background: #237553 !important;
  color: #ffffff !important;
}

.green-chip {
  display: inline-block;
  background: var(--accent-soft);
  color: var(--accent-dark);
  border: 1px solid #b6dcc8;
  border-radius: 999px;
  padding: 0.1rem 0.55rem;
  font-size: 0.82rem;
  margin-left: 0.35rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("NABirds Test Viewer: Checkpoint A vs Checkpoint B")

    st.markdown(
        "Inspect test images, compare top-5 predictions for two checkpoints, "
        "and view overall/per-species accuracy metrics."
    )
    st.markdown(
        '<span class="green-chip">Theme: Emerald</span>'
        '<span class="green-chip">BBox Crop + 240x240</span>'
        '<span class="green-chip">Top-5 + Species Metrics</span>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Paths")
        dataset_root = st.text_input("NABirds root", value="NABirds Dataset/nabirds")
        artifacts_dir = st.text_input("Artifacts dir", value="artifacts")
        artifacts_path = Path(artifacts_dir)
        ckpt_files = sorted(artifacts_path.glob("*.pt"))
        ckpt_options = [str(p) for p in ckpt_files]

        default_a = str(artifacts_path / "resnet50_nabirds_head_only.pt")
        default_b = str(artifacts_path / "resnet50_nabirds_layer4_finetuned.pt")
        default_a_idx = ckpt_options.index(default_a) if default_a in ckpt_options else 0
        default_b_idx = ckpt_options.index(default_b) if default_b in ckpt_options else 0

        if not ckpt_options:
            st.error(f"No .pt checkpoints found in `{artifacts_dir}`.")
            return

        ckpt_a = st.selectbox(
            "Checkpoint A",
            options=ckpt_options,
            index=default_a_idx,
        )
        ckpt_b = st.selectbox(
            "Checkpoint B",
            options=ckpt_options,
            index=default_b_idx,
        )

        device_name = st.selectbox(
            "Device",
            options=[get_device_choice(), "cpu", "cuda", "mps"],
            index=0,
        )

        eval_batch_size = st.slider("Eval batch size", min_value=8, max_value=128, value=32, step=8)

    try:
        data = load_frontend_data(dataset_root=dataset_root, artifacts_dir=artifacts_dir)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    ckpt_a_exists = Path(ckpt_a).exists()
    ckpt_b_exists = Path(ckpt_b).exists()

    if not ckpt_a_exists:
        st.error(f"Checkpoint A not found: {ckpt_a}")
        return
    if not ckpt_b_exists:
        st.error(f"Checkpoint B not found: {ckpt_b}")
        return

    try:
        model_a = load_model(ckpt_a, num_classes=len(data.label_names), device_name=device_name)
    except Exception as e:
        st.error(f"Failed to load Checkpoint A model: {e}")
        return

    try:
        model_b = load_model(ckpt_b, num_classes=len(data.label_names), device_name=device_name)
    except Exception as e:
        st.error(f"Failed to load Checkpoint B model: {e}")
        return

    if ckpt_a == ckpt_b:
        st.warning("Checkpoint A and B are the same file; select two different checkpoints to compare.")

    st.subheader("Test Image Viewer")

    species_options = ["(all)"] + sorted(data.test_df["species"].unique().tolist())
    selected_species = st.selectbox("Filter by ground-truth species", options=species_options)

    filtered_df = data.test_df
    if selected_species != "(all)":
        filtered_df = filtered_df[filtered_df["species"] == selected_species].reset_index(drop=True)

    if filtered_df.empty:
        st.warning("No images match current filter.")
        return

    idx = st.slider("Test sample index", min_value=0, max_value=len(filtered_df) - 1, value=0)
    row = filtered_df.iloc[idx]

    img = Image.open(row["image_path"]).convert("RGB")
    bbox = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
    cropped = crop_resize_pad(img, bbox=bbox, size=TARGET_SIZE, pad_rgb=IMAGENET_PAD_RGB)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.caption("Original image")
        st.image(img, use_container_width=True)
    with col_img2:
        st.caption("Model input (bbox crop -> resize+pad 240x240)")
        st.image(cropped, use_container_width=True)

    st.write(f"Ground truth: **{row['species']}**")
    st.write(f"Image path: `{row['image_path']}`")

    device = torch.device(device_name)
    pred_a = predict_topk(model_a, img, bbox, data.label_names, device=device, k=5)
    pred_b = predict_topk(model_b, img, bbox, data.label_names, device=device, k=5)
    ckpt_a_name = Path(ckpt_a).name
    ckpt_b_name = Path(ckpt_b).name

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Checkpoint A Top-5**  \n`{ckpt_a_name}`")
        st.dataframe(pred_a, hide_index=True, use_container_width=True)
        st.write(
            "Top-1 correct:",
            bool(pred_a.iloc[0]["species"] == row["species"]),
        )

    with c2:
        st.markdown(f"**Checkpoint B Top-5**  \n`{ckpt_b_name}`")
        st.dataframe(pred_b, hide_index=True, use_container_width=True)
        st.write(
            "Top-1 correct:",
            bool(pred_b.iloc[0]["species"] == row["species"]),
        )

    st.subheader("Evaluation Metrics")
    run_eval = st.button("Run full test-split evaluation")

    if run_eval:
        with st.spinner("Evaluating Checkpoint A..."):
            res_a = evaluate_model(
                model_a,
                data.test_df,
                device=device,
                batch_size=eval_batch_size,
                show_progress=True,
            )

        with st.spinner("Evaluating Checkpoint B..."):
            res_b = evaluate_model(
                model_b,
                data.test_df,
                device=device,
                batch_size=eval_batch_size,
                show_progress=True,
            )

        overall_rows = [
            {
                "model": f"A: {ckpt_a_name}",
                "n_test": res_a["n_samples"],
                "top1_acc": res_a["top1_acc"],
                "top5_acc": res_a["top5_acc"],
            },
            {
                "model": f"B: {ckpt_b_name}",
                "n_test": res_b["n_samples"],
                "top1_acc": res_b["top1_acc"],
                "top5_acc": res_b["top5_acc"],
            },
        ]

        overall_df = pd.DataFrame(overall_rows)
        st.markdown("**Overall metrics**")
        st.dataframe(overall_df, hide_index=True, use_container_width=True)

        st.markdown("**Per-species top-1 accuracy**")
        per_species_df = pd.concat(
            [
                per_species_stats_table(res_a, data.label_names, f"A: {ckpt_a_name}"),
                per_species_stats_table(res_b, data.label_names, f"B: {ckpt_b_name}"),
            ],
            ignore_index=True,
        )

        species_filter = st.selectbox(
            "Per-species table filter",
            options=["all", "lowest-20 checkpoint A", "lowest-20 checkpoint B"],
        )

        model_a_label = f"A: {ckpt_a_name}"
        model_b_label = f"B: {ckpt_b_name}"

        if species_filter == "lowest-20 checkpoint A":
            keep = (
                per_species_df[per_species_df["model"] == model_a_label]
                .sort_values("top1_acc", na_position="last")
                .head(20)["species"]
                .tolist()
            )
            per_species_df = per_species_df[per_species_df["species"].isin(keep)]
        elif species_filter == "lowest-20 checkpoint B":
            keep = (
                per_species_df[per_species_df["model"] == model_b_label]
                .sort_values("top1_acc", na_position="last")
                .head(20)["species"]
                .tolist()
            )
            per_species_df = per_species_df[per_species_df["species"].isin(keep)]

        st.dataframe(
            per_species_df.sort_values(["model", "top1_acc"], na_position="last"),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
