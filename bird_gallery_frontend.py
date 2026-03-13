from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from bird_pipeline import (
    ALL_SPECIFIC_LABEL_NAMES_CSV,
    DEFAULT_LABEL_NAMES_CSV,
    PipelineConfig,
    get_default_device,
    input_items_from_uploaded_files,
    input_items_from_folder,
    list_classifier_checkpoints,
    run_inference_batch,
)
from training_engine import (
    LOGS_DIR,
    QUEUE_JSON,
    SUMMARY_CSV,
    TrainingConfig,
    _update_queue_status,
    add_to_queue,
    cancel_running_job,
    clear_finished_jobs,
    get_progress,
    get_training_log,
    load_queue,
    remove_from_queue,
)


COLOR_MAP = {
    "green": "#0f9d58",
    "yellow": "#f4b400",
    "red": "#db4437",
}


def confidence_badge(conf: float | None, color_name: str | None) -> str:
    if conf is None or color_name is None:
        return '<span class="badge na">n/a</span>'
    color = COLOR_MAP.get(color_name, "#777")
    return f'<span class="badge" style="background:{color};">{conf*100:.1f}%</span>'


_VERDICT_STYLE = (
    "color:#fff;border-radius:999px;padding:0.15rem 0.6rem;"
    "font-weight:700;font-size:0.8rem;"
)


def _verdict_badge(bg_color: str, text: str) -> str:
    return f'<span style="background:{bg_color};{_VERDICT_STYLE}">{text}</span>'


def crop_megapixels_from_row(row: pd.Series) -> float | None:
    try:
        x1 = float(row.get("bbox_x1"))
        y1 = float(row.get("bbox_y1"))
        x2 = float(row.get("bbox_x2"))
        y2 = float(row.get("bbox_y2"))
    except Exception:
        return None

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return None
    return (w * h) / 1_000_000.0


def apply_css() -> None:
    # The base light theme (colors, fonts, widget styling) is declared in
    # .streamlit/config.toml so Streamlit renders everything consistently in
    # light mode without CSS fighting the defaults.  This stylesheet only adds
    # things config.toml cannot express: the gradient background, inline-code
    # appearance, dropdown portal menus (rendered outside Streamlit's theme
    # scope), the caret color fix, and the custom badge/card classes.
    st.markdown(
        """
<style>
/* Subtle gradient on the main canvas */
.stApp {
  background: radial-gradient(1200px 500px at 10% -10%, #e8f5ec 0%, #f7fbf8 60%);
}

/* Inline code: tinted background so backtick spans are distinct but readable */
[data-testid="stMarkdownContainer"] code {
  background: #dceee3 !important;
  color: #1a3829 !important;
  border-radius: 4px;
  padding: 0.1em 0.3em;
}

/* Explicit caret so the text cursor is visible in inputs */
input, textarea {
  caret-color: #1f3b2f !important;
}

/* Dropdown portal menus render outside .stApp so they don't inherit the
   Streamlit theme; force them to match the light palette manually. */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
  background: #ffffff !important;
}
div[role="listbox"], ul[role="listbox"] {
  background: #ffffff !important;
  border: 1px solid #b7d3c3 !important;
}
div[role="option"], li[role="option"] {
  background: #ffffff !important;
  color: #163528 !important;
}
div[role="option"]:hover, li[role="option"]:hover {
  background: #f1f9f4 !important;
}
div[role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] {
  background: #e6f4eb !important;
}

/* Custom badge and card classes used in the gallery */
.badge {
  display: inline-block;
  color: white;
  border-radius: 999px;
  padding: 0.18rem 0.6rem;
  font-weight: 700;
  font-size: 0.85rem;
}
.badge.na {
  background: #64748b;
}
.card {
  border: 1px solid #d6e7dc;
  border-radius: 12px;
  background: #ffffff;
  padding: 0.7rem;
  margin-bottom: 0.9rem;
}
.meta {
  color: #274c3a;
  font-size: 0.9rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def build_app_logger() -> logging.Logger:
    log_dir = Path("artifacts/pipeline_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bird_gallery_frontend")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_dir / "frontend.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _path_keys(path_str: str) -> set[str]:
    p = Path(path_str)
    keys = {str(p), p.name}
    try:
        keys.add(str(p.resolve()))
    except Exception:
        pass
    return keys


def _checkpoint_stage_rank(ckpt_path: str) -> int:
    """Return a stage rank so stage 3 > stage 2 > stage 1."""
    name = Path(ckpt_path).name.lower()
    if "layer3_layer4" in name:
        return 3
    if "layer4" in name:
        return 2
    return 1


def _checkpoint_test_acc(
    checkpoint_path: str,
    run_summary_csv: Path = Path("artifacts/logs/run_summary.csv"),
) -> float | None:
    """Return the best test_acc recorded for this checkpoint in run_summary.csv, or None."""
    if not run_summary_csv.exists():
        return None
    try:
        df = pd.read_csv(run_summary_csv)
        if not {"checkpoint_path", "test_acc"}.issubset(df.columns):
            return None
        keys = _path_keys(checkpoint_path)
        mask = df["checkpoint_path"].apply(
            lambda p: bool(_path_keys(str(p)) & keys) if pd.notna(p) else False
        )
        matched = pd.to_numeric(df.loc[mask, "test_acc"], errors="coerce").dropna()
        return float(matched.max()) if not matched.empty else None
    except Exception:
        return None


def _filter_checkpoints_by_mode(checkpoints: list[str], all_specific: bool) -> list[str]:
    if all_specific:
        filtered = [c for c in checkpoints if "all_specific" in Path(c).name]
    else:
        filtered = [c for c in checkpoints if "all_specific" not in Path(c).name]
    return filtered if filtered else checkpoints


def choose_best_checkpoint(
    checkpoints: Iterable[str],
    run_summary_csv: Path = Path("artifacts/logs/run_summary.csv"),
) -> str:
    ckpts = list(checkpoints)
    if not ckpts:
        raise ValueError("No checkpoints provided.")

    checkpoint_by_key: dict[str, str] = {}
    for ckpt in ckpts:
        for key in _path_keys(ckpt):
            checkpoint_by_key[key] = ckpt

    if run_summary_csv.exists():
        try:
            summary_df = pd.read_csv(run_summary_csv)
            if {"checkpoint_path", "test_acc"}.issubset(summary_df.columns):
                valid_df = summary_df.dropna(subset=["checkpoint_path", "test_acc"]).copy()
                valid_df["test_acc"] = pd.to_numeric(valid_df["test_acc"], errors="coerce")
                valid_df = valid_df.dropna(subset=["test_acc"])
                valid_df = valid_df.sort_values("test_acc", ascending=False)
                for _, row in valid_df.iterrows():
                    for key in _path_keys(str(row["checkpoint_path"])):
                        if key in checkpoint_by_key:
                            return checkpoint_by_key[key]
        except Exception:
            pass

    # Fallback: prefer stage3 > stage2 > stage1 by filename
    return max(ckpts, key=_checkpoint_stage_rank)



def render_summary(df: pd.DataFrame) -> None:
    total = len(df)
    birds = int(df["yolo_detected"].fillna(False).sum())
    no_bird = total - birds
    mean_conf = float(df["pred_confidence"].dropna().mean()) if birds > 0 else 0.0
    crop_mps = [mp for mp in (crop_megapixels_from_row(row) for _, row in df.iterrows()) if mp is not None]
    mean_crop_mp = float(np.mean(crop_mps)) if crop_mps else None
    if "sharpness_score_100" in df.columns:
        sharp_vals = pd.to_numeric(df["sharpness_score_100"], errors="coerce")
        mean_sharp = float(sharp_vals.dropna().mean()) if sharp_vals.notna().any() else None
    elif "sharpness_tenengrad" in df.columns:
        # Backward compatibility for older run outputs.
        sharp_vals = pd.to_numeric(df["sharpness_tenengrad"], errors="coerce")
        mean_sharp = float(sharp_vals.dropna().mean()) if sharp_vals.notna().any() else None
    else:
        mean_sharp = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total images", total)
    c2.metric("Bird detected", birds)
    c3.metric("No bird", no_bird)
    c4.metric("Mean top-1 confidence", f"{mean_conf*100:.1f}%")
    c5.metric("Mean Sharpness (0-100)", f"{mean_sharp:.1f}" if mean_sharp is not None else "n/a")
    c6.metric("Mean crop MP", f"{mean_crop_mp:.2f}" if mean_crop_mp is not None else "n/a")

    if "metadata_written" in df.columns:
        tagged = int(df["metadata_written"].fillna(False).sum())
        st.caption(f"Original JPEGs tagged with metadata: {tagged}")

    if "ground_truth" in df.columns:
        labeled = int(df["ground_truth"].notna().sum())
        if labeled > 0:
            in_model_mask = df["label_in_model"] == True
            correct = int((df.loc[in_model_mask, "is_correct"] == True).sum())
            wrong = int((df.loc[in_model_mask, "is_correct"] == False).sum())
            evaluable = correct + wrong
            accuracy = correct / evaluable if evaluable > 0 else None

            no_bird_excluded = int(
                ((df["label_in_model"] == False) & (df["yolo_detected"].fillna(False) == False)
                 & df["ground_truth"].notna()).sum()
            )
            out_of_scope = int(
                ((df["label_in_model"] == False) & (df["yolo_detected"].fillna(True) == True)
                 & df["ground_truth"].notna()).sum()
            )

            in_top5 = int((df.loc[in_model_mask, "is_in_top5"] == True).sum())
            top5_acc = in_top5 / evaluable if evaluable > 0 else None

            st.divider()
            st.markdown("**Ground-truth accuracy** (from labels.csv in photo folder)")
            ga1, ga2, ga3, ga4, ga5, ga6, ga7 = st.columns(7)
            ga1.metric("Labeled photos", labeled)
            ga2.metric("Correct (top-1)", correct)
            ga3.metric("Wrong", wrong)
            ga4.metric("No bird detected", no_bird_excluded,
                       help="Bird detector found nothing — excluded from accuracy")
            ga5.metric("Species not in model", out_of_scope,
                       help="Ground-truth species not in this model's label set — excluded from accuracy")
            ga6.metric(
                "Top-1 accuracy",
                f"{accuracy * 100:.1f}%" if accuracy is not None else "n/a",
                help="correct / (correct + wrong), photos with no detection or out-of-scope species excluded",
            )
            ga7.metric(
                "Top-5 accuracy",
                f"{top5_acc * 100:.1f}%" if top5_acc is not None else "n/a",
                help="Ground truth appears anywhere in model's top-5 predictions",
            )


def render_gallery(df: pd.DataFrame, default_page_size: int = 200) -> None:
    show_df = df.copy()

    has_ground_truth = "ground_truth" in show_df.columns and show_df["ground_truth"].notna().any()
    if has_ground_truth:
        col_species, col_verdict = st.columns(2)
    else:
        col_species = st

    species_options = ["(all)"] + sorted(show_df["pred_species"].dropna().unique())
    species_filter = col_species.selectbox("Filter by predicted species", options=species_options)
    if species_filter != "(all)":
        show_df = show_df[show_df["pred_species"] == species_filter].copy()

    if has_ground_truth:
        verdict_filter = col_verdict.selectbox(
            "Filter by verdict",
            options=["(all)", "Correct", "Wrong", "No bird detected", "Not in model subset", "Unlabeled"],
        )
        if verdict_filter == "Correct":
            show_df = show_df[show_df["is_correct"] == True].copy()
        elif verdict_filter == "Wrong":
            show_df = show_df[(show_df["is_correct"] == False) & (show_df["label_in_model"] == True)].copy()
        elif verdict_filter == "No bird detected":
            show_df = show_df[
                (show_df["label_in_model"] == False) & (show_df["yolo_detected"].fillna(False) == False)
                & show_df["ground_truth"].notna()
            ].copy()
        elif verdict_filter == "Not in model subset":
            show_df = show_df[
                (show_df["label_in_model"] == False) & (show_df["yolo_detected"].fillna(True) == True)
                & show_df["ground_truth"].notna()
            ].copy()
        elif verdict_filter == "Unlabeled":
            show_df = show_df[show_df["ground_truth"].isna()].copy()

    sort_options = [
        "Species (A to Z)",
        "Species confidence (high to low)",
        "Species confidence (low to high)",
        "Sharpness (high to low)",
        "Sharpness (low to high)",
    ]
    sort_choice = st.selectbox("Sort results", options=sort_options, index=1)

    if sort_choice == "Species (A to Z)":
        show_df = show_df.sort_values(
            by=["pred_species", "pred_confidence"],
            ascending=[True, False],
            na_position="last",
            kind="mergesort",
        )
    elif sort_choice == "Species confidence (high to low)":
        show_df = show_df.sort_values("pred_confidence", ascending=False, na_position="last", kind="mergesort")
    elif sort_choice == "Species confidence (low to high)":
        show_df = show_df.sort_values("pred_confidence", ascending=True, na_position="last", kind="mergesort")
    else:
        sharpness_col = "sharpness_score_100" if "sharpness_score_100" in show_df.columns else "sharpness_tenengrad"
        show_df = show_df.sort_values(
            sharpness_col,
            ascending=(sort_choice == "Sharpness (low to high)"),
            na_position="last",
            kind="mergesort",
        )

    total_rows = len(show_df)
    if total_rows == 0:
        st.warning("No gallery items match the current filter.")
        return

    page_size = st.select_slider("Results per page", options=[25, 50, 100, 200, 500], value=default_page_size)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    page_key = f"gallery_page_{species_filter}_{page_size}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = min(max(1, int(st.session_state[page_key])), total_pages)

    c_prev, c_page, c_next = st.columns([1, 2, 1])
    with c_page:
        page = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=int(st.session_state[page_key]),
            step=1,
            key=f"{page_key}_input",
        )
    page = int(page)
    with c_prev:
        prev_clicked = st.button("Prev", disabled=(page <= 1))
    with c_next:
        next_clicked = st.button("Next", disabled=(page >= total_pages))

    if prev_clicked and page > 1:
        page -= 1
    if next_clicked and page < total_pages:
        page += 1

    st.session_state[page_key] = page

    start = (page - 1) * page_size
    end = min(start + page_size, total_rows)
    page_df = show_df.iloc[start:end]

    st.caption(f"Showing {start + 1}-{end} of {total_rows} results (page {page}/{total_pages})")

    for _, row in page_df.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            left, right = st.columns([1.4, 1.2])

            with left:
                st.markdown(f"**{row['image_name']}**")
                st.markdown(f"<div class='meta'>Source: {row['source_type']} | Bird detected: {bool(row['yolo_detected'])}</div>", unsafe_allow_html=True)
                if "metadata_written" in row.index and pd.notna(row.get("metadata_written")):
                    meta_state = "yes" if bool(row.get("metadata_written")) else "no"
                    meta_note = row.get("metadata_error") if pd.notna(row.get("metadata_error")) else ""
                    st.markdown(
                        f"<div class='meta'>Metadata tagged: {meta_state}{(' | ' + str(meta_note)) if meta_note else ''}</div>",
                        unsafe_allow_html=True,
                    )
                if pd.notna(row.get("ground_truth")):
                    gt = row["ground_truth"]
                    in_model = row.get("label_in_model")
                    is_correct = row.get("is_correct")
                    if not in_model:
                        if not row.get("yolo_detected", True):
                            verdict = _verdict_badge("#64748b", "⊘ no bird detected")
                        else:
                            verdict = _verdict_badge("#b45309", "⊘ not in model subset")
                    elif is_correct:
                        verdict = _verdict_badge("#0f9d58", "✓ Correct")
                    else:
                        verdict = _verdict_badge("#db4437", "✗ Wrong")
                    st.markdown(
                        f"Ground truth: **{gt}** &nbsp; {verdict}",
                        unsafe_allow_html=True,
                    )

                if pd.notna(row.get("pred_species")):
                    st.markdown(f"Predicted species: **{row['pred_species']}**")
                    st.markdown(
                        "Confidence: "
                        + confidence_badge(
                            row.get("pred_confidence"),
                            row.get("pred_confidence_color"),
                        ),
                        unsafe_allow_html=True,
                    )
                    top5 = _parse_top5(row.get("pred_top5"))
                    if top5:
                        gt_base = _strip_qualifier(str(row.get("ground_truth", ""))) if pd.notna(row.get("ground_truth")) else None
                        rank1_conf = row.get("pred_confidence", 0) * 100
                        rank1_match = gt_base and _strip_qualifier(str(row["pred_species"])) == gt_base
                        rank1_line = f"1. {row['pred_species']} ({rank1_conf:.1f}%)"
                        if rank1_match:
                            rank1_line += " ← ground truth"

                        remaining = []
                        for e in top5[1:]:  # skip rank 1, already shown above
                            sp = e.get("species", "?")
                            cf = e.get("confidence", 0.0)
                            match = gt_base and _strip_qualifier(sp) == gt_base
                            highlight = " ← ground truth" if match else ""
                            remaining.append(f"{e['rank']}. {sp} ({cf*100:.1f}%){highlight}")
                        st.caption("Top 5:\n" + "\n".join([rank1_line] + remaining))
                    sharpness_100 = row.get("sharpness_score_100")
                    if pd.notna(sharpness_100):
                        sharpness_color = row.get("sharpness_color")
                        sharpness_level = row.get("sharpness_level")
                        st.markdown(
                            "Bird sharpness: "
                            + confidence_badge(
                                float(sharpness_100) / 100.0,
                                sharpness_color,
                            ),
                            unsafe_allow_html=True,
                        )
                        if pd.notna(sharpness_level):
                            st.markdown(f"Sharpness level: **{sharpness_level}**")
                    else:
                        # Backward compatibility for older run outputs.
                        legacy_sharpness = row.get("sharpness_tenengrad")
                        if pd.notna(legacy_sharpness):
                            st.markdown(f"Bird sharpness (legacy): **{float(legacy_sharpness):.1f}**")
                    crop_mp = crop_megapixels_from_row(row)
                    if crop_mp is not None:
                        st.markdown(f"Bird crop size: **{crop_mp:.2f} MP**")
                else:
                    st.markdown("Predicted species: **n/a**")
                    st.markdown("Confidence: " + confidence_badge(None, None), unsafe_allow_html=True)

                if pd.notna(row.get("source_path")):
                    st.caption(f"{row['source_path']}")

            with right:
                r1, r2 = st.columns(2)
                with r1:
                    if isinstance(row.get("original_path"), str) and row["original_path"] and Path(row["original_path"]).exists():
                        st.image(Image.open(row["original_path"]), caption="Original", use_container_width=True)
                    else:
                        st.info("Original unavailable")
                with r2:
                    if isinstance(row.get("crop_path"), str) and row["crop_path"] and Path(row["crop_path"]).exists():
                        st.image(Image.open(row["crop_path"]), caption="Bird crop", use_container_width=True)
                    else:
                        st.info("No crop")

            st.markdown("</div>", unsafe_allow_html=True)


def _load_folder_labels(folder_path: str) -> dict[str, str]:
    """Load labels.csv from folder_path, return {filename: species}."""
    labels_path = Path(folder_path) / "labels.csv"
    if not labels_path.exists():
        return {}
    try:
        df = pd.read_csv(labels_path)
        if {"filename", "species"}.issubset(df.columns):
            return dict(zip(df["filename"].astype(str), df["species"].astype(str)))
    except Exception:
        pass
    return {}


def _infer_labels_from_results(result_df: pd.DataFrame) -> dict[str, str]:
    """Try to find labels.csv by inspecting the source_path column of results."""
    if "source_path" not in result_df.columns:
        return {}
    parents = result_df["source_path"].dropna().map(lambda p: str(Path(p).parent))
    if parents.empty:
        return {}
    folder = parents.mode().iloc[0]
    return _load_folder_labels(folder)


def _parse_top5(value) -> list | None:
    """Return pred_top5 as a list of dicts whether it's already a list or a CSV-serialised string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None


def _strip_qualifier(name: str) -> str:
    """Strip sex/morph qualifier: 'Hooded Merganser (Breeding male)' -> 'Hooded Merganser'."""
    return re.sub(r"\s*\([^)]*\)\s*", "", str(name)).strip()


def _annotate_with_ground_truth(
    result_df: pd.DataFrame,
    labels: dict[str, str],
    label_names_csv: str | Path,
) -> pd.DataFrame:
    """Add ground_truth, label_in_model, is_correct columns to result_df.

    Ground truth labels are base species names (no sex/morph qualifier).
    Comparison is done at base-species level so 'Hooded Merganser' matches
    'Hooded Merganser (Breeding male)'.
    """
    if not labels:
        return result_df
    df = result_df.copy()
    try:
        model_species_full = set(pd.read_csv(label_names_csv)["species"].dropna().astype(str).tolist())
    except Exception:
        model_species_full = set()

    model_base_species = {_strip_qualifier(s) for s in model_species_full}

    df["ground_truth"] = df["image_name"].map(labels)

    def _label_in_model(row):
        gt = row.get("ground_truth")
        if pd.isna(gt) or gt is None:
            return None
        # No bird detected → exclude from accuracy (same treatment as out-of-scope species)
        if not row.get("yolo_detected", True):
            return False
        return _strip_qualifier(str(gt)) in model_base_species

    df["label_in_model"] = df.apply(_label_in_model, axis=1)

    def _is_correct(row):
        gt = row.get("ground_truth")
        pred = row.get("pred_species")
        if pd.isna(gt) or gt is None:
            return None
        if not row.get("yolo_detected", True):
            return None  # excluded, not wrong
        return _strip_qualifier(str(pred)) == _strip_qualifier(str(gt))

    def _is_in_top5(row):
        gt = row.get("ground_truth")
        if pd.isna(gt) or gt is None:
            return None
        if not row.get("yolo_detected", True):
            return None
        top5 = _parse_top5(row.get("pred_top5"))
        if not top5:
            return None
        gt_base = _strip_qualifier(str(gt))
        try:
            return any(_strip_qualifier(str(e["species"])) == gt_base for e in top5)
        except Exception:
            return None

    df["is_correct"] = df.apply(_is_correct, axis=1)
    df["is_in_top5"] = df.apply(_is_in_top5, axis=1)
    return df


def list_previous_run_dirs(root: Path = Path("artifacts/pipeline_runs")) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Training tab helpers
# ---------------------------------------------------------------------------
_SWEEP_PARAMS = {
    "stage1_lr": "Stage 1 LR",
    "stage2_lr": "Stage 2 LR",
    "stage3_lr": "Stage 3 LR",
    "stage1_epochs": "Stage 1 Epochs",
    "stage2_epochs": "Stage 2 Epochs",
    "stage3_epochs": "Stage 3 Epochs",
    "weight_decay": "Weight Decay",
    "label_smoothing": "Label Smoothing",
    "batch_size": "Batch Size",
    "seed": "Seed",
    "crop_scale_min": "Crop Scale Min",
    "jitter_brightness": "Jitter Brightness",
    "jitter_hue": "Jitter Hue",
    "random_erasing_p": "Random Erasing P",
}


def _generate_sweep_values(start: float, stop: float, multiplier: float) -> list[float]:
    """Generate start, start*mult, start*mult^2, ... while in range [start, stop] (or [stop, start] if mult<1)."""
    if multiplier <= 0 or multiplier == 1:
        return [start]
    values = []
    v = start
    if multiplier > 1:
        while v <= stop + 1e-12:
            values.append(v)
            v *= multiplier
    else:
        while v >= stop - 1e-12:
            values.append(v)
            v *= multiplier
    return values if values else [start]


def _launch_next_job() -> bool:
    """Start the next pending job if none is running. Returns True if a job was launched."""
    queue = load_queue()
    running = [j for j in queue if j["status"] == "running"]
    pending = [j for j in queue if j["status"] == "pending"]

    # Defense: if a job is stuck as "running" but the subprocess already exited
    # (e.g. killed by SIGINT before the KeyboardInterrupt handler could update the
    # queue), mark it failed so the queue doesn't stay permanently blocked.
    proc = st.session_state.get("_training_proc")
    if running and proc is not None and proc.poll() is not None:
        for stuck_job in running:
            _update_queue_status(stuck_job["id"], status="failed",
                                 error="Process exited unexpectedly (exit code: "
                                       f"{proc.poll()})",
                                 completed_at=datetime.now().isoformat(timespec="seconds"))
        st.session_state.pop("_training_proc", None)
        st.session_state.pop("_training_log_f", None)
        running = []

    if running or not pending:
        return False

    # Check if our stored subprocess is still alive
    proc = st.session_state.get("_training_proc")
    if proc is not None and proc.poll() is None:
        return False  # still running

    next_job = pending[0]
    job_id = next_job["id"]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"training_{job_id}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "training_engine.py", "--job-id", job_id],
        stdout=log_f,
        stderr=log_f,
    )
    st.session_state["_training_proc"] = proc
    st.session_state["_training_log_f"] = log_f
    return True


def _config_form(prefix: str = "run") -> TrainingConfig | None:
    """Render a TrainingConfig form and return the config, or None if invalid."""
    run_label = st.text_input("Run label", value="manual", key=f"{prefix}_label",
                              help="Short name for this run, used in checkpoint filenames.")
    species_mode = st.radio("Species mode", options=["98", "555"], index=0,
                             key=f"{prefix}_species",
                             help="98 = target species subset | 555 = all NABirds-specific classes")

    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("Seed", value=42, step=1, key=f"{prefix}_seed")
    batch_size = c2.number_input("Batch size", value=32, min_value=1, step=8, key=f"{prefix}_bs")
    val_fraction = c3.number_input("Val fraction", value=0.10, min_value=0.01, max_value=0.5,
                                    step=0.01, key=f"{prefix}_vf")

    _stage_labels = {
        1: "Head (fc only)",
        2: "Layer 4 + Head",
        3: "Layer 3 + Layer 4 + Head",
    }
    stages_to_run = st.multiselect(
        "Layers to train (cumulative, head → deeper)",
        options=[1, 2, 3],
        default=[1, 2, 3],
        format_func=lambda s: _stage_labels[s],
        key=f"{prefix}_stages",
    )
    if not stages_to_run:
        st.warning("Select at least one training stage.")
        return None

    with st.expander("Per-stage settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Head (fc only)**")
            st.caption("Trains classification head, backbone frozen")
            s1_ep = st.number_input("Epochs", value=20, min_value=1, key=f"{prefix}_s1ep")
            s1_lr = st.number_input("LR", value=1e-3, format="%.2e", key=f"{prefix}_s1lr")
        with sc2:
            st.markdown("**Layer 4 + Head**")
            st.caption("Unfreezes ResNet layer4 + head")
            s2_ep = st.number_input("Epochs", value=10, min_value=1, key=f"{prefix}_s2ep")
            s2_lr = st.number_input("LR", value=2e-4, format="%.2e", key=f"{prefix}_s2lr")
        with sc3:
            st.markdown("**Layer 3 + Layer 4 + Head**")
            st.caption("Unfreezes layer3, layer4 + head")
            s3_ep = st.number_input("Epochs", value=8, min_value=1, key=f"{prefix}_s3ep")
            s3_lr = st.number_input("LR", value=1e-4, format="%.2e", key=f"{prefix}_s3lr")

    with st.expander("Shared training params"):
        p1, p2 = st.columns(2)
        wd = p1.number_input("Weight decay", value=2e-4, format="%.2e", key=f"{prefix}_wd")
        ls = p2.number_input("Label smoothing", value=0.0, min_value=0.0, max_value=0.5,
                              step=0.01, key=f"{prefix}_ls")

    with st.expander("Augmentation"):
        a1, a2 = st.columns(2)
        cs_min = a1.number_input("Crop scale min", value=0.6, min_value=0.1, max_value=1.0,
                                   step=0.05, key=f"{prefix}_csmin")
        cs_max = a2.number_input("Crop scale max", value=1.0, min_value=0.1, max_value=1.0,
                                   step=0.05, key=f"{prefix}_csmax")
        b1, b2, b3, b4 = st.columns(4)
        j_br = b1.number_input("Jitter brightness", value=0.3, min_value=0.0, step=0.05, key=f"{prefix}_jbr")
        j_co = b2.number_input("Jitter contrast", value=0.3, min_value=0.0, step=0.05, key=f"{prefix}_jco")
        j_sa = b3.number_input("Jitter saturation", value=0.3, min_value=0.0, step=0.05, key=f"{prefix}_jsa")
        j_hu = b4.number_input("Jitter hue", value=0.1, min_value=0.0, max_value=0.5, step=0.01,
                                 key=f"{prefix}_jhu")
        e1, e2 = st.columns(2)
        er_p = e1.number_input("Erasing P", value=0.3, min_value=0.0, max_value=1.0, step=0.05,
                                 key=f"{prefix}_erp")
        er_s = e2.number_input("Erasing scale max", value=0.2, min_value=0.01, max_value=0.5,
                                 step=0.01, key=f"{prefix}_ers")

    return TrainingConfig(
        run_label=run_label,
        species_mode=species_mode,
        seed=int(seed),
        batch_size=int(batch_size),
        val_fraction=float(val_fraction),
        stages_to_run=list(stages_to_run),
        stage1_epochs=int(s1_ep), stage1_lr=float(s1_lr),
        stage2_epochs=int(s2_ep), stage2_lr=float(s2_lr),
        stage3_epochs=int(s3_ep), stage3_lr=float(s3_lr),
        weight_decay=float(wd), label_smoothing=float(ls),
        crop_scale_min=float(cs_min), crop_scale_max=float(cs_max),
        jitter_brightness=float(j_br), jitter_contrast=float(j_co),
        jitter_saturation=float(j_sa), jitter_hue=float(j_hu),
        random_erasing_p=float(er_p), random_erasing_scale_max=float(er_s),
    )


def _render_queue_dashboard() -> bool:
    """Render running job progress + queue table. Returns True if a job is actively running."""
    queue = load_queue()

    # Status counts
    counts = {s: sum(1 for j in queue if j["status"] == s)
              for s in ("pending", "running", "done", "failed", "cancelled")}
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Pending", counts["pending"])
    mc2.metric("Running", counts["running"])
    mc3.metric("Done", counts["done"])
    mc4.metric("Failed", counts["failed"])
    mc5.metric("Cancelled", counts["cancelled"])

    running_jobs = [j for j in queue if j["status"] == "running"]
    any_running = bool(running_jobs)

    # Live progress for running job
    if running_jobs:
        job = running_jobs[0]
        job_id = job["id"]
        progress = get_progress(job_id)
        st.markdown(f"**Running job:** `{job_id}` — label: `{job['config'].get('run_label', '?')}`")

        if progress:
            status = progress.get("status", "?")
            stage = progress.get("stage", 0)
            epoch = progress.get("epoch", 0)
            total_epochs = progress.get("total_epochs", 1)
            train_acc = progress.get("train_acc")
            val_acc = progress.get("val_acc")
            best_val = progress.get("best_val_acc")

            if status == "loading_data":
                st.info("Loading dataset...")
            elif status in ("running", "starting"):
                frac = epoch / max(1, total_epochs)
                label = f"Stage {stage} | Preparing..." if epoch == 0 else f"Stage {stage} | Epoch {epoch}/{total_epochs}"
                st.progress(frac, text=label)
                pm1, pm2, pm3 = st.columns(3)
                if train_acc is not None:
                    pm1.metric("Train acc", f"{train_acc:.4f}")
                if val_acc is not None:
                    pm2.metric("Val acc", f"{val_acc:.4f}")
                if best_val is not None:
                    pm3.metric("Best val acc", f"{best_val:.4f}")
            else:
                st.info(f"Status: {status}")

        if st.button("Cancel running job", key="cancel_running"):
            cancel_running_job(job_id)
            st.warning(f"Cancel signal sent to job {job_id}. It will stop after the current epoch.")

        with st.expander("Training log (tail)"):
            log_text = get_training_log(job_id)
            if log_text:
                # Show last 60 lines
                lines = log_text.splitlines()
                st.code("\n".join(lines[-60:]), language=None)
            else:
                st.caption("No log output yet.")

    # Queue table
    st.markdown("### Queue")
    if not queue:
        st.caption("Queue is empty.")
    else:
        for job in queue:
            job_id = job["id"]
            status = job["status"]
            cfg = job.get("config", {})
            label = cfg.get("run_label", "?")
            species = cfg.get("species_mode", "?")
            stages = cfg.get("stages_to_run", [])
            added = job.get("added_at", "")[:16]

            status_icon = {"pending": "⏳", "running": "🔄", "done": "✅",
                           "failed": "❌", "cancelled": "🚫"}.get(status, "?")

            col_info, col_btn = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"{status_icon} `{job_id}` &nbsp; **{label}** &nbsp; "
                    f"species={species} &nbsp; stages={stages} &nbsp; added={added}",
                    unsafe_allow_html=True,
                )
                if status == "failed":
                    err = job.get("error", "")
                    if err:
                        st.caption(f"Error: {err[:120]}")
            with col_btn:
                if status == "pending":
                    if st.button("Remove", key=f"rm_{job_id}"):
                        remove_from_queue(job_id)
                        st.rerun()

    if counts["done"] + counts["failed"] + counts["cancelled"] > 0:
        if st.button("Clear finished jobs"):
            n = clear_finished_jobs()
            st.success(f"Removed {n} finished job(s).")
            st.rerun()

    return any_running


def render_training_tab() -> None:
    st.header("Training Queue")

    # Try to auto-start the next pending job
    launched = _launch_next_job()
    if launched:
        st.rerun()  # pick up the new "running" status immediately

    any_running = _render_queue_dashboard()

    # Auto-refresh while a job is running, or while a subprocess is active but
    # hasn't updated the queue file to "running" yet (covers the launch gap).
    proc = st.session_state.get("_training_proc")
    proc_active = proc is not None and proc.poll() is None
    if any_running or proc_active:
        time.sleep(3)
        st.rerun()

    st.divider()

    # ---- Add job ----
    add_tab_single, add_tab_sweep = st.tabs(["Add Single Run", "Add Parameter Sweep"])

    with add_tab_single:
        with st.form("add_single_run"):
            cfg = _config_form(prefix="single")
            submitted = st.form_submit_button("Add to Queue")
            if submitted and cfg is not None:
                job_id = add_to_queue(cfg)
                st.success(f"Job `{job_id}` added to queue.")
                st.rerun()

    with add_tab_sweep:
        st.markdown("Vary one parameter multiplicatively across a range. All other params come from the base config below.")

        sweep_param = st.selectbox(
            "Parameter to sweep",
            options=list(_SWEEP_PARAMS.keys()),
            format_func=lambda k: _SWEEP_PARAMS[k],
            key="sweep_param",
        )

        sc1, sc2, sc3 = st.columns(3)
        sweep_start = sc1.number_input("Start value", value=1e-3, format="%.2e", key="sweep_start")
        sweep_stop = sc2.number_input("Stop value", value=1e-5, format="%.2e", key="sweep_stop")
        sweep_mult = sc3.number_input("Multiplier (×)", value=0.1, format="%.4f", key="sweep_mult",
                                       help="Each step = previous × multiplier. Use <1 to decrease, >1 to increase.")

        preview_vals = _generate_sweep_values(float(sweep_start), float(sweep_stop), float(sweep_mult))
        st.caption(f"Preview ({len(preview_vals)} runs): {[f'{v:.4g}' for v in preview_vals]}")

        with st.form("add_sweep"):
            st.markdown("**Base config** (the non-swept parameters):")
            base_cfg = _config_form(prefix="sweep_base")
            sweep_submitted = st.form_submit_button(f"Add {len(preview_vals)} sweep jobs to Queue")
            if sweep_submitted and base_cfg is not None:
                added_ids = []
                for val in preview_vals:
                    cfg_dict = asdict(base_cfg)
                    cfg_dict[sweep_param] = val
                    cfg_dict["run_label"] = f"{base_cfg.run_label}_{sweep_param}_{val:.4g}"
                    job_cfg = TrainingConfig(**cfg_dict)
                    jid = add_to_queue(job_cfg)
                    added_ids.append(jid)
                st.success(f"Added {len(added_ids)} sweep jobs: {added_ids}")
                st.rerun()

    st.divider()

    # ---- Run History ----
    st.header("Run History")
    if not SUMMARY_CSV.exists():
        st.caption("No run history yet (artifacts/logs/run_summary.csv not found).")
    else:
        try:
            df = pd.read_csv(SUMMARY_CSV)
            df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
            df["best_val_acc"] = pd.to_numeric(df["best_val_acc"], errors="coerce")

            filter_col, sort_col = st.columns(2)
            stage_filter = filter_col.selectbox(
                "Filter by stage",
                options=["all"] + sorted(df["stage"].dropna().unique().tolist()),
                key="hist_stage_filter",
            )
            show_df = df if stage_filter == "all" else df[df["stage"] == stage_filter]

            st.dataframe(
                show_df.sort_values("test_acc", ascending=False, na_position="last")[
                    ["run_label", "stage", "test_acc", "best_val_acc", "num_epochs",
                     "lr", "weight_decay", "label_smoothing", "checkpoint_path", "timestamp"]
                ].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Could not load run history: {e}")


def _render_classify_tab(
    tab,
    *,
    run_mode: str,
    selected_run_dir: "Path | None",
    label_names_csv: Path,
    classifier_checkpoint: str,
    yolo_weights: str,
    yolo_conf: float,
    device: str,
    yolo_batch_size: int,
    classifier_batch_size: int,
    input_mode: str,
    recursive: bool,
    write_metadata_to_originals: bool,
    folder_path: str,
    uploaded_files: list,
    run_clicked: bool,
    logger,
) -> None:
    """All Classify tab UI. Lives in its own function so `return` statements
    don't exit main() — allowing the Training tab to always render afterward."""
    with tab:
        st.title("Bird Detection + Species Identification")
        st.write("Run YOLO bird detection, crop best bird per photo, and classify species with your ResNet checkpoint.")

        if run_mode == "View previous run":
            if selected_run_dir is None:
                st.info("Select a saved run to view results.")
                return
            try:
                results_csv_path = selected_run_dir / "results.csv"
                results_json_path = selected_run_dir / "results.json"
                errors_path = selected_run_dir / "errors.csv"

                result_df = pd.read_csv(results_csv_path)
                prev_labels = _infer_labels_from_results(result_df)
                if prev_labels:
                    result_df = _annotate_with_ground_truth(result_df, prev_labels, label_names_csv)
                st.success(f"Loaded previous run: {selected_run_dir}")
                if prev_labels:
                    st.info(f"Found labels.csv with {len(prev_labels)} labeled photo(s) — showing accuracy metrics.")
                render_summary(result_df)

                st.subheader("Gallery")
                render_gallery(result_df)

                st.subheader("Exports")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download results.csv",
                        data=results_csv_path.read_bytes(),
                        file_name=f"{selected_run_dir.name}_results.csv",
                        mime="text/csv",
                    )
                with c2:
                    if results_json_path.exists():
                        st.download_button(
                            "Download results.json",
                            data=results_json_path.read_bytes(),
                            file_name=f"{selected_run_dir.name}_results.json",
                            mime="application/json",
                        )

                if errors_path.exists():
                    st.subheader("Errors")
                    err_df = pd.read_csv(errors_path)
                    st.dataframe(err_df, use_container_width=True)

                st.caption("Frontend log: artifacts/pipeline_runs/frontend.log")
                st.caption(f"Pipeline log for this run: {selected_run_dir / 'pipeline.log'}")
            except Exception as e:
                logger.exception("Failed to load previous run")
                st.error(f"Could not load previous run: {type(e).__name__}: {e}")
            return

        if not run_clicked:
            st.info("Select inputs and click Run Inference.")
            return

        try:
            logger.info("Run requested | mode=%s | recursive=%s", input_mode, recursive)
            if input_mode == "Upload files":
                if not uploaded_files:
                    st.warning("Please upload at least one JPEG file.")
                    return
                total_bytes = sum(getattr(f, "size", 0) for f in uploaded_files)
                logger.info(
                    "Upload batch stats | files=%d | total_bytes=%d | total_gb=%.3f",
                    len(uploaded_files),
                    total_bytes,
                    total_bytes / (1024**3),
                )
                inputs = input_items_from_uploaded_files(uploaded_files)
            else:
                if not folder_path.strip():
                    st.warning("Please enter a valid folder path.")
                    return
                inputs = input_items_from_folder(folder_path.strip(), recursive=recursive)

            if not inputs:
                st.warning("No JPEG images found to process.")
                return

            logger.info(
                "Input resolved | count=%d | checkpoint=%s | yolo_weights=%s | yolo_conf=%.3f | device=%s | yolo_bs=%d | cls_bs=%d",
                len(inputs),
                classifier_checkpoint,
                yolo_weights,
                float(yolo_conf),
                device,
                int(yolo_batch_size),
                int(classifier_batch_size),
            )

            cfg = PipelineConfig(
                classifier_checkpoint=classifier_checkpoint,
                label_names_csv=label_names_csv,
                yolo_weights=yolo_weights,
                yolo_conf=float(yolo_conf),
                device=device,
                output_root="artifacts/pipeline_runs",
                write_metadata_to_originals=bool(write_metadata_to_originals),
                yolo_batch_size=int(yolo_batch_size),
                classifier_batch_size=int(classifier_batch_size),
            )

            progress = st.progress(0.0, text="Initializing...")
            status = st.empty()

            def on_progress(done: int, total: int, message: str) -> None:
                frac = float(done) / float(max(1, total))
                progress.progress(min(max(frac, 0.0), 1.0), text=f"{done}/{total}")
                status.info(message)

            with st.spinner(f"Running inference on {len(inputs)} images..."):
                results, errors, run_dir = run_inference_batch(inputs, cfg, progress_callback=on_progress)

            progress.progress(1.0, text=f"{len(inputs)}/{len(inputs)}")
            status.success(f"Finished. Run ID: {run_dir.name}")
            logger.info("Run completed | run_dir=%s | results=%d | errors=%d", run_dir, len(results), len(errors))

            st.success(f"Completed run: {run_dir}")

            result_df = pd.DataFrame([r.__dict__ for r in results])
            if input_mode == "Folder path" and folder_path.strip():
                folder_labels = _load_folder_labels(folder_path.strip())
                if folder_labels:
                    result_df = _annotate_with_ground_truth(result_df, folder_labels, label_names_csv)
                    st.info(f"Found labels.csv with {len(folder_labels)} labeled photo(s) — showing accuracy metrics.")
            render_summary(result_df)

            st.subheader("Gallery")
            render_gallery(result_df)

            st.subheader("Exports")
            results_csv_path = run_dir / "results.csv"
            results_json_path = run_dir / "results.json"

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download results.csv",
                    data=results_csv_path.read_bytes(),
                    file_name=f"{run_dir.name}_results.csv",
                    mime="text/csv",
                )
            with c2:
                st.download_button(
                    "Download results.json",
                    data=results_json_path.read_bytes(),
                    file_name=f"{run_dir.name}_results.json",
                    mime="application/json",
                )

            errors_path = run_dir / "errors.csv"
            if errors_path.exists():
                st.subheader("Errors")
                err_df = pd.read_csv(errors_path)
                st.dataframe(err_df, use_container_width=True)
                logger.warning("Run completed with errors | errors_csv=%s | count=%d", errors_path, len(err_df))

            st.caption(f"Frontend log: artifacts/pipeline_runs/frontend.log")
            st.caption(f"Pipeline log for this run: {run_dir / 'pipeline.log'}")

        except Exception as e:
            logger.exception("Pipeline failed")
            st.error(f"Pipeline failed: {type(e).__name__}: {e}")


def main() -> None:
    st.set_page_config(page_title="Bird Gallery Inference", layout="wide")
    apply_css()
    logger = build_app_logger()

    tab_classify, tab_training = st.tabs(["Classify", "Training"])

    # ---- Sidebar ----
    run_mode = "New inference"
    selected_run_dir: Path | None = None

    with st.sidebar:
        st.header("Run Mode")
        run_mode = st.radio("Mode", options=["New inference", "View previous run"], index=0)

        if run_mode == "View previous run":
            previous_runs = list_previous_run_dirs()
            if not previous_runs:
                st.warning("No previous runs found in artifacts/pipeline_runs.")
            else:
                options = [p.name for p in previous_runs]
                selected_name = st.selectbox("Previous run", options=options, index=0)
                selected_run_dir = next((p for p in previous_runs if p.name == selected_name), None)

        st.header("Models")
        all_specific = st.toggle(
            "All 555 NABirds species",
            value=False,
            help="Off = 98 target species subset  |  On = all 555 NABirds-specific classes",
        )
        label_names_csv = ALL_SPECIFIC_LABEL_NAMES_CSV if all_specific else DEFAULT_LABEL_NAMES_CSV

        all_ckpts = list_classifier_checkpoints("artifacts")
        if not all_ckpts:
            st.error("No .pt classifier checkpoints found in artifacts/")
            # Still render training tab so user can train a model
            with tab_training:
                render_training_tab()
            return

        ckpts = _filter_checkpoints_by_mode(all_ckpts, all_specific)
        if not ckpts:
            st.warning("No checkpoints found for the selected species set.")
            ckpts = all_ckpts

        default_ckpt = choose_best_checkpoint(ckpts)
        classifier_checkpoint = st.selectbox(
            "Classifier checkpoint",
            options=ckpts,
            index=ckpts.index(default_ckpt),
        )
        ckpt_acc = _checkpoint_test_acc(classifier_checkpoint)
        if ckpt_acc is not None:
            st.caption(f"Test accuracy: **{ckpt_acc*100:.2f}%** | Label names: {label_names_csv}")
        else:
            st.caption(f"Label names: {label_names_csv}")
        yolo_weights = st.text_input("YOLO weights", value="yolo11n.pt")
        yolo_conf = st.slider("YOLO confidence threshold", min_value=0.01, max_value=0.95, value=0.25, step=0.01)
        device = st.selectbox("Device", options=["auto", get_default_device(), "cpu", "cuda", "mps"], index=0)
        yolo_batch_size = st.slider("YOLO batch size", min_value=1, max_value=16, value=2, step=1)
        classifier_batch_size = st.slider("Classifier batch size", min_value=1, max_value=128, value=16, step=1)

        st.header("Input")
        input_mode = st.radio("Input mode", options=["Upload files", "Folder path"], index=0)
        recursive = st.toggle("Recursive folder scan", value=False)
        write_metadata_to_originals = st.toggle(
            "Tag original JPEGs with species/confidence (folder mode)",
            value=True,
        )

        folder_path = ""
        uploaded_files = []
        if input_mode == "Upload files":
            uploaded_files = st.file_uploader(
                "Select JPEG files",
                type=["jpg", "jpeg"],
                accept_multiple_files=True,
            )
            if write_metadata_to_originals:
                st.caption("Metadata tagging is only applied for Folder path mode (local files).")
            if uploaded_files:
                total_bytes = sum(getattr(f, "size", 0) for f in uploaded_files)
                total_gb = total_bytes / (1024**3)
                st.caption(f"Selected files: {len(uploaded_files)} | Total size: {total_gb:.2f} GB")
                if total_gb > 1.0:
                    st.warning(
                        "Large upload selected. For 1000+ photos, folder path mode is more memory-stable than upload mode."
                    )
        else:
            folder_path = st.text_input("Folder path", value="")

        run_clicked = st.button("Run Inference", type="primary", disabled=(run_mode != "New inference"))

    # ---- Classify tab ----
    # In its own function so `return` statements inside don't exit main(),
    # ensuring the Training tab always renders below.
    _render_classify_tab(
        tab_classify,
        run_mode=run_mode,
        selected_run_dir=selected_run_dir,
        label_names_csv=label_names_csv,
        classifier_checkpoint=classifier_checkpoint,
        yolo_weights=yolo_weights,
        yolo_conf=yolo_conf,
        device=device,
        yolo_batch_size=yolo_batch_size,
        classifier_batch_size=classifier_batch_size,
        input_mode=input_mode,
        recursive=recursive,
        write_metadata_to_originals=write_metadata_to_originals,
        folder_path=folder_path,
        uploaded_files=uploaded_files,
        run_clicked=run_clicked,
        logger=logger,
    )

    # ---- Training tab ----
    # Rendered last so its st.rerun() calls (for live progress) don't prevent
    # the sidebar or classify tab from rendering in the same script pass.
    with tab_training:
        render_training_tab()


if __name__ == "__main__":
    main()
