from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from bird_pipeline import (
    PipelineConfig,
    get_default_device,
    input_items_from_uploaded_files,
    input_items_from_folder,
    list_classifier_checkpoints,
    run_inference_batch,
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
    st.markdown(
        """
<style>
:root {
  --main-bg: #f7fbf8;
  --main-text: #1f3b2f;
  --side-bg: #e8f3ec;
  --side-text: #163528;
  --input-bg: #ffffff;
  --input-text: #163528;
}

html, body {
  color-scheme: light !important;
}

.stApp {
  background: radial-gradient(1200px 500px at 10% -10%, #e8f5ec 0%, var(--main-bg) 60%);
  color: var(--main-text) !important;
  color-scheme: light !important;
}

/* Main content readability */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
  color: var(--main-text) !important;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"],
[data-testid="stText"] *,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
  color: var(--main-text) !important;
}

/* Sidebar readability */
section[data-testid="stSidebar"] {
  background: var(--side-bg) !important;
  border-right: 1px solid #d3e4d8;
  color-scheme: light !important;
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
  color: var(--side-text) !important;
}

/* Inputs/selects: ensure readable text and neutral background */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
input, textarea {
  background: var(--input-bg) !important;
  color: var(--input-text) !important;
}

[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="slider"] * {
  color: var(--input-text) !important;
}

/* Force light surfaces for widgets that can inherit dark-mode backgrounds */
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTextArea"] [data-baseweb="textarea"] > div {
  background: #ffffff !important;
  border: 1px solid #b7d3c3 !important;
  border-radius: 10px !important;
}

[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  color: #163528 !important;
}

[data-testid="stSelectbox"] svg,
[data-testid="stNumberInput"] svg {
  color: #163528 !important;
  fill: currentColor !important;
}

/* Number input +/- buttons */
[data-testid="stNumberInput"] button {
  background: #e6f4eb !important;
  color: #163528 !important;
  border-left: 1px solid #b7d3c3 !important;
}
[data-testid="stNumberInput"] button:hover {
  background: #dcefe3 !important;
}
[data-testid="stNumberInput"] button:disabled {
  background: #f1f5f2 !important;
  color: #6c7f74 !important;
}

/* Header/toolbar readability */
header[data-testid="stHeader"] {
  background: #e8f3ec !important;
  border-bottom: 1px solid #d3e4d8 !important;
}
header[data-testid="stHeader"] * {
  color: #163528 !important;
}
[data-testid="stToolbar"] * {
  color: #163528 !important;
}

/* File uploader readability */
[data-testid="stFileUploaderDropzone"] {
  background: #ffffff !important;
  border: 1px dashed #b7d3c3 !important;
  border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color: #163528 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
  color: #274c3a !important;
}
[data-testid="stBaseButton-secondary"] {
  background: #e6f4eb !important;
  color: #163528 !important;
  border: 1px solid #b7d3c3 !important;
}
[data-testid="stBaseButton-secondary"] * {
  color: #163528 !important;
}

/* Dropdown menu options (portal) */
div[role="listbox"],
ul[role="listbox"] {
  background: #ffffff !important;
  color: #163528 !important;
  border: 1px solid #b7d3c3 !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
  background: #ffffff !important;
  color: #163528 !important;
}
div[role="option"],
li[role="option"] {
  color: var(--input-text) !important;
  background: #ffffff !important;
}
div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
  background: #e6f4eb !important;
  color: #103524 !important;
}
div[role="option"]:hover,
li[role="option"]:hover {
  background: #f1f9f4 !important;
  color: #103524 !important;
}
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


def choose_default_checkpoint(
    checkpoints: Iterable[str],
    run_summary_csv: Path = Path("artifacts/logs/run_summary.csv"),
) -> tuple[str, float | None]:
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

                if "timestamp" in valid_df.columns:
                    valid_df = valid_df.sort_values(["test_acc", "timestamp"], ascending=[False, False])
                else:
                    valid_df = valid_df.sort_values(["test_acc"], ascending=[False])

                for _, row in valid_df.iterrows():
                    ckpt_path = str(row["checkpoint_path"])
                    for key in _path_keys(ckpt_path):
                        if key in checkpoint_by_key:
                            return checkpoint_by_key[key], float(row["test_acc"])
        except Exception:
            # Ignore summary parsing failures and fall back to filename heuristic.
            pass

    for ckpt in ckpts:
        if "head_only" in Path(ckpt).name:
            return ckpt, None
    return ckpts[0], None


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


def render_gallery(df: pd.DataFrame, default_page_size: int = 200) -> None:
    show_df = df.copy()

    species_options = ["(all)"] + sorted(x for x in show_df["pred_species"].dropna().unique().tolist())
    species_filter = st.selectbox("Filter by predicted species", options=species_options)
    if species_filter != "(all)":
        show_df = show_df[show_df["pred_species"] == species_filter].copy()

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


def list_previous_run_dirs(root: Path = Path("artifacts/pipeline_runs")) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def main() -> None:
    st.set_page_config(page_title="Bird Gallery Inference", layout="wide")
    apply_css()
    logger = build_app_logger()

    st.title("Bird Detection + Species Identification")
    st.write("Run YOLO bird detection, crop best bird per photo, and classify species with your ResNet checkpoint.")

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
        ckpts = list_classifier_checkpoints("artifacts")
        if not ckpts:
            st.error("No .pt classifier checkpoints found in artifacts/")
            return

        default_ckpt, default_test_acc = choose_default_checkpoint(ckpts)

        classifier_checkpoint = st.selectbox("Classifier checkpoint", options=ckpts, index=ckpts.index(default_ckpt))
        if default_test_acc is not None:
            st.caption(f"Default set to highest test accuracy: {default_test_acc:.4f}")
        label_names_csv = st.text_input("Label names CSV", value="artifacts/label_names.csv")
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

    if run_mode == "View previous run":
        if selected_run_dir is None:
            st.info("Select a saved run to view results.")
            return

        try:
            results_csv_path = selected_run_dir / "results.csv"
            results_json_path = selected_run_dir / "results.json"
            errors_path = selected_run_dir / "errors.csv"

            result_df = pd.read_csv(results_csv_path)
            st.success(f"Loaded previous run: {selected_run_dir}")
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
            return
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


if __name__ == "__main__":
    main()
