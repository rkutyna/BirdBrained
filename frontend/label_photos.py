"""
Manual bird photo labeling tool.

Opens a folder of photos one at a time, lets you search & select a species,
and saves labels to labels.csv in the same folder.

Run with:  streamlit run frontend/label_photos.py
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPECIES_CSV = Path("artifacts/labels/label_names_nabirds_base_species.csv")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
LABELS_FILENAME = "labels.csv"
LABELS_COLUMNS = ["filename", "species", "labeled_at"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_species() -> list[str]:
    df = pd.read_csv(SPECIES_CSV)
    return sorted(df["species"].dropna().astype(str).tolist())


def scan_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_labels(folder: Path) -> dict[str, str]:
    """Return {filename: species} from labels.csv in the folder."""
    labels_path = folder / LABELS_FILENAME
    if not labels_path.exists():
        return {}
    try:
        df = pd.read_csv(labels_path)
        if {"filename", "species"}.issubset(df.columns):
            return dict(zip(df["filename"].astype(str), df["species"].astype(str)))
    except Exception:
        pass
    return {}


def save_label(folder: Path, filename: str, species: str) -> None:
    labels_path = folder / LABELS_FILENAME
    existing = load_labels(folder)
    existing[filename] = species

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABELS_COLUMNS)
        writer.writeheader()
        for fname, sp in sorted(existing.items()):
            writer.writerow({
                "filename": fname,
                "species": sp,
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
            })


def remove_label(folder: Path, filename: str) -> None:
    existing = load_labels(folder)
    existing.pop(filename, None)
    labels_path = folder / LABELS_FILENAME
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABELS_COLUMNS)
        writer.writeheader()
        for fname, sp in sorted(existing.items()):
            writer.writerow({
                "filename": fname,
                "species": sp,
                "labeled_at": "",
            })


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def apply_css() -> None:
    st.markdown(
        """
<style>
:root {
  --main-bg: #f7fbf8;
  --main-text: #1f3b2f;
  --side-bg: #e8f3ec;
  --side-text: #163528;
}
html, body { color-scheme: light !important; }
.stApp {
  background: radial-gradient(1200px 500px at 10% -10%, #e8f5ec 0%, var(--main-bg) 60%);
  color: var(--main-text) !important;
  color-scheme: light !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * { color: var(--main-text) !important; }
section[data-testid="stSidebar"] {
  background: var(--side-bg) !important;
  border-right: 1px solid #d3e4d8;
  color-scheme: light !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * { color: var(--side-text) !important; }
[data-baseweb="input"] > div, input, textarea {
  background: #ffffff !important;
  color: #163528 !important;
}
[data-testid="stBaseButton-secondary"] {
  background: #e6f4eb !important;
  color: #163528 !important;
  border: 1px solid #b7d3c3 !important;
}
[data-testid="stBaseButton-primary"] {
  background: #2e7d52 !important;
  color: #ffffff !important;
  border: none !important;
}
.labeled-badge {
  display: inline-block;
  background: #2e7d52;
  color: #fff;
  border-radius: 999px;
  padding: 0.2rem 0.8rem;
  font-weight: 700;
  font-size: 0.9rem;
  margin-bottom: 0.4rem;
}
.unlabeled-badge {
  display: inline-block;
  background: #888;
  color: #fff;
  border-radius: 999px;
  padding: 0.2rem 0.8rem;
  font-weight: 700;
  font-size: 0.9rem;
  margin-bottom: 0.4rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Bird Photo Labeler",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_css()

    # ---- Session state defaults ----
    if "folder" not in st.session_state:
        st.session_state["folder"] = ""
    if "idx" not in st.session_state:
        st.session_state["idx"] = 0
    if "show_unlabeled_only" not in st.session_state:
        st.session_state["show_unlabeled_only"] = False

    species_list = load_species()

    # ================================================================
    # SIDEBAR
    # ================================================================
    with st.sidebar:
        st.title("Bird Photo Labeler")
        st.caption("Label your personal photos with NABirds species names.")

        st.header("Folder")
        folder_input = st.text_input(
            "Photo folder path",
            value=st.session_state["folder"],
            placeholder="/Users/you/bird_photos",
        )
        if folder_input != st.session_state["folder"]:
            st.session_state["folder"] = folder_input
            st.session_state["idx"] = 0
            st.rerun()

        folder = Path(st.session_state["folder"]) if st.session_state["folder"] else None

        if folder is None or not folder.exists():
            if st.session_state["folder"]:
                st.error("Folder not found.")
            else:
                st.info("Enter a folder path above to get started.")
            return

        images = scan_images(folder)
        if not images:
            st.warning("No image files found in this folder.")
            return

        labels = load_labels(folder)
        n_labeled = sum(1 for img in images if img.name in labels)
        n_total = len(images)

        st.progress(n_labeled / n_total, text=f"{n_labeled} / {n_total} labeled")

        st.header("Navigate")
        show_unlabeled = st.toggle("Show unlabeled only", value=st.session_state["show_unlabeled_only"])
        if show_unlabeled != st.session_state["show_unlabeled_only"]:
            st.session_state["show_unlabeled_only"] = show_unlabeled
            st.session_state["idx"] = 0
            st.rerun()

        if show_unlabeled:
            display_images = [img for img in images if img.name not in labels]
            if not display_images:
                st.success("All images in this folder are labeled!")
                return
        else:
            display_images = images

        idx = min(st.session_state["idx"], len(display_images) - 1)
        st.session_state["idx"] = idx

        st.caption(f"Photo {idx + 1} of {len(display_images)}")

        nav_prev, nav_next = st.columns(2)
        if nav_prev.button("← Prev", width="stretch", disabled=(idx == 0)):
            st.session_state["idx"] = idx - 1
            st.rerun()
        if nav_next.button("Next →", width="stretch", disabled=(idx == len(display_images) - 1)):
            st.session_state["idx"] = idx + 1
            st.rerun()

        # Jump to next unlabeled
        if not show_unlabeled:
            unlabeled_indices = [i for i, img in enumerate(display_images) if img.name not in labels]
            if unlabeled_indices:
                next_unlabeled = next((i for i in unlabeled_indices if i > idx), unlabeled_indices[0])
                if st.button("Jump to next unlabeled →", width="stretch"):
                    st.session_state["idx"] = next_unlabeled
                    st.rerun()
            else:
                st.success("All labeled!")

        st.divider()
        st.header("Export")
        labels_path = folder / LABELS_FILENAME
        if labels_path.exists():
            st.download_button(
                "Download labels.csv",
                data=labels_path.read_bytes(),
                file_name=LABELS_FILENAME,
                mime="text/csv",
                width="stretch",
            )
            st.caption(f"Saved at: {labels_path}")
        else:
            st.caption("No labels saved yet.")

    # ================================================================
    # MAIN CONTENT
    # ================================================================
    current_image = display_images[idx]
    current_label = labels.get(current_image.name)

    # Re-load labels fresh (so edits from save_label are reflected)
    labels = load_labels(folder)
    current_label = labels.get(current_image.name)

    col_img, col_label = st.columns([3, 2])

    # ---- Image display ----
    with col_img:
        filename_display = current_image.name
        st.subheader(filename_display)

        try:
            img = Image.open(current_image)
            st.image(img, width="stretch")
        except Exception as e:
            st.error(f"Could not open image: {e}")

    # ---- Labeling panel ----
    with col_label:
        st.subheader("Label this photo")

        if current_label:
            st.markdown(
                f'<span class="labeled-badge">✓ {current_label}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="unlabeled-badge">Unlabeled</span>',
                unsafe_allow_html=True,
            )

        st.markdown("**Select species** — type to search:")
        selected_species = st.selectbox(
            "Species",
            options=[""] + species_list,
            index=([""] + species_list).index(current_label) if current_label in species_list else 0,
            key=f"species_select_{idx}_{current_image.name}",
            label_visibility="collapsed",
        )

        save_col, clear_col = st.columns([2, 1])
        with save_col:
            if st.button("Save label", type="primary", width="stretch", disabled=not selected_species):
                save_label(folder, current_image.name, selected_species)
                # Auto-advance to next image
                if idx < len(display_images) - 1:
                    st.session_state["idx"] = idx + 1
                st.rerun()
        with clear_col:
            if st.button("Clear", width="stretch", disabled=(current_label is None)):
                remove_label(folder, current_image.name)
                st.rerun()

        st.divider()

        # ---- Labeled images summary in this session ----
        st.markdown("**Recent labels**")
        recent = [(img.name, labels[img.name]) for img in display_images if img.name in labels]
        recent_near = recent[max(0, idx - 3): idx + 4]
        if recent_near:
            recent_df = pd.DataFrame(recent_near, columns=["Filename", "Species"])
            st.dataframe(recent_df, hide_index=True, width="stretch")
        else:
            st.caption("No labels saved yet.")

        # ---- Thumbnail strip: nearby images ----
        st.markdown("**Nearby photos**")
        strip_start = max(0, idx - 2)
        strip_end = min(len(display_images), idx + 5)
        strip_images = display_images[strip_start:strip_end]

        thumb_cols = st.columns(len(strip_images))
        for col, img_path in zip(thumb_cols, strip_images):
            i = display_images.index(img_path)
            is_current = i == idx
            is_labeled = img_path.name in labels
            border_color = "#2e7d52" if is_current else ("#6cb88a" if is_labeled else "#ccc")
            try:
                thumb = Image.open(img_path)
                thumb.thumbnail((120, 120))
                with col:
                    st.image(
                        thumb,
                        width="stretch",
                        caption="★" if is_current else ("✓" if is_labeled else ""),
                    )
                    if st.button(
                        f"{i + 1}",
                        key=f"thumb_{i}",
                        width="stretch",
                        type="primary" if is_current else "secondary",
                    ):
                        st.session_state["idx"] = i
                        st.rerun()
            except Exception:
                col.caption("err")


if __name__ == "__main__":
    main()
