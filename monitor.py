#!/usr/bin/env python3
"""Live autoresearch monitor — run in a separate terminal.

Usage:
    python monitor.py            # refresh every 15s
    python monitor.py --interval 5   # refresh every 5s
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_CSVS = {
    "subset98": ROOT / "artifacts" / "autoresearch_log.csv",
    "full555": ROOT / "artifacts" / "autoresearch_log_full555.csv",
}
STATUS_FILE = ROOT / "artifacts" / "autoresearch_status.json"
PROGRESS_FILE = ROOT / "artifacts" / "autoresearch_progress.json"

# Set at startup from --species flag
SPECIES_MODE = "subset98"

# ANSI helpers
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def read_status() -> dict | None:
    """Read the status JSON written by autorun.py."""
    if not STATUS_FILE.exists():
        return None
    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        data["_age_sec"] = time.time() - data.get("updated_ts", 0)
        return data
    except (json.JSONDecodeError, OSError):
        return None


def read_progress() -> dict | None:
    """Read the progress JSON written by train.py (if available)."""
    if not PROGRESS_FILE.exists():
        return None
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        data["_age_sec"] = time.time() - PROGRESS_FILE.stat().st_mtime
        return data
    except (json.JSONDecodeError, OSError):
        return None


def read_log() -> list[dict]:
    log_csv = LOG_CSVS.get(SPECIES_MODE, LOG_CSVS["subset98"])
    if not log_csv.exists():
        return []
    with open(log_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render() -> str:
    lines = []
    now = datetime.now().strftime("%H:%M:%S")

    species_label = "98 species" if SPECIES_MODE == "subset98" else "555 species"
    lines.append(f"{BOLD}{'=' * 62}{RESET}")
    lines.append(f"{BOLD}  AUTORESEARCH MONITOR{RESET}  {CYAN}[{species_label}]{RESET}  {DIM}{now}{RESET}")
    lines.append(f"{BOLD}{'=' * 62}{RESET}")

    # --- Live status from autorun.py + per-epoch progress from train.py ---
    status = read_status()
    progress = read_progress()
    has_progress = progress and progress.get("_age_sec", 999) < 300

    # Status file doesn't expire — it's valid for the whole iteration
    has_status = status is not None

    lines.append("")
    if has_progress:
        # We have live per-epoch data from train.py
        stage = progress.get("stage", "?")
        epoch = progress.get("epoch", "?")
        budget = progress.get("budget_sec", 0)
        remaining = progress.get("remaining_sec", 0)
        train_elapsed = budget - remaining
        best = progress.get("best_val_acc", 0)
        age = int(progress["_age_sec"])

        pct = (train_elapsed / budget * 100) if budget > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar = f"[{'#' * filled}{'.' * (bar_width - filled)}]"

        iter_label = ""
        if has_status:
            iteration = status.get("iteration", "?")
            total_iters = status.get("total_iterations")
            iter_label = f"Iter {iteration}"
            if total_iters:
                iter_label += f" / {total_iters}"
            iter_label += "  "

        lines.append(f"  {CYAN}{BOLD}TRAINING{RESET}  {iter_label}{DIM}(updated {age}s ago){RESET}")
        lines.append(f"  Stage: {BOLD}{stage}{RESET}  Epoch: {BOLD}{epoch}{RESET}")
        lines.append(f"  {bar}  {fmt_time(train_elapsed)} / {fmt_time(budget)}  ({pct:.0f}%)")
        lines.append(f"  Best val acc this run: {GREEN}{BOLD}{best:.4f}{RESET}")
        if has_status:
            lines.append(f"  Best before this run: {BOLD}{status.get('best_before', 0):.4f}{RESET}")
    elif has_status:
        # No per-epoch data but we know an iteration is running
        iteration = status.get("iteration", "?")
        state = status.get("state", "?")
        best_before = status.get("best_before", 0)
        iter_started = status.get("iter_started_ts", time.time())
        total_iters = status.get("total_iterations")
        iter_elapsed = time.time() - iter_started

        iter_label = f"Iter {iteration}"
        if total_iters:
            iter_label += f" / {total_iters}"

        if state == "training":
            budget_est = 1800
            pct = min(100, iter_elapsed / budget_est * 100)
            bar_width = 30
            filled = int(bar_width * pct / 100)
            bar = f"[{'#' * filled}{'.' * (bar_width - filled)}]"
            lines.append(f"  {CYAN}{BOLD}TRAINING{RESET}  {iter_label}  {DIM}({fmt_time(iter_elapsed)} into this iteration){RESET}")
            lines.append(f"  {bar}  ~{fmt_time(iter_elapsed)} / ~{fmt_time(budget_est)}  (~{pct:.0f}%)")
            lines.append(f"  Best before this run: {BOLD}{best_before:.4f}{RESET}")
        elif state == "evaluating":
            lines.append(f"  {YELLOW}{BOLD}EVALUATING{RESET}  {iter_label}  {DIM}({fmt_time(iter_elapsed)} total){RESET}")
        else:
            lines.append(f"  {DIM}State: {state}  {iter_label}{RESET}")
    else:
        lines.append(f"  {DIM}Autorun not active{RESET}")

    # --- Experiment history from CSV ---
    rows = read_log()
    lines.append("")
    lines.append(f"  {BOLD}EXPERIMENT HISTORY{RESET}  ({len(rows)} total)")
    lines.append(f"  {'─' * 56}")

    if rows:
        # Show last 8 rows
        display = rows[-8:]
        lines.append(
            f"  {DIM}{'#':>3}  {'val_acc':>8}  {'test_acc':>8}  {'epochs':>6}  "
            f"{'time':>7}  {'status':>7}  notes{RESET}"
        )
        for i, row in enumerate(display):
            idx = len(rows) - len(display) + i + 1
            val = float(row.get("top1_val_acc") or 0)
            test = float(row.get("top1_test_acc") or 0)
            epochs = row.get("total_epochs") or "?"
            secs = float(row.get("training_seconds") or 0)
            row_status = row.get("status", "?")
            notes = row.get("notes", "")[:30]

            if row_status == "keep":
                color = GREEN
            elif row_status == "discard":
                color = RED
            else:
                color = RESET

            lines.append(
                f"  {idx:>3}  {color}{val:>8.4f}{RESET}  {test:>8.4f}  "
                f"{epochs:>6}  {fmt_time(secs):>7}  {color}{row_status:>7}{RESET}  {notes}"
            )

        # Show best
        best_row = max(rows, key=lambda r: float(r.get("top1_val_acc") or 0))
        best_val = float(best_row.get("top1_val_acc") or 0)
        best_notes = best_row.get("notes", "")[:40]
        lines.append(f"  {'─' * 56}")
        lines.append(f"  {GREEN}{BOLD}Best: {best_val:.4f}{RESET}  ({best_notes})")

        # Show analysis of last row if present
        last_analysis = rows[-1].get("analysis", "")
        if last_analysis:
            lines.append("")
            lines.append(f"  {BOLD}Last analysis:{RESET} {YELLOW}{last_analysis}{RESET}")
    else:
        lines.append(f"  {DIM}No experiments logged yet.{RESET}")

    lines.append(f"\n{BOLD}{'=' * 62}{RESET}")
    lines.append(f"  {DIM}Refreshing every {{interval}}s  |  Ctrl+C to quit{RESET}")
    return "\n".join(lines)


def main():
    global SPECIES_MODE
    parser = argparse.ArgumentParser(description="Autoresearch live monitor")
    parser.add_argument("--interval", type=int, default=15, help="Refresh interval in seconds")
    parser.add_argument(
        "--species",
        choices=list(LOG_CSVS),
        default="subset98",
        help="Species mode to monitor (default: subset98)",
    )
    args = parser.parse_args()
    SPECIES_MODE = args.species

    try:
        while True:
            output = render().replace("{interval}", str(args.interval))
            print(CLEAR + output, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RESET}")


if __name__ == "__main__":
    main()
