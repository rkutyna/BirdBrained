#!/usr/bin/env python3
"""Outer-loop runner for unattended Codex-driven autoresearch.

This script launches fresh `codex exec` sessions in a loop. Each Codex session
is instructed to perform exactly one experiment cycle in this repository:

1. Read `program.md` and the current experiment history.
2. Modify only `train.py`.
3. Run `python train.py`.
4. Keep the new `train.py` only if the latest logged score improved.

The outer loop snapshots `train.py` before each run and enforces the keep /
restore decision based on `artifacts/autoresearch_log.csv`, so the overnight
process does not depend purely on prompt compliance.
"""
from __future__ import annotations

import argparse
import csv
import json
import selectors
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"
PROGRAM_MD = ROOT / "program.md"
AUTORESEARCH_LOG = ROOT / "artifacts" / "autoresearch_log.csv"
FALLBACK_SUMMARY = ROOT / "artifacts" / "logs" / "run_summary.csv"
RUNNER_DIR = ROOT / "artifacts" / "autoresearch_runner"
PROMPT_TEMPLATE = ROOT / "prompts" / "autoresearch_codex_prompt.txt"
HISTORY_JSONL = RUNNER_DIR / "history.jsonl"
RUNNER_LOG = RUNNER_DIR / "runner.log"
STATUS_FILE = ROOT / "artifacts" / "autoresearch_status.json"


@dataclass
class IterationResult:
    iteration: int
    started_at: str
    finished_at: str
    codex_exit_code: int
    log_rows_before: int
    log_rows_after: int
    best_before: float
    best_after: float
    last_metric: float | None
    improved: bool
    restored_train_py: bool
    had_new_log_row: bool
    iter_dir: str
    dry_run: bool
    commit_hash: str | None = None


# ---------------------------------------------------------------------------
# Git helpers (Karpathy-style branch/commit tracking)
# ---------------------------------------------------------------------------
def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30,
    )


def git_ensure_branch(tag: str) -> None:
    """Create or checkout the autoresearch/<tag> branch."""
    branch = f"autoresearch/{tag}"
    result = _git("rev-parse", "--verify", branch)
    if result.returncode == 0:
        _git("checkout", branch)
        log_event(f"Checked out existing branch: {branch}")
    else:
        _git("checkout", "-b", branch)
        log_event(f"Created new branch: {branch}")


def git_commit_experiment(notes: str) -> str | None:
    """Stage train.py and commit. Returns short commit hash, or None on failure."""
    _git("add", "train.py")
    diff = _git("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        return None  # nothing to commit
    result = _git("commit", "-m", f"autoresearch: {notes}")
    if result.returncode != 0:
        return None
    rev = _git("rev-parse", "--short", "HEAD")
    return rev.stdout.strip() if rev.returncode == 0 else None


def git_revert_last_commit() -> None:
    """Undo the last commit (soft reset), then restore train.py from HEAD."""
    _git("reset", "--hard", "HEAD~1")


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_event(message: str, *, iter_dir: Path | None = None) -> None:
    line = f"[{now_ts()}] {message}"
    print(line, flush=True)
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    with RUNNER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if iter_dir is not None:
        with (iter_dir / "runner_events.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_metric(row: dict[str, str] | None) -> float | None:
    if not row:
        return None
    raw = row.get("top1_val_acc", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def best_metric(rows: list[dict[str, str]]) -> float:
    vals = [m for m in (parse_metric(r) for r in rows) if m is not None]
    return max(vals) if vals else 0.0


def last_metric(rows: list[dict[str, str]]) -> float | None:
    for row in reversed(rows):
        metric = parse_metric(row)
        if metric is not None:
            return metric
    return None


def ensure_layout() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)


def next_iteration_index() -> int:
    if not HISTORY_JSONL.exists():
        return 1
    with HISTORY_JSONL.open(encoding="utf-8") as f:
        return sum(1 for _ in f) + 1


def load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def load_extra_instructions(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


CROSS_REF_LOGS = {
    "subset98": ROOT / "artifacts" / "autoresearch_log.csv",
    "full555": ROOT / "artifacts" / "autoresearch_log_full555.csv",
}


def build_cross_reference(current_species: str | None) -> str:
    """Build a summary of the OTHER species mode's experiment log for inspiration."""
    if current_species is None:
        current_species = "subset98"
    other = "full555" if current_species == "subset98" else "subset98"
    other_log = CROSS_REF_LOGS[other]
    rows = read_rows(other_log)
    if not rows:
        return f"No experiments logged yet for {other} mode."

    # Show a compact summary: best result + last 5 experiments with notes
    best_row = max(rows, key=lambda r: float(r.get("top1_val_acc") or 0))
    best_val = float(best_row.get("top1_val_acc") or 0)
    best_notes = best_row.get("notes", "")

    lines = [
        f"The {other} dataset ({len(rows)} experiments, best val_acc={best_val:.4f}) "
        f"has been tested with these techniques. Use for inspiration — results may "
        f"not directly correlate between dataset sizes.",
        "",
        f"Best experiment: val_acc={best_val:.4f} — {best_notes}",
        "",
        "Recent experiments (status | val_acc | notes):",
    ]
    for row in rows[-8:]:
        val = float(row.get("top1_val_acc") or 0)
        status = row.get("status", "?")
        notes = row.get("notes", "")
        lines.append(f"  {status:>7} | {val:.4f} | {notes}")

    return "\n".join(lines)


def build_prompt(
    *,
    template: str,
    iteration: int,
    best_before: float,
    log_rows_before: int,
    extra_instructions: str,
    species: str | None = None,
    user_message: str | None = None,
) -> str:
    return template.format(
        iteration=iteration,
        best_before=f"{best_before:.6f}",
        log_rows_before=log_rows_before,
        autoresearch_log=str(AUTORESEARCH_LOG.relative_to(ROOT)),
        fallback_summary=str(FALLBACK_SUMMARY.relative_to(ROOT)),
        program_md=str(PROGRAM_MD.relative_to(ROOT)),
        train_py=str(TRAIN_PY.relative_to(ROOT)),
        extra_instructions=extra_instructions or "(none)",
        cross_reference_log=build_cross_reference(species),
        user_message=user_message or "(none)",
    )


def append_history(result: IterationResult) -> None:
    with HISTORY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.__dict__, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Inject CLI overrides into train.py config
# ---------------------------------------------------------------------------
def apply_train_overrides(args: argparse.Namespace) -> None:
    """Set SPECIES_MODE and TIME_BUDGET_SEC in train.py from CLI flags."""
    import re as _re
    if args.species is None and args.time_budget is None:
        return
    text = TRAIN_PY.read_text(encoding="utf-8")
    if args.species is not None:
        text = _re.sub(
            r'^(SPECIES_MODE\s*=\s*).*$',
            f'\\g<1>"{args.species}"  # set by autorun.py --species',
            text,
            count=1,
            flags=_re.MULTILINE,
        )
        log_event(f"Set SPECIES_MODE = {args.species!r} in train.py")
    if args.time_budget is not None:
        text = _re.sub(
            r'^(TIME_BUDGET_SEC\s*=\s*).*$',
            f'\\g<1>{args.time_budget}  # set by autorun.py --time-budget',
            text,
            count=1,
            flags=_re.MULTILINE,
        )
        log_event(f"Set TIME_BUDGET_SEC = {args.time_budget} in train.py")
    TRAIN_PY.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Status file — written by autorun.py so monitor.py can poll it
# ---------------------------------------------------------------------------
def write_status(
    iteration: int,
    state: str,
    best_before: float,
    iter_started: float,
    total_iterations: int | None = None,
) -> None:
    """Write a status JSON that monitor.py can poll."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps({
            "iteration": iteration,
            "state": state,
            "best_before": round(best_before, 6),
            "iter_started_ts": round(iter_started, 1),
            "total_iterations": total_iterations,
            "updated_ts": round(time.time(), 1),
        }),
        encoding="utf-8",
    )


def clear_status() -> None:
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()


def run_codex(
    prompt: str,
    iter_dir: Path,
    args: argparse.Namespace,
    *,
    iteration: int = 0,
    best_before: float = 0.0,
) -> int:
    cmd = [
        "codex",
        "exec",
        "--color",
        "never",
        "-C",
        str(ROOT),
        "-o",
        str(iter_dir / "last_message.txt"),
    ]
    if args.model:
        cmd.extend(["-m", args.model])
    if args.ephemeral:
        cmd.append("--ephemeral")
    if args.danger_full_access:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.extend(["--sandbox", "workspace-write", "--full-auto"])
    cmd.append("-")

    command_text = shlex.join(cmd)
    (iter_dir / "codex_command.txt").write_text(command_text, encoding="utf-8")
    log_event(f"Starting Codex: {command_text}", iter_dir=iter_dir)

    with (iter_dir / "codex_stdout.log").open("w", encoding="utf-8") as stdout_f, (
        iter_dir / "codex_stderr.log"
    ).open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=ROOT,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        proc.stdin.write(prompt)
        proc.stdin.close()

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_f))
        sel.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_f))

        while sel.get_map():
            events = sel.select()
            for key, _ in events:
                stream_name, file_handle = key.data
                line = key.fileobj.readline()
                if line == "":
                    sel.unregister(key.fileobj)
                    continue
                file_handle.write(line)
                file_handle.flush()
                prefix = "[codex]" if stream_name == "stdout" else "[codex:stderr]"
                print(f"{prefix} {line}", end="", flush=True)

        exit_code = proc.wait()

    log_event(f"Codex finished with exit_code={exit_code}", iter_dir=iter_dir)
    return exit_code


def write_iteration_artifacts(iter_dir: Path, prompt: str, train_before: str) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (iter_dir / "train_before.py").write_text(train_before, encoding="utf-8")


def maybe_restore_train(train_before: str) -> None:
    TRAIN_PY.write_text(train_before, encoding="utf-8")


def run_iteration(
    *,
    iteration: int,
    template: str,
    args: argparse.Namespace,
    started_wall: float,
) -> IterationResult:
    before_rows = read_rows(AUTORESEARCH_LOG)
    best_before = best_metric(before_rows)
    log_rows_before = len(before_rows)
    started_at = datetime.now().isoformat(timespec="seconds")

    iter_dir = RUNNER_DIR / f"iter_{iteration:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_before = TRAIN_PY.read_text(encoding="utf-8")
    extra_instructions = load_extra_instructions(args.extra_instructions)
    prompt = build_prompt(
        template=template,
        iteration=iteration,
        best_before=best_before,
        log_rows_before=log_rows_before,
        extra_instructions=extra_instructions,
        species=args.species,
        user_message=args.user_message,
    )
    write_iteration_artifacts(iter_dir, prompt, train_before)

    log_event(
        f"[iter {iteration}] prepared iter_dir={iter_dir.relative_to(ROOT)} "
        f"best_before={best_before:.6f} log_rows_before={log_rows_before}",
        iter_dir=iter_dir,
    )

    if args.dry_run:
        log_event(f"[iter {iteration}] dry-run: skipping Codex invocation", iter_dir=iter_dir)
        after_rows = before_rows
        best_after = best_before
        last_after = last_metric(after_rows)
        return IterationResult(
            iteration=iteration,
            started_at=started_at,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            codex_exit_code=0,
            log_rows_before=log_rows_before,
            log_rows_after=log_rows_before,
            best_before=best_before,
            best_after=best_after,
            last_metric=last_after,
            improved=False,
            restored_train_py=False,
            had_new_log_row=False,
            iter_dir=str(iter_dir.relative_to(ROOT)),
            dry_run=True,
        )

    iter_started = time.time()
    write_status(iteration, "training", best_before, iter_started, getattr(args, "iterations", None))
    exit_code = run_codex(prompt, iter_dir, args, iteration=iteration, best_before=best_before)
    write_status(iteration, "evaluating", best_before, iter_started, getattr(args, "iterations", None))
    after_rows = read_rows(AUTORESEARCH_LOG)
    best_after = best_metric(after_rows)
    last_after = last_metric(after_rows)
    had_new_log_row = len(after_rows) > log_rows_before
    new_rows = after_rows[log_rows_before:]
    improved = False
    restored = False

    commit_hash = None
    if had_new_log_row and last_after is not None and last_after > best_before:
        improved = True
        (iter_dir / "decision.txt").write_text(
            f"keep\nlast_metric={last_after:.6f}\nbest_before={best_before:.6f}\n",
            encoding="utf-8",
        )
        # Git: commit the successful experiment
        if args.tag:
            notes_text = ""
            if new_rows:
                notes_text = new_rows[-1].get("notes", "")
            commit_hash = git_commit_experiment(notes_text or f"iter {iteration}")
            if commit_hash:
                log_event(f"[iter {iteration}] git commit {commit_hash}", iter_dir=iter_dir)
    else:
        maybe_restore_train(train_before)
        restored = True
        reason = "no_new_log_row"
        if had_new_log_row and last_after is not None:
            reason = f"no_improvement last_metric={last_after:.6f} <= best_before={best_before:.6f}"
        elif exit_code != 0:
            reason = f"codex_exit_code={exit_code}"
        (iter_dir / "decision.txt").write_text(
            f"restore\nreason={reason}\n",
            encoding="utf-8",
        )

    if new_rows:
        (iter_dir / "new_log_rows.json").write_text(
            json.dumps(new_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for idx, row in enumerate(new_rows, start=1):
            log_event(
                f"[iter {iteration}] new_log_row_{idx}: {json.dumps(row, sort_keys=True)}",
                iter_dir=iter_dir,
            )
    else:
        log_event(f"[iter {iteration}] no new rows appended to {AUTORESEARCH_LOG.name}", iter_dir=iter_dir)

    (iter_dir / "train_after.py").write_text(
        TRAIN_PY.read_text(encoding="utf-8"), encoding="utf-8"
    )

    finished_at = datetime.now().isoformat(timespec="seconds")
    elapsed = time.time() - started_wall
    log_event(
        f"[iter {iteration}] exit_code={exit_code} "
        f"had_new_log_row={had_new_log_row} last_metric={last_after} "
        f"best_after={best_after:.6f} improved={improved} "
        f"restored={restored} total_elapsed_s={elapsed:.1f}",
        iter_dir=iter_dir,
    )

    return IterationResult(
        iteration=iteration,
        started_at=started_at,
        finished_at=finished_at,
        codex_exit_code=exit_code,
        log_rows_before=log_rows_before,
        log_rows_after=len(after_rows),
        best_before=best_before,
        best_after=best_after,
        last_metric=last_after,
        improved=improved,
        restored_train_py=restored,
        had_new_log_row=had_new_log_row,
        iter_dir=str(iter_dir.relative_to(ROOT)),
        dry_run=False,
        commit_hash=commit_hash,
    )


def should_stop(
    *,
    completed_iterations: int,
    started_wall: float,
    args: argparse.Namespace,
    consecutive_no_improve: int,
) -> bool:
    if args.iterations is not None and completed_iterations >= args.iterations:
        return True
    if args.hours is not None and (time.time() - started_wall) >= args.hours * 3600:
        return True
    if args.patience is not None and consecutive_no_improve >= args.patience:
        return True
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Codex autoresearch overnight.")
    p.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Maximum number of Codex experiment cycles to run.",
    )
    p.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Maximum wall-clock hours to run before stopping.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Stop after this many consecutive non-improving iterations.",
    )
    p.add_argument(
        "--cooldown-seconds",
        type=float,
        default=5.0,
        help="Sleep between iterations.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional Codex model override passed to `codex exec -m`.",
    )
    p.add_argument(
        "--extra-instructions",
        type=Path,
        default=None,
        help="Optional text file appended to the Codex prompt each iteration.",
    )
    p.add_argument(
        "--prompt-template",
        type=Path,
        default=PROMPT_TEMPLATE,
        help="Prompt template file for each Codex iteration.",
    )
    p.add_argument(
        "--danger-full-access",
        action="store_true",
        help=(
            "Run Codex with --dangerously-bypass-approvals-and-sandbox. "
            "Required if you want PyTorch to see MPS in this repo."
        ),
    )
    p.add_argument(
        "--ephemeral",
        action="store_true",
        help="Run Codex sessions without persisting Codex session files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and iteration directories without invoking Codex.",
    )
    p.add_argument(
        "--species",
        choices=["subset98", "full555"],
        default=None,
        help="Species mode. Sets SPECIES_MODE in train.py before each run.",
    )
    p.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Training time budget in seconds. Sets TIME_BUDGET_SEC in train.py.",
    )
    p.add_argument(
        "--user-message",
        type=str,
        default=None,
        help="A note or direction for the agent (e.g. 'revisit timed-out runs with larger budget').",
    )
    p.add_argument(
        "--tag",
        default=None,
        help=(
            "Git branch tag (e.g. 'mar13'). Creates/checks out branch "
            "'autoresearch/<tag>' and commits each successful experiment. "
            "Omit to skip git integration."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations is None and args.hours is None:
        log_event(
            "Specify at least one stopping condition with --iterations or --hours.",
        )
        return 2
    if not TRAIN_PY.exists():
        log_event(f"Missing {TRAIN_PY}")
        return 2
    ensure_layout()
    apply_train_overrides(args)
    # Update log CSV path if using full555 mode
    global AUTORESEARCH_LOG
    if args.species == "full555":
        AUTORESEARCH_LOG = ROOT / "artifacts" / "autoresearch_log_full555.csv"
    if args.tag:
        git_ensure_branch(args.tag)
    template = load_template(args.prompt_template)
    started_wall = time.time()
    iteration = next_iteration_index()
    consecutive_no_improve = 0
    completed_iterations = 0

    log_event(
        "Runner starting with "
        f"iterations={args.iterations} hours={args.hours} patience={args.patience} "
        f"cooldown_seconds={args.cooldown_seconds} danger_full_access={args.danger_full_access} "
        f"species={args.species} time_budget={args.time_budget} "
        f"dry_run={args.dry_run}"
    )

    while True:
        result = run_iteration(
            iteration=iteration,
            template=template,
            args=args,
            started_wall=started_wall,
        )
        completed_iterations += 1
        if not args.dry_run:
            append_history(result)

        if result.improved:
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

        if should_stop(
            completed_iterations=completed_iterations,
            started_wall=started_wall,
            args=args,
            consecutive_no_improve=consecutive_no_improve,
        ):
            break

        if args.cooldown_seconds > 0:
            log_event(
                f"Sleeping {args.cooldown_seconds:.1f}s before next iteration"
            )
            time.sleep(args.cooldown_seconds)
        iteration += 1

    log_event(
        f"Stopped after {completed_iterations} iterations. "
        f"History: {HISTORY_JSONL.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
