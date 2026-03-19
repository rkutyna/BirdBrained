#!/usr/bin/env python3
"""Replay saved train.py variants with optional budget and notes overrides.

This runner is separate from autorun.py on purpose: autorun remains the Codex
outer loop, while this script replays concrete historical train.py variants or
manifests of config deltas for controlled re-evaluation.

Typical use:
    python replay_train_variants.py artifacts/autoresearch_replays/manifests/last8_one_hour.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import pprint
import selectors
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"
AUTORESEARCH_LOG = ROOT / "artifacts" / "autoresearch_log.csv"
REPLAYS_DIR = ROOT / "artifacts" / "autoresearch_replays"


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(text: str) -> str:
    chars = []
    last_dash = False
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    return "".join(chars).strip("-") or "variant"


def log_event(message: str, campaign_dir: Path) -> None:
    line = f"[{now_ts()}] {message}"
    print(line, flush=True)
    with (campaign_dir / "runner.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_log_rows() -> list[dict[str, str]]:
    if not AUTORESEARCH_LOG.exists():
        return []
    with AUTORESEARCH_LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for base in (manifest_path.parent, ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ROOT / candidate).resolve()


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list):
        raise ValueError(f"Manifest must be an object with a 'variants' list: {path}")
    return data


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for ch in text:
        offsets.append(offsets[-1] + 1)
    return offsets


def assignment_spans(text: str) -> dict[str, tuple[int, int, ast.Assign]]:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    starts = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)

    spans: dict[str, tuple[int, int, ast.Assign]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        start = starts[node.lineno - 1]
        if node.end_lineno is None:
            raise ValueError(f"Missing end position for assignment: {name}")
        end = starts[node.end_lineno - 1] + len(lines[node.end_lineno - 1])
        spans[name] = (start, end, node)
    return spans


def literal_assignment_value(text: str, node: ast.Assign):
    segment = ast.get_source_segment(text, node.value)
    if segment is None:
        raise ValueError(f"Could not extract assignment value for {ast.dump(node)}")
    return ast.literal_eval(segment)


def render_assignment(name: str, value) -> str:
    rendered = pprint.pformat(value, sort_dicts=False, width=88)
    return f"{name} = {rendered}\n"


def parse_nested_path(path: str) -> tuple[str, list[str | int]]:
    root = []
    idx = 0
    while idx < len(path) and path[idx] not in ".[":
        root.append(path[idx])
        idx += 1
    root_name = "".join(root)
    if not root_name:
        raise ValueError(f"Invalid nested path: {path}")

    tokens: list[str | int] = []
    while idx < len(path):
        if path[idx] == ".":
            idx += 1
            start = idx
            while idx < len(path) and path[idx] not in ".[":
                idx += 1
            tokens.append(path[start:idx])
        elif path[idx] == "[":
            idx += 1
            start = idx
            while idx < len(path) and path[idx] != "]":
                idx += 1
            if idx >= len(path):
                raise ValueError(f"Unclosed index in nested path: {path}")
            tokens.append(int(path[start:idx]))
            idx += 1
        else:
            raise ValueError(f"Unexpected character in nested path: {path}")
    return root_name, tokens


def set_nested_value(container, tokens: list[str | int], value) -> None:
    current = container
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = value


def apply_variant_overrides(
    source_text: str,
    *,
    set_values: dict[str, object],
    nested_set: dict[str, object],
) -> str:
    spans = assignment_spans(source_text)
    root_values: dict[str, object] = {}

    for name, value in set_values.items():
        if name not in spans:
            raise KeyError(f"Assignment not found in source: {name}")
        root_values[name] = value

    for path, value in nested_set.items():
        root_name, tokens = parse_nested_path(path)
        if root_name not in spans:
            raise KeyError(f"Assignment not found in source for path {path}: {root_name}")
        if root_name not in root_values:
            root_values[root_name] = literal_assignment_value(source_text, spans[root_name][2])
        set_nested_value(root_values[root_name], tokens, value)

    updated_text = source_text
    replacements: list[tuple[int, int, str]] = []
    for name, value in root_values.items():
        start, end, _ = spans[name]
        replacements.append((start, end, render_assignment(name, value)))

    for start, end, replacement in sorted(replacements, reverse=True):
        updated_text = updated_text[:start] + replacement + updated_text[end:]
    return updated_text


def stream_command(cmd: list[str], exp_dir: Path) -> int:
    command_text = " ".join(cmd)
    (exp_dir / "command.txt").write_text(command_text, encoding="utf-8")

    with (exp_dir / "stdout.log").open("w", encoding="utf-8") as stdout_f, (
        exp_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, ("stdout", stdout_f))
        sel.register(proc.stderr, selectors.EVENT_READ, ("stderr", stderr_f))

        try:
            while sel.get_map():
                for key, _ in sel.select():
                    stream_name, file_handle = key.data
                    line = key.fileobj.readline()
                    if line == "":
                        sel.unregister(key.fileobj)
                        continue
                    file_handle.write(line)
                    file_handle.flush()
                    prefix = "[train]" if stream_name == "stdout" else "[train:stderr]"
                    print(f"{prefix} {line}", end="", flush=True)
        except KeyboardInterrupt:
            proc.terminate()
            raise

        return proc.wait()


def prepare_variant_text(
    source_path: Path,
    variant: dict,
    *,
    budget_sec: int,
    notes_suffix: str,
) -> str:
    source_text = source_path.read_text(encoding="utf-8")
    set_values = dict(variant.get("set", {}))
    nested_set = dict(variant.get("nested_set", {}))
    set_values["TIME_BUDGET_SEC"] = budget_sec

    if "NOTES" not in set_values:
        spans = assignment_spans(source_text)
        if "NOTES" not in spans:
            raise KeyError(f"NOTES assignment not found in source: {source_path}")
        set_values["NOTES"] = literal_assignment_value(source_text, spans["NOTES"][2])

    if notes_suffix:
        notes = str(set_values["NOTES"])
        if notes_suffix not in notes:
            set_values["NOTES"] = f"{notes}{notes_suffix}"

    return apply_variant_overrides(source_text, set_values=set_values, nested_set=nested_set)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay historical train.py variants.")
    p.add_argument(
        "manifest",
        type=Path,
        help="JSON manifest describing the variants to replay.",
    )
    p.add_argument(
        "--budget-sec",
        type=int,
        default=3600,
        help="Override TIME_BUDGET_SEC for every replayed variant.",
    )
    p.add_argument(
        "--notes-suffix",
        default=" | 1h replay",
        help="Suffix appended to NOTES so replay runs are distinguishable in the log.",
    )
    p.add_argument(
        "--cooldown-seconds",
        type=float,
        default=5.0,
        help="Sleep between runs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the campaign plan and effective train.py snapshots without launching training.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the campaign if a replay exits non-zero or does not append a log row.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)

    campaign_name = manifest.get("name") or manifest_path.stem
    campaign_dir = REPLAYS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(campaign_name)}"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    original_train = TRAIN_PY.read_text(encoding="utf-8")
    variants = manifest["variants"]
    log_event(
        f"Starting replay campaign '{campaign_name}' with {len(variants)} variants "
        f"at budget={args.budget_sec}s dry_run={args.dry_run}",
        campaign_dir,
    )

    try:
        for idx, variant in enumerate(variants, start=1):
            variant_name = str(variant.get("name") or f"variant {idx}")
            source_path = resolve_path(str(variant["source"]), manifest_path)
            exp_dir = campaign_dir / f"exp_{idx:02d}_{slugify(variant_name)}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            effective_text = prepare_variant_text(
                source_path,
                variant,
                budget_sec=args.budget_sec,
                notes_suffix=args.notes_suffix,
            )
            (exp_dir / "source_train.py").write_text(
                source_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (exp_dir / "effective_train.py").write_text(effective_text, encoding="utf-8")
            (exp_dir / "variant.json").write_text(
                json.dumps(variant, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            log_event(
                f"[{idx}/{len(variants)}] prepared '{variant_name}' from {source_path.relative_to(ROOT)}",
                campaign_dir,
            )

            if args.dry_run:
                continue

            before_rows = read_log_rows()
            TRAIN_PY.write_text(effective_text, encoding="utf-8")
            log_event(f"[{idx}/{len(variants)}] launching '{variant_name}'", campaign_dir)
            exit_code = stream_command([sys.executable, "train.py"], exp_dir)
            after_rows = read_log_rows()
            new_rows = after_rows[len(before_rows):]
            (exp_dir / "new_log_rows.json").write_text(
                json.dumps(new_rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log_event(
                f"[{idx}/{len(variants)}] exit_code={exit_code} new_rows={len(new_rows)}",
                campaign_dir,
            )

            if new_rows:
                latest = new_rows[-1]
                log_event(
                    f"[{idx}/{len(variants)}] latest row: "
                    f"val={latest.get('top1_val_acc')} test={latest.get('top1_test_acc')} "
                    f"notes={latest.get('notes')}",
                    campaign_dir,
                )

            TRAIN_PY.write_text(original_train, encoding="utf-8")

            failed = exit_code != 0 or not new_rows
            if failed and args.stop_on_error:
                log_event(f"Stopping early after '{variant_name}'", campaign_dir)
                return 1

            if idx < len(variants) and args.cooldown_seconds > 0:
                log_event(
                    f"Sleeping {args.cooldown_seconds:.1f}s before next replay",
                    campaign_dir,
                )
                time.sleep(args.cooldown_seconds)
    finally:
        TRAIN_PY.write_text(original_train, encoding="utf-8")
        log_event("Restored working train.py", campaign_dir)

    log_event("Replay campaign complete", campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
