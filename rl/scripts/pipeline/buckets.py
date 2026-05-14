"""Verdict-bucketed task suite layout.

Tasks live in one of four subfolders depending on their current verdict.
This is the *physical* layout, not a symlink view.

    harbor/tasks/<suite>/
    ├── pending/<id>/          in-flight or never-decided
    ├── verified/<id>/         verified* terminal
    ├── dropped/<id>/          dropped terminal
    └── phase_b_failed/<id>/   doctor never finalized

The orchestrator moves a task's spec dir between buckets after the run.
build_spec() always creates new specs in `pending/`. find_task_dir() locates
an existing spec wherever it currently lives.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


# Order matters: when refreshing, we walk in this order. Pending is first
# so a freshly built spec doesn't get accidentally categorized.
BUCKETS = ("pending", "verified", "dropped", "phase_b_failed")

VERDICT_TO_BUCKET = {
    "verified":                  "verified",
    "verified_after_rewrite":    "verified",
    "verified_gold_corrected":   "verified",
    "verifiable_judge":          "verified",
    "dropped":                   "dropped",
    "phase_b_failed":            "phase_b_failed",
}


def find_task_dir(suite_dir: Path, id_safe: str) -> Path | None:
    """Locate a task's spec dir across all buckets. Returns None if absent."""
    for bucket in BUCKETS:
        p = suite_dir / bucket / id_safe
        if p.exists():
            return p
    return None


def ensure_bucket(suite_dir: Path, bucket: str) -> Path:
    p = suite_dir / bucket
    p.mkdir(parents=True, exist_ok=True)
    return p


def move_task(suite_dir: Path, id_safe: str, target_bucket: str) -> Path:
    """Move (or create empty target dir for) a task to `target_bucket/<id>/`."""
    current = find_task_dir(suite_dir, id_safe)
    target_dir = ensure_bucket(suite_dir, target_bucket) / id_safe
    if current is None:
        return target_dir
    if current == target_dir:
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(current), str(target_dir))
    return target_dir


def migrate_flat_to_buckets(suite_dir: Path, rollup_csv: Path) -> dict[str, int]:
    """One-shot: convert legacy flat layout → bucketed.

    Walks task dirs at the suite root and moves each into its verdict bucket
    based on `rollup_csv`. Anything without a known verdict goes to pending.
    """
    decisions: dict[str, str] = {}
    if rollup_csv.exists():
        df = pd.read_csv(rollup_csv)
        for _, r in df.iterrows():
            from .build import id_safe
            decisions[id_safe(str(r["task_id"]))] = str(r["verdict"])

    counts: dict[str, int] = {}
    for entry in sorted(suite_dir.iterdir()):
        if not entry.is_dir() or entry.name in BUCKETS or entry.name.startswith("_"):
            continue
        verdict = decisions.get(entry.name)
        bucket = VERDICT_TO_BUCKET.get(verdict, "pending")
        target = ensure_bucket(suite_dir, bucket) / entry.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(entry), str(target))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts
