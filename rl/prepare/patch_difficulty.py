"""One-shot: add `difficulty_level` to the [metadata] block of every task.toml.

Reads `rl/cache/difficulty_lookup.json` (built by surveying decisions.csv files).
For each task.toml under `rl/harbor/tasks/`:
  1. Parse `source_row_id` from the [metadata] block.
  2. Look up its difficulty (0 = uncategorized).
  3. If task.toml already has `difficulty_level`, skip.
  4. Otherwise insert `difficulty_level = N` at the end of the [metadata] block.

Idempotent — re-runs are no-ops for already-patched files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def find_source_row_id(text: str) -> str | None:
    m = re.search(r'^source_row_id\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    return m.group(1) if m else None


def patch_file(path: Path, value: int) -> str:
    text = path.read_text()
    if re.search(r'^difficulty_level\s*=', text, flags=re.MULTILINE):
        return "skip-already-has"
    meta = re.search(r'^\[metadata\]\s*$', text, flags=re.MULTILINE)
    if not meta:
        return "skip-no-metadata-section"
    after = text[meta.end():]
    next_section = re.search(r'^\s*\[[a-zA-Z]', after, flags=re.MULTILINE)
    insert_pos = meta.end() + (next_section.start() if next_section else len(after))
    # back up over trailing blank lines so the new field stays inside [metadata]
    while insert_pos > 0 and text[insert_pos - 1] == "\n" and text[insert_pos - 2:insert_pos] == "\n\n":
        insert_pos -= 1
    new_text = text[:insert_pos] + f"difficulty_level = {value}\n" + text[insert_pos:]
    path.write_text(new_text)
    return "patched"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-root", default="rl/harbor/tasks")
    ap.add_argument("--lookup", default="rl/cache/difficulty_lookup.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lookup: dict[str, int] = json.loads(Path(args.lookup).read_text())
    print(f"Loaded {len(lookup)} difficulty entries from {args.lookup}")

    task_tomls = sorted(Path(args.tasks_root).rglob("task.toml"))
    print(f"Found {len(task_tomls)} task.toml files under {args.tasks_root}\n")

    outcomes: Counter[str] = Counter()
    by_suite: dict[str, Counter] = {}
    unmatched: list[str] = []

    for p in task_tomls:
        suite = p.relative_to(args.tasks_root).parts[0]
        by_suite.setdefault(suite, Counter())
        text = p.read_text()
        src_id = find_source_row_id(text)
        if not src_id:
            outcomes["skip-no-source-row-id"] += 1
            by_suite[suite]["skip-no-source-row-id"] += 1
            continue
        value = lookup.get(src_id, 0)
        if value == 0 and src_id not in lookup:
            unmatched.append(src_id)
        if args.dry_run:
            already = bool(re.search(r"^difficulty_level\s*=", text, flags=re.MULTILINE))
            verdict = "would-skip-already-has" if already else f"would-patch-with-{value}"
            outcomes[verdict] += 1
            by_suite[suite][verdict] += 1
            continue
        verdict = patch_file(p, value)
        outcomes[verdict] += 1
        by_suite[suite][verdict] += 1

    print("=== Overall outcomes ===")
    for k, v in sorted(outcomes.items()):
        print(f"  {k:35s} {v}")

    print("\n=== Per-suite breakdown ===")
    for suite, c in sorted(by_suite.items()):
        items = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print(f"  {suite}: {items}")

    if unmatched:
        print(f"\n[info] {len(unmatched)} unique source_row_ids had no entry in the lookup "
              f"(written as difficulty_level=0). Sample:")
        for s in unmatched[:5]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
