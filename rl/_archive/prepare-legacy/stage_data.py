"""Sync the local data mirror into the HF Bucket.

After `prepare/build_harbor_tasks.py --name <slug>` has run, each unique Kaggle
dataset's files live at:

    rl/harbor/tasks/jupyter-agent-<slug>/data/<bucket_prefix>/

This script reads `manifest.jsonl`, dedupes bucket prefixes, and uploads each
to `hf://buckets/<user>/jupyter-agent-<slug>-data/<bucket_prefix>/` via
`sync_bucket` (Xet-chunked, dedup-aware, parallel via hf_xet).

It does NOT touch Kaggle — the kagglehub step happens in build_harbor_tasks.

Usage:
  uv run python -m prepare.stage_data --name test
  uv run python -m prepare.stage_data --name test --dry-run
  uv run python -m prepare.stage_data --name test --only mirichoi0218__insurance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from prepare.build_harbor_tasks import derive_names

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--user", default=None,
        help="HF namespace (defaults to whatever build_harbor_tasks used).",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Restrict to one or more bucket_prefix value(s).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if not os.environ.get("HF_TOKEN"):
        print("[err] HF_TOKEN missing in .env", file=sys.stderr)
        return 1

    names = derive_names(args.name, user=args.user) if args.user else derive_names(args.name)
    suite_dir = Path(names["local_dir"])
    data_dir = Path(names["local_data_dir"])
    manifest_path = suite_dir / "manifest.jsonl"

    if not manifest_path.exists():
        print(f"[miss] {manifest_path} — run prepare.build_harbor_tasks --name {args.name} first.")
        return 1
    if not data_dir.exists():
        print(f"[miss] {data_dir} — local data mirror missing.")
        return 1

    print(f"[plan] suite={names['base']}")
    print(f"  source: {data_dir}/<bucket_prefix>/")
    print(f"  bucket: hf://buckets/{names['bucket_id']}")

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    by_prefix: dict[str, Path] = {}
    for r in rows:
        prefix = r["bucket_prefix"]
        src = data_dir / prefix
        if src.exists() and src.is_dir():
            by_prefix.setdefault(prefix, src)

    if args.only:
        wanted = set(args.only)
        by_prefix = {k: v for k, v in by_prefix.items() if k in wanted}

    n_prefixes = len(by_prefix)
    print(f"  unique bucket prefixes: {n_prefixes}")

    if not args.dry_run:
        from huggingface_hub import create_bucket
        try:
            create_bucket(names["bucket_id"], exist_ok=True)
        except Exception as exc:
            print(f"[err] create_bucket({names['bucket_id']}): {exc}", file=sys.stderr)
            return 1
        print(f"[bucket] ensured {names['bucket_id']}")

    from huggingface_hub import sync_bucket

    t0 = time.time()
    ok, errs = 0, 0
    for prefix, src in by_prefix.items():
        files = [p for p in src.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        dest = f"hf://buckets/{names['bucket_id']}/{prefix}"
        if args.dry_run:
            print(f"  [would-upload] {prefix}: {len(files)} file(s), {total/1024/1024:.1f} MB → {dest}")
            ok += 1
            continue
        try:
            sync_bucket(str(src), dest, verbose=False)
            print(f"  [uploaded]    {prefix}: {len(files)} file(s), {total/1024/1024:.1f} MB → {dest}")
            ok += 1
        except Exception as exc:
            print(f"  [err]         {prefix}: {exc}", file=sys.stderr)
            errs += 1

    elapsed = time.time() - t0
    print(f"\n[done] {ok}/{n_prefixes} synced in {elapsed:.1f}s. errors={errs}.")
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
