"""Bulk download every unique Kaggle dataset in the source parquet, with
back-links to source row ids and full success/skip/fail logging, and
optionally upload each to an HF Bucket as it lands.

Designed to be run **once** end-to-end (~40 GB across ~791 unique datasets
for the v1 source). Resumable via the `--report` JSONL — re-running the
same command skips datasets already marked `uploaded` or `downloaded` and
retries everything else.

What gets logged (one JSON line per dataset in the report file):

    {
      "kaggle_dataset_name": "uciml/iris",
      "status":  "uploaded",       // uploaded | downloaded | skipped | failed
      "files":   2,
      "bytes":   10240,
      "size_mb": 0.01,
      "local_path":     "/.../kagglehub/datasets/uciml/iris/versions/2",
      "bucket_id":      "AdithyaSK/jupyter-agent-kaggle-all",
      "bucket_prefix":  "uciml__iris",
      "bucket_url":     "hf://buckets/AdithyaSK/jupyter-agent-kaggle-all/uciml__iris",
      "source_row_ids": ["0108/152/108152163.ipynb_qa_1", ...],   // back-link
      "n_source_rows":  12,
      "error":          null,
      "elapsed_sec":    2.3,
      "ts":             "2026-05-12T19:23:00"
    }

Why sync_bucket and not hf-mount: `sync_bucket` is xet-chunked, parallel,
dedup-aware, and lives in our existing dep tree. `hf-mount` is great for
*lazy reads* — agents touching only part of a big dataset — but for the
upload direction it would just translate to writes through a FUSE layer
with no throughput win. See https://github.com/huggingface/hf-mount

Usage:
  # Smoke (5 datasets, just download — don't upload):
  uv run python -m prepare.fetch_all_kaggle --limit 5 --report cache/fetch.jsonl

  # Full run, parallel download + upload:
  uv run python -m prepare.fetch_all_kaggle \\
      --report cache/fetch.jsonl \\
      --upload-bucket AdithyaSK/jupyter-agent-kaggle-all \\
      --max-workers 8

  # Resume after interrupt — same command. Already-uploaded skip instantly.

  # Keep disk lean — delete local copy after successful upload:
  uv run python -m prepare.fetch_all_kaggle ... --purge-local-after-upload
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO_ROOT / "rl" / "cache" / "raw" / "non_thinking.parquet"


# ---------------------------------------------------------------------------
# Kaggle auth (same as fetch_kaggle.py / stage_data.py)
# ---------------------------------------------------------------------------


def setup_kaggle_auth() -> str:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy KAGGLE_USERNAME/KAGGLE_KEY"
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return f"kaggle.json at {cfg}"
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY_TOKEN")
    if token and token.startswith("KGAT_"):
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"username": "anonymous", "key": token}))
        cfg.chmod(0o600)
        os.environ["KAGGLE_USERNAME"] = "anonymous"
        os.environ["KAGGLE_KEY"] = token
        return f"wrote {cfg} from KAGGLE_API_TOKEN"
    raise RuntimeError(
        "No Kaggle credentials. Set KAGGLE_USERNAME+KAGGLE_KEY, place "
        "~/.kaggle/kaggle.json, or export KAGGLE_API_TOKEN=KGAT_..."
    )


# ---------------------------------------------------------------------------
# Source parquet scan — dataset → list of source_row_ids using it
# ---------------------------------------------------------------------------


def scan_source(parquet_path: Path, executor_filter: str | None) -> dict[str, list[str]]:
    """Return {kaggle_dataset_name: [source_row_id, ...]} from the source parquet."""
    pf = pq.ParquetFile(parquet_path)
    out: dict[str, list[str]] = defaultdict(list)
    for batch in pf.iter_batches(batch_size=8192, columns=["id", "kaggle_dataset_name", "executor_type"]):
        for r in batch.to_pylist():
            name = r.get("kaggle_dataset_name")
            if not name:
                continue
            if executor_filter and r.get("executor_type") != executor_filter:
                continue
            out[name].append(r["id"])
    return dict(out)


# ---------------------------------------------------------------------------
# Bucket prefix convention (matches build_harbor_tasks.py / stage_data.py)
# ---------------------------------------------------------------------------


def _bucket_prefix(kaggle_name: str) -> str:
    return kaggle_name.replace("/", "__")


# ---------------------------------------------------------------------------
# Report file: append-only JSONL. Indexed by kaggle_dataset_name (last-wins).
# ---------------------------------------------------------------------------


_TERMINAL_OK = {"uploaded", "downloaded", "skipped-already"}


def load_report(path: Path) -> dict[str, dict]:
    """Return {kaggle_dataset_name: latest_entry} from the report JSONL."""
    if not path.exists():
        return {}
    idx: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx[rec.get("kaggle_dataset_name", "?")] = rec
    return idx


def append_report(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------


def process_one(
    name: str,
    source_row_ids: list[str],
    bucket_id: str | None,
    purge_local: bool,
    max_size_mb: int | None,
) -> dict:
    """Download (+optional upload) one dataset; return a report entry.

    If `max_size_mb` is set and the downloaded dataset exceeds it, the local
    copy is removed and the entry is marked `oversize` (no bucket upload).
    """
    import kagglehub

    entry: dict = {
        "kaggle_dataset_name": name,
        "status": "failed",
        "n_source_rows": len(source_row_ids),
        "source_row_ids": source_row_ids[:50],  # cap to keep report rows tractable
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    t0 = time.time()

    # ── Download ─────────────────────────────────────────────────────────
    try:
        local = Path(kagglehub.dataset_download(name))
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"kagglehub: {type(exc).__name__}: {exc}"
        entry["elapsed_sec"] = round(time.time() - t0, 2)
        return entry

    files = [p for p in local.rglob("*") if p.is_file()]
    if not files:
        entry["error"] = f"kagglehub returned empty dir: {local}"
        entry["elapsed_sec"] = round(time.time() - t0, 2)
        return entry

    total = sum(p.stat().st_size for p in files)
    size_mb = total / (1024 * 1024)
    entry.update(
        local_path=str(local),
        files=len(files),
        bytes=total,
        size_mb=round(size_mb, 2),
        status="downloaded",
    )

    # ── Size guard ───────────────────────────────────────────────────────
    if max_size_mb is not None and size_mb > max_size_mb:
        entry["status"] = "oversize"
        entry["error"] = f"size {size_mb:.1f}MB exceeds max {max_size_mb}MB; not uploaded"
        # Free disk immediately; user opted out of this dataset via the cap
        shutil.rmtree(local, ignore_errors=True)
        entry["local_path"] = None
        entry["purged_local"] = True
        entry["elapsed_sec"] = round(time.time() - t0, 2)
        return entry

    # ── Upload (optional) ────────────────────────────────────────────────
    if bucket_id:
        from huggingface_hub import sync_bucket

        prefix = _bucket_prefix(name)
        dest = f"hf://buckets/{bucket_id}/{prefix}"
        try:
            sync_bucket(str(local), dest, verbose=False)
            entry["bucket_id"] = bucket_id
            entry["bucket_prefix"] = prefix
            entry["bucket_url"] = dest
            entry["status"] = "uploaded"
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"sync_bucket: {type(exc).__name__}: {exc}"
            entry["elapsed_sec"] = round(time.time() - t0, 2)
            return entry

    # ── Optional local cleanup ────────────────────────────────────────────
    if purge_local and entry["status"] == "uploaded":
        try:
            # Delete the version dir; kagglehub will re-fetch on demand
            shutil.rmtree(local, ignore_errors=True)
            entry["local_path"] = None
            entry["purged_local"] = True
        except Exception as exc:  # noqa: BLE001
            entry["purge_warning"] = str(exc)

    entry["elapsed_sec"] = round(time.time() - t0, 2)
    return entry


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_PARQUET,
                        help=f"Source parquet (default: {DEFAULT_PARQUET.relative_to(REPO_ROOT)})")
    parser.add_argument("--executor-filter", default="e2b",
                        help="Only include rows whose executor_type matches "
                             "(use empty string '' to include all). Default: e2b.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N unique datasets (smoke testing).")
    parser.add_argument(
        "--sort-by", default="usage", choices=["usage", "name", "random"],
        help="Order to process datasets in. `usage` puts most-referenced datasets "
             "first (Pareto-fast); `name` is alphabetical (deterministic); "
             "`random` is shuffled deterministically by --seed.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for --sort-by random.")
    parser.add_argument("--max-size-mb", type=int, default=None,
                        help="Skip (and immediately delete) datasets larger than this. "
                             "Useful to avoid multi-GB medical-imaging outliers. "
                             "Default: no limit. Try 500 for tabular-only.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--upload-bucket", default=None,
                        help="If set, sync each successful download to this HF bucket id.")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "rl" / "cache" / "fetch_all_kaggle.jsonl",
                        help="Resumable report file (JSONL).")
    parser.add_argument("--purge-local-after-upload", action="store_true",
                        help="Delete the kagglehub cache dir for each dataset after a "
                             "successful bucket upload. Saves disk; kagglehub re-fetches if needed.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt datasets currently marked failed in the report.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only — list what would be processed, exit.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if not args.source.exists():
        print(f"[err] source parquet not found: {args.source}", file=sys.stderr)
        return 1

    # 1. Scan source for unique kaggle datasets + back-links
    print(f"[scan] {args.source.relative_to(REPO_ROOT)}")
    exec_filter = args.executor_filter or None
    by_name = scan_source(args.source, exec_filter)
    print(f"  {sum(len(v) for v in by_name.values()):,} source rows "
          f"({len(by_name):,} unique kaggle datasets, executor_filter={exec_filter!r})")

    # 2. Load existing report; figure out who to skip / retry
    existing = load_report(args.report)
    print(f"[report] {args.report}: {len(existing)} entries on disk")

    skip_names: set[str] = set()
    retry_names: list[str] = []
    for name, rec in existing.items():
        status = rec.get("status")
        if status in _TERMINAL_OK:
            skip_names.add(name)
        elif status == "failed" and not args.retry_failed:
            skip_names.add(name)
        elif status == "failed" and args.retry_failed:
            retry_names.append(name)

    # 3. Build the work list — ordered by the chosen sort
    candidates = [name for name in by_name if name not in skip_names]
    if args.sort_by == "usage":
        # Most-referenced first → biggest impact for fixed compute budget
        todo = sorted(candidates, key=lambda n: (-len(by_name[n]), n))
    elif args.sort_by == "random":
        import random as _random
        rng = _random.Random(args.seed)
        todo = list(candidates)
        rng.shuffle(todo)
    else:
        todo = sorted(candidates)
    if args.limit is not None:
        todo = todo[: args.limit]

    n_skip = len(by_name) - len(todo)
    print(f"[plan] todo={len(todo)}  skip={n_skip} (already in report)  "
          f"upload={'YES → '+args.upload_bucket if args.upload_bucket else 'NO (download only)'}")

    if args.dry_run:
        for n in todo[:20]:
            print(f"  - {n}  (n_source_rows={len(by_name[n])})")
        if len(todo) > 20:
            print(f"  ... +{len(todo) - 20} more")
        return 0

    # 4. Auth + (optional) bucket ensure
    auth_src = setup_kaggle_auth()
    print(f"[auth] kaggle: {auth_src}")
    if args.upload_bucket:
        from huggingface_hub import create_bucket

        if not os.environ.get("HF_TOKEN"):
            print("[err] HF_TOKEN required for --upload-bucket", file=sys.stderr)
            return 1
        create_bucket(args.upload_bucket, exist_ok=True)
        print(f"[bucket] ensured hf://buckets/{args.upload_bucket}")

    # 5. Parallel pipeline
    print(f"[run] {args.max_workers} workers, {len(todo)} datasets", flush=True)
    t0 = time.time()
    status_counts: Counter[str] = Counter()
    err_reasons: Counter[str] = Counter()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                process_one,
                name,
                by_name[name],
                args.upload_bucket,
                args.purge_local_after_upload,
                args.max_size_mb,
            ): name
            for name in todo
        }
        pbar = tqdm(
            total=len(todo),
            desc="kaggle",
            unit="ds",
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for fut in as_completed(futures):
            entry = fut.result()
            append_report(args.report, entry)
            name = entry["kaggle_dataset_name"]
            status = entry["status"]
            status_counts[status] += 1
            if entry.get("error"):
                err_reasons[entry["error"][:60]] += 1
            if "bytes" in entry:
                total_bytes += entry["bytes"]
            tag = {"uploaded": "ok+up", "downloaded": "ok",
                   "failed": "drop", "skipped-already": "skip",
                   "oversize": "oversz"}.get(status, status)
            extra = ""
            if "size_mb" in entry:
                extra += f" {entry['size_mb']:.1f}MB"
            if "bucket_url" in entry:
                extra += f" → {entry['bucket_prefix']}"
            if entry.get("error"):
                extra += f"  err={entry['error'][:60]}"
            # Use plain print to avoid tqdm.write's WeakSet race across worker threads
            # (sync_bucket + kagglehub each spawn their own tqdm instances).
            try:
                tqdm.write(f"  [{tag:7s}] {name}  ({entry['n_source_rows']} rows){extra}")
            except RuntimeError:
                print(f"  [{tag:7s}] {name}  ({entry['n_source_rows']} rows){extra}", flush=True)
            pbar.update(1)
            pbar.set_postfix(
                ok=status_counts.get("uploaded", 0) + status_counts.get("downloaded", 0),
                fail=status_counts.get("failed", 0),
                oversz=status_counts.get("oversize", 0),
                gb=f"{total_bytes/(1024**3):.2f}",
            )
        pbar.close()

    elapsed = time.time() - t0

    # 6. Summary
    print()
    print("─" * 70)
    print(f"[done] {sum(status_counts.values())} processed in {elapsed/60:.1f} min")
    for k, v in status_counts.most_common():
        print(f"  {k:18s} {v}")
    print(f"  total bytes:       {total_bytes/(1024**3):.2f} GB")
    print(f"  report:            {args.report}")
    if err_reasons:
        print()
        print("top error reasons:")
        for reason, n in err_reasons.most_common(10):
            print(f"  {n:4d}  {reason}")
    return 0 if not status_counts.get("failed") else 0  # don't hard-fail; failures are in report


if __name__ == "__main__":
    raise SystemExit(main())
