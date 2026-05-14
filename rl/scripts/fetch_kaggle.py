"""Download Kaggle dataset(s) locally, optionally push to a HF Bucket.

Standalone, focused. Useful for:
  - inspecting a dataset before deciding to include it in a task suite
  - pre-warming the local kagglehub cache (`~/.cache/kagglehub/`)
  - one-off bucket uploads outside the full build_harbor_tasks → stage_data flow

Usage:
  # Just download one dataset locally:
  uv run python -m prepare.fetch_kaggle --name uciml/iris

  # Multiple datasets, parallel:
  uv run python -m prepare.fetch_kaggle --name uciml/iris --name CooperUnion/cardataset

  # Download AND upload to a bucket prefix:
  uv run python -m prepare.fetch_kaggle --name uciml/iris \\
      --upload-bucket AdithyaSK/jupyter-agent-test-data

  # Read names from a file (one name per line, # for comments):
  uv run python -m prepare.fetch_kaggle --names-from kaggle_list.txt --upload-bucket ...

Auth: KAGGLE_USERNAME+KAGGLE_KEY, ~/.kaggle/kaggle.json, or KAGGLE_API_TOKEN=KGAT_...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Auth
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
# Per-dataset: download (+optional upload)
# ---------------------------------------------------------------------------


def _bucket_prefix(kaggle_name: str) -> str:
    """`uciml/pima-indians-diabetes-database` → `uciml__pima-indians-diabetes-database`."""
    return kaggle_name.replace("/", "__")


def fetch(name: str, bucket_id: str | None = None) -> dict:
    """Download `name`. If `bucket_id` is given, also sync to the bucket.
    Returns a result dict.
    """
    import kagglehub

    info: dict = {"kaggle_dataset_name": name}
    try:
        local = Path(kagglehub.dataset_download(name))
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"kagglehub: {type(exc).__name__}: {exc}"
        return info

    files = [p for p in local.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    info.update(
        local_path=str(local),
        files=len(files),
        bytes=total,
        size_mb=round(total / (1024 * 1024), 2),
    )

    if bucket_id:
        from huggingface_hub import sync_bucket

        prefix = _bucket_prefix(name)
        dest = f"hf://buckets/{bucket_id}/{prefix}"
        try:
            sync_bucket(str(local), dest, verbose=False)
            info["bucket_id"] = bucket_id
            info["bucket_prefix"] = prefix
            info["bucket_url"] = dest
        except Exception as exc:  # noqa: BLE001
            info["upload_error"] = f"sync_bucket: {type(exc).__name__}: {exc}"

    return info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _read_names_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", action="append", default=[],
        help="Kaggle dataset name (owner/slug). Repeatable.",
    )
    parser.add_argument(
        "--names-from", type=Path, default=None,
        help="Read dataset names from a file (one per line; `#` comments allowed).",
    )
    parser.add_argument(
        "--upload-bucket", default=None,
        help="If set, sync each download to `hf://buckets/<this-bucket-id>/<prefix>/`.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write JSONL report (one row per dataset) to this path.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    names: list[str] = list(args.name)
    if args.names_from:
        names.extend(_read_names_file(args.names_from))
    if not names:
        print("[err] no datasets given (use --name or --names-from)", file=sys.stderr)
        return 1

    auth_src = setup_kaggle_auth()
    print(f"[auth] kaggle: {auth_src}")

    if args.upload_bucket:
        from huggingface_hub import create_bucket

        if not os.environ.get("HF_TOKEN"):
            print("[err] HF_TOKEN required for --upload-bucket", file=sys.stderr)
            return 1
        try:
            create_bucket(args.upload_bucket, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[err] create_bucket({args.upload_bucket}): {exc}", file=sys.stderr)
            return 1
        print(f"[bucket] ensured hf://buckets/{args.upload_bucket}")
    print(f"[plan] {len(names)} dataset(s), max_workers={args.max_workers}")

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(fetch, n, args.upload_bucket): n for n in names}
        for fut in as_completed(futures):
            r = fut.result()
            name = r["kaggle_dataset_name"]
            if "error" in r:
                print(f"  [drop] {name}: {r['error']}")
            elif "upload_error" in r:
                print(f"  [partial] {name}: downloaded ({r['size_mb']} MB) but upload failed — {r['upload_error']}")
            elif args.upload_bucket:
                print(f"  [ok+upload] {name}: {r['files']} file(s), {r['size_mb']} MB → {r['bucket_url']}")
            else:
                print(f"  [ok] {name}: {r['files']} file(s), {r['size_mb']} MB at {r['local_path']}")
            results.append(r)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"[report] {args.report}")

    n_ok = sum(1 for r in results if "error" not in r)
    elapsed = time.time() - t0
    print(f"\n[done] {n_ok}/{len(names)} downloaded in {elapsed:.1f}s.")
    return 0 if n_ok == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())
