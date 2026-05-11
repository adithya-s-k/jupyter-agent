"""Cache the public jupyter-agent dataset locally so subsequent steps run offline.

Pulls both splits of `jupyter-agent/jupyter-agent-dataset` and materializes them to
`rl/cache/raw/{thinking,non_thinking}.parquet`. Full dataset is ~72 GB; use
`--limit N` for a smoke run (each split capped at N rows).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
CACHE_ROOT = RL_ROOT / "cache"
RAW_DIR = CACHE_ROOT / "raw"
HF_CACHE = CACHE_ROOT / "hf-datasets"

SPLITS = ["thinking", "non_thinking"]
DATASET_ID = "jupyter-agent/jupyter-agent-dataset"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap each split at N rows (for smoke runs).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for a full download (~72 GB).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help="Which splits to cache.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
    # huggingface_hub 1.x uses hf_xet for parallel chunked downloads; the old
    # HF_HUB_ENABLE_HF_TRANSFER flag is a no-op now. Enable max-throughput xet.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    if args.limit is None and not args.confirm:
        print(
            "Full download is ~72 GB. Re-run with --confirm to proceed, "
            "or pass --limit N for a smoke run.",
            file=sys.stderr,
        )
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for split in args.splits:
        out_path = RAW_DIR / f"{split}.parquet"
        if out_path.exists() and args.limit is None:
            print(f"[skip] {out_path} already exists.")
            continue
        print(f"[load] {DATASET_ID} split={split} limit={args.limit}")
        ds = load_dataset(DATASET_ID, split=split, cache_dir=str(HF_CACHE))
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[write] {out_path} ({len(ds)} rows)")
        ds.to_parquet(str(out_path))

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
