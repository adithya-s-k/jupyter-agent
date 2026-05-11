"""Push the local Harbor task suite to HuggingFace Hub as a dataset repo.

For `--name test`:
  rl/harbor/tasks/jupyter-agent-test/   →   hf://datasets/<user>/jupyter-agent-test-harbor

This is what makes the suite shareable / clone-and-runnable / connectable
to openreward.ai (their Harbor-mode auto-deploy reads from a GitHub or HF
dataset repo with this exact layout).

Usage:
  uv run python -m prepare.push_harbor --name test
  uv run python -m prepare.push_harbor --name test --private
  uv run python -m prepare.push_harbor --name test --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from prepare.build_harbor_tasks import derive_names

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", required=True,
        help="Slug (must match prepare.build_harbor_tasks --name).",
    )
    parser.add_argument(
        "--user", default=None,
        help="HF namespace (defaults to whatever build_harbor_tasks used).",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message", default=None,
        help="Commit message. Defaults to 'sync N tasks for <slug>'.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("HF_TOKEN"):
        print("[err] HF_TOKEN missing in .env", file=sys.stderr)
        return 1

    names = derive_names(args.name, user=args.user) if args.user else derive_names(args.name)
    suite_dir = Path(names["local_dir"])
    if not suite_dir.exists():
        print(f"[miss] {suite_dir} — run prepare.build_harbor_tasks --name {args.name} first.")
        return 1

    # Quick sanity count.
    task_dirs = [p for p in suite_dir.iterdir() if p.is_dir() and (p / "task.toml").exists()]
    n_tasks = len(task_dirs)
    repo_id = names["repo_id"]

    print(f"[plan] push {n_tasks} task(s) from {suite_dir}")
    print(f"  → hf://datasets/{repo_id}  (private={args.private})")

    if args.dry_run:
        print("[dry-run] nothing pushed.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()

    # 1. create_repo idempotent
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"[err] create_repo({repo_id}): {exc}", file=sys.stderr)
        return 1
    print(f"[repo] ensured {repo_id} (dataset)")

    # 2. upload the folder. Note: `data/**` is EXCLUDED — that mirror is the
    #    local-only copy; the canonical remote copy is the HF Bucket. Tasks
    #    pull from `HF_BUCKET` at runtime, not from the dataset repo.
    commit_msg = args.commit_message or f"sync {n_tasks} task(s) for slug={args.name}"
    try:
        api.upload_folder(
            folder_path=str(suite_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
            ignore_patterns=[
                "data/**",                 # local data mirror; bucket is canonical
                "__pycache__", "*.pyc",
                ".DS_Store", "stage_manifest.jsonl",
                "dropped.jsonl",            # implementation detail, not part of the suite
            ],
        )
    except Exception as exc:
        print(f"[err] upload_folder: {exc}", file=sys.stderr)
        return 1

    print(f"\n[done] pushed to https://huggingface.co/datasets/{repo_id}")
    print(f"  - manifest:        manifest.jsonl")
    print(f"  - tasks:           {n_tasks}")
    print(f"  - data bucket:     hf://buckets/{names['bucket_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
