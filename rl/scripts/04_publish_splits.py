"""Publish data/splits/{eval,train}_manifest.parquet to the HF Hub.

Creates / updates `AdithyaSK/data_agent_rl` as a private dataset with two
splits, plus splits.yaml and a generated README. This is the source of truth
for which rows go into eval vs. train.

Usage:
    uv run --project rl python rl/scripts/04_publish_splits.py
    # or with overrides:
    uv run --project rl python rl/scripts/04_publish_splits.py \
        --repo-id AdithyaSK/data_agent_rl --private
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi


# Repo-relative paths
RL = Path(__file__).resolve().parents[1]


def _load_env() -> str:
    """Load HF_TOKEN from the repo's .env (project root or rl/.env)."""
    for candidate in [RL.parent / ".env", RL / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN not found in .env or environment")
    return tok


def _build_readme(splits_yaml: dict, repo_id: str) -> str:
    eval_dist = splits_yaml.get("reward_mode_distribution_eval", {})
    train_dist = splits_yaml.get("reward_mode_distribution_train", {})
    eval_n = splits_yaml.get("eval_size_actual", 0)
    train_n = splits_yaml.get("train_size_actual", 0)
    seed = splits_yaml.get("seed")

    def _dist_row(eval_d: dict, train_d: dict) -> str:
        keys = sorted(set(eval_d) | set(train_d), key=lambda k: -(eval_d.get(k, 0) or 0))
        rows = []
        for k in keys:
            ev = eval_d.get(k, 0) or 0
            tr = train_d.get(k, 0) or 0
            rows.append(f"| `{k}` | {ev:,} | {tr:,} |")
        return "\n".join(rows)

    return f"""---
license: mit
language:
- en
pretty_name: data-agent RL splits (v1)
tags:
- data-science
- code-agent
- reinforcement-learning
- jupyter
configs:
- config_name: default
  data_files:
  - split: eval
    path: data/eval-*.parquet
  - split: train
    path: data/train-*.parquet
---

# {repo_id}

Source-of-truth eval/train split for the **data-agent** RL pipeline.

Derived from `jupyter-agent/jupyter-agent-dataset` by:
1. Filtering to `executor_type == "e2b"` (29,555 rows survive; `executor_type == "llm"` rows have mismatched Kaggle metadata and are dropped).
2. Per-row classification of the gold answer into a default reward grading mode (see `reward_mode_initial`).
3. Stratified sampling by `(reward_mode_initial × package_tier)` with a max-K-per-Kaggle cap on the eval split (K={splits_yaml.get('eval_k_per_kaggle')}) to prevent dataset leakage / dominance.

## Splits

| Split | Rows |
|---|---|
| `eval` | {eval_n:,} (candidate pool — the actual eval set is whatever survives Stage-2 frontier verification) |
| `train` | {train_n:,} |

Reproducibility: `seed = {seed}`. Full config in [`splits.yaml`](splits.yaml).

## Per-row schema

| Column | Type | Source |
|---|---|---|
| `id` | str | original dataset |
| `question` | str | original |
| `answer` | str | original gold (may be wrong — see verification stage) |
| `kaggle_dataset_name` | str | original |
| `executor_type` | str | original (always `"e2b"` here) |
| `files_used` | list\\[str\\] | original |
| `packages_used` | list\\[str\\] | original |
| `edu_score` | int | original |
| `answer_norm` | str | classifier — answer with `%`, parens, trailing units stripped |
| `reward_mode_initial` | str | classifier — see below |
| `q_word_count`, `answer_len`, `n_files`, `n_packages` | int | classifier |
| `package_tier` | int | classifier (0 pandas-only / 1 sklearn-tier / 2 deep-learning / 3 other) |

## Reward-mode taxonomy (`reward_mode_initial`)

| Mode | Eval | Train |
|---|---|---|
{_dist_row(eval_dist, train_dist)}

### Grader behaviour per mode

| Mode | Description |
|---|---|
| `numeric` | float comparison with abs + rel tolerance — free, deterministic |
| `exact_short` | case-insensitive string equality, ≤5 tokens — free |
| `exact_bool` | yes/no/true/false normalization — free |
| `list` / `list_csv` | parse as list, set/order compare — free |
| `flexible` | exact → numeric → llm-judge fallback — cheap |
| `llm_judge_long` | judge-only, for multi-sentence answers — judge call required |

After Stage-2 frontier verification, additional columns are added: `verifiable`, `reward_mode_final`, `gold_corrected`, `gold_original`, `pass_rate`.

## Citation

```bibtex
@dataset{{adithyask_data_agent_rl_2026,
  author = {{AdithyaSK}},
  title = {{data_agent_rl}},
  year = {{2026}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--repo-id", default="AdithyaSK/data_agent_rl")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()

    splits_dir = RL / args.splits_dir
    eval_parquet = splits_dir / "eval_manifest.parquet"
    train_parquet = splits_dir / "train_manifest.parquet"
    splits_yaml_path = splits_dir / "splits.yaml"

    for p in [eval_parquet, train_parquet, splits_yaml_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    print(f"Loading HF token from .env …")
    token = _load_env()

    print(f"Reading manifests …")
    eval_df = pd.read_parquet(eval_parquet)
    train_df = pd.read_parquet(train_parquet)
    print(f"  eval:  {len(eval_df):,} rows × {len(eval_df.columns)} cols")
    print(f"  train: {len(train_df):,} rows × {len(train_df.columns)} cols")

    # build HF datasets
    ds = DatasetDict({
        "eval":  Dataset.from_pandas(eval_df,  preserve_index=False),
        "train": Dataset.from_pandas(train_df, preserve_index=False),
    })

    print(f"\nPushing to {args.repo_id} (private={args.private}) …")
    ds.push_to_hub(args.repo_id, private=args.private, token=token)
    print("  parquet uploaded")

    # README + splits.yaml
    splits_yaml = yaml.safe_load(splits_yaml_path.read_text())
    api = HfApi(token=token)
    readme = _build_readme(splits_yaml, args.repo_id)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=splits_yaml_path,
        path_in_repo="splits.yaml",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"  README.md + splits.yaml uploaded")

    print(f"\nDone. View at:  https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
