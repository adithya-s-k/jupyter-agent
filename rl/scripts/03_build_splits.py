"""Sample a 1,000-row eval pool and ~28,500-row train pool from classified rows.

Stratifies by (reward_mode_initial × package_tier) and caps the number of rows
per Kaggle dataset to prevent leakage / dominance. Deterministic given the
seed in `splits.yaml`.

Outputs:
    data/splits/eval_ids.txt              (1000 lines)
    data/splits/train_ids.txt             (~28,500 lines)
    data/splits/eval_manifest.parquet     (per-id rows from classified.parquet)
    data/splits/train_manifest.parquet
    data/splits/splits.yaml               (seed, strata config, content hashes)

Usage:
    uv run --project rl python rl/scripts/03_build_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---- defaults (overridable via CLI) ------------------------------------------

DEFAULT_EVAL_SIZE = 1000
DEFAULT_SEED = 42
DEFAULT_EVAL_K_PER_KAGGLE = 4   # at most 4 rows per Kaggle dataset in eval
DEFAULT_TRAIN_K_PER_KAGGLE = 0  # 0 = no cap; train keeps all remaining rows.
# Median Kaggle dataset has 8 rows, top has 1,698. Capping aggressively
# (K=8) drops 80% of the train pool. Keep them all and let RL sampling weight.


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sample_eval(df: pd.DataFrame, target: int, k_per_kaggle: int, seed: int) -> pd.DataFrame:
    """Stratified sample, max-K-per-Kaggle, proportional to natural distribution."""
    rng = np.random.default_rng(seed)
    pool_size = len(df)

    # Strata = (reward_mode_initial, package_tier)
    strata = df.groupby(["reward_mode_initial", "package_tier"]).size().reset_index(name="n")
    strata["share"] = strata["n"] / pool_size
    # Target rows per stratum, floored — then top up the deficit from largest strata.
    strata["target_float"] = strata["share"] * target
    strata["target"] = np.floor(strata["target_float"]).astype(int)
    deficit = target - int(strata["target"].sum())
    # Hand the deficit to strata with the largest fractional remainders
    if deficit > 0:
        rem = (strata["target_float"] - strata["target"]).sort_values(ascending=False)
        for i in rem.head(deficit).index:
            strata.at[i, "target"] = int(strata.at[i, "target"]) + 1

    picked = []
    for _, s in strata.iterrows():
        target_n = int(s["target"])
        if target_n <= 0:
            continue
        sub = df[(df["reward_mode_initial"] == s["reward_mode_initial"]) &
                 (df["package_tier"] == s["package_tier"])].copy()
        # Greedy: shuffle within stratum, walk in order, cap per-Kaggle.
        sub = sub.sample(frac=1, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
        per_kaggle: dict[str, int] = {}
        chosen = []
        for _, r in sub.iterrows():
            k = r["kaggle_dataset_name"]
            if per_kaggle.get(k, 0) >= k_per_kaggle:
                continue
            chosen.append(r)
            per_kaggle[k] = per_kaggle.get(k, 0) + 1
            if len(chosen) >= target_n:
                break
        # If still under target (every Kaggle in this stratum hit the cap),
        # relax cap and fill from remaining rows in this stratum.
        if len(chosen) < target_n:
            already = {r["id"] for r in chosen}
            extras = [r for _, r in sub.iterrows() if r["id"] not in already][: target_n - len(chosen)]
            chosen.extend(extras)
        picked.extend(chosen)

    return pd.DataFrame(picked).reset_index(drop=True)


def _cap_per_kaggle(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Cap a dataframe at k rows per Kaggle dataset, keeping the rest in order."""
    rng = np.random.default_rng(seed)
    shuffled = df.sample(frac=1, random_state=rng.integers(0, 2**31 - 1)).reset_index(drop=True)
    per_kaggle: dict[str, int] = {}
    keep_mask = np.zeros(len(shuffled), dtype=bool)
    for i, kag in enumerate(shuffled["kaggle_dataset_name"]):
        if per_kaggle.get(kag, 0) < k:
            keep_mask[i] = True
            per_kaggle[kag] = per_kaggle.get(kag, 0) + 1
    return shuffled[keep_mask].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="data/classified.parquet")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval-k-per-kaggle", type=int, default=DEFAULT_EVAL_K_PER_KAGGLE)
    ap.add_argument("--train-k-per-kaggle", type=int, default=DEFAULT_TRAIN_K_PER_KAGGLE)
    args = ap.parse_args()

    rl_root = Path(__file__).resolve().parents[1]
    src = rl_root / args.classified
    out_dir = rl_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src} …")
    df = pd.read_parquet(src)
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    # --- eval split: stratified sample, max-K-per-Kaggle
    print(f"\nSampling eval pool (target {args.eval_size}, "
          f"max {args.eval_k_per_kaggle} per Kaggle, seed {args.seed})…")
    eval_df = _sample_eval(df, args.eval_size, args.eval_k_per_kaggle, args.seed)
    print(f"  eval rows: {len(eval_df):,}")

    # --- train split: rest, optionally per-Kaggle capped
    train_pool = df[~df["id"].isin(eval_df["id"])].copy()
    if args.train_k_per_kaggle > 0:
        print(f"\nCapping train pool at {args.train_k_per_kaggle} per Kaggle…")
        train_df = _cap_per_kaggle(train_pool, args.train_k_per_kaggle, args.seed + 1)
        print(f"  train rows: {len(train_df):,} (from {len(train_pool):,} candidates)")
    else:
        print(f"\nNo train cap — keeping all {len(train_pool):,} remaining rows.")
        train_df = train_pool.sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)

    # --- sanity: zero leakage of Kaggle datasets? we DO allow it (a Kaggle can
    # appear in both eval and train) but cap per-side prevents one dataset
    # from dominating either side. Report overlap for transparency.
    shared_kag = set(eval_df["kaggle_dataset_name"]) & set(train_df["kaggle_dataset_name"])
    print(f"\n  Kaggle datasets unique to eval:  {len(set(eval_df['kaggle_dataset_name']) - shared_kag):,}")
    print(f"  Kaggle datasets unique to train: {len(set(train_df['kaggle_dataset_name']) - shared_kag):,}")
    print(f"  Kaggle datasets shared:          {len(shared_kag):,}")

    # --- write
    print("\nWriting outputs…")
    eval_ids = out_dir / "eval_ids.txt"
    train_ids = out_dir / "train_ids.txt"
    eval_manifest = out_dir / "eval_manifest.parquet"
    train_manifest = out_dir / "train_manifest.parquet"

    eval_ids.write_text("\n".join(eval_df["id"].astype(str).tolist()) + "\n")
    train_ids.write_text("\n".join(train_df["id"].astype(str).tolist()) + "\n")
    eval_df.to_parquet(eval_manifest, index=False)
    train_df.to_parquet(train_manifest, index=False)

    # --- splits.yaml (reproducibility record)
    splits_yaml = {
        "version": "data_agent_rl_v1",
        "seed": args.seed,
        "eval_size_target": args.eval_size,
        "eval_size_actual": len(eval_df),
        "train_size_actual": len(train_df),
        "eval_k_per_kaggle": args.eval_k_per_kaggle,
        "train_k_per_kaggle": args.train_k_per_kaggle,
        "stratification": "(reward_mode_initial, package_tier)",
        "classifier_input_sha256_16": _file_hash(src),
        "eval_ids_sha256_16": _file_hash(eval_ids),
        "train_ids_sha256_16": _file_hash(train_ids),
        "reward_mode_distribution_eval": eval_df["reward_mode_initial"].value_counts().to_dict(),
        "reward_mode_distribution_train": train_df["reward_mode_initial"].value_counts().to_dict(),
        "package_tier_distribution_eval": eval_df["package_tier"].value_counts().to_dict(),
        "kaggle_unique_eval": int(eval_df["kaggle_dataset_name"].nunique()),
        "kaggle_unique_train": int(train_df["kaggle_dataset_name"].nunique()),
        "kaggle_shared": len(shared_kag),
    }
    (out_dir / "splits.yaml").write_text(yaml.safe_dump(splits_yaml, sort_keys=False))

    print(f"  {eval_ids}")
    print(f"  {train_ids}")
    print(f"  {eval_manifest}  ({eval_manifest.stat().st_size / 1024:.1f} KB)")
    print(f"  {train_manifest} ({train_manifest.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {out_dir / 'splits.yaml'}")

    # --- summary
    print("\nEval reward-mode distribution:")
    for k, v in eval_df["reward_mode_initial"].value_counts().items():
        print(f"  {k:18s} {v:5d}  ({v/len(eval_df)*100:5.1f}%)")

    print("\nTrain reward-mode distribution:")
    for k, v in train_df["reward_mode_initial"].value_counts().items():
        print(f"  {k:18s} {v:6d}  ({v/len(train_df)*100:5.1f}%)")


if __name__ == "__main__":
    main()
