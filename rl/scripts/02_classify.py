"""Classify every row in the cached dataset into a default REWARD_MODE.

Reads `data/raw/thinking.parquet` (non_thinking is identical for our purposes —
same id/question/answer/kaggle; only the `messages` trace differs).

Outputs `data/classified.parquet` with these per-row columns added:

    reward_mode_initial    str   — bucket from the classifier rules below
    answer_norm            str   — answer with %, trailing units, parens stripped
    q_word_count           int
    answer_len             int
    n_files                int
    n_packages             int
    package_tier           int   — 0 pandas-only / 1 sklearn-tier / 2 deep-learning / 3 other

Drops rows where `executor_type != "e2b"` (broken Kaggle↔file metadata) and
rows with no Kaggle dataset name. The result is the runnable pool of ~29,561.

Usage:
    uv run --project rl python rl/scripts/02_classify.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# ---- classifier rules ---------------------------------------------------------

# A number: optionally signed, plain int or decimal, optional thousands commas,
# optional scientific notation. Examples: "0.072297", "-12", "1,000", "1.5e-3".
_NUMERIC_RE = re.compile(r"-?\d{1,3}(,\d{3})*(\.\d+)?|-?\d+(\.\d+)?([eE][-+]?\d+)?")

_BOOL_SET = {"yes", "no", "true", "false", "y", "n", "t", "f"}

# Trailing-unit suffixes we strip to normalise answers. Order matters
# (longest first) to avoid eating into the number itself.
_UNIT_SUFFIXES = ["%", " %", " percent", " bytes", " seconds", " sec", " s",
                  " ms", " minutes", " min", " hours", " hr", " kg", " g",
                  " mb", " gb", " kb", " mph", " mhz", " ghz"]


def _normalize_answer(a: str) -> str:
    """Strip common formatting noise without changing semantic content."""
    if a is None:
        return ""
    s = str(a).strip()
    if s == "":
        return ""
    # strip parenthetical commentary like "0.97 (between radius_mean and …)"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    # strip trailing units
    for u in _UNIT_SUFFIXES:
        if s.lower().endswith(u):
            s = s[: -len(u)].rstrip()
            break
    return s


def classify_answer(a: str | None) -> str:
    """Return one of the reward-mode buckets defined in PLAN.md."""
    if a is None:
        return "missing"
    s = str(a).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "missing"

    # 1. pure number (with or without trailing units handled by the unit-strip pass
    #    via answer_norm — but for the initial classification we use the raw form)
    if _NUMERIC_RE.fullmatch(s):
        return "numeric"

    s_low = s.lower()
    # 2. boolean shorthand
    if s_low in _BOOL_SET:
        return "exact_bool"

    # 3. list-literal forms
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        return "list"
    # 4. comma-separated short items (3+ commas, no sentence punctuation, short)
    if s.count(",") >= 2 and "." not in s and len(s) < 200 and " and " not in s_low:
        return "list_csv"

    # 5. short canonical string (≤ 5 tokens, no sentence terminator)
    toks = s.split()
    if len(toks) <= 5 and not re.search(r"[.!?]", s):
        return "exact_short"

    # 6. long / multi-sentence → llm-judge
    if len(toks) > 20 or re.search(r"[.!?]\s+[A-Z]", s):
        return "llm_judge_long"

    # 7. everything else → flexible (exact → numeric → judge fallback at run time)
    return "flexible"


# ---- package-tier feature ----------------------------------------------------

_TIER_0 = {"pandas", "numpy", "matplotlib"}
_TIER_1 = {"sklearn", "scikit-learn", "scipy", "seaborn", "statsmodels", "plotly"}
_TIER_2 = {"tensorflow", "tf", "torch", "pytorch", "keras", "transformers",
           "xgboost", "lightgbm", "catboost"}


def _package_tier(pkgs) -> int:
    if pkgs is None:
        return 3
    names = {str(p).strip().lower() for p in pkgs}
    if names & _TIER_2:
        return 2
    if names & _TIER_1:
        return 1
    if names and names.issubset(_TIER_0):
        return 0
    return 3


# ---- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/raw/thinking.parquet",
                    help="path to the cached parquet (defaults to thinking split)")
    ap.add_argument("--output", default="data/classified.parquet")
    args = ap.parse_args()

    rl_root = Path(__file__).resolve().parents[1]
    src = rl_root / args.input
    dst = rl_root / args.output

    print(f"Reading {src} …")
    df = pd.read_parquet(src, columns=[
        "id", "question", "answer", "kaggle_dataset_name",
        "executor_type", "files_used", "packages_used", "edu_score",
    ])
    print(f"  loaded {len(df):,} rows")

    # dedup by id (one row in thinking has a duplicate)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    print(f"  after dedup: {len(df):,}")

    # filter to runnable pool
    before = len(df)
    df = df[df["executor_type"] == "e2b"]
    print(f"  after executor_type=='e2b' filter: {len(df):,} (-{before - len(df):,})")
    before = len(df)
    df = df[df["kaggle_dataset_name"].notna() & (df["kaggle_dataset_name"].astype(str) != "")]
    print(f"  after kaggle non-null filter:      {len(df):,} (-{before - len(df):,})")

    # feature columns
    df = df.copy()
    df["answer_norm"] = df["answer"].apply(_normalize_answer)
    df["reward_mode_initial"] = df["answer"].apply(classify_answer)
    df["q_word_count"] = df["question"].astype(str).str.split().str.len()
    df["answer_len"] = df["answer"].astype(str).str.len()
    df["n_files"] = df["files_used"].apply(lambda x: len(x) if x is not None else 0)
    df["n_packages"] = df["packages_used"].apply(lambda x: len(x) if x is not None else 0)
    df["package_tier"] = df["packages_used"].apply(_package_tier)

    # drop the few "missing" rows
    before = len(df)
    df = df[df["reward_mode_initial"] != "missing"]
    print(f"  after dropping reward_mode_initial=='missing': {len(df):,} (-{before - len(df):,})")

    # report distribution
    print("\nReward-mode distribution:")
    counts = df["reward_mode_initial"].value_counts()
    for k, v in counts.items():
        print(f"  {k:18s} {v:7,d}  ({v/len(df)*100:5.1f}%)")

    print("\nPackage-tier distribution:")
    for tier, label in [(0, "pandas-only"), (1, "+sklearn/scipy"), (2, "deep-learning"), (3, "other")]:
        n = (df["package_tier"] == tier).sum()
        print(f"  tier {tier} ({label:18s}): {n:7,d}  ({n/len(df)*100:5.1f}%)")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    sz_mb = dst.stat().st_size / 1024 / 1024
    print(f"\nWrote {dst}  ({len(df):,} rows, {sz_mb:.1f} MB, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
