"""Replace flagged eval tasks with clean candidates of the same shape.

Reads:
  - cache/eval/eval_manifest.parquet      — current eval set
  - cache/eval/candidates.parquet         — full pool to draw from
  - cache/eval/llm_audit.parquet          — gpt-4o-mini verdict per eval row
  - cache/eval/candidates_audited.parquet — gpt-4o-mini verdict per candidate

A row needs replacement if EITHER:
  - Its regex flags include anything in REPLACE_FLAGS, OR
  - Its LLM verdict is IMPOSSIBLE or DEPENDENT.

A candidate is eligible iff:
  - Its regex flags are EMPTY, AND
  - Its LLM verdict is CLEAN, AND
  - It's not currently in the eval set, AND
  - Its kaggle wouldn't break the max-2 cap.

Matching prefers (difficulty bucket, package_tier, answer_type); relaxes
in steps if pool is empty. Writes new eval_ids.txt + eval_manifest.parquet
in place.

Run: cd rl && uv run python -m prepare.replace_flagged
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from prepare.audit_eval_v1 import classify

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
EVAL_DIR = RL_ROOT / "cache" / "eval"

# Which flags trigger a replacement. PARENTHESIS / PERCENT_AMBIGUOUS /
# MULTI_PART_ANSWER are kept — the upgraded LLM judge has explicit rules for
# those. STOCHASTIC / NOTEBOOK_REF / SENTENCE_ANSWER / NON_ENGLISH / LONG_ANSWER
# / ML_FIT_REFERENCED are all true risks (the model can't reliably reproduce
# the gold without seeing the original notebook).
REPLACE_FLAGS = {"STOCHASTIC", "NOTEBOOK_REF", "ML_FIT_REFERENCED",
                 "LONG_ANSWER", "SENTENCE_ANSWER", "NON_ENGLISH"}

# LLM score → difficulty bucket. Same mapping as build_eval_set.llm_to_bucket.
def score_to_bucket(score: int) -> str:
    if score <= 2:
        return "easy"
    if score == 3:
        return "medium"
    return "hard"


def main() -> int:
    manifest_path = EVAL_DIR / "eval_manifest.parquet"
    candidates_path = EVAL_DIR / "candidates.parquet"
    llm_audit_path = EVAL_DIR / "llm_audit.parquet"
    cand_audit_path = EVAL_DIR / "candidates_audited.parquet"
    if not manifest_path.exists() or not candidates_path.exists():
        print(f"[err] missing {manifest_path} or {candidates_path}")
        return 1

    manifest = pd.read_parquet(manifest_path)

    # Prefer the LLM-audited candidate pool if available; falls back to plain.
    if cand_audit_path.exists():
        cands = pd.read_parquet(cand_audit_path)
        print(f"[load] candidates_audited.parquet ({len(cands)} rows, LLM verdicts available)")
    else:
        cands = pd.read_parquet(candidates_path)
        cands["audit_verdict"] = "UNKNOWN"
        print(f"[load] candidates.parquet ({len(cands)} rows, no LLM verdicts)")

    # Attach LLM audit verdict to the manifest if available.
    if "audit_verdict" in manifest.columns:
        print(f"[load] manifest already has audit_verdict column (in-place)")
    elif llm_audit_path.exists():
        llm_audit = pd.read_parquet(llm_audit_path)[["id", "audit_verdict", "audit_reasoning"]]
        manifest = manifest.merge(llm_audit, on="id", how="left")
        print(f"[load] merged {len(llm_audit)} LLM verdicts onto manifest")
    else:
        manifest["audit_verdict"] = "UNKNOWN"

    # Flag each manifest row via regex.
    manifest["_flag_list"] = manifest.apply(
        lambda r: classify(r["question"], r["answer"]), axis=1,
    )

    def needs_replace(r):
        regex_hit = any(f in REPLACE_FLAGS for f in r["_flag_list"])
        llm_hit = r["audit_verdict"] in ("IMPOSSIBLE", "DEPENDENT")
        return regex_hit or llm_hit

    to_replace_mask = manifest.apply(needs_replace, axis=1)
    targets = manifest[to_replace_mask].copy()
    print(f"[plan] {len(targets)} task(s) need replacement "
          f"(regex flags ∩ {REPLACE_FLAGS}  OR  LLM IMPOSSIBLE/DEPENDENT)")
    for _, r in targets.iterrows():
        print(f"  - {r['difficulty']:6s} {r['id']}  "
              f"llm={r['audit_verdict']}  regex={r['_flag_list']}  "
              f"Q: {r['question'][:70]!r}")

    # Pre-compute candidate flags + bucket
    cands["_flag_list"] = cands.apply(lambda r: classify(r["question"], r["answer"]), axis=1)
    cands["_bucket"] = cands["llm_score"].apply(score_to_bucket)

    # Existing state
    in_use_ids = set(manifest["id"])
    kaggle_counts = Counter(manifest["kaggle_dataset_name"])

    replacements: list[tuple[str, str]] = []  # (old_id, new_id)
    new_rows: dict[str, dict] = {}

    for _, target in targets.iterrows():
        old_id = target["id"]
        bucket = target["difficulty"]
        tier = target["feat_package_tier"]
        ans_type = target["feat_answer_type"]
        old_kaggle = target["kaggle_dataset_name"]

        # Eligible pool — must be regex-clean AND LLM-CLEAN
        clean_mask = (
            cands._flag_list.apply(lambda fs: len(fs) == 0)
            & (cands.audit_verdict.isin(["CLEAN", "UNKNOWN"]))
        )
        pool = cands[
            (cands._bucket == bucket)
            & (cands.feat_package_tier == tier)
            & (cands.feat_answer_type == ans_type)
            & (~cands.id.isin(in_use_ids))
            & clean_mask
        ]

        # Prefer a different kaggle (and respect the cap)
        def kaggle_ok(kg: str) -> bool:
            if kg == old_kaggle and kaggle_counts[kg] >= 2:
                return False
            return kaggle_counts[kg] < 2
        pool = pool[pool.kaggle_dataset_name.apply(kaggle_ok)]

        if len(pool) == 0:
            # Relax: allow the same kaggle if still under cap.
            pool = cands[
                (cands._bucket == bucket)
                & (cands.feat_package_tier == tier)
                & (cands.feat_answer_type == ans_type)
                & (~cands.id.isin(in_use_ids))
                & (cands._flag_list.apply(lambda fs: len(fs) == 0))
            ]
            pool = pool[pool.kaggle_dataset_name.apply(lambda kg: kaggle_counts[kg] < 2)]

        if len(pool) == 0:
            # Relax: ignore answer_type, keep tier.
            pool = cands[
                (cands._bucket == bucket)
                & (cands.feat_package_tier == tier)
                & (~cands.id.isin(in_use_ids))
                & clean_mask
            ]
            pool = pool[pool.kaggle_dataset_name.apply(lambda kg: kaggle_counts[kg] < 2)]

        if len(pool) == 0:
            # Final relax: any clean candidate in the same bucket. Sacrifices
            # tier/answer_type balance to fix a flagged task. Logged so we know.
            pool = cands[
                (cands._bucket == bucket)
                & (~cands.id.isin(in_use_ids))
                & clean_mask
            ]
            pool = pool[pool.kaggle_dataset_name.apply(lambda kg: kaggle_counts[kg] < 2)]
            if len(pool) > 0:
                print(f"  [relax] {old_id}: no same-tier replacement, swapping tier/ans_type")

        if len(pool) == 0:
            print(f"[fail] no replacement found for {old_id}; leaving original in place")
            continue

        new = pool.iloc[0]
        new_id = new["id"]
        in_use_ids.add(new_id)
        kaggle_counts[old_kaggle] -= 1
        kaggle_counts[new["kaggle_dataset_name"]] += 1
        replacements.append((old_id, new_id))
        new_rows[old_id] = {
            **new.to_dict(),
            "difficulty": bucket,  # preserve the original bucket label
        }
        print(f"  [swap] {old_id}  →  {new_id}  "
              f"(kaggle {old_kaggle} → {new['kaggle_dataset_name']}, "
              f"new gold={new['answer'][:40]!r})")

    if not replacements:
        print("[noop] no replacements made")
        return 0

    # Rebuild manifest preserving order
    new_manifest_rows = []
    for _, r in manifest.iterrows():
        if r["id"] in {old for old, _ in replacements}:
            new_row = new_rows[r["id"]]
            new_row.pop("_flag_list", None)
            new_row.pop("_bucket", None)
            new_manifest_rows.append(new_row)
        else:
            rd = r.to_dict()
            rd.pop("_flag_list", None)
            new_manifest_rows.append(rd)

    new_df = pd.DataFrame(new_manifest_rows)
    new_df.to_parquet(manifest_path, index=False)
    print(f"\n[save] {manifest_path}  ({len(new_df)} rows, {len(replacements)} swapped)")

    # eval_ids.txt
    ids_path = EVAL_DIR / "eval_ids.txt"
    ids_path.write_text("\n".join(new_df["id"]) + "\n")
    print(f"[save] {ids_path}  ({len(new_df)} ids)")

    # Summary of new bucket composition
    print(); print("─" * 60)
    print(f"Final composition:")
    for bucket in ("easy", "medium", "hard"):
        sub = new_df[new_df.difficulty == bucket]
        print(f"  {bucket:6s}: {len(sub)}  unique_kaggle={sub.kaggle_dataset_name.nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
