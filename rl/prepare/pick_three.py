"""Pick 3 representative tasks from the 150-task eval set — 1 per difficulty.

Reads `cache/eval/eval_manifest.parquet` (produced by `build_eval_set.py`) and
writes a tiny `three_ids.txt` consumable by `build_harbor_tasks --ids-from`.

Selection rule (deterministic given the seed):
  - One row from each (difficulty, package_tier) it can hit, preferring
    answer_type=numeric (graders are most reliable there).
  - Falls back to whatever the bucket has if numeric isn't available.

Usage:
  uv run python -m prepare.pick_three
  uv run python -m prepare.pick_three --seed 7 --out cache/eval/three_ids.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=RL_ROOT / "cache" / "eval" / "eval_manifest.parquet")
    p.add_argument("--out", type=Path,
                   default=RL_ROOT / "cache" / "eval" / "three_ids.txt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"[err] {args.manifest} missing — run prepare.build_eval_set first")
        return 1

    df = pd.read_parquet(args.manifest)
    rng = random.Random(args.seed)

    picks: list[dict] = []
    for bucket in ("easy", "medium", "hard"):
        sub = df[df.difficulty == bucket]
        # Prefer numeric answers (cleanest grader signal).
        numeric = sub[sub.feat_answer_type == "numeric"]
        pool = numeric if len(numeric) >= 1 else sub
        ids = list(pool.id.values)
        rng.shuffle(ids)
        chosen = pool[pool.id == ids[0]].iloc[0].to_dict()
        picks.append(chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(p["id"] for p in picks) + "\n")

    print(f"[picked] {len(picks)} ids → {args.out}\n")
    for p_ in picks:
        print(f"  [{p_['difficulty']:6s}] {p_['id']}")
        print(f"           kaggle:   {p_['kaggle_dataset_name']}")
        print(f"           question: {(p_['question'] or '')[:120]}")
        print(f"           gold:     {p_['answer']!r}")
        print(f"           llm:      score={p_['llm_score']}  "
              f"tier={p_['feat_package_tier']}  type={p_['feat_answer_type']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
