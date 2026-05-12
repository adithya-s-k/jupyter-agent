"""LLM-based audit pass over eval tasks.

For each `(question, gold_answer)` row, ask gpt-4o-mini whether the gold is
**reproducible from the dataset alone**, OR whether it depends on the
original notebook's specific choices (which model, which split, which
random seed, which features kept after manual selection, etc.).

Verdict (Pydantic + structured output):
  CLEAN     — agent can reproduce the gold from data + question alone
  DEPENDENT — agent needs the original notebook's specific choices to match
  IMPOSSIBLE— gold is essentially un-graderable (sentence-shaped, ambiguous,
              or required external knowledge / different language)

Outputs a `cache/eval/llm_audit.parquet` with one row per task; intended
to be consumed by `replace_flagged.py` (which now reads both regex flags
AND LLM verdict).

Run: cd rl && uv run python -m prepare.llm_audit [--candidates] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
EVAL_DIR = RL_ROOT / "cache" / "eval"


class AuditVerdict(str, Enum):
    CLEAN = "CLEAN"
    DEPENDENT = "DEPENDENT"
    IMPOSSIBLE = "IMPOSSIBLE"


class AuditOut(BaseModel):
    # reasoning first → forces chain-of-thought before commit
    reasoning: str = Field(description=(
        "One sentence: name the SPECIFIC notebook-side choice (which model, "
        "which split, which features, which seed, etc.) that the gold depends "
        "on — OR explain why the gold is derivable from the data alone."
    ))
    verdict: AuditVerdict = Field(description=(
        "CLEAN if the gold can be reproduced from the dataset + question "
        "alone (e.g. counts, means, group-bys, standard library functions "
        "with deterministic output). "
        "DEPENDENT if the gold depends on the original notebook's specific "
        "choices: which model, which train/test split (random seed), which "
        "features selected, which hyperparameters, which preprocessing. "
        "IMPOSSIBLE if the gold is sentence-shaped, ambiguous, in a non-English "
        "language without English equivalence, or otherwise not graderable."
    ))


AUDIT_PROMPT = """You are auditing a data-science eval task for *reproducibility*.

The task gives an agent a Kaggle dataset and a question. The gold answer was
extracted from the *original notebook* a human wrote. You decide whether an
independent agent, working from the dataset alone (no access to the original
notebook), can reproduce the gold answer.

**CLEAN** examples:
  Q: "How many rows in the dataset?"           A: "5000"
  Q: "What is the mean of column X?"           A: "42.3"
  Q: "Which category appears most often?"      A: "Action"
  Q: "What is the ADF test critical value at 5%%?" A: "-2.8631"
    (deterministic statsmodels output)
  Q: "Which feature has the highest pandas .corr() with Y?"  A: "TAX"

**DEPENDENT** examples — agent CAN'T reliably reproduce without the notebook:
  Q: "Test accuracy of the final logistic regression after feature selection?"
    → "feature selection" is a notebook-specific choice
  Q: "Validation accuracy of the LSTM model after one epoch?"
    → LSTM init is random; different runs differ
  Q: "Best hyperparameter from grid search?"
    → depends on grid + scoring
  Q: "Which feature ranked second by RFE?"
    → depends on which estimator the original notebook chose for RFE
  Q: "How many features removed after VIF-based selection?"
    → depends on VIF threshold the notebook picked

**IMPOSSIBLE** examples:
  Q: "What conclusion about stationarity?"  A: "The time series is stationary"
    (sentence-shaped, graderable but the agent's phrasing will differ)
  Q (in Russian): "..."                     A: "Нет"
    (cross-language equivalence is fragile)
  Q: "Which row exhibits the most unusual pattern?"
    (no rigorous definition of "unusual")

Rule of thumb: if the question references "the model" / "the final" /
"the best-performing" / "selected as optimal" / "after feature selection" /
"the original notebook" / a specific model class (LSTM, Random Forest, GBM)
with a metric — almost always DEPENDENT.

Now classify this task:

Question: {question}

Gold answer: {answer}

Files agent has access to: {files}
"""


def _audit_one_call(client, question: str, answer: str, files: list[str]) -> dict:
    try:
        files_str = ", ".join(files[:5]) or "(none)"
        resp = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": AUDIT_PROMPT.format(question=question, answer=answer, files=files_str),
            }],
            response_format=AuditOut,
            temperature=0,
        )
        out = resp.choices[0].message.parsed
        return {"verdict": out.verdict.value, "reasoning": out.reasoning}
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "ERROR", "reasoning": f"{type(exc).__name__}: {exc}"}


def _audit_one(client, question: str, answer: str, files: list[str], votes: int = 1) -> dict:
    """Run `votes` independent audits and take the majority verdict.

    Tie-break order: DEPENDENT > IMPOSSIBLE > CLEAN. This makes the audit
    conservative — a task only labelled CLEAN when a strict majority agrees.
    """
    if votes == 1:
        return _audit_one_call(client, question, answer, files)

    calls = [_audit_one_call(client, question, answer, files) for _ in range(votes)]
    verdicts = [c["verdict"] for c in calls]
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    # Pick the verdict with highest count; tie-break: DEPENDENT > IMPOSSIBLE > CLEAN > ERROR
    rank = {"DEPENDENT": 4, "IMPOSSIBLE": 3, "CLEAN": 2, "ERROR": 1}
    best = max(counts.items(), key=lambda kv: (kv[1], rank.get(kv[0], 0)))[0]
    # Stitch reasonings together for inspection
    joined = " || ".join(f"[{c['verdict']}] {c['reasoning']}" for c in calls)
    return {
        "verdict": best,
        "reasoning": joined,
        "vote_breakdown": json.dumps(counts),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=EVAL_DIR / "eval_manifest.parquet",
                   help="Parquet of tasks to audit.")
    p.add_argument("--output", type=Path,
                   default=EVAL_DIR / "llm_audit.parquet")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--votes", type=int, default=1,
                   help="Run N audits per task; take majority (tie → DEPENDENT). "
                        "Use 3 for stable verdicts; 1 is fast/cheap.")
    p.add_argument("--limit", type=int, default=None,
                   help="Audit only the first N rows (smoke).")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY missing"); return 1
    from openai import OpenAI
    client = OpenAI()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"[load] {len(df)} tasks from {args.input}")

    # We need files_used for the prompt; if not present (candidates.parquet),
    # use an empty list.
    has_files = "files_used" in df.columns
    if not has_files:
        print("[info] no files_used column; passing '(none)' to judge")

    t0 = time.time()
    results = [None] * len(df)
    rows = list(df.iterrows())

    def _runner(args_t):
        i, row = args_t
        if has_files:
            raw = row.get("files_used", None)
            files = [str(x) for x in raw] if raw is not None and len(raw) > 0 else []
        else:
            files = []
        return i, _audit_one(client, row["question"], row["answer"], files, votes=args.votes)

    done = 0
    last_print = t0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_runner, t) for t in rows]
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res
            done += 1
            now = time.time()
            if now - last_print >= 5 or done == len(df):
                print(f"  [{done}/{len(df)}]  {done / (now - t0):.1f} req/s")
                last_print = now

    df["audit_verdict"] = [r["verdict"] for r in results]
    df["audit_reasoning"] = [r["reasoning"] for r in results]
    if args.votes > 1:
        df["audit_votes"] = [r.get("vote_breakdown", "") for r in results]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    # Summary
    print(); print("─" * 60)
    vc = df.audit_verdict.value_counts()
    for v in ("CLEAN", "DEPENDENT", "IMPOSSIBLE", "ERROR"):
        print(f"  {v:<11s} {vc.get(v, 0):>3d}")
    if "difficulty" in df.columns:
        print()
        for bucket in ("easy", "medium", "hard"):
            sub = df[df.difficulty == bucket]
            sub_vc = sub.audit_verdict.value_counts()
            print(f"  {bucket:6s}  CLEAN={sub_vc.get('CLEAN', 0)}  "
                  f"DEP={sub_vc.get('DEPENDENT', 0)}  IMP={sub_vc.get('IMPOSSIBLE', 0)}")

    not_clean = df[df.audit_verdict != "CLEAN"]
    if len(not_clean):
        print()
        print(f"[non-CLEAN sample (first 10):]")
        for _, r in not_clean.head(10).iterrows():
            tid = r.get("id", "?")
            print(f"  [{r['audit_verdict']}] {tid}")
            print(f"    Q: {r['question'][:140]}")
            print(f"    A: {r['answer']!r}")
            print(f"    why: {r['audit_reasoning'][:200]}")
    print(f"\n[save] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
