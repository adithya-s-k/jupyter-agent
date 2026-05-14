"""Shared three-tier grader.

Used by both runtimes:
  * `env/server.py` — inline inside the `final_answer` ORS @tool method
  * `tests/test.sh` — Harbor CLI path; this module is mounted at /tests/grader.py
    and invoked with `python3 /tests/grader.py < /workdir/answer.txt`

Tiers (cheapest → most expensive):
  1. Exact (case-insensitive, whitespace-collapsed)
  2. Numeric (parse a float, accept abs/rel diff ≤ 1e-3)
  3. LLM-judge (gpt-4o-mini, A/B/C verdict, OpenAI simple-evals prompt)
     — only invoked if OPENAI_API_KEY is set

CLI form (used by tests/test.sh):
  EXPECTED_ANSWER='5' QUESTION='How many...' python3 grader.py
    → reads predicted answer from stdin
    → writes a single float ('1.0' or '0.0') to stdout
    → exits 0 always; harbor reads stdout from /logs/verifier/reward.txt
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass


_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")

# Locked in PLAN.md §"Locked decisions" — adapted from
# https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
JUDGE_PROMPT = """You are grading short answers from a data-science agent.
Decide whether the predicted answer is semantically equivalent to the gold answer.

Rules:
- Case, punctuation, whitespace, articles ("the"), and trailing units don't matter.
- For numbers: predicted must match to the last significant figure of the gold
  (gold "0.544341", predicted "0.544" -> CORRECT; "0.5" -> INCORRECT).
- Extra surrounding prose is fine if the gold value is clearly stated
  (gold "5", predicted "There are 5 distinct categories" -> CORRECT).
- Common synonyms/abbreviations count (gold "Not Applicable", predicted "N/A" -> CORRECT).
- Parenthetical annotations in the gold are equivalence hints, not required
  (gold "Gandalf (Ainur)", predicted "Gandalf" -> CORRECT;
   gold "No (correlation coefficient = 0.02)", predicted "No" -> CORRECT).
- Percent + qualifier: if the gold is "X% in YEAR", a prediction of X or X%
  is CORRECT as long as the core numeric value matches
  (gold "21.334% in 2014", predicted "21.334" -> CORRECT;
   gold "21.334% in 2014", predicted "21.334% in 2013" -> INCORRECT — year mismatch).
- Multi-part gold like "0.987 (XGBoost and LGBM)" — if the question asks for one
  thing (e.g. "which model?"), predicting either listed value is CORRECT.
- If the predicted answer hedges without committing to the gold value -> NOT_ATTEMPTED.
- If it commits to a different value -> INCORRECT.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Reply with exactly one token: A (CORRECT), B (INCORRECT), or C (NOT_ATTEMPTED)."""


@dataclass
class GradeResult:
    reward: float
    method: str  # "exact" | "numeric" | "llm" | "miss"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _to_float(s: str) -> float | None:
    if not s:
        return None
    m = _NUMERIC_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def grade(
    gold: str,
    candidate: str,
    *,
    question: str = "",
    judge: bool = True,
    judge_model: str | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> GradeResult:
    """Run the three-tier match. `judge=False` skips the LLM tier (useful
    in tests or when the OpenAI API key isn't available)."""

    if not gold or candidate is None:
        return GradeResult(0.0, "miss")

    # ── Tier 1: exact (case-insensitive, whitespace-collapsed) ─────────
    if _normalize(gold) == _normalize(candidate):
        return GradeResult(1.0, "exact")

    # ── Tier 2: numeric ────────────────────────────────────────────────
    g, c = _to_float(gold), _to_float(candidate)
    if g is not None and c is not None:
        if abs(g - c) <= abs_tol or abs(g - c) / max(abs(g), 1e-9) <= rel_tol:
            return GradeResult(1.0, "numeric")

    # ── Tier 3: LLM-judge (opt-in, requires OPENAI_API_KEY) ────────────
    if judge and os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            model = judge_model or os.environ.get("GRADER_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question, gold=gold, pred=candidate
                        ),
                    }
                ],
                max_tokens=4,
                temperature=0,
            )
            verdict = (resp.choices[0].message.content or "").strip().upper()
            letter = next((c for c in verdict if c in "ABC"), "C")
            return GradeResult(1.0 if letter == "A" else 0.0, "llm")
        except Exception as exc:  # noqa: BLE001
            # Fall through to miss on any client/network failure.
            print(f"[grader] llm-judge failed: {exc}", file=sys.stderr)

    return GradeResult(0.0, "miss")


def main_cli() -> int:
    """CLI shim for `tests/test.sh` — env-driven, stdout-only.

    Inputs (via env vars set by Harbor's [verifier.env]):
      EXPECTED_ANSWER — the gold
      QUESTION        — the original question (for the LLM judge)
      OPENAI_API_KEY  — enables tier 3 if set
      GRADER_MODEL    — optional override (default gpt-4o-mini)

    Predicted answer is read from stdin.
    Single float written to stdout. Exit code is always 0.
    """
    gold = (os.environ.get("EXPECTED_ANSWER") or "").strip()
    question = (os.environ.get("QUESTION") or "").strip()
    candidate = sys.stdin.read().strip()
    result = grade(gold, candidate, question=question)
    print(f"{result.reward:.1f}")
    print(f"[grader] gold={gold!r} pred={candidate[:80]!r} reward={result.reward} method={result.method}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
