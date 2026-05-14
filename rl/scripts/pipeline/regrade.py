"""In-process grader used to AUTO-ESCALATE failed Phase B trials.

When Sonnet+seta finishes with reward 0.0, before invoking the (expensive)
doctor we try the same `predicted_answer` against successively looser
grader modes. The agent doesn't re-run; only the comparison logic does.

Free cascade (no LLM):
    1. exact          case-insensitive string equality
    2. numeric        parse float, tolerance match
    3. list/list_csv  set/order compare for list answers
    4. unit-strip     strip %, parentheticals, trailing units, then retry

Paid fallback (~$0.0005 each via gpt-5.4-nano):
    5. llm-judge      one-shot judge call

If any mode promotes the trial to reward ≥ 1.0, the caller persists the
new REWARD_MODE to task.toml so future runs skip the strict mode.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from . import llm_client


_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_UNIT_SUFFIXES = ["%", " %", " percent", " years", " seconds", " s", " ms",
                  " kg", " g", " mb", " gb"]


@dataclass
class RegradeOutcome:
    passed: bool
    mode: str               # name of the mode that produced this verdict
    judge_cost_usd: float   # LLM cost if llm-judge was invoked


def _strip_units(s: str) -> str:
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    for u in _UNIT_SUFFIXES:
        if s.lower().endswith(u):
            return s[: -len(u)].rstrip()
    return s


def _try_exact(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def _try_numeric(pred: str, gold: str, atol: float = 1e-3, rtol: float = 0.01) -> bool:
    try:
        p = float(pred.strip().replace(",", ""))
        g = float(gold.strip().replace(",", ""))
    except ValueError:
        return False
    if abs(p - g) <= atol:
        return True
    if g != 0 and abs(p - g) / abs(g) <= rtol:
        return True
    return False


def _try_list_csv(pred: str, gold: str) -> bool:
    p = {s.strip().lower() for s in pred.split(",") if s.strip()}
    g = {s.strip().lower() for s in gold.split(",") if s.strip()}
    return bool(g) and p == g


def _try_list_literal(pred: str, gold: str) -> bool:
    try:
        import ast
        p = ast.literal_eval(pred.strip())
        g = ast.literal_eval(gold.strip())
        return list(p) == list(g) or set(map(str, p)) == set(map(str, g))
    except Exception:  # noqa: BLE001
        return False


JUDGE_SYSTEM = """You are grading short answers from a data-science agent.
Decide whether the predicted answer is semantically equivalent to the gold answer.

Rules:
- Case, punctuation, whitespace, articles ("the") don't matter.
- For numbers: predicted matches if it agrees with the gold to a reasonable
  precision (extra trailing zeros / fewer decimals are fine).
- Predictions containing extra prose are OK if they clearly state the right value.
- Parenthetical annotations in the gold are equivalence hints, not required
  (gold "Gandalf (Ainur)", predicted "Gandalf" → CORRECT).

Answer with exactly one token: CORRECT or INCORRECT."""


def _try_llm_judge(pred: str, gold: str, question: str = "",
                   model: str = "openai/gpt-5.4-nano",
                   state=None, task_id: str = "",
                   call_kind: str = "regrade_judge") -> tuple[bool, float]:
    user_msg = (f"QUESTION:  {question}\n"
                f"GOLD:      {gold}\n"
                f"PREDICTED: {pred}\n\n"
                "CORRECT or INCORRECT?")
    resp = llm_client.call(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0, max_tokens=10,
        state=state, task_id=task_id, phase="B-regrade", call_kind=call_kind,
    )
    verdict = (resp.content or "").strip().upper()
    return verdict.startswith("CORRECT"), resp.cost_usd


def regrade(*, predicted: str, gold: str,
            current_mode: str,
            question: str = "",
            atol: float = 1e-3, rtol: float = 0.01,
            state=None, task_id: str = "",
            judge_model: str = "openai/gpt-5.4-nano") -> RegradeOutcome | None:
    """Try cheaper-first escalation. Returns a passing RegradeOutcome on first
    hit, or None if no mode promotes the trial.

    `current_mode` controls what we even try — if the task was already
    `llm-judge` or `vote-judge`, we don't bother re-trying free modes that
    were essentially already part of that.
    """
    if not predicted or not gold:
        return None
    if predicted.strip() == "" or gold.strip() == "":
        return None

    # 1. exact (case-insensitive)
    if _try_exact(predicted, gold):
        return RegradeOutcome(passed=True, mode="exact", judge_cost_usd=0.0)

    # 1b. exact after unit-strip
    if _try_exact(_strip_units(predicted), _strip_units(gold)):
        return RegradeOutcome(passed=True, mode="exact_normalized",
                              judge_cost_usd=0.0)

    # 2. numeric
    if _try_numeric(predicted, gold, atol=atol, rtol=rtol):
        return RegradeOutcome(passed=True, mode="numeric", judge_cost_usd=0.0)
    # 2b. numeric after unit-strip
    if _try_numeric(_strip_units(predicted), _strip_units(gold), atol=atol, rtol=rtol):
        return RegradeOutcome(passed=True, mode="numeric_normalized",
                              judge_cost_usd=0.0)

    # 3. list comparisons (only if gold looks list-like)
    if "," in gold or gold.strip().startswith(("[", "(")):
        if _try_list_csv(predicted, gold):
            return RegradeOutcome(passed=True, mode="list_csv", judge_cost_usd=0.0)
        if _try_list_literal(predicted, gold):
            return RegradeOutcome(passed=True, mode="list", judge_cost_usd=0.0)

    # 5. llm-judge fallback. Only if it's plausibly close enough; skip when
    # predicted is obviously empty / very short.
    if current_mode in ("llm-judge", "vote-judge"):
        return None              # already at the loosest paid mode
    passed, cost = _try_llm_judge(
        predicted, gold, question=question,
        model=judge_model,
        state=state, task_id=task_id,
    )
    if passed:
        return RegradeOutcome(passed=True, mode="llm-judge", judge_cost_usd=cost)

    return None


# ----- task.toml hot-edit so future runs use the recovered mode -----

def promote_reward_mode_in_toml(task_toml_path, new_mode: str) -> None:
    """Update REWARD_MODE in [verifier.env] to the new (looser) value."""
    text = task_toml_path.read_text()
    pat = re.compile(r'^(REWARD_MODE\s*=\s*)("[^"]*"|\S+)\s*$', re.MULTILINE)
    new_line = f'REWARD_MODE = "{new_mode}"'
    if pat.search(text):
        text = pat.sub(new_line, text, count=1)
        task_toml_path.write_text(text)
