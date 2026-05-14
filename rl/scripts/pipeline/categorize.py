"""Phase D — categorize a verified task on a 1-5 difficulty scale.

D1 (RUBRIC, Sonnet): one judge call. Required.
D2 (EMPIRICAL, gpt-4o): one extra seta trial. Optional, off by default.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import llm_client


CATEGORIZE_MODEL = "anthropic/claude-sonnet-4-6"
EMPIRICAL_PROBE_MODEL = "openai/gpt-4o"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "categorize.md"
CATEGORIZE_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


@dataclass
class CategorizeResult:
    level: int                  # 1-5; 0 if parsing failed
    reasoning: str
    confidence: float
    signal: str
    raw: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    # D2 — only set when run_empirical=True
    empirical_easy: bool | None = None
    empirical_probe_cost_usd: float = 0.0
    empirical_predicted: str = ""


def _passing_trajectory_excerpt(trial_dir: Path, max_chars: int = 6000) -> str:
    """Pull out the code cells (and key outputs) from a passing trial."""
    if trial_dir is None:
        return "(no passing trial dir captured)"
    for f in (trial_dir / "agent").glob("*.trajectory.json"):
        try:
            msgs = json.loads(f.read_text())
        except Exception:
            return "(could not parse trajectory)"
        break
    else:
        return "(no trajectory.json found)"

    # Extract only the tool calls (code cells) — that's what determines difficulty
    out: list[str] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            # Truncate each cell to keep total size bounded
            args_s = str(args)[:600]
            out.append(f"--- TOOL {name}\n{args_s}")
    joined = "\n".join(out)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + f"\n... [truncated, full {len(joined)} chars]"
    return joined or "(no tool calls found)"


def _parse_json(text: str) -> dict | None:
    # Strip ```json fences if present
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # try to find the first { ... } block
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def categorize_rubric(*, row: Mapping, passing_trial_dir: Path,
                      model: str = CATEGORIZE_MODEL,
                      temperature: float = 0.0) -> CategorizeResult:
    excerpt = _passing_trajectory_excerpt(passing_trial_dir)
    user_msg = (
        f"QUESTION: {row['question']}\n"
        f"GOLD: {row['answer']}\n\n"
        f"PASSING TRAJECTORY EXCERPT (tool calls in order):\n{excerpt}\n\n"
        "Categorize. Output JSON only — no prose."
    )
    messages = [
        {"role": "system", "content": CATEGORIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    resp = llm_client.call(model=model, messages=messages,
                           temperature=temperature, max_tokens=400)

    parsed = _parse_json(resp.content)
    if parsed is None:
        return CategorizeResult(
            level=0, reasoning=f"parse_error: {resp.content[:200]}",
            confidence=0.0, signal="parse_error",
            raw=resp.content, cost_usd=resp.cost_usd,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached_tokens=resp.cached_tokens,
        )

    return CategorizeResult(
        level=int(parsed.get("level") or 0),
        reasoning=str(parsed.get("reasoning") or ""),
        confidence=float(parsed.get("confidence") or 0.0),
        signal=str(parsed.get("signal") or ""),
        raw=resp.content,
        cost_usd=resp.cost_usd,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        cached_tokens=resp.cached_tokens,
    )


def empirical_probe(*, row: Mapping, suite_path: Path, jobs_dir: Path,
                    state, k_max: int = 1) -> dict:
    """Run gpt-4o + seta + K=1 as a cheap difficulty cross-check.

    Returns: {passed: bool, predicted: str, cost_usd: float, trial_dir}
    """
    from .verify import run_trial
    from .build import id_safe

    task_id = str(row["id"])
    slug = EMPIRICAL_PROBE_MODEL.replace("/", "-").replace(".", "-")
    job_name = f"empirical-{id_safe(task_id)}-{slug}"
    state.append_event(event="empirical_start", task_id=task_id, phase="D",
                       model=EMPIRICAL_PROBE_MODEL, job_name=job_name)
    t0 = time.time()
    result = run_trial(
        suite_path=suite_path, task_id=task_id, model=EMPIRICAL_PROBE_MODEL,
        job_name=job_name, jobs_dir=jobs_dir,
        sandbox="docker", log_dir=state.state_dir / "logs",
    )
    state.append_event(
        event="empirical_finish", task_id=task_id, phase="D",
        model=EMPIRICAL_PROBE_MODEL, job_name=job_name,
        reward=result.reward, predicted=result.predicted_answer,
        elapsed_sec=time.time() - t0, cost_usd=result.cost_usd,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cached_tokens=result.cached_tokens,
    )
    return {
        "passed": result.reward >= 1.0,
        "predicted": result.predicted_answer,
        "cost_usd": result.cost_usd,
        "trial_dir": result.trial_dir,
    }


def categorize(*, row: Mapping, passing_trial_dir: Path,
               suite_path: Path | None = None, jobs_dir: Path | None = None,
               state=None, run_empirical: bool = False) -> CategorizeResult:
    """Phase D entry point: rubric (always) + empirical (optional)."""
    result = categorize_rubric(row=row, passing_trial_dir=passing_trial_dir)

    if run_empirical and suite_path and jobs_dir and state is not None:
        probe = empirical_probe(row=row, suite_path=suite_path,
                                jobs_dir=jobs_dir, state=state)
        result.empirical_easy = probe["passed"]
        result.empirical_probe_cost_usd = probe["cost_usd"]
        result.empirical_predicted = probe["predicted"]

    return result
