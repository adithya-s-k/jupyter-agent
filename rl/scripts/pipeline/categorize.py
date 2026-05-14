"""Phase D — categorize a verified task on a 1-5 difficulty scale.

D1 (RUBRIC, Sonnet): one judge call. Required. Structured output via Pydantic
                     validation; the rubric prompt requests exact JSON shape.
D2 (EMPIRICAL, gpt-4o): one extra seta trial. Optional, off by default.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field, ValidationError, conint, confloat

from . import llm_client


CATEGORIZE_MODEL = "anthropic/claude-sonnet-4-6"
EMPIRICAL_PROBE_MODEL = "openai/gpt-4o"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "categorize.md"
CATEGORIZE_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


# Structured output schema. Sonnet returns a JSON object that must validate.
class CategorizeOutput(BaseModel):
    level: conint(ge=1, le=5)   # type: ignore[valid-type]
    reasoning: str = Field(..., min_length=1, max_length=2000)
    confidence: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    signal: str = ""


@dataclass
class CategorizeResult:
    level: int
    reasoning: str
    confidence: float
    signal: str
    raw: str
    parse_ok: bool
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    empirical_easy: bool | None = None
    empirical_probe_cost_usd: float = 0.0
    empirical_predicted: str = ""


def _passing_trajectory_excerpt(trial_dir: Path, max_chars: int = 6000) -> str:
    if trial_dir is None:
        return "(no passing trial dir captured)"
    msgs = None
    for f in (trial_dir / "agent").glob("*.trajectory.json"):
        try:
            msgs = json.loads(f.read_text())
        except Exception:
            return "(could not parse trajectory)"
        break
    if msgs is None:
        return "(no trajectory.json found)"

    out: list[str] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            args_s = str(args)[:600]
            out.append(f"--- TOOL {name}\n{args_s}")
    joined = "\n".join(out)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + f"\n... [truncated, full {len(joined)} chars]"
    return joined or "(no tool calls found)"


def _extract_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def categorize_rubric(*, row: Mapping, passing_trial_dir: Path,
                      model: str = CATEGORIZE_MODEL,
                      temperature: float = 0.0,
                      state=None) -> CategorizeResult:
    excerpt = _passing_trajectory_excerpt(passing_trial_dir)
    user_msg = (
        f"QUESTION: {row['question']}\n"
        f"GOLD: {row['answer']}\n\n"
        f"PASSING TRAJECTORY EXCERPT (tool calls in order):\n{excerpt}\n\n"
        "Output a JSON object matching this exact schema and nothing else:\n"
        '  {"level": <1|2|3|4|5>, "reasoning": "<one sentence>", '
        '"confidence": <0.0-1.0>, "signal": "<what tipped you off>"}'
    )
    messages = [
        {"role": "system", "content": CATEGORIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    resp = llm_client.call(
        model=model, messages=messages,
        temperature=temperature, max_tokens=400,
        state=state, task_id=str(row["id"]), phase="D", call_kind="categorize_rubric",
        response_format={"type": "json_object"},  # ignored on Anthropic, hint for OpenAI
    )

    parsed_raw = _extract_json(resp.content)
    if parsed_raw is None:
        return CategorizeResult(
            level=0, reasoning=f"parse_error: {resp.content[:200]}",
            confidence=0.0, signal="parse_error",
            raw=resp.content, parse_ok=False,
            cost_usd=resp.cost_usd, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, cached_tokens=resp.cached_tokens,
        )

    try:
        validated = CategorizeOutput.model_validate(parsed_raw)
    except ValidationError as e:
        return CategorizeResult(
            level=0, reasoning=f"schema_error: {e!s}",
            confidence=0.0, signal="schema_error",
            raw=resp.content, parse_ok=False,
            cost_usd=resp.cost_usd, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, cached_tokens=resp.cached_tokens,
        )

    return CategorizeResult(
        level=int(validated.level),
        reasoning=validated.reasoning,
        confidence=float(validated.confidence),
        signal=validated.signal,
        raw=resp.content,
        parse_ok=True,
        cost_usd=resp.cost_usd,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        cached_tokens=resp.cached_tokens,
    )


def empirical_probe(*, row: Mapping, suite_path: Path, jobs_dir: Path,
                    state, k_max: int = 1) -> dict:
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
        sandbox="docker", log_dir=state.logs_dir,
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
    result = categorize_rubric(row=row, passing_trial_dir=passing_trial_dir, state=state)

    if run_empirical and suite_path and jobs_dir and state is not None:
        probe = empirical_probe(row=row, suite_path=suite_path,
                                jobs_dir=jobs_dir, state=state)
        result.empirical_easy = probe["passed"]
        result.empirical_probe_cost_usd = probe["cost_usd"]
        result.empirical_predicted = probe["predicted"]

    return result
