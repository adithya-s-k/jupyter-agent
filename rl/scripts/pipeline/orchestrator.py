"""Per-task orchestrator: Phase A (build) → B (verify) → C (doctor) → D (categorize)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .build import build_spec, id_safe, TASKS_DIR, DEFAULT_SUITE
from .verify import run_trial, TrialResult
from .state import StateStore


@dataclass
class RunConfig:
    state_store: StateStore
    suite_name: str = DEFAULT_SUITE
    model: str = "anthropic/claude-sonnet-4-6"
    k_max: int = 2                            # default: one Sonnet retry before doctor
    sandbox: str = "docker"
    rewrite_spec: bool = False
    subprocess_timeout_sec: int = 900         # outer wall-time per harbor invocation
    max_retries_per_trial: int = 1            # retry transient errors with backoff
    task_timeout_sec: int = 1500              # orchestrator-level soft cap per task (25 min)
    # Phase C (doctor)
    enable_doctor: bool = True
    max_rewrites: int = 1
    doctor_budget_usd: float = 0.50
    doctor_max_calls: int = 20
    doctor_model: str = "anthropic/claude-sonnet-4-6"
    # Phase D (categorize)
    enable_categorize: bool = True
    categorize_model: str = "anthropic/claude-sonnet-4-6"
    run_empirical_probe: bool = False        # gpt-4o probe; off by default

    # Allowed doctor probe aliases (None = all aliases in doctor.ALLOWED_PROBE_MODELS)
    probe_aliases: list[str] | None = None
    # LLM used by Phase B regrade's llm-judge fallback
    regrade_judge_model: str = "openai/gpt-5.4-nano"


def _trial_job_name(suite: str, task_id: str, model: str, k: int, rewrite_idx: int = 0) -> str:
    model_slug = model.replace("/", "-").replace(".", "-").replace(":", "-")
    suffix = f"-r{rewrite_idx}" if rewrite_idx else ""
    return f"pl-{suite}-{id_safe(task_id)}-{model_slug}-k{k}{suffix}"


def _run_phase_b(*, row, cfg: RunConfig, spec_dir: Path, state: StateStore,
                 jobs_dir: Path, rewrite_idx: int = 0,
                 budget_remaining_fn=None) -> dict:
    """Run K trials (adaptive — stops on first pass). Returns:
       { passing_trial, trial_results: [TrialResult...], total_cost, ... }"""
    task_id = str(row["id"])
    trial_results: list[TrialResult] = []
    passing = None
    total_cost = 0.0
    total_prompt = total_completion = total_cached = 0

    for k in range(1, cfg.k_max + 1):
        job_name = _trial_job_name(cfg.suite_name, task_id, cfg.model, k, rewrite_idx)
        state.append_event(event="trial_start", task_id=task_id, phase="B",
                           model=cfg.model, k_attempt=k, rewrite_idx=rewrite_idx,
                           job_name=job_name)
        t0 = time.time()
        result = run_trial(
            suite_path=spec_dir.parent, task_id=task_id, model=cfg.model,
            job_name=job_name, jobs_dir=jobs_dir,
            sandbox=cfg.sandbox, log_dir=state.logs_dir,
            subprocess_timeout_sec=cfg.subprocess_timeout_sec,
            max_retries=cfg.max_retries_per_trial,
            budget_remaining_sec=budget_remaining_fn() if budget_remaining_fn else None,
        )
        trial_results.append(result)

        # AUTO-ESCALATE: if the trial failed but we got a non-empty predicted
        # answer, re-grade it with looser modes BEFORE handing off to doctor.
        # Recovers numeric/format false-negatives essentially for free.
        if result.reward < 1.0 and result.predicted_answer:
            from .regrade import regrade, promote_reward_mode_in_toml
            from pathlib import Path
            current_mode = str(row.get("reward_mode_initial") or "flexible")
            outcome = regrade(
                predicted=result.predicted_answer,
                gold=str(row["answer"]),
                current_mode=current_mode,
                question=str(row["question"]),
                state=state, task_id=task_id,
                judge_model=cfg.regrade_judge_model,
            )
            if outcome and outcome.passed:
                _MODE_TO_TOML = {
                    "exact":              "exact",
                    "exact_normalized":   "flexible",
                    "numeric":            "numeric",
                    "numeric_normalized": "flexible",
                    "list":               "list",
                    "list_csv":           "list_csv",
                    "llm-judge":          "flexible",
                }
                new_mode = _MODE_TO_TOML.get(outcome.mode, "flexible")
                try:
                    promote_reward_mode_in_toml(
                        Path(spec_dir) / "task.toml", new_mode,
                    )
                except Exception:  # noqa: BLE001
                    pass
                # Patch the trial: now-passing, with judge cost (if any) folded in.
                result.reward = 1.0
                result.error_kind = f"ok_regrade_{outcome.mode}"
                if outcome.judge_cost_usd:
                    result.cost_usd += outcome.judge_cost_usd
                state.append_event(
                    event="regrade_promoted", task_id=task_id, phase="B",
                    model=cfg.model, k_attempt=k, rewrite_idx=rewrite_idx,
                    from_mode=current_mode, to_mode=new_mode,
                    regrade_mode=outcome.mode,
                    judge_cost_usd=outcome.judge_cost_usd,
                )

        # Accumulate cost (post-regrade so judge calls are included)
        total_cost += result.cost_usd
        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens
        total_cached += result.cached_tokens

        state.append_event(
            event="trial_finish", task_id=task_id, phase="B",
            model=cfg.model, k_attempt=k, rewrite_idx=rewrite_idx,
            job_name=job_name,
            reward=result.reward, predicted=result.predicted_answer,
            error_kind=result.error_kind, elapsed_sec=result.elapsed_sec,
            cost_usd=result.cost_usd,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            trial_dir=str(result.trial_dir) if result.trial_dir else None,
        )

        if result.reward >= 1.0:
            passing = (k, result)
            break

    return {
        "passing": passing,                # (k, TrialResult) or None
        "trials": trial_results,
        "cost_usd": total_cost,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "cached_tokens": total_cached,
    }


def process_task(row: Mapping, cfg: RunConfig) -> dict:
    """Run one full per-task pipeline. Returns the final decision dict."""
    task_id = str(row["id"])
    state = cfg.state_store
    jobs_dir = state.trials_dir   # per-run trials/ folder

    t_start = time.time()
    def _budget_left() -> float:
        return cfg.task_timeout_sec - (time.time() - t_start)
    def _budget_exhausted() -> bool:
        return _budget_left() <= 0

    state.append_event(event="task_start", task_id=task_id, phase="A",
                       task_timeout_sec=cfg.task_timeout_sec)

    # --- PHASE A: build spec
    try:
        spec_dir = build_spec(row, suite=cfg.suite_name, overwrite=cfg.rewrite_spec)
    except Exception as e:
        state.append_event(event="task_finish", task_id=task_id, phase="A",
                           error_kind="spec_build_error", error_msg=str(e))
        state.upsert_decision(task_id, verdict="spec_build_error",
                              total_cost_usd=0.0, total_trials=0,
                              error_msg=str(e))
        state.flush()
        return state.get_decision(task_id) or {}
    state.append_event(event="spec_built", task_id=task_id, phase="A",
                       spec_dir=str(spec_dir))

    cum = {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
           "cached_tokens": 0, "total_trials": 0,
           "doctor_cost_usd": 0.0, "probe_cost_usd": 0.0,
           "categorize_cost_usd": 0.0}

    # spec_dir.parent is the BUCKET folder (e.g. pending/), which is what
    # we pass to `harbor run -p` since Harbor expects -p to be the parent
    # of <task_id_safe>/.

    # --- PHASE B (round 1)
    b = _run_phase_b(row=row, cfg=cfg, spec_dir=spec_dir, state=state,
                     jobs_dir=jobs_dir, rewrite_idx=0,
                     budget_remaining_fn=_budget_left)
    cum["cost_usd"] += b["cost_usd"]
    cum["prompt_tokens"] += b["prompt_tokens"]
    cum["completion_tokens"] += b["completion_tokens"]
    cum["cached_tokens"] += b["cached_tokens"]
    cum["total_trials"] += len(b["trials"])
    passing_round = "B"
    passing_model = cfg.model
    passing_k = b["passing"][0] if b["passing"] else None
    passing_predicted = b["passing"][1].predicted_answer if b["passing"] else ""
    passing_trial_dir = b["passing"][1].trial_dir if b["passing"] else None
    doctor_verdict = None
    doctor_reasoning = None
    spec_rewrite_count = 0
    gold_corrected = False
    gold_original = None

    # --- PHASE C: doctor (if Phase B failed, enabled, AND budget remains)
    if b["passing"] is None and cfg.enable_doctor and not _budget_exhausted():
        from .doctor import run_doctor
        specs_archive_dir = state.specs_dir / id_safe(task_id)
        trial_dirs = [t.trial_dir for t in b["trials"]]
        d = run_doctor(
            row=row, spec_dir=spec_dir, trial_dirs=trial_dirs,
            state=state, jobs_dir=jobs_dir,
            specs_archive_dir=specs_archive_dir,
            max_calls=cfg.doctor_max_calls,
            max_budget=cfg.doctor_budget_usd,
            model=cfg.doctor_model,
            sandbox=cfg.sandbox,
            subprocess_timeout_sec=cfg.subprocess_timeout_sec,
            budget_remaining_fn=_budget_left,
            probe_aliases=cfg.probe_aliases,
        )
        cum["doctor_cost_usd"] += d.total_cost_usd
        cum["probe_cost_usd"] += d.probe_cost_usd
        doctor_verdict = d.verdict
        doctor_reasoning = d.reasoning

        if d.verdict in ("spec_fixed", "gold_corrected") and spec_rewrite_count < cfg.max_rewrites and not _budget_exhausted():
            spec_rewrite_count += 1
            if d.verdict == "gold_corrected":
                gold_corrected = True
                gold_original = str(row["answer"])
            # --- PHASE B (round 2 with edited spec)
            b2 = _run_phase_b(row=row, cfg=cfg, spec_dir=spec_dir, state=state,
                              jobs_dir=jobs_dir, rewrite_idx=1)
            cum["cost_usd"] += b2["cost_usd"]
            cum["prompt_tokens"] += b2["prompt_tokens"]
            cum["completion_tokens"] += b2["completion_tokens"]
            cum["cached_tokens"] += b2["cached_tokens"]
            cum["total_trials"] += len(b2["trials"])
            if b2["passing"]:
                passing_round = "B2"
                passing_k = b2["passing"][0]
                passing_predicted = b2["passing"][1].predicted_answer
                passing_trial_dir = b2["passing"][1].trial_dir

        elif d.verdict == "verifiable_judge":
            # Doctor confirmed cross-model consensus on the gold; mark verified.
            passing_round = "C-judge"
            passing_predicted = "(cross-model consensus)"

    # Decide final verdict
    if passing_predicted and (
        b["passing"] or
        passing_round == "B2" or
        passing_round == "C-judge"
    ):
        if gold_corrected:
            final_verdict = "verified_gold_corrected"
        elif passing_round == "B2":
            final_verdict = "verified_after_rewrite"
        elif passing_round == "C-judge":
            final_verdict = "verifiable_judge"
        else:
            final_verdict = "verified"
    elif _budget_exhausted():
        # Soft wall-time cap tripped — short-circuit to dropped with reason.
        final_verdict = "dropped"
        doctor_reasoning = (doctor_reasoning or "") + \
            f" [task_timeout: exceeded {cfg.task_timeout_sec}s budget]"
    elif doctor_verdict == "unverifiable":
        final_verdict = "dropped"
    elif doctor_verdict in ("spec_fixed", "gold_corrected") and spec_rewrite_count >= cfg.max_rewrites:
        # Doctor wanted to edit but we'd already hit the rewrite budget AND
        # the post-edit Phase B didn't pass — treat as dropped, not glitched.
        final_verdict = "dropped"
    elif cfg.enable_doctor and doctor_verdict is None:
        # Doctor was enabled but never ran (e.g., regrade promoted the trial
        # and we never entered Phase C). This shouldn't reach here.
        final_verdict = "dropped"
    elif not cfg.enable_doctor:
        # Doctor explicitly disabled by user — keep phase_b_failed as the
        # honest signal that Phase B exhausted without recovery.
        final_verdict = "phase_b_failed"
    else:
        # Unrecognised doctor verdict — log and drop rather than glitch.
        final_verdict = "dropped"
        if doctor_verdict and doctor_verdict != "unverifiable":
            doctor_reasoning = (doctor_reasoning or "") + \
                f" [orchestrator: unrecognized doctor_verdict={doctor_verdict!r}]"

    # --- PHASE D: categorize
    diff_level = 0
    diff_confidence = 0.0
    diff_reasoning = ""
    diff_signal = ""
    empirical_easy = None
    if final_verdict.startswith("verified") and cfg.enable_categorize and passing_trial_dir:
        from .categorize import categorize
        c = categorize(
            row=row,
            passing_trial_dir=passing_trial_dir,
            suite_path=spec_dir.parent,
            jobs_dir=jobs_dir,
            state=state,
            run_empirical=cfg.run_empirical_probe,
            model=cfg.categorize_model,
        )
        cum["categorize_cost_usd"] += c.cost_usd + c.empirical_probe_cost_usd
        diff_level = c.level
        diff_confidence = c.confidence
        diff_reasoning = c.reasoning
        diff_signal = c.signal
        empirical_easy = c.empirical_easy
        state.append_event(
            event="categorize_finish", task_id=task_id, phase="D",
            level=c.level, confidence=c.confidence,
            reasoning=c.reasoning, signal=c.signal,
            cost_usd=c.cost_usd,
            empirical_easy=c.empirical_easy,
            empirical_predicted=c.empirical_predicted,
            empirical_probe_cost_usd=c.empirical_probe_cost_usd,
        )

    total_cost = (cum["cost_usd"] + cum["doctor_cost_usd"]
                  + cum["probe_cost_usd"] + cum["categorize_cost_usd"])

    state.upsert_decision(
        task_id,
        verdict=final_verdict,
        passing_round=passing_round if final_verdict.startswith("verified") else None,
        passing_model=passing_model if final_verdict.startswith("verified") else None,
        passing_k=passing_k,
        passing_predicted=passing_predicted,
        gold_corrected=gold_corrected,
        gold_original=gold_original,
        spec_rewrite_count=spec_rewrite_count,
        doctor_verdict=doctor_verdict,
        doctor_reasoning=doctor_reasoning,
        difficulty_level=diff_level,
        difficulty_confidence=diff_confidence,
        difficulty_reasoning=diff_reasoning,
        difficulty_signal=diff_signal,
        empirical_easy=empirical_easy,
        total_cost_usd=round(total_cost, 6),
        phase_b_cost_usd=round(cum["cost_usd"], 6),
        doctor_cost_usd=round(cum["doctor_cost_usd"], 6),
        probe_cost_usd=round(cum["probe_cost_usd"], 6),
        categorize_cost_usd=round(cum["categorize_cost_usd"], 6),
        total_trials=cum["total_trials"],
        prompt_tokens=cum["prompt_tokens"],
        completion_tokens=cum["completion_tokens"],
        cached_tokens=cum["cached_tokens"],
    )
    state.append_event(event="task_finish", task_id=task_id,
                       verdict=final_verdict, total_cost_usd=total_cost,
                       difficulty_level=diff_level)

    # Move the spec dir to its verdict bucket. Up to this point it lived in
    # pending/<id>/. Now that we know the final verdict, move it to
    # verified/dropped/phase_b_failed/. Anything else stays in pending.
    try:
        from .buckets import VERDICT_TO_BUCKET, move_task
        target_bucket = VERDICT_TO_BUCKET.get(final_verdict)
        if target_bucket:
            suite_dir = spec_dir.parent.parent     # <suite>/pending/ → <suite>/
            move_task(suite_dir, id_safe(task_id), target_bucket)
    except Exception as e:  # noqa: BLE001
        state.append_event(event="bucket_move_error", task_id=task_id,
                           verdict=final_verdict, error=str(e))

    state.flush()
    return state.get_decision(task_id) or {}
