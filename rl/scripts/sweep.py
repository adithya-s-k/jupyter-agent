"""Model-size sweep across the eval set.

Two phases:
  3A — Qwen sweep: bottom-up through 7 Qwen3 models × 4 agents per task,
       stop when ≥ STOP_AT (default 2) wins land.
  3B — frontier fallback: for tasks with zero Qwen passes, run Claude
       Sonnet 4.6 then GPT-5.5 across 4 agents.

Tasks are processed bucket-by-bucket: --buckets easy,medium,hard (default
easy,medium). Concurrency default 32 — see SWEEP_PLAN.md §9 for the math.

State lives in `rl/cache/sweep/v1/state.jsonl` (append-only event log). Every
run emits `start` + `finish` events; if a run crashes mid-flight, the next
invocation detects the orphaned start and re-launches.

Composable CLI — one subcommand per phase / inspection task:

    sweep run             # Phase 3A + 3B (default --buckets easy,medium)
    sweep run --no-fallback
    sweep fallback        # Phase 3B alone (assumes 3A finished)
    sweep status          # what's done, what's pending
    sweep cost            # rolling cost totals
    sweep diagnose        # error histogram + top failure modes
    sweep retry-failed    # re-launch error'd runs
    sweep report          # write REPORT.md
    sweep dry-run         # print planned launches without executing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
SWEEP_DIR = RL_ROOT / "cache" / "sweep"
DEFAULT_SWEEP_NAME = "v1"

# ---------------------------------------------------------------------------
# Config — the 7-Qwen spine + 2 frontier fallback
# ---------------------------------------------------------------------------

QWEN_TIERS: list[list[dict]] = [
    # T1
    [
        {"name": "qwen3-4b-inst",       "model": "hf/Qwen/Qwen3-4B-Instruct-2507:nscale"},
        {"name": "qwen3-4b-thinking",   "model": "hf/Qwen/Qwen3-4B-Thinking-2507:nscale"},
    ],
    # T2
    [{"name": "qwen3-8b",               "model": "hf/Qwen/Qwen3-8B:nscale"}],
    # T3
    [{"name": "qwen3-14b",              "model": "hf/Qwen/Qwen3-14B:nscale"}],
    # T4
    [
        {"name": "qwen3-30b-coder",     "model": "hf/Qwen/Qwen3-Coder-30B-A3B-Instruct:scaleway"},
        {"name": "qwen3-32b",           "model": "hf/Qwen/Qwen3-32B:nscale"},
    ],
    # T5
    [{"name": "qwen3-235b-inst",        "model": "hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale"}],
]

FRONTIER_MODELS: list[dict] = [
    {"name": "sonnet-4-6",  "model": "anthropic/claude-sonnet-4-6"},  # cheaper, runs first
    {"name": "gpt-5.5",     "model": "openai/gpt-5.5"},               # frontier, only if sonnet also fails
]

# (agent_label, agent_flag, uses_opencode_model_format)
AGENTS: list[dict] = [
    {"name": "jupy",  "flag": "--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent", "oc_fmt": False},
    {"name": "bash",  "flag": "--agent-import-path rl.harbor_agents.bash:BashOnlyAgent",       "oc_fmt": False},
    {"name": "seta",  "flag": "--agent-import-path rl.harbor_agents.seta:SetaToolAgent",       "oc_fmt": False},
    {"name": "oc",    "flag": "--agent opencode",                                              "oc_fmt": True},
]

ERROR_CATEGORIES = {
    "rate_limit_429":   re.compile(r"\b(429|rate.?limit|too many requests)\b", re.I),
    "e2b_timeout":      re.compile(r"\b(timeout|timed out|deadline exceeded)\b", re.I),
    "kernel_failed":    re.compile(r"kernel_server failed|didn'?t bind", re.I),
    "tool_parse_err":   re.compile(r"tool.?call.*(parse|invalid|format)", re.I),
    "api_error":        re.compile(r"(api|openai|anthropic).*(error|connection|refused)", re.I),
    "no_answer":        re.compile(r"no answer at /workdir/answer\.txt", re.I),
    "judge_failed":     re.compile(r"llm-judge failed|grader.*error", re.I),
}


# ---------------------------------------------------------------------------
# State (append-only JSONL)
# ---------------------------------------------------------------------------


@dataclass
class State:
    """In-memory view of state.jsonl. Always re-read from disk on startup."""
    sweep_dir: Path
    rows: list[dict] = field(default_factory=list)

    @property
    def state_path(self) -> Path:
        return self.sweep_dir / "state.jsonl"

    def load(self) -> "State":
        self.rows = []
        if self.state_path.exists():
            for line in self.state_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return self

    def append(self, row: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)

    # ── Queries ───────────────────────────────────────────────────────────

    def finished_triples(self) -> set[tuple[str, str, str]]:
        """(task_id, model, agent) triples with a 'finish' event."""
        return {
            (r["task_id"], r["model"], r["agent"])
            for r in self.rows if r.get("event") == "finish"
        }

    def started_but_not_finished(self) -> set[tuple[str, str, str]]:
        starts = {(r["task_id"], r["model"], r["agent"])
                  for r in self.rows if r.get("event") == "start"}
        finished = self.finished_triples()
        return starts - finished

    def passes_per_task(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for r in self.rows:
            if r.get("event") == "finish" and r.get("reward", 0) >= 1.0:
                c[r["task_id"]] += 1
        return dict(c)

    def passes_for_task_in_phase(self, task_id: str, phase: str) -> int:
        c = 0
        for r in self.rows:
            if (
                r.get("event") == "finish"
                and r["task_id"] == task_id
                and r.get("phase") == phase
                and r.get("reward", 0) >= 1.0
            ):
                c += 1
        return c

    def successful_agents_for_task(self, task_id: str, phase: str | None = None) -> set:
        """Return the SET of unique agent names that have passed for this task.

        Used as the new graduation criterion: a task graduates when ≥STOP_AT
        distinct harnesses have shown at least one win (not just total wins).
        """
        agents: set = set()
        for r in self.rows:
            if r.get("event") != "finish":
                continue
            if r["task_id"] != task_id:
                continue
            if phase is not None and r.get("phase") != phase:
                continue
            if r.get("reward", 0) >= 1.0:
                agents.add(r.get("agent"))
        return agents

    def qwen_failed_tasks(self, all_tasks: list[str]) -> list[str]:
        """Tasks with zero passes in any Qwen run."""
        qwen_pass = defaultdict(int)
        for r in self.rows:
            if (
                r.get("event") == "finish"
                and r.get("phase") == "3A"
                and r.get("reward", 0) >= 1.0
            ):
                qwen_pass[r["task_id"]] += 1
        return [t for t in all_tasks if qwen_pass.get(t, 0) == 0]

    def errored_runs(self) -> list[dict]:
        return [r for r in self.rows
                if r.get("event") == "finish" and r.get("error_kind") not in (None, "ok")]

    def cost_total(self) -> dict:
        total_cost = 0.0
        pt = ct = cached = 0
        n = 0
        by_model = defaultdict(lambda: {"runs": 0, "cost": 0.0, "passes": 0, "errors": 0})
        by_agent = defaultdict(lambda: {"runs": 0, "cost": 0.0, "passes": 0, "errors": 0})
        by_tier = defaultdict(lambda: {"runs": 0, "cost": 0.0, "passes": 0, "errors": 0})
        for r in self.rows:
            if r.get("event") != "finish":
                continue
            cost = float(r.get("cost_usd") or 0.0)
            passed = r.get("reward", 0) >= 1.0
            err = r.get("error_kind") not in (None, "ok")
            total_cost += cost
            pt += int(r.get("prompt_tokens") or 0)
            ct += int(r.get("completion_tokens") or 0)
            cached += int(r.get("cached_tokens") or 0)
            n += 1
            for store, key in (
                (by_model, r["model"]), (by_agent, r["agent"]), (by_tier, r.get("tier", "?")),
            ):
                store[key]["runs"] += 1
                store[key]["cost"] += cost
                store[key]["passes"] += int(passed)
                store[key]["errors"] += int(err)
        return {
            "total_runs": n,
            "total_cost_usd": round(total_cost, 4),
            "tokens": {"input": pt, "output": ct, "cached": cached},
            "by_model": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_model.items()},
            "by_agent": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_agent.items()},
            "by_tier":  {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_tier.items()},
        }


# ---------------------------------------------------------------------------
# Harbor invocation
# ---------------------------------------------------------------------------


def get_keys(env_file: Path) -> dict[str, str]:
    keys: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys


def safe_id(raw_id: str) -> str:
    cleaned = raw_id.replace(".ipynb", "")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", cleaned).strip("_")


def categorize_error(text: str) -> str:
    """Best-effort error categorization from log text. Returns 'unknown' otherwise."""
    for kind, rx in ERROR_CATEGORIES.items():
        if rx.search(text):
            return kind
    return "unknown"


def model_for_agent(model_spec: str, agent: dict) -> str:
    """If agent is opencode, convert `hf/X:p` → `huggingface/X:p`. Else unchanged."""
    if not agent.get("oc_fmt"):
        return model_spec
    if model_spec.startswith("hf/"):
        return "huggingface/" + model_spec[len("hf/"):]
    if model_spec.startswith("anthropic/"):
        return model_spec  # opencode also accepts anthropic/...
    if model_spec.startswith("openai/"):
        return model_spec
    return model_spec


def job_name_for(sweep_name: str, task_id: str, model_slug: str, agent_name: str) -> str:
    """Deterministic job-name → idempotent job dir."""
    return f"sweep-{sweep_name}-{safe_id(task_id)}-{model_slug}-{agent_name}"


def build_cmd(*, suite: Path, task_id: str, model_spec: str, model_slug: str,
              agent: dict, keys: dict[str, str], job_name: str, jobs_dir: Path,
              bill_to: str | None) -> list[str]:
    aes = []
    for env_key in ("OPENAI_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        v = keys.get(env_key)
        if v:
            aes += ["--ae", f"{env_key}={v}"]
    # Always pass OPENAI for the verifier (LLM judge tier).
    ve_openai = keys.get("OPENAI_API_KEY", "")
    cmd = ["harbor", "run", "-p", str(suite)]
    cmd += shlex.split(agent["flag"])
    cmd += ["--model", model_for_agent(model_spec, agent)]
    cmd += aes
    cmd += ["--ve", f"OPENAI_API_KEY={ve_openai}"]
    cmd += ["--env", "e2b", "--env-file", str(REPO_ROOT / ".env"), "--yes"]
    cmd += ["--job-name", job_name, "--jobs-dir", str(jobs_dir)]
    cmd += ["-i", safe_id(task_id), "-n", "1"]
    # If bill-to set, pass through as env var the agent reads
    return cmd


def _opencode_token_totals(job_dir: Path) -> dict | None:
    """Harbor's built-in `opencode` agent stores per-step token counts in
    `<trial>/agent/opencode.txt` (one JSON event per line). Harbor's own cost
    populator only knows OpenAI pricing — for Qwen via HF it lands as 0. We
    parse the raw events and total tokens here, then compute cost from our
    own table in parse_result().
    """
    for trial in job_dir.iterdir():
        if not trial.is_dir():
            continue
        oc_log = trial / "agent" / "opencode.txt"
        if not oc_log.exists():
            continue
        total_in = total_out = total_cached = 0
        for line in oc_log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "step_finish":
                continue
            tokens = (ev.get("part") or {}).get("tokens", {}) or {}
            total_in += int(tokens.get("input", 0) or 0)
            total_out += int(tokens.get("output", 0) or 0)
            cache = tokens.get("cache", {}) or {}
            total_cached += int(cache.get("read", 0) or 0)
        if total_in or total_out:
            return {"prompt_tokens": total_in, "completion_tokens": total_out,
                    "cached_tokens": total_cached}
    return None


def parse_result(job_dir: Path, model_spec: str | None = None) -> dict:
    """Pull reward + cost + tokens out of a finished job's result.json."""
    rj = job_dir / "result.json"
    if not rj.exists():
        return {"finished": False}
    try:
        data = json.loads(rj.read_text())
    except Exception:  # noqa: BLE001
        return {"finished": True, "error_kind": "parse_err"}

    out: dict = {"finished": True}
    stats = data.get("stats", {})
    evals = list((stats.get("evals") or {}).values())
    if not evals:
        out["error_kind"] = "no_eval"
        return out
    e = evals[0]
    n_err = e.get("n_errors", 0) or 0
    rs = e.get("reward_stats", {}).get("reward", {})
    excs = e.get("exception_stats", {})
    if excs and not rs:
        out["error_kind"] = list(excs.keys())[0][:40] if excs else "unknown_exc"
        return out
    # Take the max reward seen (only 1 trial per job in our case)
    if rs:
        try:
            best = max(float(k) for k in rs.keys() if rs[k])
        except ValueError:
            best = 0.0
        out["reward"] = best
    else:
        out["reward"] = 0.0
    # Cost + tokens — sum across trials (only 1 here)
    out["cost_usd"] = float(stats.get("cost_usd") or 0.0)
    out["prompt_tokens"] = int(stats.get("n_input_tokens") or 0)
    out["completion_tokens"] = int(stats.get("n_output_tokens") or 0)
    out["cached_tokens"] = int(stats.get("n_cache_tokens") or 0)

    # Opencode fallback: if cost is 0 but the harbor opencode log exists,
    # parse it ourselves and price from our table.
    # If Harbor didn't price this run (cost_usd 0.0), compute it ourselves
    # from token counts. Tokens may live in either result.json's n_* fields
    # (Harbor's opencode adapter populates them) or in the per-trial
    # opencode.txt log (raw JSON-line events).
    if out["cost_usd"] == 0.0 and model_spec is not None:
        from harbor_agents._shared.cost import compute_cost
        # Prefer in-memory tokens; if missing, parse opencode.txt
        if out["prompt_tokens"] == 0 and out["completion_tokens"] == 0:
            oc_totals = _opencode_token_totals(job_dir)
            if oc_totals is not None:
                out["prompt_tokens"] = oc_totals["prompt_tokens"]
                out["completion_tokens"] = oc_totals["completion_tokens"]
                out["cached_tokens"] = oc_totals["cached_tokens"]
        if out["prompt_tokens"] or out["completion_tokens"]:
            out["cost_usd"] = round(compute_cost(
                model_spec, out["prompt_tokens"],
                out["completion_tokens"], out["cached_tokens"],
            ), 6)
    # Categorize "no answer" as a special non-error
    if out["reward"] == 0.0 and not out.get("error_kind"):
        # Check for "no answer" in verifier logs
        for trial_dir in job_dir.iterdir():
            if trial_dir.is_dir():
                test_stdout = trial_dir / "verifier" / "test-stdout.txt"
                if test_stdout.exists() and "no answer" in test_stdout.read_text().lower():
                    out["error_kind"] = "no_answer"
                    break
    if not out.get("error_kind"):
        out["error_kind"] = "ok"
    return out


async def run_one(
    *, sem: asyncio.Semaphore, sweep_name: str, suite: Path, task_id: str,
    model_spec: str, model_slug: str, agent: dict, tier: str, phase: str,
    keys: dict[str, str], jobs_dir: Path, state: State, bill_to: str | None,
    timeout_sec: int, sweep_log, stagger_jitter: float = 3.0,
    max_retries_on_build_exc: int = 3,
):
    import random as _random
    job_name = job_name_for(sweep_name, task_id, model_slug, agent["name"])
    job_dir = jobs_dir / job_name
    triple = (task_id, model_spec, agent["name"])

    # Idempotency: skip if already finished
    if triple in state.finished_triples():
        return

    async with sem:
        # Light jitter to avoid bursty E2B sandbox-creation rate (saw
        # "Response 404/403/400" + BuildException when 8+ harbor processes
        # spawn within ms of each other).
        if stagger_jitter > 0:
            await asyncio.sleep(_random.uniform(0, stagger_jitter))

        # Record start
        state.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "start", "phase": phase, "tier": tier,
            "task_id": task_id, "model": model_spec, "agent": agent["name"],
            "job_name": job_name,
        })
        msg = f"[start] {phase}/{tier} {task_id[:30]:<30s} {model_slug:<18s} {agent['name']:<5s}"
        print(msg); sweep_log.write(msg + "\n"); sweep_log.flush()

        # If a previous orphaned job dir exists, nuke it for a clean state
        if job_dir.exists():
            subprocess.run(["rm", "-rf", str(job_dir)], check=False)

        cmd = build_cmd(
            suite=suite, task_id=task_id, model_spec=model_spec, model_slug=model_slug,
            agent=agent, keys=keys, job_name=job_name, jobs_dir=jobs_dir, bill_to=bill_to,
        )
        log_path = jobs_dir / "_sweep_logs" / f"{job_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if bill_to:
            env["HF_BILL_TO"] = bill_to  # custom — agents can read this if they want

        t0 = time.time()
        with log_path.open("w") as logf:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT, env=env,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    error_kind = "harbor_timeout"
                    finish_row = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "event": "finish", "phase": phase, "tier": tier,
                        "task_id": task_id, "model": model_spec, "agent": agent["name"],
                        "job_name": job_name, "elapsed_sec": round(time.time() - t0, 1),
                        "reward": 0.0, "error_kind": error_kind, "log_path": str(log_path),
                    }
                    state.append(finish_row)
                    print(f"[fail] {job_name} → timeout"); sweep_log.write(f"[fail] {job_name} timeout\n")
                    return
            except Exception as exc:  # noqa: BLE001
                state.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "finish", "phase": phase, "tier": tier,
                    "task_id": task_id, "model": model_spec, "agent": agent["name"],
                    "job_name": job_name, "elapsed_sec": round(time.time() - t0, 1),
                    "reward": 0.0, "error_kind": f"launch_err: {exc}",
                    "log_path": str(log_path),
                })
                return

        # Parse result.json
        result = parse_result(job_dir, model_spec=model_spec)
        elapsed = time.time() - t0
        # If we couldn't read result.json, scan the harbor log for hints
        if not result.get("finished"):
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            error_kind = categorize_error(log_text[-4000:]) if log_text else "no_result"
            result = {"finished": False, "error_kind": error_kind, "reward": 0.0}

        # Retry transient E2B failures up to max_retries_on_build_exc times.
        # SandboxException = creation-rate / throughput throttle; back off
        # harder than BuildException (which is per-build error).
        TRANSIENT_E2B = ("BuildException", "SandboxException", "InternalServerError")
        retries_left = max_retries_on_build_exc
        while (
            any((result.get("error_kind") or "").startswith(x) for x in TRANSIENT_E2B)
            and retries_left > 0
        ):
            retries_left -= 1
            kind = result.get("error_kind", "?")
            backoff = (15 if kind.startswith("SandboxException") else 5) + _random.uniform(0, 15)
            print(f"  [retry] {job_name}: {kind} → wait {backoff:.1f}s and re-run ({retries_left} left)")
            sweep_log.write(f"[retry] {job_name}: {kind} → backoff {backoff:.1f}s\n")
            await asyncio.sleep(backoff)
            subprocess.run(["rm", "-rf", str(job_dir)], check=False)
            with log_path.open("w") as logf:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT, env=env,
                    )
                    await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
                except asyncio.TimeoutError:
                    proc.kill(); await proc.wait()
                    result = {"finished": False, "error_kind": "harbor_timeout", "reward": 0.0}
                    break
            result = parse_result(job_dir, model_spec=model_spec)
            if not result.get("finished"):
                log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
                result = {"finished": False, "error_kind": categorize_error(log_text[-4000:]) or "no_result", "reward": 0.0}
        elapsed = time.time() - t0

        finish_row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "finish", "phase": phase, "tier": tier,
            "task_id": task_id, "model": model_spec, "agent": agent["name"],
            "job_name": job_name, "elapsed_sec": round(elapsed, 1),
            "reward": float(result.get("reward") or 0.0),
            "cost_usd": float(result.get("cost_usd") or 0.0),
            "prompt_tokens": int(result.get("prompt_tokens") or 0),
            "completion_tokens": int(result.get("completion_tokens") or 0),
            "cached_tokens": int(result.get("cached_tokens") or 0),
            "error_kind": result.get("error_kind", "unknown"),
            "log_path": str(log_path),
        }
        state.append(finish_row)
        tag = "✓" if finish_row["reward"] >= 1.0 else ("✗" if finish_row["error_kind"] == "ok" else "!")
        msg = (f"[{tag}] {phase}/{tier} {task_id[:30]:<30s} {model_slug:<18s} "
               f"{agent['name']:<5s} r={finish_row['reward']} ${finish_row['cost_usd']:.4f} "
               f"t={elapsed:.0f}s ({finish_row['error_kind']})")
        print(msg); sweep_log.write(msg + "\n"); sweep_log.flush()


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------


def get_task_ids_by_bucket(buckets: list[str], manifest_path: Path) -> list[tuple[str, str]]:
    """[(raw_id, bucket), …] in bucket order (easy→medium→hard)."""
    df = pd.read_parquet(manifest_path)
    out: list[tuple[str, str]] = []
    for b in buckets:
        sub = df[df.difficulty == b]
        for tid in sub.id:
            out.append((str(tid), b))
    return out


async def phase_3a(
    *, tasks_in_order: list[tuple[str, str]], state: State, sem: asyncio.Semaphore,
    cfg: dict, sweep_log,
) -> None:
    """Per-task pipelined tier escalation.

    Each task runs as an independent coroutine that walks T1→T2→…→T5,
    bailing as soon as it accumulates `STOP_AT` passes. All tasks run
    concurrently; the global semaphore caps total in-flight harbor runs.

    Strictly better than wave-based escalation because tier transitions
    don't pause the worker pool — when an easy task finishes T1 with 2
    wins, the freed slots go to harder tasks already in T2/T3 rather
    than sitting idle.
    """
    excluded_agents = set(cfg.get("exclude_agents", []))
    active_agents = [a for a in AGENTS if a["name"] not in excluded_agents]
    if excluded_agents:
        msg = f"  [config] excluding agents: {sorted(excluded_agents)} → running {[a['name'] for a in active_agents]}"
        print(msg); sweep_log.write(msg + "\n")

    async def sweep_one_task(task_id: str, bucket: str) -> None:
        # Graduation criterion: ≥ STOP_AT *unique* agents have passed.
        # E.g. with STOP_AT=2, we want at least 2 different harnesses to
        # have solved the task before stopping escalation. A single harness
        # passing on 2 different model sizes is NOT enough — we want
        # cross-harness diversity.
        for tier_idx, tier_models in enumerate(QWEN_TIERS, start=1):
            tier_name = f"T{tier_idx}"
            if len(state.successful_agents_for_task(task_id, "3A")) >= cfg["stop_at"]:
                return  # graduated (≥ STOP_AT unique harnesses passed)
            tier_pending = []
            for m in tier_models:
                for a in active_agents:
                    if (task_id, m["model"], a["name"]) in state.finished_triples():
                        continue
                    tier_pending.append(asyncio.create_task(run_one(
                        sem=sem, sweep_name=cfg["sweep_name"], suite=cfg["suite"],
                        task_id=task_id, model_spec=m["model"], model_slug=m["name"],
                        agent=a, tier=tier_name, phase="3A", keys=cfg["keys"],
                        jobs_dir=cfg["jobs_dir"], state=state, bill_to=cfg["bill_to"],
                        timeout_sec=cfg["timeout"], sweep_log=sweep_log,
                    )))
            if tier_pending:
                await asyncio.gather(*tier_pending, return_exceptions=True)
            # Recheck after the tier completes; bail if graduated
            if len(state.successful_agents_for_task(task_id, "3A")) >= cfg["stop_at"]:
                return

    msg = (f"\n=== Phase 3A — pipelined across {len(tasks_in_order)} task(s) "
           f"(buckets: {sorted(set(b for _, b in tasks_in_order))}) ===")
    print(msg); sweep_log.write(msg + "\n")
    task_coros = [sweep_one_task(t, b) for t, b in tasks_in_order]
    await asyncio.gather(*task_coros, return_exceptions=True)


async def phase_3b(
    *, tasks_in_order: list[tuple[str, str]], state: State, sem: asyncio.Semaphore,
    cfg: dict, sweep_log,
) -> None:
    """Frontier fallback on QWEN-FAIL tasks. Sonnet first, then GPT-5.5 only if Sonnet fails."""
    all_task_ids = [t for t, _ in tasks_in_order]
    failed = state.qwen_failed_tasks(all_task_ids)
    if not failed:
        msg = "[phase 3B] no QWEN-FAIL tasks; skipping frontier fallback."
        print(msg); sweep_log.write(msg + "\n")
        return
    msg = f"\n=== Phase 3B — frontier fallback on {len(failed)} task(s) ==="
    print(msg); sweep_log.write(msg + "\n")
    for f_model in FRONTIER_MODELS:
        pending: list = []
        for tid in failed:
            # Skip if already graduated in 3B (≥ STOP_AT unique frontier harnesses passed)
            if len(state.successful_agents_for_task(tid, "3B")) >= cfg["stop_at"]:
                continue
            for a in AGENTS:
                if (tid, f_model["model"], a["name"]) in state.finished_triples():
                    continue
                pending.append(asyncio.create_task(run_one(
                    sem=sem, sweep_name=cfg["sweep_name"], suite=cfg["suite"],
                    task_id=tid, model_spec=f_model["model"], model_slug=f_model["name"],
                    agent=a, tier="F", phase="3B", keys=cfg["keys"],
                    jobs_dir=cfg["jobs_dir"], state=state, bill_to=cfg["bill_to"],
                    timeout_sec=cfg["timeout"], sweep_log=sweep_log,
                )))
        if pending:
            sweep_log.write(f"\n=== Phase 3B — {f_model['name']}: {len(pending)} runs ===\n")
            await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


def cfg_from_args(args) -> dict:
    sweep_dir = SWEEP_DIR / args.sweep
    sweep_dir.mkdir(parents=True, exist_ok=True)
    return {
        "sweep_name": args.sweep,
        "sweep_dir": sweep_dir,
        "suite": REPO_ROOT / args.suite,
        "manifest": REPO_ROOT / "rl" / "cache" / "eval" / "eval_manifest.parquet",
        "jobs_dir": REPO_ROOT / "rl" / "jobs",
        "stop_at": args.stop_at,
        "timeout": args.timeout,
        "bill_to": args.bill_to,
        "keys": get_keys(REPO_ROOT / ".env"),
        "exclude_agents": getattr(args, "exclude_agents", []) or [],
    }


def cmd_run(args) -> int:
    load_dotenv(REPO_ROOT / ".env")
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    tasks = get_task_ids_by_bucket(args.buckets, cfg["manifest"])
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[sweep] {len(tasks)} tasks across {args.buckets}  concurrency={args.concurrency}  state has {len(state.rows)} prior events")
    sweep_log_path = cfg["sweep_dir"] / "sweep.log"
    with sweep_log_path.open("a") as sweep_log:
        sweep_log.write(f"\n\n=== sweep run @ {datetime.now(timezone.utc).isoformat()} ===\n")
        sem = asyncio.Semaphore(args.concurrency)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(phase_3a(
                tasks_in_order=tasks, state=state, sem=sem, cfg=cfg, sweep_log=sweep_log,
            ))
            if not args.no_fallback:
                loop.run_until_complete(phase_3b(
                    tasks_in_order=tasks, state=state, sem=sem, cfg=cfg, sweep_log=sweep_log,
                ))
        finally:
            loop.close()
    print_status(state, tasks)
    return 0


def cmd_status(args) -> int:
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    tasks = get_task_ids_by_bucket(args.buckets, cfg["manifest"])
    print_status(state, tasks)
    return 0


def print_status(state: State, tasks: list[tuple[str, str]]) -> None:
    finished = state.finished_triples()
    started_not_finished = state.started_but_not_finished()
    errs = state.errored_runs()
    print()
    print(f"  total runs (finished events): {sum(1 for r in state.rows if r.get('event')=='finish')}")
    print(f"  unique (task, model, agent) completed: {len(finished)}")
    print(f"  orphaned (start without finish): {len(started_not_finished)}")
    print(f"  errored runs: {len(errs)}")
    # Per-task summary
    passes = state.passes_per_task()
    if tasks:
        graduated = sum(1 for t, _ in tasks if passes.get(t, 0) >= 2)
        print(f"  tasks graduated (≥2 passes): {graduated}/{len(tasks)}")


def cmd_cost(args) -> int:
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    summary = state.cost_total()
    out_path = cfg["sweep_dir"] / "cost_running.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[save] {out_path}")
    return 0


def cmd_diagnose(args) -> int:
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    errs = state.errored_runs()
    print(f"errored runs: {len(errs)}\n")
    kinds = Counter(r.get("error_kind", "?") for r in errs)
    print("by error_kind:")
    for k, c in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24s} {c:>4d}")
    print()
    by_agent = Counter(r["agent"] for r in errs)
    by_model = Counter(r["model"] for r in errs)
    print("by agent:")
    for k, c in by_agent.most_common(10):
        print(f"  {k:<10s} {c}")
    print()
    print("by model:")
    for k, c in by_model.most_common(10):
        print(f"  {k:<60s} {c}")
    print()
    # Print a few example crashes with log paths
    print("first 5 example crashes (with log paths):")
    for r in errs[:5]:
        print(f"  [{r['error_kind']}] {r['task_id']}  {r['model']}  {r['agent']}")
        print(f"    log: {r.get('log_path', '(no log)')}")
    return 0


def cmd_retry_failed(args) -> int:
    """Re-launch errored runs by deleting their state finish events."""
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    errs = state.errored_runs()
    if args.error:
        errs = [e for e in errs if e.get("error_kind") == args.error]
    if not errs:
        print(f"no errored runs match {args.error or '(any)'}")
        return 0
    print(f"will retry {len(errs)} errored run(s)")
    # The simplest way to "retry" is to remove the finish event from state.jsonl
    # so the next sweep run picks it up. Rewrite state.jsonl excluding them.
    bad_keys = {(r["task_id"], r["model"], r["agent"], r.get("ts")) for r in errs}
    kept = [r for r in state.rows
            if not (r.get("event") == "finish" and (r["task_id"], r["model"], r["agent"], r.get("ts")) in bad_keys)]
    backup = state.state_path.with_suffix(".jsonl.bak")
    state.state_path.rename(backup)
    with state.state_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"  rewrote state.jsonl (backup at {backup}). re-run `sweep run` to retry.")
    return 0


def cmd_dry_run(args) -> int:
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    tasks = get_task_ids_by_bucket(args.buckets, cfg["manifest"])
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[dry-run] {len(tasks)} tasks; would launch in tiers below.\n")
    total = 0
    for tier_idx, tier_models in enumerate(QWEN_TIERS, start=1):
        for_task = 0
        for tid, _ in tasks:
            if len(state.successful_agents_for_task(tid, "3A")) >= cfg["stop_at"]:
                continue
            for m in tier_models:
                for a in AGENTS:
                    if (tid, m["model"], a["name"]) in state.finished_triples():
                        continue
                    for_task += 1
        print(f"  Phase 3A T{tier_idx} (models={[m['name'] for m in tier_models]}): {for_task} runs")
        total += for_task
    failed = state.qwen_failed_tasks([t for t, _ in tasks])
    if not args.no_fallback:
        f_runs = len(failed) * len(FRONTIER_MODELS) * len(AGENTS)
        print(f"  Phase 3B frontier (QWEN-FAIL={len(failed)}): up to {f_runs} runs")
        total += f_runs
    print(f"\n  Total estimated runs: {total}")
    return 0


def cmd_fallback(args) -> int:
    args.no_fallback = False  # ensure
    load_dotenv(REPO_ROOT / ".env")
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    tasks = get_task_ids_by_bucket(args.buckets, cfg["manifest"])
    if args.limit:
        tasks = tasks[: args.limit]
    sweep_log_path = cfg["sweep_dir"] / "sweep.log"
    with sweep_log_path.open("a") as sweep_log:
        sweep_log.write(f"\n\n=== fallback @ {datetime.now(timezone.utc).isoformat()} ===\n")
        sem = asyncio.Semaphore(args.concurrency)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(phase_3b(
                tasks_in_order=tasks, state=state, sem=sem, cfg=cfg, sweep_log=sweep_log,
            ))
        finally:
            loop.close()
    print_status(state, tasks)
    return 0


def cmd_report(args) -> int:
    cfg = cfg_from_args(args)
    state = State(cfg["sweep_dir"]).load()
    tasks = get_task_ids_by_bucket(args.buckets, cfg["manifest"])
    summary = state.cost_total()
    passes = state.passes_per_task()
    failed = state.qwen_failed_tasks([t for t, _ in tasks])

    lines = []
    lines.append(f"# Sweep `{args.sweep}` — report\n")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}_\n")
    lines.append(f"\n## Run counts\n")
    lines.append(f"- Total finished events: {summary['total_runs']}")
    lines.append(f"- Unique (task, model, agent): {len(state.finished_triples())}")
    lines.append(f"- Tasks graduated (≥2 wins): {sum(1 for t,_ in tasks if passes.get(t,0)>=2)}/{len(tasks)}")
    lines.append(f"- QWEN-FAIL tasks (zero Qwen passes): {len(failed)}\n")

    lines.append(f"\n## Cost & tokens\n")
    lines.append(f"- Total cost: **${summary['total_cost_usd']}**")
    lines.append(f"- Tokens — input: {summary['tokens']['input']:,}, output: {summary['tokens']['output']:,}, cached: {summary['tokens']['cached']:,}\n")

    lines.append(f"\n## By model\n")
    lines.append("| Model | Runs | Passes | Errors | Cost ($) |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, v in sorted(summary["by_model"].items(), key=lambda kv: -kv[1]["runs"]):
        lines.append(f"| {m} | {v['runs']} | {v['passes']} | {v['errors']} | {v['cost']:.4f} |")
    lines.append(f"\n## By agent\n")
    lines.append("| Agent | Runs | Passes | Errors | Cost ($) |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, v in sorted(summary["by_agent"].items(), key=lambda kv: -kv[1]["runs"]):
        lines.append(f"| {m} | {v['runs']} | {v['passes']} | {v['errors']} | {v['cost']:.4f} |")

    out = cfg["sweep_dir"] / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"[report] {out}")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def add_common(p):
    p.add_argument("--sweep", default=DEFAULT_SWEEP_NAME)
    p.add_argument("--buckets", default="easy,medium",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    p.add_argument("--suite", default="rl/harbor/tasks/jupyter-agent-eval-v1")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--stop-at", type=int, default=2)
    p.add_argument("--timeout", type=int, default=480, help="Per-run timeout seconds (default 8 min)")
    p.add_argument("--bill-to", default=None, help="X-HF-Bill-To header (e.g. 'huggingface'). Currently informational.")
    p.add_argument("--limit", type=int, default=None, help="Take only the first N tasks (smoke).")
    p.add_argument("--exclude-agents", default="",
                   type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   help="Comma-list of agent slugs to skip (e.g. 'oc'). Useful when one harness is too slow.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Full sweep (Phase 3A + 3B).")
    add_common(p_run)
    p_run.add_argument("--no-fallback", action="store_true", help="Skip Phase 3B.")

    p_status = sub.add_parser("status", help="Quick state.jsonl summary.")
    add_common(p_status)

    p_cost = sub.add_parser("cost", help="Rolling cost totals → cost_running.json.")
    add_common(p_cost)

    p_diag = sub.add_parser("diagnose", help="Error histogram + top failure modes.")
    add_common(p_diag)

    p_retry = sub.add_parser("retry-failed", help="Strip errored finish events; next `run` re-launches.")
    add_common(p_retry)
    p_retry.add_argument("--error", default=None, help="Only retry errors of this kind.")

    p_dry = sub.add_parser("dry-run", help="Show planned launches without executing.")
    add_common(p_dry)
    p_dry.add_argument("--no-fallback", action="store_true")

    p_fb = sub.add_parser("fallback", help="Phase 3B alone (assumes 3A done).")
    add_common(p_fb)

    p_rep = sub.add_parser("report", help="Generate REPORT.md.")
    add_common(p_rep)

    args = parser.parse_args()
    dispatch = {
        "run": cmd_run, "status": cmd_status, "cost": cmd_cost,
        "diagnose": cmd_diagnose, "retry-failed": cmd_retry_failed,
        "dry-run": cmd_dry_run, "fallback": cmd_fallback, "report": cmd_report,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
