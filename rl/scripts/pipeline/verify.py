"""Phase B — run one Harbor trial and parse its result.

Shells out to the `harbor` CLI. We could call its Python API directly, but
shelling has two upsides:
  (a) easy to log the exact reproducible command for debugging
  (b) we already know it works from sweep.py

Per-trial output lives under <state-dir>/trials/<job_name>/.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Where the agents live (still under the legacy `harbor_agents/` name —
# rename to `harbor.agents.*` deferred until after M1 works).
SETA_AGENT_IMPORT = "rl.harbor_agents.seta:SetaToolAgent"

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


@dataclass
class TrialResult:
    job_name: str
    trial_dir: Path | None        # the per-trial subfolder Harbor creates
    reward: float
    predicted_answer: str
    elapsed_sec: float
    error_kind: str               # "ok" | "no_answer" | "harbor_error" | "harbor_timeout" | ...
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    stdout_path: Path
    stderr_path: Path


def _read_env_keys() -> dict[str, str]:
    """Pluck the keys we need from the project .env (without dotenv reload)."""
    keys: dict[str, str] = {}
    if not ENV_FILE.exists():
        return keys
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys


def safe_id(task_id: str) -> str:
    return task_id.replace("/", "_").replace(".ipynb", "")


def build_command(*, suite_path: Path, task_id: str, model: str,
                  job_name: str, jobs_dir: Path,
                  sandbox: str = "docker",
                  keys: dict[str, str] | None = None) -> list[str]:
    keys = keys or _read_env_keys()
    cmd = [
        "harbor", "run",
        "-p", str(suite_path),
        "--agent-import-path", SETA_AGENT_IMPORT,
        "--model", model,
        "--env", sandbox,
        "--env-file", str(ENV_FILE),
        "--yes",
        "--job-name", job_name,
        "--jobs-dir", str(jobs_dir),
        "-i", safe_id(task_id),
        "-n", "1",
    ]
    # Pass through API keys to the agent and verifier
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        v = keys.get(k)
        if v:
            cmd += ["--ae", f"{k}={v}"]
    if keys.get("OPENAI_API_KEY"):
        cmd += ["--ve", f"OPENAI_API_KEY={keys['OPENAI_API_KEY']}"]
    return cmd


def _find_trial_dir(job_dir: Path) -> Path | None:
    """Harbor creates exactly one subfolder like `<task>__<random6>` per trial."""
    if not job_dir.exists():
        return None
    for child in job_dir.iterdir():
        if child.is_dir() and re.search(r"__[A-Za-z0-9]{6,}$", child.name):
            return child
    return None


def _read_text(p: Path, max_chars: int | None = None) -> str:
    try:
        t = p.read_text(errors="replace")
        if max_chars and len(t) > max_chars:
            return t[:max_chars] + f"\n... [truncated, full {len(t)} chars]"
        return t
    except FileNotFoundError:
        return ""


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def parse_trial(job_dir: Path) -> dict:
    """Extract reward, predicted answer, cost, tokens from a Harbor trial dir.

    Returns a partial dict; the caller wraps it into a TrialResult.
    """
    out = {"reward": 0.0, "predicted_answer": "", "error_kind": "no_answer",
           "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
           "cached_tokens": 0, "trial_dir": None}

    trial = _find_trial_dir(job_dir)
    if trial is None:
        out["error_kind"] = "no_trial_dir"
        return out
    out["trial_dir"] = trial

    # reward
    rw = _read_text(trial / "verifier" / "reward.txt").strip()
    if rw:
        try:
            out["reward"] = float(rw)
        except ValueError:
            pass

    # Predicted answer — Harbor doesn't copy /workdir/answer.txt out by default,
    # but the grader prints `pred='<value>'` in its stdout. Parse it from there.
    grader_stdout = _read_text(trial / "verifier" / "test-stdout.txt")
    pred = ""
    m = re.search(r"pred=(['\"])(.*?)\1", grader_stdout)
    if m:
        pred = m.group(2)
    out["predicted_answer"] = pred

    # cost / tokens via Harbor's result.json
    res = _read_json(trial / "result.json") or {}
    stats = (res.get("stats") or {})
    out["cost_usd"] = float(stats.get("cost_usd") or 0.0)
    out["prompt_tokens"] = int(stats.get("n_input_tokens") or 0)
    out["completion_tokens"] = int(stats.get("n_output_tokens") or 0)
    out["cached_tokens"] = int(stats.get("n_cache_tokens") or 0)

    # Cost-shim: seta also writes its own usage; prefer the higher non-zero source.
    seta_usage = _read_json(trial / "agent" / "seta_agent.usage.json")
    if seta_usage and out["cost_usd"] == 0:
        out["cost_usd"] = float(seta_usage.get("cost_usd") or 0.0)
        out["prompt_tokens"] = int(seta_usage.get("prompt_tokens") or out["prompt_tokens"])
        out["completion_tokens"] = int(seta_usage.get("completion_tokens") or out["completion_tokens"])
        out["cached_tokens"] = int(seta_usage.get("cached_tokens") or out["cached_tokens"])

    # error kind: if reward == 1.0 ok, else look at grader for hints
    if out["reward"] >= 1.0:
        out["error_kind"] = "ok"
    elif not pred:
        out["error_kind"] = "no_answer"
    else:
        out["error_kind"] = "wrong_answer"

    return out


def run_trial(*, suite_path: Path, task_id: str, model: str,
              job_name: str, jobs_dir: Path,
              sandbox: str = "docker",
              log_dir: Path | None = None) -> TrialResult:
    """Invoke harbor and parse the result. Returns a TrialResult."""
    cmd = build_command(
        suite_path=suite_path, task_id=task_id, model=model,
        job_name=job_name, jobs_dir=jobs_dir, sandbox=sandbox,
    )
    log_dir = log_dir or jobs_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job_name}.stdout.log"
    stderr_path = log_dir / f"{job_name}.stderr.log"

    t0 = time.time()
    try:
        with stdout_path.open("w") as so, stderr_path.open("w") as se:
            # Write the exact reproducible command at the top of stdout
            so.write("# CMD: " + " ".join(shlex.quote(p) for p in cmd) + "\n")
            so.flush()
            proc = subprocess.run(cmd, stdout=so, stderr=se,
                                  timeout=1800, check=False, cwd=REPO_ROOT)
        elapsed = time.time() - t0
        job_dir = jobs_dir / job_name
        parsed = parse_trial(job_dir)
        # If harbor exited non-zero and we have no trial dir, that's a harbor error
        if proc.returncode != 0 and parsed["trial_dir"] is None:
            parsed["error_kind"] = "harbor_error"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        parsed = {"reward": 0.0, "predicted_answer": "", "error_kind": "harbor_timeout",
                  "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                  "cached_tokens": 0, "trial_dir": None}

    return TrialResult(
        job_name=job_name,
        trial_dir=parsed["trial_dir"],
        reward=parsed["reward"],
        predicted_answer=parsed["predicted_answer"],
        elapsed_sec=elapsed,
        error_kind=parsed["error_kind"],
        cost_usd=parsed["cost_usd"],
        prompt_tokens=parsed["prompt_tokens"],
        completion_tokens=parsed["completion_tokens"],
        cached_tokens=parsed["cached_tokens"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
