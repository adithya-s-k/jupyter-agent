"""Build a self-contained HTML dashboard for sweep v2 results.

Walks `state.jsonl` + per-trial dirs + `eval_manifest.parquet`, packages
everything as one JSON blob, and renders `visualization.html` with embedded
data so it can be opened directly (no server, no CDN).

Usage:
    uv run --project rl python -m prepare.build_visualization \
        --sweep-dir cache/sweep/v2 \
        --jobs-dir jobs \
        --tasks-dir harbor/tasks/jupyter-agent-eval-v1 \
        --eval-manifest cache/eval/eval_manifest.parquet \
        --out cache/sweep/v2/visualization.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODEL_SHORT: dict[str, str] = {
    "hf/Qwen/Qwen3-4B-Instruct-2507:nscale": "qwen3-4b-inst",
    "hf/Qwen/Qwen3-4B-Thinking-2507:nscale": "qwen3-4b-think",
    "hf/Qwen/Qwen3-8B:nscale": "qwen3-8b",
    "hf/Qwen/Qwen3-14B:nscale": "qwen3-14b",
    "hf/Qwen/Qwen3-Coder-30B-A3B-Instruct:nscale": "qwen3-coder-30b",
    "hf/Qwen/Qwen3-32B:nscale": "qwen3-32b",
    "hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale": "qwen3-235b",
    "anthropic/claude-sonnet-4-6": "sonnet-4-6",
    "openai/gpt-5.5": "gpt-5.5",
}

AGENT_LABEL: dict[str, str] = {
    "bash": "bash-only",
    "jupy": "jupyter-tool",
    "seta": "seta-tool",
    "oc": "opencode",
}

# Ordered tier groups for the heatmap columns. Each item:
#   (tier_code, tier_label, [model_short, ...])
TIER_GROUPS: list[tuple[str, str, list[str]]] = [
    ("T1", "Tier 1 — 4B",   ["qwen3-4b-inst", "qwen3-4b-think"]),
    ("T2", "Tier 2 — 8B",   ["qwen3-8b"]),
    ("T3", "Tier 3 — 14B",  ["qwen3-14b"]),
    ("T4", "Tier 4 — 30B",  ["qwen3-coder-30b", "qwen3-32b"]),
    ("T5", "Tier 5 — 235B", ["qwen3-235b"]),
    ("F",  "Frontier",      ["sonnet-4-6", "gpt-5.5"]),
]
AGENTS_ORDER: list[str] = ["bash", "jupy", "seta"]
EXCLUDE_AGENTS: set[str] = {"oc"}


def _task_id_safe(task_id: str) -> str:
    s = task_id.replace("/", "_").replace(".ipynb", "")
    return s


def _load_state(sweep_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (sweep_dir / "state.jsonl").read_text().splitlines() if l.strip()]


def _read_text(p: Path, max_chars: int | None = None) -> str:
    try:
        t = p.read_text(errors="replace")
        if max_chars and len(t) > max_chars:
            return t[:max_chars] + f"\n... [truncated to {max_chars} chars]"
        return t
    except FileNotFoundError:
        return ""


def _read_json(p: Path) -> dict | list | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _find_trial_dir(job_dir: Path) -> Path | None:
    if not job_dir.exists():
        return None
    for child in job_dir.iterdir():
        if child.is_dir() and re.search(r"__[A-Za-z0-9]{6,}$", child.name):
            return child
    return None


def _load_trial_artifacts(job_name: str, jobs_dir: Path, agent: str) -> dict:
    job_dir = jobs_dir / job_name
    out: dict = {
        "trajectory": None,
        "grader_text": "",
        "reward_raw": None,
        "usage": None,
        "trial_log_tail": "",
    }
    trial = _find_trial_dir(job_dir)
    if not trial:
        return out

    out["grader_text"] = _read_text(trial / "verifier" / "test-stdout.txt", max_chars=4000)
    rw = _read_text(trial / "verifier" / "reward.txt").strip()
    if rw:
        try:
            out["reward_raw"] = float(rw)
        except ValueError:
            out["reward_raw"] = rw

    trial_log = trial / "trial.log"
    if trial_log.exists():
        full = _read_text(trial_log)
        # last ~80 lines
        out["trial_log_tail"] = "\n".join(full.splitlines()[-80:])

    agent_dir = trial / "agent"
    if not agent_dir.exists():
        return out

    if agent == "oc":
        traj = _read_json(agent_dir / "trajectory.json")
        out["trajectory"] = traj
        out["opencode_log"] = _read_text(agent_dir / "opencode.txt", max_chars=20000)
    else:
        # jupy/bash/seta → <agent>_agent.trajectory.json + <agent>_agent.usage.json
        names = {
            "bash": "bash_agent",
            "jupy": "jupyter_agent",
            "seta": "seta_agent",
        }
        prefix = names.get(agent, agent)
        traj = _read_json(agent_dir / f"{prefix}.trajectory.json")
        if traj is None:
            # fall back: any *.trajectory.json
            for f in agent_dir.glob("*.trajectory.json"):
                traj = _read_json(f)
                break
        out["trajectory"] = traj
        usage = _read_json(agent_dir / f"{prefix}.usage.json")
        if usage is None:
            for f in agent_dir.glob("*.usage.json"):
                usage = _read_json(f)
                break
        out["usage"] = usage
    return out


def _normalize_trajectory(traj, agent: str) -> list[dict]:
    """Return [{role, content, tool_calls?}, ...] with content as plain string.

    Handles both list-of-dicts (our custom agents) and opencode's nested format.
    Truncates very long tool outputs to keep HTML reasonable.
    """
    if traj is None:
        return []
    if not isinstance(traj, list):
        return []
    out: list[dict] = []
    for m in traj:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("type") or "?"
        content = m.get("content")
        if isinstance(content, list):
            # opencode-style content list (blocks)
            parts = []
            for b in content:
                if isinstance(b, dict):
                    parts.append(b.get("text") or b.get("content") or json.dumps(b)[:500])
                else:
                    parts.append(str(b))
            content = "\n".join(p for p in parts if p)
        elif content is None:
            content = ""
        else:
            content = str(content)
        if len(content) > 12000:
            content = content[:12000] + f"\n... [truncated, full length {len(content)}]"
        entry: dict = {"role": role, "content": content}
        tc = m.get("tool_calls")
        if tc:
            entry["tool_calls"] = []
            for t in tc:
                fn = (t or {}).get("function") or {}
                entry["tool_calls"].append({
                    "id": t.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
        tcid = m.get("tool_call_id")
        if tcid:
            entry["tool_call_id"] = tcid
        out.append(entry)
    return out


def _as_list(v) -> list:
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _load_eval_manifest(manifest: Path) -> dict[str, dict]:
    df = pd.read_parquet(manifest)
    by_id: dict[str, dict] = {}
    for _, r in df.iterrows():
        by_id[r["id"]] = {
            "difficulty": r.get("difficulty"),
            "question": r.get("question"),
            "answer": r.get("answer"),
            "kaggle": r.get("kaggle_dataset_name"),
            "files": _as_list(r.get("files_used")),
            "packages": _as_list(r.get("packages_used")),
            "answer_type": r.get("feat_answer_type"),
            "edu_score": int(r.get("edu_score") or 0),
        }
    return by_id


def build_data(sweep_dir: Path, jobs_dir: Path, tasks_dir: Path, manifest: Path) -> dict:
    events = _load_state(sweep_dir)
    finishes = [
        e for e in events
        if e.get("event") == "finish" and e.get("agent") not in EXCLUDE_AGENTS
    ]
    eval_meta = _load_eval_manifest(manifest)

    # group by (task_id, model, agent), keep latest finish per triple
    by_triple: dict[tuple, dict] = {}
    for e in finishes:
        key = (e["task_id"], e["model"], e["agent"])
        if key not in by_triple or e["ts"] > by_triple[key]["ts"]:
            by_triple[key] = e

    tasks: dict[str, dict] = {}
    for (task_id, model, agent), e in by_triple.items():
        if task_id not in tasks:
            meta = eval_meta.get(task_id, {})
            tasks[task_id] = {
                "id": task_id,
                "id_safe": _task_id_safe(task_id),
                "difficulty": meta.get("difficulty", "?"),
                "question": meta.get("question", ""),
                "answer": str(meta.get("answer", "")),
                "kaggle": meta.get("kaggle", ""),
                "files": meta.get("files", []),
                "packages": meta.get("packages", []),
                "answer_type": meta.get("answer_type", ""),
                "edu_score": meta.get("edu_score", 0),
                "trials": [],
            }
        artifacts = _load_trial_artifacts(e["job_name"], jobs_dir, agent)
        trial = {
            "model": model,
            "model_short": MODEL_SHORT.get(model, model.split("/")[-1]),
            "agent": agent,
            "agent_label": AGENT_LABEL.get(agent, agent),
            "tier": e.get("tier", ""),
            "phase": e.get("phase", ""),
            "reward": float(e.get("reward") or 0.0),
            "cost": float(e.get("cost_usd") or 0.0),
            "prompt_tokens": int(e.get("prompt_tokens") or 0),
            "completion_tokens": int(e.get("completion_tokens") or 0),
            "cached_tokens": int(e.get("cached_tokens") or 0),
            "elapsed_sec": float(e.get("elapsed_sec") or 0.0),
            "error_kind": e.get("error_kind") or "",
            "job_name": e["job_name"],
            "ts": e["ts"],
            "grader_text": artifacts["grader_text"],
            "trajectory": _normalize_trajectory(artifacts["trajectory"], agent),
            "opencode_log": artifacts.get("opencode_log", "") if agent == "oc" else "",
            "trial_log_tail": artifacts["trial_log_tail"],
        }
        tasks[task_id]["trials"].append(trial)

    # task-level aggregates
    for t in tasks.values():
        passing = [tr for tr in t["trials"] if tr["reward"] >= 1.0]
        passing_harnesses = sorted({tr["agent"] for tr in passing})
        t["passing_harnesses"] = passing_harnesses
        t["graduated"] = len(passing_harnesses) >= 2
        # graduation tier = the tier at which the 2nd distinct harness passed
        if t["graduated"]:
            seen: set[str] = set()
            grad_tier = None
            for tr in sorted(passing, key=lambda x: (x["phase"], x["tier"], x["ts"])):
                seen.add(tr["agent"])
                if len(seen) >= 2:
                    grad_tier = tr["tier"]
                    break
            t["graduated_tier"] = grad_tier
        else:
            t["graduated_tier"] = None
        t["total_cost"] = round(sum(tr["cost"] for tr in t["trials"]), 4)
        t["total_runs"] = len(t["trials"])
        t["any_pass"] = len(passing_harnesses) >= 1
        # best per (agent): first passing or worst-by-tier
        best: dict[str, dict] = {}
        for tr in t["trials"]:
            cur = best.get(tr["agent"])
            if cur is None:
                best[tr["agent"]] = tr
            else:
                # prefer passing, else most recent
                if tr["reward"] > cur["reward"] or (
                    tr["reward"] == cur["reward"] and tr["ts"] > cur["ts"]
                ):
                    best[tr["agent"]] = tr
        t["best_per_agent"] = {
            a: {"tier": tr["tier"], "model_short": tr["model_short"], "reward": tr["reward"],
                "error_kind": tr["error_kind"]}
            for a, tr in best.items()
        }
        # sort trials by tier order then phase
        tier_order = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5, "F": 6}
        t["trials"].sort(key=lambda x: (tier_order.get(x["tier"], 99), x["phase"], x["ts"]))

    task_list = sorted(tasks.values(), key=lambda x: (
        0 if x["difficulty"] == "easy" else 1,
        x["id"],
    ))

    # summary
    summary = {
        "tasks_total": len(task_list),
        "tasks_graduated": sum(1 for t in task_list if t["graduated"]),
        "tasks_any_pass": sum(1 for t in task_list if t["any_pass"]),
        "tasks_failed": sum(1 for t in task_list if not t["any_pass"]),
        "runs_total": len(by_triple),
        "runs_passed": sum(1 for tr in by_triple.values() if (tr.get("reward") or 0) >= 1.0),
        "cost_total": round(sum(float(e.get("cost_usd") or 0) for e in by_triple.values()), 4),
        "tokens_prompt": sum(int(e.get("prompt_tokens") or 0) for e in by_triple.values()),
        "tokens_completion": sum(int(e.get("completion_tokens") or 0) for e in by_triple.values()),
        "tokens_cached": sum(int(e.get("cached_tokens") or 0) for e in by_triple.values()),
        "first_ts": min(e["ts"] for e in finishes),
        "last_ts": max(e["ts"] for e in finishes),
    }

    # per-model, per-agent slices
    by_model: dict[str, dict] = {}
    by_agent: dict[str, dict] = {}
    for e in by_triple.values():
        m = e["model"]
        a = e["agent"]
        passed = (e.get("reward") or 0) >= 1.0
        cost = float(e.get("cost_usd") or 0)
        for d, k in ((by_model, m), (by_agent, a)):
            d.setdefault(k, {"runs": 0, "passes": 0, "cost": 0.0, "tokens_in": 0, "tokens_out": 0})
            d[k]["runs"] += 1
            d[k]["passes"] += int(passed)
            d[k]["cost"] += cost
            d[k]["tokens_in"] += int(e.get("prompt_tokens") or 0)
            d[k]["tokens_out"] += int(e.get("completion_tokens") or 0)

    by_model_list = [
        {"model": m, "model_short": MODEL_SHORT.get(m, m), **v, "cost": round(v["cost"], 4)}
        for m, v in by_model.items()
    ]
    by_model_list.sort(key=lambda x: x["model_short"])
    by_agent_list = [
        {"agent": a, "agent_label": AGENT_LABEL.get(a, a), **v, "cost": round(v["cost"], 4)}
        for a, v in by_agent.items()
    ]

    # build per-task heatmap matrix: matrix[agent][model_short] -> trial_idx or null
    for t in task_list:
        cell_map: dict[str, dict[str, int]] = {a: {} for a in AGENTS_ORDER}
        for i, tr in enumerate(t["trials"]):
            cell_map.setdefault(tr["agent"], {})[tr["model_short"]] = i
        t["cell_map"] = cell_map

    return {
        "summary": summary,
        "by_model": by_model_list,
        "by_agent": by_agent_list,
        "agents_order": AGENTS_ORDER,
        "models_order": list(MODEL_SHORT.values()),
        "tier_groups": [
            {"tier": tier, "label": label, "models": models}
            for tier, label, models in TIER_GROUPS
        ],
        "tasks": task_list,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jupyter-Agent Sweep v2 — Eval Dashboard</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --panel2: #1e242c;
    --border: #2c333d;
    --text: #d7dadf;
    --muted: #8a9099;
    --accent: #59a6ff;
    --pass: #3ec97a;
    --fail: #e25c5c;
    --warn: #e6b455;
    --gray: #565d68;
  }
  * { box-sizing: border-box; }
  /* ===== Minimal scrollbars (firefox + webkit) ===== */
  * { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.18) transparent; }
  *::-webkit-scrollbar { width: 8px; height: 8px; }
  *::-webkit-scrollbar-track { background: transparent; }
  *::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 4px;
                               border: 2px solid transparent; background-clip: content-box; }
  *::-webkit-scrollbar-thumb:hover { background-color: rgba(255,255,255,0.32); }
  *::-webkit-scrollbar-corner { background: transparent; }
  html, body { height: 100%; margin: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif;
         background: var(--bg); color: var(--text); font-size: 13px; line-height: 1.45;
         display: flex; flex-direction: column; min-height: 100vh; overflow: hidden; }

  /* ===== Collapsible header ===== */
  header { background: var(--panel); border-bottom: 1px solid var(--border); flex: 0 0 auto; }
  .hdr-bar { padding: 10px 18px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .hdr-bar h1 { margin: 0; font-size: 15px; font-weight: 600; }
  .hdr-bar .sub { color: var(--muted); font-size: 11px; }
  .hdr-bar .pills { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  .hdr-bar .pill-stat { background: var(--panel2); padding: 3px 9px; border-radius: 4px;
                        font-size: 11px; color: var(--text); border: 1px solid var(--border); }
  .hdr-bar .pill-stat b { color: var(--accent); font-weight: 600; }
  .hdr-bar .toggle { background: var(--panel2); color: var(--text); border: 1px solid var(--border);
                     padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; }
  .hdr-bar .toggle:hover { background: var(--border); }

  .hdr-expanded { padding: 4px 18px 14px; border-top: 1px solid var(--border); }
  .hdr-expanded.hidden { display: none; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin-top: 10px; }
  .stat { background: var(--panel2); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); }
  .stat .num { font-size: 18px; font-weight: 600; }
  .stat .lbl { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .agg-tables { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }
  .agg-table { background: var(--panel2); padding: 10px; border-radius: 6px; border: 1px solid var(--border); }
  .agg-table h4 { margin: 0 0 6px; font-size: 12px; color: var(--muted); }
  .agg-table table { width: 100%; border-collapse: collapse; font-size: 11px; }
  .agg-table th { text-align: left; padding: 3px 5px; color: var(--muted); border-bottom: 1px solid var(--border); }
  .agg-table td { padding: 3px 5px; border-bottom: 1px solid var(--border); font-family: ui-monospace, SF Mono, Menlo, monospace; }
  .agg-table .pct { color: var(--accent); }

  /* ===== Filter / view bar ===== */
  .filters { padding: 8px 18px; background: var(--panel); border-bottom: 1px solid var(--border);
             display: flex; gap: 10px; flex-wrap: wrap; align-items: center; flex: 0 0 auto; }
  .filters input, .filters select { background: var(--panel2); color: var(--text);
        border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
  .filters input[type=search] { width: 240px; }
  .filters label { color: var(--muted); font-size: 11px; }
  .view-tabs { display: flex; gap: 0; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
  .view-tabs button { background: var(--panel2); color: var(--muted); border: 0; padding: 4px 12px; cursor: pointer; font-size: 11px; }
  .view-tabs button.active { background: var(--accent); color: white; }

  /* ===== Main split (horizontal: grid on left, optional side pane on right) ===== */
  main { flex: 1 1 auto; display: flex; flex-direction: row; min-height: 0; overflow: hidden; }
  .view-pane { flex: 1 1 auto; overflow: auto; min-width: 0; min-height: 0; }
  .view-pane.hidden { display: none; }
  .side-pane { flex: 0 0 640px; overflow: hidden; border-left: 1px solid var(--border);
               background: var(--panel); display: flex; flex-direction: column; min-height: 0;
               position: relative; }
  .side-resizer { position: absolute; left: -3px; top: 0; bottom: 0; width: 7px; cursor: col-resize;
                   z-index: 10; background: transparent; transition: background 0.15s; }
  .side-resizer:hover, .side-resizer.active { background: rgba(89,166,255,0.35); }
  body.resizing { cursor: col-resize !important; user-select: none; }
  body.resizing * { user-select: none; pointer-events: none; }
  body.resizing .side-resizer { pointer-events: auto; }
  .side-pane.hidden { display: none; }
  .side-head { padding: 8px 14px; background: var(--panel2); border-bottom: 1px solid var(--border);
               display: flex; gap: 10px; align-items: center; flex: 0 0 auto; }
  .side-head h2 { font-size: 13px; margin: 0; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .side-head button { background: var(--panel); color: var(--text); border: 1px solid var(--border);
                      padding: 3px 10px; border-radius: 3px; cursor: pointer; font-size: 11px; }
  .side-head button:hover { background: var(--border); }
  .side-body { padding: 12px 16px; overflow-y: auto; flex: 1 1 auto; min-height: 0; }

  /* ===== Heatmap grid ===== */
  .heatmap-wrap { padding: 0; }
  /* width: max-content forces the table to size to its colgroup widths exactly,
     not stretch to fill the view-pane. Without this, table-layout:fixed
     distributes any extra space across columns, breaking sticky-left offsets. */
  table.heatmap { border-collapse: separate; border-spacing: 0; font-size: 11px;
                  table-layout: fixed; width: max-content; }
  table.heatmap col.cg-task { width: 280px; }
  table.heatmap col.cg-h    { width: 44px;  }
  table.heatmap col.cg-cell { width: 56px;  }
  table.heatmap th, table.heatmap td { border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
                                       padding: 0; vertical-align: middle; overflow: hidden; }
  table.heatmap thead th { position: sticky; top: 0; z-index: 3; background: var(--panel); padding: 4px 6px;
                            text-align: center; font-weight: 600; color: var(--text); font-size: 10.5px;
                            white-space: nowrap; }
  table.heatmap thead tr:nth-child(2) th { top: 26px; z-index: 3; color: var(--muted); font-weight: 500; }
  table.heatmap thead th.tg-T1 { background: rgba(89,166,255,0.12); }
  table.heatmap thead th.tg-T2 { background: rgba(89,166,255,0.16); }
  table.heatmap thead th.tg-T3 { background: rgba(89,166,255,0.20); }
  table.heatmap thead th.tg-T4 { background: rgba(89,166,255,0.24); }
  table.heatmap thead th.tg-T5 { background: rgba(89,166,255,0.28); }
  table.heatmap thead th.tg-F  { background: rgba(230,180,85,0.20); }

  table.heatmap th.col-task, table.heatmap td.col-task {
    position: sticky; left: 0; z-index: 5;
    background-color: #0e1116;
    text-align: left; padding: 4px 8px;
    border-right: 2px solid var(--border);
  }
  table.heatmap thead th.col-task { z-index: 7; background-color: #161b22; }
  table.heatmap th.col-h, table.heatmap td.col-h {
    position: sticky; left: 280px; z-index: 5;
    background-color: #0e1116;
    padding: 0 4px;
    font-family: ui-monospace, monospace; font-size: 10px;
    border-right: 2px solid var(--border); color: var(--muted); text-align: center;
  }
  table.heatmap thead th.col-h { z-index: 7; background-color: #161b22; }

  table.heatmap td.col-task { font-size: 11px; line-height: 1.3; vertical-align: middle; }
  table.heatmap td.col-task .tid { font-family: ui-monospace, monospace; font-size: 10px; color: var(--muted); }
  table.heatmap td.col-task .q { color: var(--text); margin-top: 2px;
       display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  table.heatmap td.col-task .grad-mini { font-size: 9.5px; padding: 1px 5px; border-radius: 3px; margin-right: 4px; }
  .gm-yes { background: rgba(62,201,122,0.18); color: var(--pass); }
  .gm-partial { background: rgba(230,180,85,0.18); color: var(--warn); }
  .gm-no { background: rgba(226,92,92,0.18); color: var(--fail); }

  table.heatmap td.cell { width: 40px; height: 22px; cursor: pointer; text-align: center;
                          font-family: ui-monospace, monospace; font-size: 11px;
                          transition: filter 0.1s; }
  table.heatmap td.cell:hover { filter: brightness(1.4); outline: 1px solid var(--accent); outline-offset: -1px; }
  td.cell.empty { background: transparent; color: var(--gray); cursor: default; }
  td.cell.empty:hover { filter: none; outline: 0; }
  td.cell.pass { background: rgba(62,201,122,0.45); color: white; font-weight: 600; }
  td.cell.partial { background: rgba(230,180,85,0.35); color: white; }
  td.cell.fail { background: rgba(226,92,92,0.25); color: rgba(255,255,255,0.85); }
  td.cell.err { background: rgba(230,180,85,0.18); color: var(--warn); }
  td.cell.no_answer { background: rgba(226,92,92,0.12); color: rgba(226,92,92,0.7); }
  tr.task-spacer td { border: 0 !important; background-color: #0e1116; height: 8px; padding: 0; }

  tr.selected td.col-task, tr.selected td.col-h { background: rgba(89,166,255,0.10); }
  tr.selected td.col-task { border-left: 3px solid var(--accent); padding-left: 5px; }

  /* ===== List view (legacy compact) ===== */
  #view-list { padding: 0; }
  .task-row { padding: 8px 14px; border-bottom: 1px solid var(--border); cursor: pointer;
              display: grid; grid-template-columns: 28px 1fr auto; gap: 10px; align-items: center; }
  .task-row:hover { background: var(--panel); }
  .task-row.selected { background: var(--panel2); border-left: 3px solid var(--accent); padding-left: 11px; }
  .diff { font-size: 10px; text-transform: uppercase; font-weight: 600; padding: 1px 6px;
                    border-radius: 3px; letter-spacing: 0.5px; display: inline-block; }
  .diff.easy { background: rgba(62, 201, 122, 0.15); color: var(--pass); }
  .diff.medium { background: rgba(230, 180, 85, 0.15); color: var(--warn); }
  .diff.hard { background: rgba(226, 92, 92, 0.15); color: var(--fail); }
  .task-row .id { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }
  .task-row .q { color: var(--text); margin-top: 2px; }
  .grad-badge { font-size: 10px; padding: 2px 7px; border-radius: 3px; font-weight: 600; white-space: nowrap; }
  .grad-yes { background: rgba(62, 201, 122, 0.2); color: var(--pass); }
  .grad-partial { background: rgba(230, 180, 85, 0.2); color: var(--warn); }
  .grad-no { background: rgba(226, 92, 92, 0.2); color: var(--fail); }
  .harness-strip { display: flex; gap: 4px; margin-top: 5px; }
  .harness-cell { font-size: 10px; padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, monospace;
                  border: 1px solid var(--border); }
  .harness-cell.pass { background: rgba(62, 201, 122, 0.15); color: var(--pass); border-color: rgba(62,201,122,0.4); }
  .harness-cell.fail { background: rgba(226, 92, 92, 0.1); color: var(--fail); border-color: rgba(226,92,92,0.3); }
  .harness-cell.none { background: transparent; color: var(--gray); }

  /* ===== Detail / side body ===== */
  .side-body h2 { font-size: 14px; margin: 0 0 6px; }
  .side-body .question { background: var(--panel2); padding: 10px; border-radius: 5px;
                       border-left: 3px solid var(--accent); margin-bottom: 10px; }
  .side-body .meta { color: var(--muted); font-size: 11px; margin-bottom: 8px; }
  .side-body .meta b { color: var(--text); }
  .side-body .gold { background: rgba(62,201,122,0.08); padding: 6px 10px; border-radius: 4px;
                  font-family: ui-monospace, monospace; font-size: 12px; margin-bottom: 10px;
                  border-left: 3px solid var(--pass); }
  .trial-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 6px; }
  .trial-table th { text-align: left; padding: 5px 6px; color: var(--muted); font-weight: 500;
                    border-bottom: 1px solid var(--border); font-size: 10px; text-transform: uppercase; }
  .trial-table td { padding: 5px 6px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .trial-table tr:hover { background: var(--panel2); }
  .trial-table .tier { font-family: ui-monospace, monospace; font-weight: 600; }
  .trial-table .reward.pass { color: var(--pass); font-weight: 600; }
  .trial-table .reward.fail { color: var(--fail); }
  .trial-table .reward.partial { color: var(--warn); }
  .trial-table .view-btn { background: var(--panel2); color: var(--accent); border: 1px solid var(--border);
                           padding: 2px 8px; border-radius: 3px; cursor: pointer; font-size: 10px; }
  .trial-table .view-btn:hover { background: var(--border); }
  .agent-pill { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: var(--panel2);
                border: 1px solid var(--border); font-family: ui-monospace, monospace; }

  /* ===== Modal ===== */
  .modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 100; display: none;
              align-items: center; justify-content: center; padding: 30px; }
  .modal-bg.open { display: flex; }
  .modal { background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
           width: min(1200px, 100%); height: 100%; max-height: calc(100vh - 60px); display: flex; flex-direction: column; }
  .modal header.modal-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 10px 18px; background: var(--panel); border-bottom: 1px solid var(--border); }
  .modal header.modal-head h3 { margin: 0; font-size: 14px; }
  .modal header.modal-head button { background: var(--panel2); color: var(--text); border: 1px solid var(--border);
                          padding: 4px 12px; border-radius: 4px; cursor: pointer; }
  .modal .body { overflow-y: auto; padding: 12px 18px; flex: 1; }
  .msg { margin-bottom: 8px; border-radius: 5px; overflow: hidden; }
  .msg .msg-head { padding: 4px 10px; font-size: 10px; text-transform: uppercase; font-weight: 600;
                   letter-spacing: 0.5px; }
  .msg .msg-body { padding: 8px 12px; white-space: pre-wrap; word-break: break-word;
                   font-family: ui-monospace, SF Mono, Menlo, monospace; font-size: 11.5px; line-height: 1.5; }
  .msg.system .msg-head { background: #2c3a4f; color: #b6c4e2; }
  .msg.system .msg-body { background: rgba(44, 58, 79, 0.3); }
  .msg.user .msg-head { background: #1f3a4d; color: #9bcce8; }
  .msg.user .msg-body { background: rgba(31, 58, 77, 0.3); }
  .msg.assistant .msg-head { background: #2e4030; color: #a8d5af; }
  .msg.assistant .msg-body { background: rgba(46, 64, 48, 0.25); }
  .msg.tool .msg-head { background: #4a3e23; color: #e6c98a; }
  .msg.tool .msg-body { background: rgba(74, 62, 35, 0.25); }
  .msg .tool-calls { margin-top: 8px; padding: 0 4px; }
  .tool-call { margin-top: 6px; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; background: var(--panel2); }
  .tool-call .tc-head { padding: 5px 10px; background: rgba(89,166,255,0.08); display: flex; gap: 10px;
                        align-items: center; font-size: 11px; flex-wrap: wrap; border-bottom: 1px solid var(--border); }
  .tool-call .tc-name { color: var(--accent); font-weight: 600; font-family: ui-monospace, monospace; }
  .tool-call .tc-id   { color: var(--gray); font-size: 9.5px; font-family: ui-monospace, monospace; }
  .tool-call .tc-meta { color: var(--muted); font-size: 10.5px; font-family: ui-monospace, monospace;
                        margin-left: auto; }
  .tool-call .tc-code { margin: 0; padding: 8px 12px; background: #0a0d12; font-family: ui-monospace, SF Mono, Menlo, monospace;
                        font-size: 11.5px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
                        max-height: 360px; overflow: auto; color: #e8eaef; border: 0; }
  .tool-call .tc-code.lang-python { border-left: 3px solid #4584b6; }
  .tool-call .tc-code.lang-bash   { border-left: 3px solid #4eaa25; }
  .tool-call .tc-code.lang-json   { border-left: 3px solid #cb7d29; }

  .tool-output { margin: 0; padding: 8px 12px; background: rgba(0,0,0,0.35);
                 font-family: ui-monospace, SF Mono, Menlo, monospace; font-size: 11.5px; line-height: 1.5;
                 white-space: pre-wrap; word-break: break-word; max-height: 360px; overflow: auto;
                 color: #c9d0d8; border-top: 1px solid var(--border); }
  .tool-output.empty { color: var(--gray); font-style: italic; }
  .tool-output-head { padding: 4px 10px; background: rgba(74,62,35,0.25); color: #e6c98a;
                      font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
                      display: flex; gap: 8px; }
  .tool-output-head .rc { margin-left: auto; color: var(--muted); }

  .turn { border: 1px solid var(--border); border-radius: 6px; padding: 0; margin-bottom: 10px;
          background: var(--panel); overflow: hidden; }
  .turn-head { padding: 4px 12px; background: var(--panel2); color: var(--muted); font-size: 10px;
               text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;
               border-bottom: 1px solid var(--border); }
  .turn-thinking { padding: 8px 12px; color: #a8d5af; white-space: pre-wrap; word-break: break-word;
                   font-size: 12px; line-height: 1.55; }
  .turn-thinking.empty { display: none; }

  .pill { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px;
          background: var(--panel2); color: var(--muted); border: 1px solid var(--border); margin-right: 4px; }
  .pill.tier { font-family: ui-monospace, monospace; }
  .pill.ok { background: rgba(62,201,122,0.15); color: var(--pass); border-color: transparent; }
  .pill.err { background: rgba(226,92,92,0.15); color: var(--fail); border-color: transparent; }
  .pill.warn { background: rgba(230,180,85,0.15); color: var(--warn); border-color: transparent; }

  .grader-box { background: var(--panel2); padding: 8px 10px; border-radius: 4px; font-family: ui-monospace, monospace;
                font-size: 11px; white-space: pre-wrap; color: var(--muted); margin-bottom: 8px; }
  details { margin: 10px 0; }
  details summary { cursor: pointer; color: var(--muted); padding: 4px 0; }
  details > pre { background: var(--panel2); padding: 8px; border-radius: 4px; overflow-x: auto;
                  font-size: 11px; max-height: 300px; }

  .legend { display: inline-flex; gap: 8px; align-items: center; font-size: 10.5px; color: var(--muted); margin-left: 8px; }
  .legend .swatch { display: inline-block; width: 14px; height: 12px; border-radius: 2px; margin-right: 3px; vertical-align: middle; }
</style>
</head>
<body>

<header>
  <div class="hdr-bar">
    <h1>Jupyter-Agent Sweep v2</h1>
    <span class="sub" id="dt-range"></span>
    <div class="pills" id="hdr-pills"></div>
    <button class="toggle" onclick="toggleHeader()" id="toggle-btn">▼ details</button>
  </div>
  <div class="hdr-expanded hidden" id="hdr-expanded">
    <div class="stats" id="stats"></div>
    <div class="agg-tables" id="agg-tables"></div>
  </div>
</header>

<div class="filters">
  <input type="search" id="search" placeholder="Search task id or question…" />
  <label>Difficulty</label>
  <select id="diff"><option value="">all</option><option>easy</option><option>medium</option></select>
  <label>Status</label>
  <select id="status"><option value="">all</option><option value="grad">graduated</option><option value="partial">partial</option><option value="fail">failed</option></select>
  <label>Pass on harness</label>
  <select id="harness"><option value="">any</option><option value="bash">bash-only</option><option value="jupy">jupyter-tool</option><option value="seta">seta-tool</option></select>
  <div class="view-tabs" style="margin-left:auto">
    <button id="tab-heatmap" class="active" onclick="setView('heatmap')">heatmap</button>
    <button id="tab-list" onclick="setView('list')">list</button>
  </div>
  <span class="legend">
    <span><span class="swatch" style="background:rgba(62,201,122,0.45)"></span>pass</span>
    <span><span class="swatch" style="background:rgba(226,92,92,0.25)"></span>fail</span>
    <span><span class="swatch" style="background:rgba(230,180,85,0.18)"></span>err/no-answer</span>
    <span><span class="swatch" style="background:transparent;border:1px solid var(--border)"></span>not run</span>
  </span>
  <span id="count" style="color:var(--muted);font-size:11px"></span>
</div>

<main>
  <div class="view-pane" id="view-heatmap">
    <div class="heatmap-wrap" id="heatmap-wrap"></div>
  </div>
  <div class="view-pane hidden" id="view-list"></div>
  <aside class="side-pane hidden" id="side-pane">
    <div class="side-resizer" id="side-resizer" title="drag to resize"></div>
    <div class="side-head">
      <button id="side-back" onclick="sideBack()" style="display:none">← back</button>
      <h2 id="side-title"></h2>
      <button onclick="sideClose()">close ✕</button>
    </div>
    <div class="side-body" id="side-body"></div>
  </aside>
</main>

<script>
const DATA = __DATA_PLACEHOLDER__;
let selectedTaskIdx = null;
let currentView = 'heatmap';

function fmtCost(c) { return '$' + (c||0).toFixed(4); }
function fmtT(n) { if(!n) return '0'; if(n>=1e6) return (n/1e6).toFixed(1)+'M'; if(n>=1e3) return (n/1e3).toFixed(1)+'k'; return ''+n; }
function fmtDur(s) { s = s||0; if(s<60) return s.toFixed(0)+'s'; return (s/60).toFixed(1)+'m'; }
function escape(s) { return String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/"/g,'&quot;'); }

function gradeClass(t) { if (t.graduated) return 'grad-yes'; if (t.any_pass) return 'grad-partial'; return 'grad-no'; }
function gradeLabel(t) {
  if (t.graduated) return '✓ graduated ' + (t.graduated_tier||'');
  if (t.any_pass) return '~ partial (1 harness)';
  return '✗ all failed';
}
function gradeMiniClass(t) { if (t.graduated) return 'gm-yes'; if (t.any_pass) return 'gm-partial'; return 'gm-no'; }
function gradeMiniLabel(t) {
  if (t.graduated) return '✓' + (t.graduated_tier||'');
  if (t.any_pass) return '~';
  return '✗';
}

function renderHeaderStats() {
  const s = DATA.summary;
  document.getElementById('dt-range').textContent =
    `${s.first_ts.slice(0,16).replace('T',' ')} → ${s.last_ts.slice(0,16).replace('T',' ')} UTC`;
  document.getElementById('hdr-pills').innerHTML = [
    ['Tasks', s.tasks_total],
    ['Graduated', s.tasks_graduated],
    ['Any pass', s.tasks_any_pass],
    ['Failed', s.tasks_failed],
    ['Runs', s.runs_total],
    ['Cost', '$' + s.cost_total.toFixed(2)],
    ['Tok in/out', fmtT(s.tokens_prompt) + '/' + fmtT(s.tokens_completion)],
  ].map(([l,v]) => `<span class="pill-stat">${l} <b>${v}</b></span>`).join('');

  // expanded body (built once; toggled visible)
  const stats = [
    ['Tasks', s.tasks_total],
    ['Graduated (≥2 harnesses)', s.tasks_graduated],
    ['Any pass (≥1)', s.tasks_any_pass],
    ['Failed (0)', s.tasks_failed],
    ['Runs', s.runs_total],
    ['Passes', s.runs_passed],
    ['Total cost', '$' + s.cost_total.toFixed(2)],
    ['Tokens (in/out)', `${fmtT(s.tokens_prompt)} / ${fmtT(s.tokens_completion)}`],
  ];
  document.getElementById('stats').innerHTML = stats.map(([l,n]) =>
    `<div class="stat"><div class="num">${n}</div><div class="lbl">${l}</div></div>`
  ).join('');

  const byModel = DATA.by_model.slice().sort((a,b) => (b.passes/b.runs) - (a.passes/a.runs));
  const byAgent = DATA.by_agent.slice();
  const modelTable = `
    <div class="agg-table"><h4>By model</h4>
    <table><tr><th>Model</th><th>Runs</th><th>Pass</th><th>Pass%</th><th>Cost</th><th>In/Out tok</th></tr>
    ${byModel.map(m => `<tr>
      <td>${m.model_short}</td><td>${m.runs}</td><td>${m.passes}</td>
      <td class="pct">${(m.passes/m.runs*100).toFixed(0)}%</td>
      <td>${fmtCost(m.cost)}</td>
      <td>${fmtT(m.tokens_in)}/${fmtT(m.tokens_out)}</td>
    </tr>`).join('')}
    </table></div>`;
  const agentTable = `
    <div class="agg-table"><h4>By harness</h4>
    <table><tr><th>Harness</th><th>Runs</th><th>Pass</th><th>Pass%</th><th>Cost</th><th>In/Out tok</th></tr>
    ${byAgent.map(a => `<tr>
      <td>${a.agent_label}</td><td>${a.runs}</td><td>${a.passes}</td>
      <td class="pct">${(a.passes/a.runs*100).toFixed(0)}%</td>
      <td>${fmtCost(a.cost)}</td>
      <td>${fmtT(a.tokens_in)}/${fmtT(a.tokens_out)}</td>
    </tr>`).join('')}
    </table></div>`;
  document.getElementById('agg-tables').innerHTML = modelTable + agentTable;
}

function toggleHeader() {
  const ex = document.getElementById('hdr-expanded');
  ex.classList.toggle('hidden');
  document.getElementById('toggle-btn').textContent =
    ex.classList.contains('hidden') ? '▼ details' : '▲ hide details';
}

function applyFilters(tasks) {
  const q = document.getElementById('search').value.toLowerCase();
  const diff = document.getElementById('diff').value;
  const status = document.getElementById('status').value;
  const harness = document.getElementById('harness').value;
  return tasks.filter(t => {
    if (q && !(t.id.toLowerCase().includes(q) || (t.question||'').toLowerCase().includes(q))) return false;
    if (diff && t.difficulty !== diff) return false;
    if (status === 'grad' && !t.graduated) return false;
    if (status === 'partial' && (t.graduated || !t.any_pass)) return false;
    if (status === 'fail' && t.any_pass) return false;
    if (harness && !t.passing_harnesses.includes(harness)) return false;
    return true;
  });
}

function cellClass(tr) {
  if (!tr) return 'empty';
  if (tr.reward >= 1) return 'pass';
  if (tr.reward > 0) return 'partial';
  if (tr.error_kind === 'no_answer') return 'no_answer';
  if (tr.error_kind && tr.error_kind !== 'ok' && tr.error_kind !== '') return 'err';
  return 'fail';
}
function cellSymbol(tr) {
  if (!tr) return '·';
  if (tr.reward >= 1) return '✓';
  if (tr.reward > 0) return '~';
  if (tr.error_kind === 'no_answer') return '∅';
  if (tr.error_kind && tr.error_kind !== 'ok') return '⚠';
  return '✗';
}

function renderHeatmap() {
  const visible = applyFilters(DATA.tasks);
  document.getElementById('count').textContent = `${visible.length} of ${DATA.tasks.length} tasks`;
  const groups = DATA.tier_groups;
  const agents = DATA.agents_order;

  // colgroup: explicit widths so table-layout:fixed works
  let cgroup = `<col class="cg-task"><col class="cg-h">`;
  for (const g of groups) for (const _ of g.models) cgroup += `<col class="cg-cell">`;

  // build column header (two rows: tier groups, models)
  let h1 = `<tr><th class="col-task" rowspan="2">Task</th><th class="col-h" rowspan="2">H</th>`;
  let h2 = `<tr>`;
  for (const g of groups) {
    h1 += `<th class="tg-${g.tier}" colspan="${g.models.length}">${g.tier} · ${g.label}</th>`;
    for (const m of g.models) {
      h2 += `<th class="tg-${g.tier}">${m}</th>`;
    }
  }
  h1 += `</tr>`;
  h2 += `</tr>`;

  // rows: rowspan merges the col-task cell across harness rows of each task,
  // col-h shows the per-row harness label, all rows are equal height.
  const nCols = 2 + groups.reduce((s,g) => s + g.models.length, 0);
  let body = '';
  visible.forEach((t, ti) => {
    if (ti > 0) body += `<tr class="task-spacer"><td colspan="${nCols}"></td></tr>`;
    const realIdx = DATA.tasks.indexOf(t);
    const selCls = realIdx === selectedTaskIdx ? 'selected' : '';
    agents.forEach((a, ai) => {
      const isFirst = ai === 0;
      let row = `<tr class="task-h${ai} ${selCls}" data-task="${realIdx}">`;
      if (isFirst) {
        row += `<td class="col-task" rowspan="${agents.length}" onclick="selectTask(${realIdx})" style="cursor:pointer">
          <div><span class="diff ${t.difficulty}">${t.difficulty.charAt(0)}</span>
               <span class="grad-mini ${gradeMiniClass(t)}">${gradeMiniLabel(t)}</span>
               <span class="tid">${escape(t.id)}</span></div>
          <div class="q">${escape(t.question)}</div>
        </td>`;
      }
      row += `<td class="col-h">${a}</td>`;
      for (const g of groups) {
        for (const m of g.models) {
          const tridx = (t.cell_map[a] || {})[m];
          const tr = tridx != null ? t.trials[tridx] : null;
          const cls = cellClass(tr);
          const sym = cellSymbol(tr);
          if (tr) {
            const tooltip = `${m} / ${a}\n${tr.error_kind || 'ok'}\nreward=${tr.reward.toFixed(2)}, $${tr.cost.toFixed(4)}`;
            row += `<td class="cell ${cls}" title="${escape(tooltip)}" onclick="openTrace(${realIdx},${tridx})">${sym}</td>`;
          } else {
            row += `<td class="cell empty">·</td>`;
          }
        }
      }
      row += `</tr>`;
      body += row;
    });
  });

  document.getElementById('heatmap-wrap').innerHTML =
    `<table class="heatmap"><colgroup>${cgroup}</colgroup><thead>${h1}${h2}</thead><tbody>${body}</tbody></table>`;
}

function renderList() {
  const visible = applyFilters(DATA.tasks);
  document.getElementById('count').textContent = `${visible.length} of ${DATA.tasks.length} tasks`;
  const harnessOrder = DATA.agents_order;
  const html = visible.map(t => {
    const realIdx = DATA.tasks.indexOf(t);
    const harness = harnessOrder.map(a => {
      const b = t.best_per_agent[a];
      if (!b) return `<span class="harness-cell none">${a} —</span>`;
      const cls = b.reward >= 1 ? 'pass' : 'fail';
      const sym = b.reward >= 1 ? '✓' : '✗';
      return `<span class="harness-cell ${cls}" title="${b.model_short} @ ${b.tier} — ${b.error_kind}">${a} ${sym} ${b.tier}</span>`;
    }).join('');
    return `<div class="task-row ${realIdx===selectedTaskIdx?'selected':''}" onclick="selectTask(${realIdx})">
      <span class="diff ${t.difficulty}">${t.difficulty.charAt(0)}</span>
      <div>
        <div class="id">${escape(t.id)}</div>
        <div class="q">${escape(t.question).slice(0,140)}${(t.question||'').length>140?'…':''}</div>
        <div class="harness-strip">${harness}</div>
      </div>
      <div style="text-align:right">
        <div class="grad-badge ${gradeClass(t)}">${gradeLabel(t)}</div>
        <div style="color:var(--muted);font-size:10px;margin-top:4px">${fmtCost(t.total_cost)} · ${t.total_runs}r</div>
      </div>
    </div>`;
  }).join('');
  document.getElementById('view-list').innerHTML = html || `<div style="padding:20px;color:var(--muted)">No tasks match filters.</div>`;
}

function renderCurrent() {
  if (currentView === 'heatmap') renderHeatmap();
  else renderList();
}

function setView(v) {
  currentView = v;
  document.getElementById('view-heatmap').classList.toggle('hidden', v !== 'heatmap');
  document.getElementById('view-list').classList.toggle('hidden', v !== 'list');
  document.getElementById('tab-heatmap').classList.toggle('active', v === 'heatmap');
  document.getElementById('tab-list').classList.toggle('active', v === 'list');
  renderCurrent();
}

let sideStack = []; // history of side-pane views: 'task' or {trace: [taskIdx, trialIdx]}

function openSide() {
  document.getElementById('side-pane').classList.remove('hidden');
}
function sideClose() {
  document.getElementById('side-pane').classList.add('hidden');
  sideStack = [];
  selectedTaskIdx = null;
  document.getElementById('side-back').style.display = 'none';
  renderCurrent();
}
function sideBack() {
  if (sideStack.length <= 1) { sideClose(); return; }
  sideStack.pop();
  applySideTop();
}
function applySideTop() {
  const top = sideStack[sideStack.length - 1];
  document.getElementById('side-back').style.display = sideStack.length > 1 ? '' : 'none';
  if (top.kind === 'task') renderTaskDetail(DATA.tasks[top.taskIdx]);
  else if (top.kind === 'trace') renderTraceSide(top.taskIdx, top.trialIdx);
}

function selectTask(idx) {
  selectedTaskIdx = idx;
  sideStack = [{kind: 'task', taskIdx: idx}];
  openSide();
  applySideTop();
  renderCurrent();
}

function renderTaskDetail(t) {
  const taskIdx = DATA.tasks.indexOf(t);
  const trialRows = t.trials.map((tr, i) => {
    const rcls = tr.reward >= 1 ? 'pass' : (tr.reward > 0 ? 'partial' : 'fail');
    const sym = tr.reward >= 1 ? '✓' : (tr.reward > 0 ? '~' : '✗');
    const errPill = tr.error_kind && tr.error_kind !== 'ok'
      ? `<span class="pill err">${tr.error_kind}</span>` : '';
    return `<tr class="trial-row" onclick="openTrace(${taskIdx}, ${i})" style="cursor:pointer">
      <td><span class="pill tier">${tr.tier}</span></td>
      <td>${tr.model_short}</td>
      <td><span class="agent-pill">${tr.agent}</span></td>
      <td class="reward ${rcls}">${sym} ${tr.reward.toFixed(2)}</td>
      <td>${fmtCost(tr.cost)}</td>
      <td>${fmtT(tr.prompt_tokens)}/${fmtT(tr.completion_tokens)}</td>
      <td>${fmtDur(tr.elapsed_sec)}</td>
      <td>${errPill}</td>
      <td><button class="view-btn">view trace →</button></td>
    </tr>`;
  }).join('');

  document.getElementById('side-title').textContent = t.id;
  document.getElementById('side-body').innerHTML = `
    <div class="meta">
      <span class="diff ${t.difficulty}">${t.difficulty}</span> ·
      <b>Kaggle:</b> ${escape(t.kaggle || '—')} ·
      <b>Answer type:</b> ${escape(t.answer_type || '?')}
    </div>
    <div class="question">${escape(t.question)}</div>
    <div class="gold"><b>Gold:</b> ${escape(t.answer)}</div>
    <div class="meta">
      <b>Files:</b> ${(t.files||[]).map(escape).join(', ') || '—'} ·
      <b>Packages:</b> ${(t.packages||[]).map(escape).join(', ') || '—'}
    </div>
    <div class="meta">
      <span class="grad-badge ${gradeClass(t)}">${gradeLabel(t)}</span>
      &nbsp; <b>Passing:</b> ${t.passing_harnesses.join(', ') || '—'}
      &nbsp; <b>Cost:</b> ${fmtCost(t.total_cost)} &nbsp; <b>Trials:</b> ${t.total_runs}
    </div>
    <table class="trial-table">
      <thead><tr><th>Tier</th><th>Model</th><th>H</th><th>Reward</th><th>Cost</th><th>Tok i/o</th><th>Dur</th><th>Error</th><th></th></tr></thead>
      <tbody>${trialRows}</tbody>
    </table>
  `;
}

function openTrace(taskIdx, trialIdx) {
  selectedTaskIdx = taskIdx;
  // If no side-pane open, start with the task view so "back" returns to it.
  if (sideStack.length === 0) sideStack.push({kind: 'task', taskIdx});
  sideStack.push({kind: 'trace', taskIdx, trialIdx});
  openSide();
  applySideTop();
  renderCurrent();
}

function renderTraceSide(taskIdx, trialIdx) {
  const t = DATA.tasks[taskIdx];
  const tr = t.trials[trialIdx];
  const passClass = tr.reward >= 1 ? 'ok' : 'err';
  document.getElementById('side-title').innerHTML =
    `${escape(t.id)} · ${tr.tier} · ${tr.model_short} · ${tr.agent_label}
     <span class="pill ${passClass}">${tr.reward >= 1 ? 'PASS' : 'FAIL'} ${tr.reward.toFixed(2)}</span>
     <span class="pill">${fmtCost(tr.cost)} · ${fmtT(tr.prompt_tokens)}/${fmtT(tr.completion_tokens)} tok</span>`;

  const msgs = renderTrajectory(tr.trajectory || []);
  const graderHtml = tr.grader_text
    ? `<h4 style="margin:6px 0;color:var(--muted);font-size:11px;text-transform:uppercase">Grader output</h4>
       <div class="grader-box">${escape(tr.grader_text)}</div>` : '';
  const ocHtml = tr.opencode_log
    ? `<details><summary>opencode.txt (${tr.opencode_log.length} chars)</summary>
       <pre>${escape(tr.opencode_log)}</pre></details>` : '';
  const logHtml = tr.trial_log_tail
    ? `<details><summary>trial.log (last 80 lines)</summary>
       <pre>${escape(tr.trial_log_tail)}</pre></details>` : '';

  document.getElementById('side-body').innerHTML = graderHtml + msgs + ocHtml + logHtml;
}

// ===== Trace rendering: group each turn (assistant + paired tool outputs) =====
function renderTrajectory(traj) {
  // Index tool results by their tool_call_id, and build callMap for cross-ref.
  const resultsById = {};
  for (const m of traj) {
    if (m.role === 'tool' && m.tool_call_id) resultsById[m.tool_call_id] = m;
  }

  // Walk linearly. Render system/user as-is. For assistant: render the turn
  // (thinking text + each tool call paired with its result). Skip standalone
  // tool messages (already attached to their assistant turn).
  let out = '';
  let turnIdx = 0;
  for (let i = 0; i < traj.length; i++) {
    const m = traj[i];
    if (m.role === 'tool') continue;          // emitted with its assistant turn
    if (m.role === 'system' || m.role === 'user') {
      out += renderSimpleMessage(m);
    } else if (m.role === 'assistant') {
      turnIdx += 1;
      out += renderAssistantTurn(m, resultsById, turnIdx);
    } else {
      out += renderSimpleMessage(m);
    }
  }
  return out;
}

function renderSimpleMessage(m) {
  const body = escape(m.content || '(empty)');
  return `<div class="msg ${m.role}">
    <div class="msg-head">${m.role}</div>
    <div class="msg-body">${body}</div>
  </div>`;
}

function renderAssistantTurn(m, resultsById, idx) {
  const thinking = (m.content || '').trim();
  const calls = m.tool_calls || [];

  // No tool calls — just an assistant message
  if (!calls.length) {
    return `<div class="msg assistant">
      <div class="msg-head">assistant · turn ${idx}</div>
      <div class="msg-body">${escape(thinking || '(no content)')}</div>
    </div>`;
  }

  const thinkingHtml = thinking
    ? `<div class="turn-thinking">${escape(thinking)}</div>`
    : '';

  const callsHtml = calls.map(tc => {
    const result = resultsById[tc.id];
    return renderToolCall(tc, result);
  }).join('');

  return `<div class="turn">
    <div class="turn-head">▶ turn ${idx} · assistant</div>
    ${thinkingHtml}
    ${callsHtml}
  </div>`;
}

function renderToolCall(tc, resultMsg) {
  let parsed = null;
  try { parsed = JSON.parse(tc.arguments || '{}'); }
  catch { parsed = null; }

  let codeBody = '';
  let langHint = '';
  let metaParts = [];

  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    // Pull out the "body" field (code / command / answer) for the big block
    if (typeof parsed.code === 'string') { codeBody = parsed.code; langHint = 'python'; }
    else if (typeof parsed.command === 'string') { codeBody = parsed.command; langHint = 'bash'; }
    else if (typeof parsed.cell === 'string') { codeBody = parsed.cell; langHint = 'python'; }
    else if (typeof parsed.answer === 'string') { codeBody = parsed.answer; langHint = ''; }

    // Everything else → meta line
    const bodyKeys = new Set(['code', 'command', 'cell', 'answer']);
    for (const [k, v] of Object.entries(parsed)) {
      if (bodyKeys.has(k)) continue;
      let val;
      if (typeof v === 'string') val = v.length > 120 ? v.slice(0,120) + '…' : v;
      else val = JSON.stringify(v);
      if (val != null && val !== '') metaParts.push(`${k}=${val}`);
    }
    if (!codeBody && Object.keys(parsed).length === 0) {
      // empty args
    } else if (!codeBody) {
      // No recognized body field — show full JSON as code
      codeBody = JSON.stringify(parsed, null, 2);
      langHint = 'json';
    }
  } else {
    codeBody = tc.arguments || '';
    langHint = 'json';
  }

  const meta = metaParts.length ? `<span class="tc-meta">${escape(metaParts.join(' · '))}</span>` : '';
  const idLabel = tc.id ? `<span class="tc-id">${escape(tc.id.slice(0,16))}</span>` : '';

  const codeBlock = codeBody
    ? `<pre class="tc-code lang-${langHint}">${escape(codeBody)}</pre>`
    : `<div style="padding:8px 12px;color:var(--gray);font-style:italic">(no arguments)</div>`;

  let resultHtml = '';
  if (resultMsg) {
    const content = resultMsg.content || '';
    const lines = content.split('\n');
    const lineInfo = `${lines.length} line${lines.length===1?'':'s'} · ${content.length} char${content.length===1?'':'s'}`;
    const truncMatch = content.match(/\[truncated, full length (\d+)\]/);
    const truncated = truncMatch ? `<span style="color:var(--warn)">truncated · full ${truncMatch[1]}ch</span>` : '';
    const empty = content.trim() === '' ? ' empty' : '';
    resultHtml = `
      <div class="tool-output-head">output${idLabel ? '' : ''} <span class="rc">${lineInfo} ${truncated}</span></div>
      <pre class="tool-output${empty}">${empty ? '(empty)' : escape(content)}</pre>
    `;
  } else {
    resultHtml = `<div class="tool-output-head" style="color:var(--gray)">(no matching tool result)</div>`;
  }

  return `<div class="tool-call">
    <div class="tc-head">
      <span class="tc-name">→ ${escape(tc.name || '?')}()</span>
      ${idLabel}
      ${meta}
    </div>
    ${codeBlock}
    ${resultHtml}
  </div>`;
}

['search','diff','status','harness'].forEach(id => document.getElementById(id).addEventListener('input', renderCurrent));
document.addEventListener('keydown', e => { if (e.key==='Escape') sideClose(); });

// ===== Side pane resizer =====
const SIDE_MIN = 380;
const SIDE_MAX_FRAC = 0.85;  // never wider than 85% of viewport
function sideMax() { return Math.max(SIDE_MIN + 100, Math.floor(window.innerWidth * SIDE_MAX_FRAC)); }
function clampSideWidth(w) { return Math.min(sideMax(), Math.max(SIDE_MIN, w)); }
function applySideWidth(w) {
  const sp = document.getElementById('side-pane');
  sp.style.flex = `0 0 ${clampSideWidth(w)}px`;
}
// restore persisted width
try {
  const stored = parseInt(localStorage.getItem('sidePaneWidth') || '', 10);
  if (stored && !isNaN(stored)) applySideWidth(stored);
} catch {}

(function setupResizer() {
  const handle = document.getElementById('side-resizer');
  const sp = document.getElementById('side-pane');
  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener('mousedown', e => {
    dragging = true; startX = e.clientX; startW = sp.getBoundingClientRect().width;
    document.body.classList.add('resizing');
    handle.classList.add('active');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const newW = startW + (startX - e.clientX);  // dragging left grows side pane
    applySideWidth(newW);
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove('resizing');
    handle.classList.remove('active');
    try { localStorage.setItem('sidePaneWidth', String(sp.getBoundingClientRect().width)); } catch {}
  });
  window.addEventListener('resize', () => {
    // re-clamp if viewport shrank below current width
    const cur = sp.getBoundingClientRect().width;
    applySideWidth(cur);
  });
})();

renderHeaderStats();
renderCurrent();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="cache/sweep/v2")
    ap.add_argument("--jobs-dir", default="jobs")
    ap.add_argument("--tasks-dir", default="harbor/tasks/jupyter-agent-eval-v1")
    ap.add_argument("--eval-manifest", default="cache/eval/eval_manifest.parquet")
    ap.add_argument("--out", default="cache/sweep/v2/visualization.html")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    jobs_dir = Path(args.jobs_dir).resolve()
    tasks_dir = Path(args.tasks_dir).resolve()
    manifest = Path(args.eval_manifest).resolve()
    out = Path(args.out).resolve()

    print(f"sweep-dir: {sweep_dir}")
    print(f"jobs-dir: {jobs_dir}")
    print(f"manifest: {manifest}")
    print("Loading sweep state and trial artifacts…")
    data = build_data(sweep_dir, jobs_dir, tasks_dir, manifest)

    print(f"  tasks: {len(data['tasks'])}")
    print(f"  runs: {data['summary']['runs_total']}")
    print(f"  graduated: {data['summary']['tasks_graduated']}")
    print(f"  total cost: ${data['summary']['cost_total']}")

    blob = json.dumps(data, separators=(",", ":"), default=str)
    # Escape sequences that would break out of <script>...</script> if they appear
    # inside trajectory content (agents discussing HTML sometimes contain "</script>").
    blob = blob.replace("</", "<\\/").replace("<!--", "<\\!--").replace("-->", "--\\>")
    print(f"  JSON size: {len(blob)/1024/1024:.1f} MB")

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out} ({len(html)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
