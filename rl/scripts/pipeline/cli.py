"""CLI for the data-agent verification pipeline."""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .build import DEFAULT_SUITE
from .orchestrator import RunConfig, process_task
from .state import StateStore


_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

RL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = RL_ROOT / "data" / "splits" / "eval_manifest.parquet"
DEFAULT_STATE = RL_ROOT / "data" / "verification" / "eval"


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return pd.read_parquet(path)


def _select_rows(df: pd.DataFrame, *, ids, frm, to, limit) -> pd.DataFrame:
    if ids:
        return df[df["id"].isin(ids)].copy()
    n = len(df)
    if frm is None and limit is not None:
        return df.iloc[: limit].copy()
    if frm is None:
        frm = 0
    if to is None and limit is not None:
        to = frm + limit
    if to is None:
        to = n
    return df.iloc[frm: to].copy()


def cmd_run(args) -> int:
    df = _load_manifest(Path(args.manifest))
    ids = list(args.ids) if args.ids else []
    if args.ids_from:
        text = Path(args.ids_from).read_text().strip()
        if text.startswith("["):
            import json
            ids.extend(json.loads(text))
        else:
            ids.extend([l.strip() for l in text.splitlines() if l.strip()])
    rows = _select_rows(df, ids=ids or None, frm=args.frm, to=args.to, limit=args.limit)
    if rows.empty:
        print("no rows selected — check --ids / --from / --to / --limit / --manifest", file=sys.stderr)
        return 2

    # --all-oss preset: swap every LLM call site that was still on the
    # closed-source default. Explicit user-supplied values are respected.
    OSS_DEFAULT = "hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale"
    OSS_PROBES = ["qwen", "glm", "kimi", "deepseek"]
    if args.all_oss:
        if args.model == "anthropic/claude-sonnet-4-6":
            args.model = OSS_DEFAULT
        if args.doctor_model == "anthropic/claude-sonnet-4-6":
            args.doctor_model = OSS_DEFAULT
        if args.categorize_model == "anthropic/claude-sonnet-4-6":
            args.categorize_model = OSS_DEFAULT
        if args.regrade_judge_model == "openai/gpt-5.4-nano":
            args.regrade_judge_model = OSS_DEFAULT
        if not args.probe_aliases:
            args.probe_aliases = OSS_PROBES

    state = StateStore(state_dir=Path(args.state_dir))
    cli_args_clean = {k: v for k, v in vars(args).items() if not callable(v)}
    state.record_run_start(cli_args=cli_args_clean)

    cfg = RunConfig(
        state_store=state,
        suite_name=args.suite,
        model=args.model,
        k_max=args.k_max,
        sandbox=args.sandbox,
        rewrite_spec=args.rewrite_spec,
        subprocess_timeout_sec=args.subprocess_timeout_sec,
        max_retries_per_trial=args.max_retries_per_trial,
        task_timeout_sec=args.task_timeout_sec,
        enable_doctor=not args.skip_doctor,
        max_rewrites=args.max_rewrites,
        doctor_budget_usd=args.doctor_budget,
        doctor_max_calls=args.doctor_max_calls,
        doctor_model=args.doctor_model,
        enable_categorize=not args.skip_categorize,
        categorize_model=args.categorize_model,
        run_empirical_probe=args.empirical_probe,
        probe_aliases=args.probe_aliases,
        regrade_judge_model=args.regrade_judge_model,
    )

    # Pre-filter for resume
    if args.resume:
        before = len(rows)
        rows = rows[~rows["id"].apply(state.is_terminal)].reset_index(drop=True)
        print(f"[pipeline] resume: skipped {before - len(rows)} already-terminal id(s)")

    print(f"[pipeline] run_id:    {state.run_id}")
    print(f"[pipeline] tasks:     {len(rows)} selected")
    print(f"[pipeline] state dir: {state.state_dir}")
    print(f"[pipeline] suite:     {args.suite}  (sandbox={args.sandbox})")
    print(f"[pipeline] model:     {args.model}  (k_max={args.k_max})")
    print(f"[pipeline] doctor:    {'on' if not args.skip_doctor else 'off'}    "
          f"categorize: {'on' if not args.skip_categorize else 'off'}    "
          f"concurrent: {args.concurrent}")
    if args.total_cost_cap is not None:
        print(f"[pipeline] cost cap:  ${args.total_cost_cap:.2f}")
    print()

    # --- Run (sequential or parallel)
    cumulative_cost = 0.0
    cost_lock = threading.Lock()
    cap_hit = threading.Event()
    n_verified = 0
    n_attempted = 0

    def _runner(idx: int, row_dict: dict) -> dict:
        nonlocal cumulative_cost, n_verified, n_attempted
        if cap_hit.is_set():
            return {"task_id": row_dict.get("id"), "verdict": "skipped_cost_cap"}
        decision = process_task(row_dict, cfg)
        c = float(decision.get("total_cost_usd", 0.0) or 0.0)
        v = str(decision.get("verdict", ""))
        with cost_lock:
            cumulative_cost += c
            n_attempted += 1
            if v.startswith("verified"):
                n_verified += 1
            print(f"  [{idx}/{len(rows)}] {row_dict['id']}")
            print(f"      → {v}  $ {c:.4f}  trials={decision.get('total_trials', 0)}  "
                  f"diff={decision.get('difficulty_level', 0)}  "
                  f"cum=${cumulative_cost:.2f}")
            if args.total_cost_cap is not None and cumulative_cost >= args.total_cost_cap:
                cap_hit.set()
                print(f"  [pipeline] !! total-cost-cap reached "
                      f"(${cumulative_cost:.2f} ≥ ${args.total_cost_cap:.2f}); "
                      f"halting new task starts. State is preserved.")
        return decision

    pairs = [(i + 1, r.to_dict()) for i, (_, r) in enumerate(rows.iterrows())]

    if args.concurrent <= 1:
        for idx, rd in pairs:
            if cap_hit.is_set():
                break
            _runner(idx, rd)
    else:
        # Stagger the initial submissions to avoid pounding E2B's
        # create-sandbox endpoint when --concurrent is large. After the
        # first wave, ThreadPoolExecutor naturally smooths things out
        # because workers finish at different times.
        import time as _t
        stagger_sec = float(args.stagger_sec)
        with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
            futures = {}
            for i, (idx, rd) in enumerate(pairs):
                # Stagger only the first `concurrent` submissions.
                if i > 0 and i < args.concurrent and stagger_sec > 0:
                    _t.sleep(stagger_sec)
                futures[ex.submit(_runner, idx, rd)] = (idx, rd)
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    idx, rd = futures[fut]
                    print(f"  [{idx}] {rd['id']}  pipeline_error: {e}")

    state.flush(force_rollup=True)   # rebuild cross-run rollup at end of run
    state.record_run_end(n_attempted=n_attempted, n_verified=n_verified,
                         total_cost=cumulative_cost)

    # Each task's spec is already in its bucket (orchestrator moves on verdict).
    # Print a small summary of bucket counts so the run output reflects it.
    try:
        from .buckets import BUCKETS
        from .build import TASKS_DIR
        bucket_counts = {}
        for b in BUCKETS:
            p = TASKS_DIR / args.suite / b
            bucket_counts[b] = sum(1 for x in p.iterdir() if x.is_dir()) if p.exists() else 0
        bc = "  ".join(f"{k}={v}" for k, v in bucket_counts.items())
        print(f"[pipeline] spec buckets: {bc}")
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARN: failed to count buckets: {e}")

    print()
    print(f"[pipeline] run complete. run_id={state.run_id}")
    print(f"[pipeline]   attempted: {n_attempted}    verified: {n_verified}    "
          f"cost: ${cumulative_cost:.4f}")
    print(f"[pipeline]   run dir:        {state.run_dir}")
    print(f"[pipeline]   ↳ decisions:    {state.run_dir / 'decisions.csv'}")
    print(f"[pipeline]   ↳ state:        {state.run_dir / 'state.jsonl'}")
    print(f"[pipeline]   ↳ cost:         {state.run_dir / 'cost.jsonl'}")
    print(f"[pipeline]   ↳ trials:       {state.trials_dir}")
    print(f"[pipeline]   cross-run rollup: {state.state_dir / 'decisions.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.pipeline",
                                description="data-agent verification pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="full pipeline per task (Phase A → D)")
    # selection
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    run.add_argument("--suite", default=DEFAULT_SUITE)
    run.add_argument("--ids", nargs="+", default=None,
                     help="specific task ids; overrides --from/--to/--limit")
    run.add_argument("--ids-from", type=Path, default=None,
                     help="read ids from a file (one per line, or a JSON array)")
    run.add_argument("--from", dest="frm", type=int, default=None,
                     help="start row index (0-based)")
    run.add_argument("--to", type=int, default=None)
    run.add_argument("--limit", type=int, default=None)
    # state
    run.add_argument("--state-dir", default=str(DEFAULT_STATE))
    run.add_argument("--resume", action="store_true",
                     help="skip IDs already terminal in decisions.csv")
    run.add_argument("--rewrite-spec", action="store_true",
                     help="regenerate the Harbor spec even if it exists")
    # execution
    run.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    run.add_argument("--k-max", type=int, default=2,
                     help="adaptive K upper bound — one Sonnet retry before the doctor "
                          "fires (default 2). Catches flaky single failures cheaply.")
    run.add_argument("--sandbox", default="docker", choices=["docker", "e2b"])
    run.add_argument("--concurrent", type=int, default=1,
                     help="parallel task workers (default 1 = sequential)")
    run.add_argument("--subprocess-timeout-sec", type=int, default=900,
                     help="outer wall-time cap (sec) on each `harbor run` "
                          "subprocess. Default 900s (15min).")
    run.add_argument("--task-timeout-sec", type=int, default=1500,
                     help="orchestrator-level soft cap per task in seconds. "
                          "Includes ALL phases (trials + doctor + categorize). "
                          "On expiry, task is marked dropped with reason task_timeout. "
                          "Default 1500s (25min).")
    run.add_argument("--max-retries-per-trial", type=int, default=1,
                     help="retry transient errors (SandboxException, TimeoutException, "
                          "5xx) per trial with exponential backoff. Default 1.")
    run.add_argument("--stagger-sec", type=float, default=0.5,
                     help="sleep N seconds between submitting initial workers "
                          "(to smooth E2B sandbox-creation pressure). Default 0.5.")
    run.add_argument("--total-cost-cap", type=float, default=None,
                     help="USD cumulative LLM-spend cap across the run; "
                          "halts new task starts (does not interrupt running tasks)")
    # Phase C (doctor)
    run.add_argument("--skip-doctor", action="store_true",
                     help="skip Phase C — Phase B failures become terminal")
    run.add_argument("--max-rewrites", type=int, default=1)
    run.add_argument("--doctor-budget", type=float, default=0.50)
    run.add_argument("--doctor-max-calls", type=int, default=20)
    run.add_argument("--doctor-model", default="anthropic/claude-sonnet-4-6")
    # Phase D (categorize)
    run.add_argument("--skip-categorize", action="store_true")
    run.add_argument("--empirical-probe", action="store_true",
                     help="run gpt-4o + seta + K=1 as a cheap difficulty cross-check (D2)")
    run.add_argument("--categorize-model", default="anthropic/claude-sonnet-4-6",
                     help="model used by Phase D rubric judge")
    # Doctor probe set + regrade judge
    run.add_argument("--probe-aliases", nargs="+", default=None,
                     help="restrict doctor probes to these aliases (e.g. qwen glm kimi)")
    run.add_argument("--regrade-judge-model", default="openai/gpt-5.4-nano",
                     help="LLM used by Phase B regrade's llm-judge fallback")
    # All-OSS preset
    run.add_argument("--all-oss", action="store_true",
                     help="shortcut: swap all LLM call sites to open-source models. "
                          "Anchor=Qwen3-235B, doctor=Qwen3-235B, probes={qwen,glm,kimi,deepseek}, "
                          "categorize=Qwen3-235B, regrade-judge=Qwen3-235B. Overrides "
                          "any defaults; explicit --model/--doctor-model still win.")
    run.set_defaults(func=cmd_run)

    # ----- ping subcommand -----
    pg = sub.add_parser(
        "ping",
        help="quick connectivity + tool-calling check for one or more models",
    )
    pg.add_argument("--models", nargs="+", default=None,
                    help="model IDs to test (default: a curated open-source set)")
    pg.add_argument("--with-tools", action="store_true", default=True,
                    help="also test a trivial tool call (default true)")
    pg.add_argument("--no-tools", dest="with_tools", action="store_false")
    pg.set_defaults(func=cmd_ping)

    # ----- migrate-buckets subcommand -----
    mb = sub.add_parser(
        "migrate-buckets",
        help="convert flat suite → pending/verified/dropped/phase_b_failed subfolders",
    )
    mb.add_argument("--state-dir", default=str(DEFAULT_STATE))
    mb.add_argument("--suite", default=DEFAULT_SUITE)
    mb.set_defaults(func=cmd_migrate_buckets)

    return p


DEFAULT_PING_MODELS = [
    # Closed-source (for sanity)
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.4-nano",
    # Open-source candidates
    "hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale",
    "hf/Qwen/Qwen3-8B:nscale",
    "hf/moonshotai/Kimi-K2-Instruct-0905:novita",
    "hf/zai-org/GLM-4.6:novita",
    "hf/deepseek-ai/DeepSeek-V3.1:novita",
]


def cmd_ping(args) -> int:
    """Minimal connectivity test: one chat completion (+ optional tool call) per model."""
    import time as _t
    from . import llm_client

    models = args.models or DEFAULT_PING_MODELS
    tools = None
    if args.with_tools:
        tools = [{"type": "function", "function": {
            "name": "echo",
            "description": "Return the input verbatim.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string"},
            }, "required": ["text"]},
        }}]

    print(f"[ping] testing {len(models)} model(s){'  (with tool call)' if tools else ''}\n")
    print(f"  {'model':60s}  {'status':12s} {'latency':>10s}  {'cost':>10s}  notes")
    print(f"  {'-'*60}  {'-'*12} {'-'*10}  {'-'*10}  {'-'*40}")

    rc = 0
    for m in models:
        t0 = _t.time()
        try:
            resp = llm_client.call(
                model=m,
                messages=[
                    {"role": "system", "content": "You are a connectivity-test bot. Be terse."},
                    {"role": "user", "content": (
                        "If you have an `echo` tool, call it with text='ok'. "
                        "Otherwise just reply 'ok'."
                    )},
                ],
                tools=tools,
                temperature=0.0,
                max_tokens=64,
            )
            elapsed = _t.time() - t0
            content = (resp.content or "").strip()
            tool_call = resp.tool_calls[0] if resp.tool_calls else None
            status = "OK"
            notes = []
            if tool_call:
                notes.append(f"tool_call={tool_call['name']}")
            elif content:
                notes.append(f"text={content[:30]!r}")
            else:
                notes.append("empty response")
                status = "WARN"
            notes.append(f"in={resp.prompt_tokens} out={resp.completion_tokens}")
            print(f"  {m:60s}  {status:12s} {elapsed:>9.1f}s  ${resp.cost_usd:>8.5f}  {'  '.join(notes)}")
        except Exception as e:  # noqa: BLE001
            elapsed = _t.time() - t0
            msg = str(e).replace("\n", " ")
            if len(msg) > 80:
                msg = msg[:77] + "..."
            print(f"  {m:60s}  {'FAIL':12s} {elapsed:>9.1f}s  {'-':>10s}  {msg}")
            rc = 1
    return rc


def cmd_migrate_buckets(args) -> int:
    from .buckets import migrate_flat_to_buckets
    from .build import TASKS_DIR
    suite_dir = TASKS_DIR / args.suite
    rollup = Path(args.state_dir) / "decisions.csv"
    counts = migrate_flat_to_buckets(suite_dir, rollup)
    if not counts:
        print(f"[migrate-buckets] nothing to migrate at {suite_dir}")
        return 0
    print(f"[migrate-buckets] suite: {args.suite}")
    for k, v in sorted(counts.items()):
        print(f"  {k:20s} {v:>4d}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
