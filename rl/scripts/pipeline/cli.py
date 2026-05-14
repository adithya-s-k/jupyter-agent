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
    rows = _select_rows(df, ids=args.ids, frm=args.frm, to=args.to, limit=args.limit)
    if rows.empty:
        print("no rows selected — check --ids / --from / --to / --limit / --manifest", file=sys.stderr)
        return 2

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
        enable_doctor=not args.skip_doctor,
        max_rewrites=args.max_rewrites,
        doctor_budget_usd=args.doctor_budget,
        doctor_max_calls=args.doctor_max_calls,
        doctor_model=args.doctor_model,
        enable_categorize=not args.skip_categorize,
        run_empirical_probe=args.empirical_probe,
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
        with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
            futures = {ex.submit(_runner, idx, rd): (idx, rd) for idx, rd in pairs}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    idx, rd = futures[fut]
                    print(f"  [{idx}] {rd['id']}  pipeline_error: {e}")

    state.flush()
    state.record_run_end(n_attempted=n_attempted, n_verified=n_verified,
                         total_cost=cumulative_cost)

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
    run.add_argument("--k-max", type=int, default=1,
                     help="adaptive K upper bound (default 1 — fail-fast to doctor)")
    run.add_argument("--sandbox", default="docker", choices=["docker", "e2b"])
    run.add_argument("--concurrent", type=int, default=1,
                     help="parallel task workers (default 1 = sequential)")
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
    run.set_defaults(func=cmd_run)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
