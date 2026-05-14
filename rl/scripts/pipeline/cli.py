"""CLI for the data-agent verification pipeline.

For M1 we only support the `run` subcommand on a small set of tasks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .build import DEFAULT_SUITE
from .orchestrator import RunConfig, process_task
from .state import StateStore

# Load .env from the project root so the doctor (which runs in this process,
# not inside Harbor) can see API keys.
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

    print(f"[pipeline] {len(rows)} task(s) selected")
    print(f"[pipeline] state dir: {state.state_dir}")
    print(f"[pipeline] suite:     {args.suite}  (sandbox={args.sandbox})")
    print(f"[pipeline] model:     {args.model}")
    print()

    for i, (_, row) in enumerate(rows.iterrows(), 1):
        task_id = str(row["id"])
        if args.resume and state.is_terminal(task_id):
            print(f"  [{i}/{len(rows)}] {task_id}  SKIP (already final)")
            continue
        print(f"  [{i}/{len(rows)}] {task_id}")
        decision = process_task(row.to_dict(), cfg)
        v = decision.get("verdict", "?")
        c = decision.get("total_cost_usd", 0.0)
        t = decision.get("total_trials", 0)
        print(f"      → {v}  $ {c:.4f}  trials={t}")

    state.flush()
    print(f"\n[pipeline] done. decisions.csv at {state.state_dir / 'decisions.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.pipeline",
                                description="data-agent verification pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="full pipeline per task (Phase A → B for M1)")
    # selection
    run.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                     help="path to manifest parquet (default: eval_manifest.parquet)")
    run.add_argument("--suite", default=DEFAULT_SUITE)
    run.add_argument("--ids", nargs="+", default=None,
                     help="specific task ids; overrides --from/--to/--limit")
    run.add_argument("--from", dest="frm", type=int, default=None,
                     help="start row index (0-based)")
    run.add_argument("--to", type=int, default=None,
                     help="end row index (exclusive)")
    run.add_argument("--limit", type=int, default=None,
                     help="number of rows from --from (or 0); alias for --to")
    # state
    run.add_argument("--state-dir", default=str(DEFAULT_STATE))
    run.add_argument("--resume", action="store_true",
                     help="skip IDs already terminal in decisions.parquet")
    run.add_argument("--rewrite-spec", action="store_true",
                     help="regenerate the Harbor spec even if it exists")
    # exec
    run.add_argument("--model", default="anthropic/claude-sonnet-4-6")
    run.add_argument("--k-max", type=int, default=3,
                     help="adaptive K upper bound (stops on first pass)")
    run.add_argument("--sandbox", default="docker", choices=["docker", "e2b"])
    # Phase C (doctor)
    run.add_argument("--skip-doctor", action="store_true",
                     help="skip Phase C — Phase B failures become terminal")
    run.add_argument("--max-rewrites", type=int, default=1,
                     help="doctor spec-rewrite budget per task")
    run.add_argument("--doctor-budget", type=float, default=0.50,
                     help="doctor's hard cost cap per task (LLM + probes)")
    run.add_argument("--doctor-max-calls", type=int, default=20,
                     help="doctor's tool-call cap per task")
    run.add_argument("--doctor-model", default="anthropic/claude-sonnet-4-6")
    # Phase D (categorize)
    run.add_argument("--skip-categorize", action="store_true",
                     help="skip Phase D (1-5 difficulty)")
    run.add_argument("--empirical-probe", action="store_true",
                     help="run gpt-4o + seta + K=1 as a cheap difficulty cross-check (D2)")
    run.set_defaults(func=cmd_run)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
