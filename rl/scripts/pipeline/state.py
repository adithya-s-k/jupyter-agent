"""State persistence with per-run isolation.

Layout:
    <state_dir>/
    ├── runs.csv                       INDEX of all runs (append-only)
    ├── decisions.csv                  DERIVED rollup across all runs
    │                                  (rebuilt at end of every run)
    └── runs/
        └── <run_id>/                  EVERYTHING for one run; never touched again
            ├── cli_args.json
            ├── summary.json
            ├── decisions.csv          per-task verdicts from this run only
            ├── state.jsonl            this run's events (append-only)
            ├── cost.jsonl             this run's per-LLM-call cost
            ├── trials/                Harbor trial dirs
            ├── logs/                  stdout/stderr per Harbor invocation
            └── specs/<id>/v0.toml    snapshots taken before doctor edits

After a run finishes, NOTHING inside runs/<run_id>/ is ever modified.
Top-level `decisions.csv` is a materialized view rebuilt at the end of each
run by unioning all runs/*/decisions.csv (latest verdict per task_id wins).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TERMINAL_VERDICTS: frozenset[str] = frozenset({
    "verified",
    "verified_after_rewrite",
    "verified_gold_corrected",
    "verifiable_judge",
    "dropped",
    "spec_build_error",
    "phase_b_failed",
})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class StateStore:
    state_dir: Path
    run_id: str = field(default_factory=make_run_id)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # In-process per-run decisions (only THIS run's verdicts).
    _decisions: dict[str, dict] = field(default_factory=dict)
    # Cross-run "already terminal" set rebuilt at startup from prior runs.
    _terminal_from_prior: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "runs").mkdir(exist_ok=True)
        # This run's home
        self.run_dir = self.state_dir / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "trials").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "specs").mkdir(exist_ok=True)
        # Cross-run terminal set from all prior runs (for --resume)
        self._terminal_from_prior = self._load_cross_run_terminal()

    # ----- paths -----

    @property
    def trials_dir(self) -> Path:
        return self.run_dir / "trials"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def specs_dir(self) -> Path:
        return self.run_dir / "specs"

    # ----- state.jsonl (this-run only, append-only) -----

    def append_event(self, **fields) -> None:
        rec = {"ts": now(), "run_id": self.run_id, **fields}
        line = json.dumps(rec, default=str)
        with self._lock:
            with (self.run_dir / "state.jsonl").open("a") as f:
                f.write(line + "\n")

    # ----- decisions (this-run, upsert + flush) -----

    def upsert_decision(self, task_id: str, **fields) -> None:
        with self._lock:
            row = self._decisions.get(task_id, {"task_id": task_id})
            row.update(fields)
            row["run_id"] = self.run_id
            row["ts_updated"] = now()
            self._decisions[task_id] = row

    def get_decision(self, task_id: str) -> dict | None:
        return self._decisions.get(task_id)

    def is_terminal(self, task_id: str) -> bool:
        """True iff this task already has a terminal verdict in any run."""
        if task_id in self._terminal_from_prior:
            return True
        d = self._decisions.get(task_id)
        return bool(d and d.get("verdict") in TERMINAL_VERDICTS)

    def _flush_run_decisions(self) -> None:
        if not self._decisions:
            return
        df = pd.DataFrame(list(self._decisions.values()))
        df = df.sort_values("task_id").reset_index(drop=True)
        df.to_csv(self.run_dir / "decisions.csv", index=False)

    def flush(self) -> None:
        """Write this run's decisions.csv + rebuild the cross-run rollup."""
        with self._lock:
            self._flush_run_decisions()
            self._rebuild_rollup()

    # ----- cross-run rollup at top level -----

    def _load_cross_run_terminal(self) -> set[str]:
        """At startup: read every prior runs/<id>/decisions.csv and collect
        task_ids whose latest verdict is terminal."""
        latest: dict[str, dict] = {}
        runs_root = self.state_dir / "runs"
        if not runs_root.exists():
            return set()
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir() or run_dir.name == self.run_id:
                continue
            csv = run_dir / "decisions.csv"
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            for r in df.to_dict("records"):
                tid = r.get("task_id")
                if not tid:
                    continue
                # latest by ts_updated
                prior = latest.get(tid)
                if prior is None or str(r.get("ts_updated", "")) > str(prior.get("ts_updated", "")):
                    latest[tid] = r
        return {tid for tid, r in latest.items() if r.get("verdict") in TERMINAL_VERDICTS}

    def _rebuild_rollup(self) -> None:
        """Union all runs/*/decisions.csv → top-level decisions.csv.

        Latest verdict per task_id (by ts_updated) wins. This is a derived
        materialized view — safe to delete and rebuild any time.
        """
        latest: dict[str, dict] = {}
        runs_root = self.state_dir / "runs"
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            csv = run_dir / "decisions.csv"
            if not csv.exists():
                continue
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            for r in df.to_dict("records"):
                tid = r.get("task_id")
                if not tid:
                    continue
                prior = latest.get(tid)
                if prior is None or str(r.get("ts_updated", "")) > str(prior.get("ts_updated", "")):
                    latest[tid] = r
        if not latest:
            return
        out = pd.DataFrame(list(latest.values()))
        out = out.sort_values("task_id").reset_index(drop=True)
        out.to_csv(self.state_dir / "decisions.csv", index=False)

    # ----- run-index bookkeeping -----

    def record_run_start(self, *, cli_args: dict | None = None) -> None:
        # cli_args snapshot lives INSIDE the run dir (immutable record)
        if cli_args is not None:
            (self.run_dir / "cli_args.json").write_text(
                json.dumps(cli_args, indent=2, default=str)
            )
        idx_path = self.state_dir / "runs.csv"
        row = {
            "run_id": self.run_id,
            "started_at": now(),
            "ended_at": "",
            "n_tasks_attempted": 0,
            "n_verified": 0,
            "total_cost_usd": 0.0,
        }
        if idx_path.exists():
            df = pd.read_csv(idx_path, dtype={"ended_at": "string"})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(idx_path, index=False)

    def record_run_end(self, *, n_attempted: int, n_verified: int,
                       total_cost: float) -> None:
        # summary.json INSIDE the run dir (immutable)
        (self.run_dir / "summary.json").write_text(json.dumps({
            "run_id": self.run_id,
            "ended_at": now(),
            "n_tasks_attempted": int(n_attempted),
            "n_verified": int(n_verified),
            "total_cost_usd": round(float(total_cost), 6),
        }, indent=2))

        idx_path = self.state_dir / "runs.csv"
        if not idx_path.exists():
            return
        df = pd.read_csv(idx_path, dtype={"ended_at": "string"})
        mask = df["run_id"].astype(str) == self.run_id
        if not mask.any():
            return
        df.loc[mask, "ended_at"] = now()
        df.loc[mask, "n_tasks_attempted"] = int(n_attempted)
        df.loc[mask, "n_verified"] = int(n_verified)
        df.loc[mask, "total_cost_usd"] = round(float(total_cost), 6)
        df.to_csv(idx_path, index=False)

    # ----- cost.jsonl (this-run, append-only) -----

    def cost_log_path(self) -> Path:
        return self.run_dir / "cost.jsonl"

    def append_cost_event(self, **fields) -> None:
        rec = {"ts": now(), "run_id": self.run_id, **fields}
        with self._lock:
            with self.cost_log_path().open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
