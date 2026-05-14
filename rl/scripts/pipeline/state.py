"""State persistence — append-only state.jsonl + in-process decisions dict.

state.jsonl is the source of truth. Every event during the pipeline (start,
finish, error, doctor turn, …) appends one line. Resumability and audit
both fall out of this.

decisions.parquet is a derived view — one row per task_id with the final
verdict. Maintained as an in-process dict; flushed on every N tasks (default
25) and on shutdown. Cheap at our scale.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StateStore:
    state_dir: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _decisions: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "trials").mkdir(exist_ok=True)
        (self.state_dir / "specs").mkdir(exist_ok=True)
        (self.state_dir / "logs").mkdir(exist_ok=True)
        # rehydrate decisions if a prior CSV exists (for --resume)
        dp = self.state_dir / "decisions.csv"
        if dp.exists():
            df = pd.read_csv(dp)
            for r in df.to_dict("records"):
                self._decisions[r["task_id"]] = r

    # ----- state.jsonl -----

    def append_event(self, **fields) -> None:
        rec = {"ts": now(), **fields}
        # ensure JSON-serializable
        line = json.dumps(rec, default=str)
        with self._lock:
            with (self.state_dir / "state.jsonl").open("a") as f:
                f.write(line + "\n")

    # ----- decisions (in-memory upsert + periodic flush) -----

    def upsert_decision(self, task_id: str, **fields) -> None:
        with self._lock:
            row = self._decisions.get(task_id, {"task_id": task_id})
            row.update(fields)
            row["ts_updated"] = now()
            self._decisions[task_id] = row

    def get_decision(self, task_id: str) -> dict | None:
        return self._decisions.get(task_id)

    def is_terminal(self, task_id: str) -> bool:
        """True iff the task has a final verdict and we should skip on --resume."""
        d = self._decisions.get(task_id)
        if not d:
            return False
        return d.get("verdict") in {
            "verified", "verified_gold_corrected", "verified_after_rewrite",
            "verifiable_judge", "dropped", "spec_build_error",
        }

    def flush(self) -> None:
        with self._lock:
            if not self._decisions:
                return
            df = pd.DataFrame(list(self._decisions.values()))
            df = df.sort_values("task_id").reset_index(drop=True)
            df.to_csv(self.state_dir / "decisions.csv", index=False)
