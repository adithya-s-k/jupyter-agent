"""Walk the Harbor task suite and return ORS-shaped task entries.

Reads `HARBOR_SUITE_DIR` (e.g. `rl/harbor/tasks/jupyter-agent-v1`), parses each
`<task_id>/task.toml` via `harbor.models.task.task.Task` (no custom parsing —
mirror what `harbor run` does internally), and emits the JSON each ORS task
spec needs.

Each row in the returned list becomes the `task_spec` passed to
`JupyterAgentEnv.__init__(task_spec, secrets)`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = REPO_ROOT / "rl" / "harbor" / "tasks" / "jupyter-agent-v1"


def _suite_dir() -> Path:
    return Path(os.environ.get("HARBOR_SUITE_DIR", str(DEFAULT_SUITE_DIR)))


def load_tasks() -> list[dict[str, Any]]:
    """Return one task dict per `<task_id>/task.toml` under HARBOR_SUITE_DIR."""
    suite = _suite_dir()
    if not suite.exists():
        raise RuntimeError(f"HARBOR_SUITE_DIR not found: {suite}")

    out: list[dict[str, Any]] = []
    for task_dir in sorted(suite.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.toml").exists():
            continue
        t = Task(task_dir)
        ver_env = dict(t.config.verifier.env or {})
        out.append(
            {
                "id": task_dir.name,
                "task_dir": str(task_dir.resolve()),
                "instruction": (task_dir / "instruction.md").read_text(encoding="utf-8"),
                "gold_answer": ver_env.get("EXPECTED_ANSWER", ""),
                "question": ver_env.get("QUESTION", ""),
            }
        )
    return out


if __name__ == "__main__":  # pragma: no cover
    import json

    for row in load_tasks():
        print(json.dumps({k: (v[:80] + "…" if k == "instruction" else v) for k, v in row.items()}))
