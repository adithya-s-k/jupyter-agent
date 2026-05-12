"""OpenReward (ORS) server for the Jupyter agent — delegates sandbox to Harbor.

Exposes the same 5 jupyter tools as `harbor_agents/jupyter.py` over the
OpenReward HTTP protocol. The container/sandbox lifecycle is handled by
`harbor.environments.factory.EnvironmentFactory` — we don't re-implement
Docker, E2B, healthcheck, bucket-pull, or any of that. Mirrors Harbor's own
`Trial.__init__` pattern.

Run:
    HARBOR_SUITE_DIR=harbor/tasks/jupyter-agent-v1 \\
    HARBOR_ENV_TYPE=docker \\
    uv run python -m ors.server

The server listens on http://0.0.0.0:8080. Clients (rollouts/, future TRL
trainer) connect via `openreward.EnvironmentsAPI`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)
from pydantic import BaseModel

# Harbor — sandbox / env factory
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.task import Task
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

# Project-local
from ors.list_tasks_helper import load_tasks
from ors.verdict import JUDGE_PROMPT, JudgeOut, JudgeVerdict

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
KERNEL_SERVER = RL_ROOT / "harbor_agents" / "jupyter" / "kernel_server.py"
RUN_CELL = RL_ROOT / "harbor_agents" / "jupyter" / "run_cell.py"

load_dotenv(REPO_ROOT / ".env")

DEFAULT_GRADER_MODEL = os.environ.get("ORS_GRADER_MODEL", "gpt-4o-mini")
DEFAULT_HARBOR_ENV_TYPE = os.environ.get("HARBOR_ENV_TYPE", "docker")
KERNEL_PORT = 8765


logger = logging.getLogger("ors.server")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------


class CodeCellParams(BaseModel):
    code: str


class CommandParams(BaseModel):
    command: str


class NotebookStateParams(BaseModel):
    include_images: bool = False


class FinalAnswerParams(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# In-agent notebook tracker (host-side state for get_notebook_state)
# ---------------------------------------------------------------------------


def _format_cell(kind: str, code: str, output: str, ok: bool) -> str:
    head = code.replace("\n", " ↵ ")[:140]
    out = output.strip()[:300]
    marker = "→" if ok else "✗"
    return f"[{kind}] {head}\n  {marker} {out}" if out else f"[{kind}] {head}"


# ---------------------------------------------------------------------------
# Grader — 3 tiers, same logic as rl/grader.py but inline + structured judge
# ---------------------------------------------------------------------------


def _exact(gold: str, pred: str) -> bool:
    import re

    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return bool(gold) and norm(gold) == norm(pred)


def _numeric(gold: str, pred: str, rel: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    import re

    rx = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")

    def to_f(s):
        m = rx.search((s or "").replace(",", ""))
        try:
            return float(m.group(0)) if m else None
        except ValueError:
            return None

    g, p = to_f(gold), to_f(pred)
    if g is None or p is None:
        return False
    diff = abs(g - p)
    return diff <= abs_tol or diff / max(abs(g), 1e-9) <= rel


async def _llm_judge(client: AsyncOpenAI, question: str, gold: str, pred: str) -> tuple[float, JudgeVerdict, str]:
    """Tier-3 fallback. Returns (reward, verdict, reasoning)."""
    resp = await client.beta.chat.completions.parse(
        model=DEFAULT_GRADER_MODEL,
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}
        ],
        response_format=JudgeOut,
        temperature=0,
    )
    parsed = resp.choices[0].message.parsed
    verdict = parsed.verdict if parsed else JudgeVerdict.NOT_ATTEMPTED
    reasoning = parsed.reasoning if parsed else ""
    reward = 1.0 if verdict == JudgeVerdict.CORRECT else 0.0
    return reward, verdict, reasoning


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------


class JupyterAgentEnv(Environment):
    """Jupyter data-science agent — Harbor-backed sandbox + 5 tools."""

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] | None = None):
        super().__init__(task_spec)
        secrets = secrets or {}

        self._task_dir = Path(task_spec["task_dir"])
        self._task = Task(self._task_dir)
        self._session_id = uuid.uuid4().hex[:12]

        # Per-session scratch — Harbor binds subdirs of this into the container.
        trial_dir = Path(tempfile.mkdtemp(prefix=f"ors-{self._session_id}-"))
        self._trial_paths = TrialPaths(trial_dir=trial_dir)
        self._trial_paths.mkdir()

        # Trial-level env config (defaults; sandbox provider via env var).
        trial_env_config = TrialEnvironmentConfig(type=DEFAULT_HARBOR_ENV_TYPE)

        # Mirror harbor.trial.trial.Trial.__init__
        self._henv = EnvironmentFactory.create_environment_from_config(
            config=trial_env_config,
            environment_dir=self._task_dir / "environment",
            environment_name=f"ors-{task_spec['id']}-{self._session_id[:8]}",
            session_id=self._session_id,
            trial_paths=self._trial_paths,
            task_env_config=self._task.config.environment,
            logger=logger.getChild(self._session_id),
        )

        # Grader client (LLM judge tier).
        self._grader_client: AsyncOpenAI | None = None
        api_key = secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self._grader_client = AsyncOpenAI(api_key=api_key)

        self._cells: list[dict] = []
        self._kernel_ready = False

    # ── ORS metadata ──────────────────────────────────────────────────────

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [Split(name="train", type="train"), Split(name="eval", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        # Both splits resolve to the same task list for now (slug controls it).
        return load_tasks()

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["instruction"])]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def setup(self):
        logger.info(f"[{self._session_id}] start sandbox ({DEFAULT_HARBOR_ENV_TYPE})")
        await self._henv.start(force_build=False)

        # Bucket pull happens here, via task.toml's [environment.healthcheck]
        logger.info(f"[{self._session_id}] run healthcheck (pulls bucket)")
        await self._henv.run_healthcheck()

        # chmod so non-root container users can write to bind-mounted dirs.
        # (Harbor 0.6.6's TrialPaths doesn't expose chmod_dir(); do it inline.)
        if self._henv.capabilities.mounted:
            for d in (
                self._trial_paths.trial_dir,
                self._trial_paths.agent_dir,
                self._trial_paths.verifier_dir,
                self._trial_paths.artifacts_dir,
            ):
                try:
                    d.chmod(0o777)
                except OSError:
                    pass

        # Drop the kernel server + client into the container.
        await self._henv.upload_file(str(KERNEL_SERVER), "/opt/kernel_server.py")
        await self._henv.upload_file(str(RUN_CELL), "/opt/run_cell.py")
        await self._henv.exec(
            "nohup setsid python3 /opt/kernel_server.py >/tmp/kernel.log 2>&1 < /dev/null &",
            timeout_sec=30,
        )

        # Wait for the kernel HTTP port to bind.
        for _ in range(40):
            r = await self._henv.exec(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{KERNEL_PORT}/",
                timeout_sec=5,
            )
            if (r.stdout or "").strip() == "200":
                self._kernel_ready = True
                logger.info(f"[{self._session_id}] kernel ready")
                return
            await asyncio.sleep(0.5)

        kl = await self._henv.exec("cat /tmp/kernel.log", timeout_sec=5)
        raise RuntimeError(
            f"kernel_server didn't bind 127.0.0.1:{KERNEL_PORT}.\n--- kernel.log ---\n{kl.stdout}"
        )

    async def teardown(self):
        try:
            await self._henv.stop(delete=True)
        finally:
            shutil.rmtree(self._trial_paths.trial_dir, ignore_errors=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _exec_cell(self, code: str) -> tuple[str, bool]:
        b64 = base64.b64encode(code.encode("utf-8")).decode()
        r = await self._henv.exec(
            f"python3 /opt/run_cell.py --code-b64 {b64}", timeout_sec=180
        )
        raw = (r.stdout or "").strip()
        if not raw:
            return f"[run_cell empty stdout rc={r.return_code} stderr={r.stderr or ''}]", False
        try:
            payload = json.loads(raw)
            return str(payload.get("output", "")), bool(payload.get("ok", False))
        except json.JSONDecodeError:
            return f"[run_cell unparseable: {raw[:500]}]", False

    # ── Tools (5) ─────────────────────────────────────────────────────────

    @tool
    async def add_and_execute_code_cell(self, params: CodeCellParams) -> ToolOutput:
        """Execute Python code in the stateful Jupyter-style kernel."""
        out, ok = await self._exec_cell(params.code)
        self._cells.append({"kind": "code", "code": params.code, "output": out, "ok": ok})
        return ToolOutput(blocks=[TextBlock(text=out[:8000])], reward=0.0, finished=False)

    @tool
    async def edit_and_execute_current_cell(self, params: CodeCellParams) -> ToolOutput:
        """Replace the last cell and re-execute it."""
        if self._cells:
            self._cells.pop()
        out, ok = await self._exec_cell(params.code)
        self._cells.append({"kind": "code", "code": params.code, "output": out, "ok": ok})
        return ToolOutput(blocks=[TextBlock(text=out[:8000])], reward=0.0, finished=False)

    @tool
    async def execute_shell_command(self, params: CommandParams) -> ToolOutput:
        """Run a shell command inside the sandbox (pip install, ls, etc.)."""
        r = await self._henv.exec(params.command, timeout_sec=120)
        out = (r.stdout or "") + (r.stderr or "")
        ok = r.return_code == 0
        self._cells.append({"kind": "shell", "code": params.command, "output": out, "ok": ok})
        return ToolOutput(blocks=[TextBlock(text=out[:8000])], reward=0.0, finished=False)

    @tool
    async def get_notebook_state(self, params: NotebookStateParams) -> ToolOutput:
        """Return a compact summary of recent cells and their outputs."""
        if not self._cells:
            text = "No cells executed yet."
        else:
            text = "\n".join(
                _format_cell(c["kind"], c["code"], c["output"], c["ok"])
                for c in self._cells[-10:]
            )
        return ToolOutput(blocks=[TextBlock(text=text)], reward=0.0, finished=False)

    @tool
    async def final_answer(self, params: FinalAnswerParams) -> ToolOutput:
        """Submit the final answer. Ends the episode.

        Three-tier grader:
          1. exact (normalized string)
          2. numeric tolerance
          3. structured-output LLM judge (gpt-4o-mini, JudgeOut schema)
        """
        gold = (self.task_spec.get("gold_answer") or "").strip()
        question = self.task_spec.get("question") or ""
        pred = (params.answer or "").strip()

        method = "miss"
        reward = 0.0

        if _exact(gold, pred):
            reward, method = 1.0, "exact"
        elif _numeric(gold, pred):
            reward, method = 1.0, "numeric"
        elif self._grader_client is not None and gold:
            try:
                reward, verdict, _ = await _llm_judge(self._grader_client, question, gold, pred)
                method = f"llm:{verdict.value.lower()}"
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{self._session_id}] llm-judge failed: {exc}")
                method = "llm:error"

        logger.info(
            f"[{self._session_id}] final_answer gold={gold!r} pred={pred[:80]!r} "
            f"reward={reward} method={method}"
        )
        return ToolOutput(
            blocks=[TextBlock(text=f"graded:{method}")], reward=reward, finished=True
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    suite_dir = os.environ.get("HARBOR_SUITE_DIR", str(RL_ROOT / "harbor" / "tasks" / "jupyter-agent-v1"))
    port = int(os.environ.get("ORS_PORT", "8080"))
    logger.info(f"booting JupyterAgentEnv server")
    logger.info(f"  suite:     {suite_dir}")
    logger.info(f"  sandbox:   {DEFAULT_HARBOR_ENV_TYPE}")
    logger.info(f"  port:      {port}")
    logger.info(f"  tasks:     {len(load_tasks())}")
    Server([JupyterAgentEnv]).run(host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
