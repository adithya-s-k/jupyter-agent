"""Harbor BaseAgent that exposes the 5 jupyter-style tools to an OpenAI model.

This is the "structured-tool" alternative to Harbor's stock CLI agents
(opencode/codex/etc.). Same task spec, same Harbor verifier — only the agent
changes. Where opencode gives the model `bash` / `read` / `edit`, this agent
gives the model:

  - add_and_execute_code_cell(code)        — stateful kernel
  - edit_and_execute_current_cell(code)    — replace last cell, re-run
  - execute_shell_command(command)         — env.exec passthrough
  - get_notebook_state(include_images)     — in-agent tracker summary
  - final_answer(answer)                   — writes /workdir/answer.txt + ends

The kernel is a tiny HTTP server (`kernel_server.py`) we upload to /opt/ and
start in the background. Each tool call → `env.exec(python /opt/run_cell.py …)`.

Wire it into a Harbor run via:
  harbor run -p <task_dir> \
    --agent-import-path harbor_agents.jupyter:JupyterToolAgent \
    --model openai/gpt-5 \
    --ae OPENAI_API_KEY=... \
    --env docker
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


# ---------------------------------------------------------------------------
# Tool schema — copied verbatim from references/RL_Envs_101/envs/jupyter_env/ors
# (same names/shapes that the SFT model was trained on)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add_and_execute_code_cell",
            "description": (
                "Execute Python code in the stateful Jupyter-style kernel. "
                "Variables, imports, and side-effects persist across calls. "
                "Use this for all computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_and_execute_current_cell",
            "description": (
                "Replace the last code cell with new code and re-execute it. "
                "Use this to fix errors in the previous cell instead of "
                "creating a new one."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": (
                "Run a shell command inside the sandbox (pip install, ls, "
                "etc.). Not stateful with the Python kernel."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_notebook_state",
            "description": (
                "Return a compact summary of recent cells and their outputs. "
                "Useful to recall earlier results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_images": {"type": "boolean", "default": False}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit your final answer. This ends the task. The Harbor "
                "verifier grades it (case-insensitive / numeric / LLM-judge)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Notebook tracker — in-agent (host-side) bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    kind: str  # "code" | "shell"
    code: str
    output: str
    ok: bool


@dataclass
class NotebookTracker:
    cells: list[Cell] = field(default_factory=list)

    def add(self, cell: Cell) -> None:
        self.cells.append(cell)

    def replace_last(self, cell: Cell) -> None:
        if self.cells:
            self.cells.pop()
        self.cells.append(cell)

    def summary(self, max_cells: int = 10) -> str:
        if not self.cells:
            return "No cells executed yet."
        lines: list[str] = []
        for i, c in enumerate(self.cells[-max_cells:]):
            head = c.code.replace("\n", " ↵ ")[:140]
            lines.append(f"[Cell {i}] ({c.kind}) {head}")
            out = c.output.strip()[:300]
            if out:
                marker = "→" if c.ok else "✗"
                lines.append(f"  {marker} {out}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


HERE = Path(__file__).resolve().parent

DEFAULT_MAX_TURNS = 25
SYSTEM_PROMPT = (
    "You are a Python data-analysis agent running in a stateful Jupyter-style "
    "kernel. Use the provided tools to load files, run code, inspect state, "
    "and submit your final answer via the `final_answer` tool. Variables and "
    "imports persist across cells. Keep the answer short and concise; the "
    "grader is a three-tier match (exact / numeric tolerance / LLM judge)."
)


class JupyterToolAgent(BaseAgent):
    """Custom Harbor agent exposing the 5-tool Jupyter abstraction."""

    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "jupyter-tool"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        *args,
        max_turns: int = DEFAULT_MAX_TURNS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _model_id_for_openai(name: str | None) -> str:
        """Accepts `openai/gpt-5`, `gpt-5`, etc. Returns the bare OpenAI id."""
        if not name:
            return "gpt-4o-mini"
        if name.startswith("openai/"):
            return name[len("openai/"):]
        return name

    async def _exec_cell(self, env: BaseEnvironment, code: str) -> tuple[str, bool]:
        """Send code to kernel_server and return (output, ok)."""
        b64 = base64.b64encode(code.encode("utf-8")).decode()
        cmd = f"python3 /opt/run_cell.py --code-b64 {shlex.quote(b64)}"
        result = await env.exec(cmd, timeout_sec=180)
        raw = (result.stdout or "").strip()
        if not raw:
            return f"[run_cell empty stdout, rc={result.return_code}, stderr={result.stderr or ''}]", False
        try:
            payload = json.loads(raw)
            return str(payload.get("output", "")), bool(payload.get("ok", False))
        except json.JSONDecodeError:
            return f"[run_cell unparseable: {raw[:500]}]", False

    async def _wait_for_kernel(self, env: BaseEnvironment, attempts: int = 30) -> bool:
        for _ in range(attempts):
            r = await env.exec(
                "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/",
                timeout_sec=5,
            )
            if (r.stdout or "").strip() == "200":
                return True
            await asyncio.sleep(0.5)
        return False

    # ── Required: setup ────────────────────────────────────────────────────

    async def setup(self, environment: BaseEnvironment) -> None:
        """Upload helpers, start the kernel server."""
        # 1. Upload kernel_server + run_cell to the container.
        await environment.upload_file(HERE / "kernel_server.py", "/opt/kernel_server.py")
        await environment.upload_file(HERE / "run_cell.py", "/opt/run_cell.py")

        # 2. Make sure curl is available (we use it for healthcheck).
        await environment.exec("which curl >/dev/null 2>&1 || apt-get install -y curl", timeout_sec=120)

        # 3. Start the kernel server in the background. setsid + detach so it
        #    survives Harbor's exec lifecycle. Logs to /tmp/kernel.log.
        await environment.exec(
            "nohup setsid python3 /opt/kernel_server.py >/tmp/kernel.log 2>&1 < /dev/null &",
            timeout_sec=30,
        )

        # 4. Wait for the HTTP port to bind.
        if not await self._wait_for_kernel(environment):
            log = await environment.exec("cat /tmp/kernel.log", timeout_sec=5)
            raise RuntimeError(
                f"kernel_server failed to bind 127.0.0.1:8765\n--- kernel.log ---\n{log.stdout}"
            )
        self.logger.info("kernel_server is up at 127.0.0.1:8765")

    # ── Required: run ──────────────────────────────────────────────────────

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Drive the OpenAI tool-calling loop against the 5 jupyter tools."""

        # Lazy imports so the module loads even when openai isn't around.
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package required (uv pip install openai)") from exc

        model_id = self._model_id_for_openai(self.model_name)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set on the host")
        client = OpenAI(api_key=api_key)

        tracker = NotebookTracker()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

        final_answer: str | None = None
        t0 = time.time()

        for turn in range(self.max_turns):
            self.logger.info(f"[turn {turn}] requesting completion (model={model_id})")
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(entry)

            if not msg.tool_calls:
                self.logger.info(f"[turn {turn}] no tool calls — stopping")
                break

            done = False
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "add_and_execute_code_cell":
                    code = args.get("code", "")
                    output, ok = await self._exec_cell(environment, code)
                    tracker.add(Cell(kind="code", code=code, output=output, ok=ok))
                elif name == "edit_and_execute_current_cell":
                    code = args.get("code", "")
                    output, ok = await self._exec_cell(environment, code)
                    tracker.replace_last(Cell(kind="code", code=code, output=output, ok=ok))
                elif name == "execute_shell_command":
                    command = args.get("command", "")
                    r = await environment.exec(command, timeout_sec=120)
                    output = (r.stdout or "") + (r.stderr or "")
                    ok = r.return_code == 0
                    tracker.add(Cell(kind="shell", code=command, output=output, ok=ok))
                elif name == "get_notebook_state":
                    output = tracker.summary()
                elif name == "final_answer":
                    final_answer = (args.get("answer") or "").strip()
                    # Write to the path Harbor's verifier reads.
                    safe = shlex.quote(final_answer)
                    await environment.exec(f"mkdir -p /workdir && printf %s {safe} > /workdir/answer.txt", timeout_sec=30)
                    output = f"Answer submitted: {final_answer}"
                    done = True
                else:
                    output = f"Unknown tool: {name}"

                # Cap to keep token usage sane.
                if len(output) > 8000:
                    output = output[:8000] + f"\n... [truncated]"
                self.logger.info(f"[turn {turn}] tool={name} out_chars={len(output)}")

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            if done:
                break

        # Log the trajectory for debugging — Harbor will surface this dir.
        try:
            (self.logs_dir / "jupyter_agent.trajectory.json").write_text(
                json.dumps(messages, default=str, indent=2)
            )
        except Exception:
            pass
        try:
            (self.logs_dir / "jupyter_agent.tracker.txt").write_text(
                tracker.summary(max_cells=200)
            )
        except Exception:
            pass

        self.logger.info(
            f"jupyter-tool agent done: turns={len(messages)//2}  "
            f"final_answer={final_answer!r}  elapsed={time.time()-t0:.1f}s"
        )
