"""Harbor BaseAgent modeled on CAMEL-AI / Eigent's SETA tool suite.

Two toolkits, 10 tools total, modeled after the SETA paper / OpenReward env
(https://openreward.ai/Eigent/SETA, https://www.camel-ai.org/blogs/seta-...).
SETA's `submit_solution` is intentionally NOT included — answer submission
goes through `/workdir/answer.txt` so the same task suite (jupyter-agent-eval-*)
is usable by every agent in this repo.

The persistent **notes** are auto-prepended to the system prompt every turn,
giving the model a "TODO always in context" working buffer.

Terminal toolkit (6 tools, all backed by Harbor `environment.exec`):
  - shell_exec(command, blocking=True)      run shell command, return stdout+stderr
  - shell_write_content_to_file(path,content) drop a file into the sandbox
  - shell_write_to_process(pid, content)    write to an interactive proc's stdin
  - shell_view(pid)                          read the captured stdout of a bg proc
  - shell_wait(pid)                          wait for a bg proc to exit
  - shell_kill_process(pid)                  kill -9 a bg proc

Note-taking toolkit (4 tools, in-agent dict — survives across turns of the
SAME episode but is dropped at teardown):
  - create_note(title, content)
  - append_note(title, content)
  - read_note(title)
  - list_note()

Routing for `--model` matches the jupyter-tool agent:
  openai/gpt-5
  anthropic/claude-sonnet-4-6
  hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale

Invoke:
  harbor run -p <suite> \
      --agent-import-path rl.harbor_agents.seta:SetaToolAgent \
      --model openai/gpt-5 \
      --ae OPENAI_API_KEY=... \
      --env e2b
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .._shared import parse_model, provider_credentials, UsageTracker


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/function format — converts to Anthropic/HF transparently
# via the same `tools=` arg)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": (
                "Execute a shell command in the sandbox. If `blocking` is True "
                "(default), the command runs synchronously and the combined "
                "stdout+stderr is returned. If False, the command is detached "
                "into the background and only the new process's PID is returned; "
                "use shell_view/shell_wait to inspect it later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "blocking": {"type": "boolean", "default": True},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_write_content_to_file",
            "description": (
                "Write `content` to `path` inside the sandbox. Overwrites the "
                "file if it exists. Use for scripts, configs, or — in our "
                "task suite — to commit your final answer to /workdir/answer.txt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_write_to_process",
            "description": (
                "Send `content` (followed by a newline) to the stdin of a "
                "background process started via `shell_exec(blocking=False)`. "
                "Useful for REPLs/installers that prompt interactively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["pid", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_view",
            "description": "Return the current captured stdout of a background process (`shell_exec(blocking=False)`).",
            "parameters": {
                "type": "object",
                "properties": {"pid": {"type": "string"}},
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_wait",
            "description": "Wait (up to 5 min) for a background process to terminate. Returns its captured stdout + exit info.",
            "parameters": {
                "type": "object",
                "properties": {"pid": {"type": "string"}},
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_kill_process",
            "description": "Send SIGKILL to a background process.",
            "parameters": {
                "type": "object",
                "properties": {"pid": {"type": "string"}},
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_note",
            "description": (
                "Create a persistent note. Notes are auto-injected into your "
                "system context every turn — use them for TODO lists, "
                "intermediate findings, and plans you want to recall."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_note",
            "description": "Append `content` (preceded by a newline) to an existing note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Return the full content of a note.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_note",
            "description": "List all note titles (with character counts).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class BgProc:
    pid: str
    log_path: str  # /tmp/seta_proc_<pid>.log inside the sandbox
    stdin_pipe: str  # /tmp/seta_proc_<pid>.in


DEFAULT_MAX_TURNS = 25
NOTE_TRUNCATE_CHARS = 2000  # cap per note when injecting into system prompt

SYSTEM_PROMPT = (
    "You are an autonomous data-analysis agent operating in a sandboxed "
    "Linux container with a Python environment pre-installed. You have "
    "shell tools (shell_exec, shell_write_content_to_file, …) and a "
    "persistent note system (create_note, append_note, read_note, list_note).\n\n"
    "**Use notes for planning.** Every turn, the contents of your notes are "
    "auto-injected into your context — treat them as a TODO list and a "
    "scratchpad. Create a 'plan' note early. Append findings as you work.\n\n"
    "**To submit your final answer:** write it to /workdir/answer.txt — "
    "either via `shell_write_content_to_file(path='/workdir/answer.txt', "
    "content=<answer>)` or via `shell_exec(\"echo … > /workdir/answer.txt\")`. "
    "Keep the answer short — the grader is exact-match / numeric-tolerance / "
    "LLM judge. Once the answer file is written, stop calling tools."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SetaToolAgent(BaseAgent):
    """Harbor agent exposing the 10-tool SETA abstraction."""

    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "seta-tool"

    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args, max_turns: int = DEFAULT_MAX_TURNS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self._notes: dict[str, str] = {}
        self._bg_procs: dict[str, BgProc] = {}

    # ── Setup (no kernel server needed — pure shell) ───────────────────────

    async def setup(self, environment: BaseEnvironment) -> None:
        # Make sure /tmp exists (it always does, but defensive) and workdir is writeable.
        await environment.exec("mkdir -p /workdir /tmp", timeout_sec=10)

    # ── Tool dispatch ──────────────────────────────────────────────────────

    async def _tool_shell_exec(self, env, command: str, blocking: bool = True) -> str:
        if blocking:
            r = await env.exec(command, timeout_sec=180)
            out = (r.stdout or "") + (r.stderr or "")
            return out or f"(empty output, rc={r.return_code})"
        # Background: setsid, redirect stdout+stderr to a logfile, capture PID.
        pid_token = uuid.uuid4().hex[:8]
        log_path = f"/tmp/seta_proc_{pid_token}.log"
        stdin_pipe = f"/tmp/seta_proc_{pid_token}.in"
        # mkfifo for stdin so shell_write_to_process can write into it.
        spawn_cmd = (
            f"mkfifo {stdin_pipe} 2>/dev/null; "
            f"( nohup setsid bash -c {shlex.quote(command)} "
            f"  <{stdin_pipe} >{log_path} 2>&1 ) & echo $!"
        )
        r = await env.exec(spawn_cmd, timeout_sec=15)
        pid = (r.stdout or "").strip().split()[-1]
        if not pid.isdigit():
            return f"[shell_exec bg] failed to spawn: {r.stderr or r.stdout}"
        self._bg_procs[pid] = BgProc(pid=pid, log_path=log_path, stdin_pipe=stdin_pipe)
        return f"Started background process PID={pid} log={log_path}"

    async def _tool_shell_write_content_to_file(self, env, path: str, content: str) -> str:
        # base64 to avoid shell-escaping pain.
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        cmd = f"mkdir -p $(dirname {shlex.quote(path)}) && echo {b64} | base64 -d > {shlex.quote(path)}"
        r = await env.exec(cmd, timeout_sec=30)
        if r.return_code != 0:
            return f"[write_file] failed rc={r.return_code}: {r.stderr or ''}"
        return f"Wrote {len(content)} bytes to {path}"

    async def _tool_shell_write_to_process(self, env, pid: str, content: str) -> str:
        proc = self._bg_procs.get(pid)
        if proc is None:
            return f"Unknown PID={pid}. Started PIDs: {list(self._bg_procs.keys())}"
        # Write to the FIFO. Use printf %s\\n for newline.
        b64 = base64.b64encode(content.encode("utf-8")).decode()
        cmd = f"echo {b64} | base64 -d > {shlex.quote(proc.stdin_pipe)}"
        r = await env.exec(cmd, timeout_sec=30)
        if r.return_code != 0:
            return f"[write_to_process] failed rc={r.return_code}: {r.stderr or ''}"
        return f"Wrote {len(content)} bytes to PID={pid}'s stdin"

    async def _tool_shell_view(self, env, pid: str) -> str:
        proc = self._bg_procs.get(pid)
        if proc is None:
            return f"Unknown PID={pid}"
        r = await env.exec(f"cat {shlex.quote(proc.log_path)} 2>/dev/null | tail -c 4000", timeout_sec=10)
        return (r.stdout or "").strip() or f"(no output yet from PID={pid})"

    async def _tool_shell_wait(self, env, pid: str) -> str:
        proc = self._bg_procs.get(pid)
        if proc is None:
            return f"Unknown PID={pid}"
        # Bash `wait` only works on child PIDs of THIS shell; use polling via /proc/<pid>.
        cmd = (
            f"for i in $(seq 1 300); do "
            f"  if [ ! -d /proc/{pid} ]; then break; fi; "
            f"  sleep 1; "
            f"done; "
            f"echo '--- exited ---'; cat {shlex.quote(proc.log_path)} 2>/dev/null | tail -c 4000"
        )
        r = await env.exec(cmd, timeout_sec=320)
        return (r.stdout or "").strip()

    async def _tool_shell_kill_process(self, env, pid: str) -> str:
        if pid not in self._bg_procs:
            return f"Unknown PID={pid}"
        r = await env.exec(f"kill -9 {pid} 2>&1", timeout_sec=10)
        return f"Sent SIGKILL to PID={pid}. rc={r.return_code}"

    # ── Note tools ─────────────────────────────────────────────────────────

    def _tool_create_note(self, title: str, content: str) -> str:
        self._notes[title] = content
        return f"Note '{title}' created ({len(content)} chars). Total notes: {len(self._notes)}."

    def _tool_append_note(self, title: str, content: str) -> str:
        if title not in self._notes:
            return f"Note '{title}' not found. Use create_note first."
        self._notes[title] += "\n" + content
        return f"Note '{title}' updated → {len(self._notes[title])} chars."

    def _tool_read_note(self, title: str) -> str:
        return self._notes.get(title, f"Note '{title}' not found.")

    def _tool_list_note(self) -> str:
        if not self._notes:
            return "(no notes yet)"
        return "\n".join(f"- {t} ({len(c)} chars)" for t, c in self._notes.items())

    async def _dispatch_tool(self, env: BaseEnvironment, name: str, args: dict) -> str:
        """Return the tool's textual output, capped at 8k chars."""
        try:
            if name == "shell_exec":
                output = await self._tool_shell_exec(env, args["command"], args.get("blocking", True))
            elif name == "shell_write_content_to_file":
                output = await self._tool_shell_write_content_to_file(env, args["path"], args["content"])
            elif name == "shell_write_to_process":
                output = await self._tool_shell_write_to_process(env, args["pid"], args["content"])
            elif name == "shell_view":
                output = await self._tool_shell_view(env, args["pid"])
            elif name == "shell_wait":
                output = await self._tool_shell_wait(env, args["pid"])
            elif name == "shell_kill_process":
                output = await self._tool_shell_kill_process(env, args["pid"])
            elif name == "create_note":
                output = self._tool_create_note(args["title"], args["content"])
            elif name == "append_note":
                output = self._tool_append_note(args["title"], args["content"])
            elif name == "read_note":
                output = self._tool_read_note(args["title"])
            elif name == "list_note":
                output = self._tool_list_note()
            else:
                output = f"Unknown tool: {name}"
        except KeyError as exc:
            output = f"[{name}] missing required argument {exc}"
        except Exception as exc:  # noqa: BLE001
            output = f"[{name}] error: {type(exc).__name__}: {exc}"
        if len(output) > 8000:
            output = output[:8000] + "\n... [truncated]"
        return output

    # ── System prompt with notes auto-injected ─────────────────────────────

    def _make_system_prompt(self) -> str:
        base = SYSTEM_PROMPT
        if not self._notes:
            return base
        block = ["\n\n=== YOUR PERSISTENT NOTES (auto-injected) ==="]
        for title, content in self._notes.items():
            snippet = content if len(content) <= NOTE_TRUNCATE_CHARS else (
                content[:NOTE_TRUNCATE_CHARS] + f"\n…[truncated, {len(content)} chars total]"
            )
            block.append(f"\n## {title}\n{snippet}")
        return base + "\n".join(block)

    # ── OpenAI-compat loop (covers OpenAI / Anthropic / HF Inference) ─────

    async def _run_loop(
        self, instruction: str, environment: BaseEnvironment,
        model_id: str, api_key: str, base_url: str | None,
    ) -> tuple[list[dict], UsageTracker]:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        usage = UsageTracker(model_name=self.model_name)
        messages: list[dict] = [
            {"role": "system", "content": self._make_system_prompt()},
            {"role": "user", "content": instruction},
        ]

        for turn in range(self.max_turns):
            # Refresh the system prompt with current notes BEFORE the call.
            messages[0] = {"role": "system", "content": self._make_system_prompt()}

            self.logger.info(f"[turn {turn}] requesting completion (model={model_id})  notes={len(self._notes)}")
            resp = client.chat.completions.create(
                model=model_id, messages=messages, tools=TOOLS, tool_choice="auto",
            )
            usage.add_response(resp)
            msg = resp.choices[0].message
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(entry)

            if not msg.tool_calls:
                self.logger.info(f"[turn {turn}] no tool calls — stopping")
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = await self._dispatch_tool(environment, name, args)
                self.logger.info(f"[turn {turn}] tool={name} out_chars={len(output)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
        return messages, usage

    # ── Required Harbor entry point ───────────────────────────────────────

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext,
    ) -> None:
        provider, model_id = parse_model(self.model_name)
        api_key, base_url = provider_credentials(provider)
        if not api_key:
            raise RuntimeError(f"API key missing for provider={provider}")

        t0 = time.time()
        messages, usage = await self._run_loop(instruction, environment, model_id, api_key, base_url)
        elapsed = time.time() - t0

        usage.populate(context)

        # Persist trajectory + notes + usage for debugging.
        try:
            (self.logs_dir / "seta_agent.trajectory.json").write_text(
                json.dumps(messages, default=str, indent=2)
            )
            (self.logs_dir / "seta_agent.notes.json").write_text(
                json.dumps(self._notes, indent=2)
            )
            (self.logs_dir / "seta_agent.usage.json").write_text(
                json.dumps(usage.as_dict(), indent=2)
            )
        except Exception:  # noqa: BLE001
            pass

        self.logger.info(
            f"seta-tool agent done: provider={provider} model={model_id}  "
            f"notes={len(self._notes)}  calls={usage.n_calls}  "
            f"tokens={usage.prompt_tokens}+{usage.completion_tokens}  "
            f"cost_usd={usage.cost_usd:.6f}  elapsed={elapsed:.1f}s"
        )
