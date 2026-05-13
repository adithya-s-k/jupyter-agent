"""Harbor BaseAgent that gives the model a single `bash` tool.

Same shape as `mini-swe-agent` (one tool, shell access), but speaks the
provider-prefix model contract used by the rest of `rl/harbor_agents/`:

  openai/gpt-5                                  → OpenAI native
  anthropic/claude-sonnet-4-6                   → Anthropic OpenAI-compat shim
  hf/Qwen/Qwen3-8B:nscale                       → HF Inference router

Why this exists: Harbor's built-in `mini-swe-agent` uses LiteLLM and doesn't
recognise our `hf/...:provider` model strings ("Unable to determine API key
for model hf/Qwen/...:nscale"). Rather than fight LiteLLM's provider
registry, this agent uses the OpenAI client with the right base_url per
provider via `_shared.providers` — same code path as JupyterToolAgent and
SetaToolAgent.

Submission protocol: write `/workdir/answer.txt`. No `final_answer` tool.
"""

from __future__ import annotations

import json
import os
import time

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .._shared import parse_model, provider_credentials, UsageTracker


TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the sandbox and return its combined "
                "stdout+stderr. The shell is non-stateful between calls "
                "(each call is a fresh `bash -c '<command>'`). Use this to "
                "explore files (ls, head, cat), execute Python one-liners "
                "(`python3 -c \"…\"`), and finally to write the answer "
                "(`echo -n \"<value>\" > /workdir/answer.txt`)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

DEFAULT_MAX_TURNS = 25

SYSTEM_PROMPT = (
    "You are an autonomous data-analysis agent operating in a sandboxed "
    "Linux container. Your only tool is `bash`. The dataset files are in "
    "/home/user/input/. Python 3 + pandas + numpy + scikit-learn + scipy "
    "are pre-installed.\n\n"
    "Work the problem step-by-step: first inspect the data (ls, head, "
    "shape, dtypes), then plan, then compute, then submit.\n\n"
    "**To submit your final answer:** write it to /workdir/answer.txt with "
    "an absolute path, e.g. `echo -n \"<value>\" > /workdir/answer.txt` or "
    "`python3 -c 'open(\"/workdir/answer.txt\",\"w\").write(str(<value>))'`. "
    "Keep the answer short and concise — the grader is exact / numeric / "
    "LLM judge. After the file is written, stop calling tools."
)


class BashOnlyAgent(BaseAgent):
    """Custom Harbor agent: single bash tool, multi-provider via OpenAI client."""

    SUPPORTS_WINDOWS: bool = False

    @staticmethod
    def name() -> str:
        return "bash-only"

    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args, max_turns: int = DEFAULT_MAX_TURNS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec("mkdir -p /workdir /home/user/input", timeout_sec=10)

    async def _exec_bash(self, env: BaseEnvironment, command: str) -> str:
        r = await env.exec(command, timeout_sec=180)
        out = (r.stdout or "") + (r.stderr or "")
        return out or f"(empty output, rc={r.return_code})"

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext,
    ) -> None:
        provider, model_id = parse_model(self.model_name)
        api_key, base_url = provider_credentials(provider)
        if not api_key:
            raise RuntimeError(f"API key missing for provider={provider}")

        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

        tracker = UsageTracker(model_name=self.model_name)
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        t0 = time.time()

        for turn in range(self.max_turns):
            self.logger.info(f"[turn {turn}] requesting completion (model={model_id})")
            resp = client.chat.completions.create(
                model=model_id, messages=messages, tools=TOOLS, tool_choice="auto",
            )
            tracker.add_response(resp)
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                cmd = (args.get("command") or "").strip()
                output = await self._exec_bash(environment, cmd) if cmd else "(empty command)"
                if len(output) > 8000:
                    output = output[:8000] + "\n... [truncated]"
                self.logger.info(f"[turn {turn}] bash chars={len(output)}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        tracker.populate(context)
        try:
            (self.logs_dir / "bash_agent.trajectory.json").write_text(
                json.dumps(messages, default=str, indent=2)
            )
            (self.logs_dir / "bash_agent.usage.json").write_text(
                json.dumps(tracker.as_dict(), indent=2)
            )
        except Exception:  # noqa: BLE001
            pass

        self.logger.info(
            f"bash-only agent done: provider={provider} model={model_id}  "
            f"calls={tracker.n_calls}  tokens={tracker.prompt_tokens}+{tracker.completion_tokens}  "
            f"cost_usd={tracker.cost_usd:.6f}  elapsed={time.time()-t0:.1f}s"
        )
