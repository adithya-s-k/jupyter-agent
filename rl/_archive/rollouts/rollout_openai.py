"""End-to-end rollout against the local OpenReward server.

Drives an OpenAI tool-calling loop against `JupyterAgentEnv` (in `env/server.py`).
Mirrors the canonical pattern from
`references/RL_Envs_101/envs/jupyter_env/ors/rollout.py`.

Run:
    # 1. start the server in another shell
    HARBOR_SUITE_DIR=harbor/tasks/jupyter-agent-v1 \\
        uv run python -m ors.server &

    # 2. run a rollout against it
    uv run python -m rollouts.rollout_openai \\
        --task-id 0082_302_82302927_qa_3 \\
        --model openai/gpt-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openreward import EnvironmentsAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

ORS_URL = os.environ.get("ORS_URL", "http://127.0.0.1:8080")
ORS_ENV_NAME = os.environ.get("ORS_ENV_NAME", "jupyteragentenv")

SYSTEM = (
    "You are a Python data-analysis agent running in a stateful Jupyter-style "
    "kernel. Use the provided tools to load files, run code, inspect state, "
    "and submit your final answer via the `final_answer` tool. Variables and "
    "imports persist across cells. Keep the answer short and concise."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ors_tools_to_openai(tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def blocks_to_text(blocks) -> str:
    return "\n".join(getattr(b, "text", str(b)) for b in (blocks or []))


def _model_id(name: str) -> str:
    """`openai/gpt-5` → `gpt-5`. Plain ids pass through."""
    if "/" in name and name.startswith("openai/"):
        return name.split("/", 1)[1]
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", default=None, help="Task id (e.g. 0082_302_82302927_qa_3)")
    p.add_argument("--task-index", type=int, default=None)
    p.add_argument("--split", default="train", choices=["train", "eval"])
    p.add_argument("--model", default="openai/gpt-5")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[err] OPENAI_API_KEY missing in .env", file=sys.stderr)
        return 1

    llm = OpenAI(api_key=api_key)
    model_id = _model_id(args.model)

    print(f"ORS:    {ORS_URL}")
    print(f"env:    {ORS_ENV_NAME}")
    print(f"model:  {model_id}")
    print("─" * 70)

    api = EnvironmentsAPI(base_url=ORS_URL, api_key="")
    env = api.get(ORS_ENV_NAME)

    # Pick the task
    tasks = env.list_tasks(args.split)
    if args.task_id is not None:
        candidates = [t for t in tasks if t.task_spec.get("id") == args.task_id]
        if not candidates:
            print(f"[err] no task with id={args.task_id}", file=sys.stderr)
            return 1
        task = candidates[0]
    elif args.task_index is not None:
        if args.task_index >= len(tasks):
            print(f"[err] task_index out of range (have {len(tasks)})", file=sys.stderr)
            return 1
        task = tasks[args.task_index]
    else:
        task = tasks[0]

    spec = task.task_spec
    print(f"task:   {spec['id']}")
    print(f"gold:   {spec['gold_answer']!r}")
    print("─" * 70)

    tools = ors_tools_to_openai(env.list_tools())

    cumulative_reward = 0.0
    finished = False
    turns = 0
    final_answer: str | None = None
    t0 = time.time()

    with env.session(task=task, secrets={"OPENAI_API_KEY": api_key}) as session:
        prompt_text = blocks_to_text(session.get_prompt())
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_text},
        ]

        for turn in range(args.max_turns):
            turns = turn + 1
            r = llm.chat.completions.create(
                model=model_id, messages=messages, tools=tools, tool_choice="auto"
            )
            msg = r.choices[0].message

            if args.verbose:
                print(f"\n── turn {turns} ──")
                if msg.content:
                    print(f"  [assistant] {msg.content[:200]}")

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
                if args.verbose:
                    print("  [stop] no tool calls, model gave up")
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    raw_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    raw_args = {}

                out = session.call_tool(name, raw_args)
                text = blocks_to_text(out.blocks)
                reward = float(out.reward or 0.0)
                cumulative_reward += reward
                if args.verbose:
                    print(f"  ↳ {name}({str(raw_args)[:120]})  reward={reward}  finished={out.finished}")
                    print(f"    {text[:200]}")

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": name, "content": text[:8000]}
                )

                if name == "final_answer":
                    final_answer = raw_args.get("answer")
                if out.finished:
                    finished = True

            if finished:
                if args.verbose:
                    print("  [done] env reported finished=True")
                break
        else:
            if args.verbose:
                print(f"  [hit max_turns={args.max_turns}]")

    elapsed = time.time() - t0
    summary = {
        "task_id": spec["id"],
        "gold": spec["gold_answer"],
        "final_answer": final_answer,
        "turns": turns,
        "cumulative_reward": cumulative_reward,
        "finished": finished,
        "elapsed_sec": round(elapsed, 1),
    }
    print()
    print("─" * 70)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
