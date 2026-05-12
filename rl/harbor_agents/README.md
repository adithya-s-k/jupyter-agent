# `rl/harbor_agents/` — custom Harbor agents

This package holds **custom Harbor agents** we use to evaluate the
jupyter-agent task suites. Harbor's built-in agents (`opencode`, `codex`,
`mini-swe-agent`, etc.) all still work via `--agent <name>` — these are for
the abstractions we want explicit control over, in particular:

- **`jupyter/`** — the 4-tool stateful Jupyter abstraction the SFT model
  was trained on. Stateful kernel + shell + state inspection. **No
  `final_answer` tool** — submit by writing to `/workdir/answer.txt`.
- **`seta/`** — 10-tool SETA-style agent (6 shell + 4 notes). Notes are
  auto-injected into the system prompt every turn so the model has a
  persistent "TODO in context".

All custom agents share **`_shared/providers.py`**, which routes
`--model <prefix>/<id>` to the right OpenAI-compat endpoint:

| Prefix | Endpoint | Env var |
|---|---|---|
| `openai/`    | OpenAI native (default base_url) | `OPENAI_API_KEY` |
| `anthropic/` | `https://api.anthropic.com/v1/` (Anthropic OpenAI-compat shim) | `ANTHROPIC_API_KEY` |
| `hf/`        | `https://router.huggingface.co/v1` (HF Inference router) | `HF_TOKEN` |

This means **every agent uses one `openai.OpenAI` client** — no need for
the `anthropic` SDK or any provider-specific library beyond `openai`.

## Layout

```
rl/harbor_agents/
├── README.md               ← you are here
├── __init__.py
├── _shared/
│   ├── __init__.py
│   └── providers.py        parse_model() + provider_credentials()
├── jupyter/
│   ├── __init__.py         re-exports JupyterToolAgent
│   ├── agent.py            the agent
│   ├── kernel_server.py    uploaded into the container at /opt/
│   └── run_cell.py         uploaded into the container at /opt/
└── seta/
    ├── __init__.py         re-exports SetaToolAgent
    └── agent.py            the agent
```

Import paths (used by `harbor run --agent-import-path …`):
- `rl.harbor_agents.jupyter:JupyterToolAgent`
- `rl.harbor_agents.seta:SetaToolAgent`

## How to add a new custom agent

1. Pick a slug (`mytool`).
2. `mkdir rl/harbor_agents/mytool && touch rl/harbor_agents/mytool/__init__.py`
3. Create `rl/harbor_agents/mytool/agent.py` based on `jupyter/agent.py` or
   `seta/agent.py`. Skeleton:

   ```python
   from harbor.agents.base import BaseAgent
   from harbor.environments.base import BaseEnvironment
   from harbor.models.agent.context import AgentContext
   from rl.harbor_agents._shared import parse_model, provider_credentials

   TOOLS = [...]            # OpenAI function-format
   SYSTEM_PROMPT = "..."

   class MyToolAgent(BaseAgent):
       SUPPORTS_WINDOWS = False

       @staticmethod
       def name() -> str: return "my-tool"
       def version(self) -> str: return "0.1.0"

       async def setup(self, environment: BaseEnvironment) -> None:
           ...   # upload helper scripts, start servers, etc.

       async def run(
           self, instruction: str, environment: BaseEnvironment,
           context: AgentContext,
       ) -> None:
           provider, model_id = parse_model(self.model_name)
           api_key, base_url = provider_credentials(provider)
           if not api_key:
               raise RuntimeError(f"API key missing for provider={provider}")
           # … drive an OpenAI-client tool-calling loop;
           # write the agent's answer to /workdir/answer.txt at the end …
   ```

4. Re-export in `__init__.py`:

   ```python
   from .agent import MyToolAgent
   ```

5. Invoke via Harbor:

   ```bash
   harbor run -p <suite> \
       --agent-import-path rl.harbor_agents.mytool:MyToolAgent \
       --model openai/gpt-5 \
       --ae OPENAI_API_KEY=… \
       --env e2b
   ```

### Conventions all custom agents follow

- **No `final_answer` / `submit` tool.** Agents submit by writing to
  `/workdir/answer.txt` inside the sandbox. The Harbor verifier in our
  task suite reads that file and grades via exact-match / numeric /
  LLM judge.
- **Provider via `parse_model`** — never hardcode the OpenAI SDK base
  URL or the env var. Use `_shared.providers`.
- **Container helpers go in the agent's folder** (e.g.
  `jupyter/kernel_server.py`), not next to the package root. Reference
  via `Path(__file__).parent / "kernel_server.py"` and
  `environment.upload_file(...)`.
- **Trajectory logging** is best-effort: write
  `self.logs_dir / "<agent>.trajectory.json"` at the end of `run()`
  so we have something to inspect when a model fails.

## Built-in Harbor agents we also use

Some tasks are run with stock Harbor agents (no code in this package):

| Harbor name | What it does |
|---|---|
| `mini-swe-agent` | Single `bash` tool — the canonical bash-only agent |
| `opencode` | `bash` / `read` / `write` / `edit` / `grep` / `glob` / `ls` TUI agent (multi-provider) |
| `codex`, `claude-code`, `gemini-cli`, … | CLI coding agents — useful for cross-baseline comparisons |

For these, pass `--agent <name>` (not `--agent-import-path`). The same
`--model <prefix>/<id>` convention applies for the providers Harbor's
opencode plugin understands (`openai/`, `anthropic/`, `huggingface/`,
`openrouter/`, etc.).
