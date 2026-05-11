# Jupyter Agent — Pipeline Plan

Two stages. **Stage 1 is shipped** (commit `83a1949`). **Stage 2 is the next build** — an OpenReward (ORS) server that exposes the same 5 jupyter tools as `JupyterToolAgent`, for RL training rollouts.

---

# Stage 1 — Harbor task suite + HF Bucket (DONE)

---

## 30-second mental model

```
ONE SUITE = one `--name <slug>`  →  one HF Bucket + one HF Dataset repo

    cache_dataset.py
       │ downloads jupyter-agent/jupyter-agent-dataset once (~72 GB)
       ▼
  cache/raw/{thinking,non_thinking}.parquet
       │
       │  build_harbor_tasks.py --name test --n-tasks 5
       │  - filter, dedupe, sample N
       │  - kagglehub download per unique dataset (parallel)
       │  - emit task folders (instruction.md, task.toml, tests/, environment/)
       ▼
  harbor/tasks/jupyter-agent-test/
       │       │
       │       ├── manifest.jsonl
       │       ├── data/<prefix>/             ← local mirror (kagglehub→local)
       │       │                                  KEPT for offline dev only;
       │       │                                  NOT baked into images.
       │       └── <safe_id>/
       │             ├── instruction.md       ← rendered prompt
       │             ├── task.toml            ← [environment.env] HF_BUCKET, BUCKET_PREFIX
       │             │                          [environment.healthcheck] pull_bucket.py
       │             │                          [verifier.env] EXPECTED_ANSWER, QUESTION, OPENAI_API_KEY
       │             ├── tests/{test.sh,grader.py}
       │             └── environment/{Dockerfile,pull_bucket.py}    ← no data/
       │
       │  stage_data.py --name test
       │  - syncs local data/<prefix>/ → hf://buckets/<user>/<slug>-data/<prefix>/
       │
       │  push_harbor.py --name test
       │  - uploads spec folders (excluding data/) → hf://datasets/<user>/<slug>-harbor
       ▼
  PUBLISHED ARTIFACTS:
    hf://buckets/<user>/jupyter-agent-<slug>-data/       ← canonical data
    hf://datasets/<user>/jupyter-agent-<slug>-harbor     ← spec only, tiny

         │ consumed at runtime:
         ▼
  harbor run -p <path> -a <agent> -m <model> -e <sandbox> --env-file ../.env
    1. Build Dockerfile (env/Dockerfile, no data baked)
    2. Start container
    3. [environment.healthcheck] runs:
          python3 /opt/pull_bucket.py
          → pulls hf://buckets/<bucket-id>/<bucket-prefix>/ into /home/user/input/
       must pass before agent setup begins
    4. Agent setup + run (opencode / codex / JupyterToolAgent / etc.)
    5. Agent writes /workdir/answer.txt OR submits via tool
    6. Verifier runs tests/test.sh → tests/grader.py → /logs/verifier/reward.txt
```

---

## Components — what each file does

### Source data (read-only)
- `cache/raw/{thinking,non_thinking}.parquet` — 51,389 rows × 2 splits, cached from `jupyter-agent/jupyter-agent-dataset`. Ground truth for tasks.

### Shared helpers
- `grader.py` — 3-tier grader (exact / numeric tolerance / LLM-judge with gpt-4o-mini, simple-evals A/B/C prompt). Used by:
  - `tests/test.sh` (Harbor CLI eval path — copy of this file per task)
  - `JupyterToolAgent` (future — same logic for in-loop reward)

### `prepare/cache_dataset.py`
Downloads source dataset to local cache. One-time, ~72 GB. Done already.

### `prepare/build_harbor_tasks.py` ← current state after recent edits
For `--name <slug> --n-tasks N`:

1. Resolve names — base=`jupyter-agent-<slug>`, bucket=`<user>/<base>-data`, repo=`<user>/<base>-harbor`, local dir=`harbor/tasks/<base>/`
2. Load `cache/raw/non_thinking.parquet`, filter (e2b + non-null kaggle + non-null answer), dedupe by (kaggle, question), keep highest edu_score
3. Pick `N × multiplier` candidate rows, deterministic shuffle
4. **kagglehub download** every UNIQUE kaggle dataset among candidates, in parallel
5. Filter rows to those whose kaggle dataset succeeded
6. Take first N
7. Mirror downloaded files to `harbor/tasks/<base>/data/<prefix>/` (shared, deduped — offline-dev convenience)
8. For each accepted row, emit `<safe_id>/`:
   - `instruction.md` — rendered from `data/pipelines/prompts/agent_prompt_e2b.md`
   - `task.toml` — schema 1.2; `[environment.env]` carries HF_BUCKET/BUCKET_PREFIX/HF_TOKEN; **`[environment.healthcheck]` calls `python3 /opt/pull_bucket.py`** (pre-agent hook); `[verifier.env]` carries EXPECTED_ANSWER/QUESTION/OPENAI_API_KEY
   - `tests/test.sh` — shells `python3 /tests/grader.py` against `/workdir/answer.txt`
   - `tests/grader.py` — copy of root `grader.py`
   - `environment/Dockerfile` — `python:3.12-slim` + huggingface_hub + pandas/numpy/etc.; **no `COPY data`**
   - `environment/pull_bucket.py` — invoked by healthcheck; reads `HF_BUCKET` + `BUCKET_PREFIX` env vars, fetches files via `download_bucket_files`
9. Write `manifest.jsonl` (safe_id → bucket_prefix → kaggle_dataset_name → files_used)
10. Write `README.md`

### `prepare/stage_data.py`
For `--name <slug>`: read `manifest.jsonl`, sync each unique prefix from local `data/<prefix>/` → `hf://buckets/<user>/<base>-data/<prefix>/` via `sync_bucket` (Xet-chunked, parallel, idempotent).

### `prepare/push_harbor.py`
For `--name <slug>`: `create_repo("<user>/<base>-harbor", type="dataset", exist_ok=True)` + `upload_folder(harbor/tasks/<base>/)` with `ignore_patterns=["data/**", ...]`. Spec-only push.

### `harbor_agents/jupyter.py` + `kernel_server.py` + `run_cell.py`
Custom Harbor `BaseAgent` exposing 5 jupyter-style tools:
- `add_and_execute_code_cell(code)` — stateful Python kernel (via in-container HTTP server)
- `edit_and_execute_current_cell(code)` — replace last cell, re-run
- `execute_shell_command(command)` — direct `env.exec(command)`
- `get_notebook_state(include_images)` — in-agent tracker summary
- `final_answer(answer)` — writes `/workdir/answer.txt`, ends episode

Invoked via `harbor run --agent-import-path harbor_agents.jupyter:JupyterToolAgent`.

---

## End-to-end command sequence (current shape)

```bash
# 0. ONE-TIME — cache source dataset (~72 GB)
uv run python -m prepare.cache_dataset --confirm

# 1. PER-SUITE — pick N tasks, download kaggle data, emit Harbor folders
uv run python -m prepare.build_harbor_tasks --name test --n-tasks 5

# 2. Sync data → HF Bucket
uv run python -m prepare.stage_data --name test

# 3. (optional) Push spec → HF Hub
uv run python -m prepare.push_harbor --name test

# 4. RUN — local Docker with stock CLI agent (opencode/codex/etc.)
harbor run \
  --path harbor/tasks/jupyter-agent-test \
  --agent opencode --env docker --model openai/gpt-5 \
  --env-file ../.env --ae OPENAI_API_KEY="$OPENAI_API_KEY"

# 4'. RUN — E2B cloud sandbox
harbor run --path ... --agent opencode --env e2b ...

# 4''. RUN — custom 5-tool agent
harbor run --path ... \
  --agent-import-path harbor_agents.jupyter:JupyterToolAgent \
  --model openai/gpt-5 \
  --env-file ../.env --ae OPENAI_API_KEY="$OPENAI_API_KEY"
```

---

## What we've verified working (today)

| Run | Sandbox | Agent | Model | Reward | Time |
|---|---|---|---|---|---|
| Insurance R² task (gold=0.7487) | docker | nop | — | 0.0 | 39 s (smoke — container/verifier wire-up) |
| Insurance R² task | docker | opencode | gpt-4o-mini | 0.0 (overfit) | 1m 20 s |
| Insurance R² task | docker | opencode | **gpt-5** | **1.0** | 1m 6 s |
| Insurance R² task | **e2b** | opencode | **claude-sonnet-4-5** | **1.0** | 1m 19 s |
| Insurance R² task | docker | **JupyterToolAgent** | gpt-5 | **1.0** | 0m 59 s |
| Insurance R² task | docker | opencode | gpt-5 | **1.0** | 1m 38 s | ← **bucket-only variant: NO `COPY data`, healthcheck pulled from bucket** |

Same task spec across all rows. Same HF Bucket as the data source. Different sandbox providers, agents, and models all converge on reward 1.0.

---

## Locked decisions (current state)

- **Slug naming** — `jupyter-agent-<slug>` → `<user>/<base>-data` (bucket) + `<user>/<base>-harbor` (dataset repo). Default user: `AdithyaSK`.
- **Bucket is canonical** — runtime data path is always the bucket; local mirror is offline-dev convenience.
- **Healthcheck = pre-agent hook** — `[environment.healthcheck]` in task.toml runs `pull_bucket.py` before agent setup. Confirmed to work with both stock CLI agents and the custom `JupyterToolAgent`.
- **5 jupyter tools** — verbatim from `references/RL_Envs_101/envs/jupyter_env/ors/server.py` for both Harbor agent and (later) OpenReward server.
- **Grader** — 3 tiers in `grader.py`: exact / numeric (rel/abs ≤ 1e-3) / LLM-judge (gpt-4o-mini, simple-evals A/B/C). Used by both `tests/test.sh` and (future) `final_answer @tool`.
- **HF SDK** — `huggingface_hub>=1.12` (bucket API + `hf_xet`). `HF_XET_HIGH_PERFORMANCE=1`. No `hf_transfer`.

---

## Pending — what to iterate on

1. **Regenerate the 5 tasks with the bucket-only Dockerfile**. The build script is updated; just need to run `--clean` + smoke-test one task.
2. **Decide whether to drop the local `data/<prefix>/` mirror entirely** (or leave it as a kagglehub cache for offline runs). Currently kept.
3. **Push to a clean `<base>-harbor` repo** to verify the bucket-only spec is fully self-contained for a 3rd-person clone.
4. **`prepare/pull_harbor.py`** for the 3rd-person flow: `snapshot_download` + ready-to-`harbor run`. Tiny — `huggingface_hub.snapshot_download(repo_id="<user>/<base>-harbor", repo_type="dataset")`.
5. **Run all 5 tasks under one model × sandbox combination** for a real pass-rate number (`harbor run -p <suite>` with `-n 4`).
6. **OpenReward server** — same 5 tools, HTTP service, per-call rewards. The RL-training-ready twin of `JupyterToolAgent`. Read from the same `harbor/tasks/<base>/` folders.
7. **TRL/SkyRL trainer** — points at the OpenReward server, RL loop. Out of scope until 6 lands.

---

---

# Stage 2 — OpenReward server that delegates to Harbor (NEXT)

## The core idea (revised)

Stage 1 already built the right sandbox abstraction: `harbor.environments.factory.create_environment(...)` understands our `task.toml` + `Dockerfile` + healthcheck-driven bucket pull, and works across `docker | e2b | modal | daytona | runloop | gke | apple-container`.

**Stage 2 reuses that.** The OpenReward server is a thin shell that:
1. Reads the same Harbor task folders.
2. Asks Harbor's factory for a sandbox per session (`await env.start()` + `await env.run_healthcheck()`).
3. Exposes 5 `@tool` methods that just `await env.exec(...)` against the Harbor-managed container.
4. Grades `final_answer` inline via `grader.py` + a structured-output LLM judge.

**Zero re-implementation of sandboxing.** Harbor handles Docker/E2B/Modal/healthcheck/bucket-pull/build. OpenReward handles HTTP protocol, sessions, per-call rewards. The 5 tools are the bridge.

## Why this exists

Stage 1 gives us **batch eval** (Harbor `run` over a static task suite). Stage 2 gives us **in-the-loop training**: each rollout is one session against a long-lived HTTP service that returns rewards per tool call. That's the shape TRL/SkyRL want.

Two consumers of the same ORS server:
1. **Direct rollout** (`rl/rollouts/`) — `openreward.EnvironmentsAPI` client + an LLM (OpenAI/Anthropic/our SFT model). Used for RL training rollouts and standalone debugging.
2. **Harbor proxy agent** (`rl/harbor_agents/ors_proxy.py`, optional) — thin Harbor `BaseAgent` that talks to the ORS server instead of running a kernel inside the Harbor container. Lets Harbor batch eval reuse the ORS env without duplicating tool logic.

The ORS server is one process. **Locally**: `python -m env.server` → `localhost:8080`, sandboxes spawned in E2B via the code-interpreter SDK. **Hosted (future)**: same code, push to HF Space, anyone hits it via `EnvironmentsAPI(base_url=...)`.

## Goal — exact deliverables

1. A working `python -m env.server` that exposes the 5 jupyter tools over the ORS HTTP protocol.
2. Tasks loaded from our Harbor folders (`harbor/tasks/jupyter-agent-<slug>/`) — using `harbor.models.task.task.Task(task_dir)` to parse, not custom code.
3. Per-session sandbox via `harbor.environments.factory.EnvironmentFactory.create_environment(...)` + `await env.start(force_build=False)` + `await env.run_healthcheck()`. Bucket pull happens inside the healthcheck, not in our code.
4. Reward computed inline in `final_answer` via shared `rl/grader.py` + structured-output LLM judge (`AsyncOpenAI.beta.chat.completions.parse(response_format=Verdict)`).
5. A standalone `rl/rollouts/rollout.py` that drives the env via `openreward.EnvironmentsAPI` + an OpenAI model. Verified by reproducing the 4/10 pass-rate from Stage 1 against `--name v1`.
6. (Stretch) `rl/harbor_agents/ors_proxy.py` so Harbor eval can reuse the ORS env. Optional in this stage.

## Architecture

```
   hf://datasets/<user>/<base>-harbor      hf://buckets/<user>/<base>-data
   (spec, Stage 1)                         (data, Stage 1)
                  │                            │
                  └──────────────┬─────────────┘
                                 │
                                 ▼  imports
        ┌────────────────────────────────────────────────────────┐
        │  harbor.environments.factory.EnvironmentFactory        │
        │    .create_environment(type=docker|e2b|…, …)           │
        │    → BaseEnvironment with:                             │
        │       async start(force_build)                         │
        │       async run_healthcheck()    ← runs pull_bucket.py │
        │       async exec(command, …)                           │
        │       async upload_file(...)                           │
        │       async stop(delete=True)                          │
        └──────────────────┬─────────────────────────────────────┘
                           │  composed inside →
                           ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  env/server.py  (openreward.environments.Server.run)            │
   │                                                                 │
   │   class JupyterAgentEnv(Environment):                           │
   │     def __init__(self, task_spec, secrets):                     │
   │       # parse task.toml via harbor.models.task.task.Task        │
   │       # build TrialPaths + EnvironmentConfig                    │
   │       # self._henv = EnvironmentFactory.create_environment(…)   │
   │       # self.grader = AsyncOpenAI(api_key=secrets["OPENAI_…"])  │
   │                                                                 │
   │     async def setup(self):                                      │
   │       await self._henv.start(force_build=False)                 │
   │       await self._henv.run_healthcheck()   ← bucket lands here  │
   │       await self._henv.upload_file(kernel_server.py, /opt/…)    │
   │       await self._henv.exec("nohup python3 /opt/kernel_server.py &")
   │                                                                 │
   │     async def teardown(self):                                   │
   │       await self._henv.stop(delete=True)                        │
   │                                                                 │
   │     @tool async def add_and_execute_code_cell(self, params):    │
   │       r = await self._henv.exec("python3 /opt/run_cell.py …")   │
   │       return ToolOutput(blocks=[…], reward=0.0, finished=False) │
   │                                                                 │
   │     @tool async def final_answer(self, params):                 │
   │       # exact + numeric tiers from grader.grade(...)            │
   │       # if miss → AsyncOpenAI.beta.chat.completions.parse(      │
   │       #     response_format=Verdict, …)                         │
   │       return ToolOutput(reward=r, finished=True)                │
   │                                                                 │
   │   Server([JupyterAgentEnv]).run(host=0.0.0.0, port=8080)        │
   └─────────────────────────────────────────────────────────────────┘
                         ▲
                         │ HTTP (REST + SSE), openreward.EnvironmentsAPI
                         │
        ┌────────────────┴────────────────┬───────────────────────┐
        │                                 │                       │
   rollouts/rollout_openai.py    (stretch) harbor_agents/   (Stage 3) trl_grpo.py
   - openreward client           ors_proxy.py               - reward_fn → ORS
   - tool loop + LLM             - Harbor BaseAgent that      session.call_tool
   - cumulative reward             talks to local ORS       - GRPO updates
```

## File layout (revised — leaner because Harbor does the sandbox work)

```
rl/
├── env/                                ← NEW (this stage)
│   ├── __init__.py
│   ├── server.py                       ← JupyterAgentEnv + Server.run(); ~200 LOC
│   ├── verdict.py                      ← Pydantic `Verdict(BaseModel)` for structured-output judge
│   ├── README.md                       ← how to run + deploy notes
│   └── (no e2b_sandbox.py, no notebook_tracker.py, no tasks_loader.py)
│        ↑ Harbor's BaseEnvironment makes those redundant.
│
├── rollouts/                           ← NEW (this stage)
│   ├── __init__.py
│   ├── rollout_openai.py               ← openreward.EnvironmentsAPI client + OpenAI tool loop
│   └── rollout_anthropic.py            ← same shape, Anthropic tool-use API
│
└── harbor_agents/                       ← Stage 1 (already shipped)
    ├── jupyter.py                      ← uses kernel_server.py too — shared infra
    ├── kernel_server.py                ← SAME file uploaded into Harbor container by both runtimes
    ├── run_cell.py                     ← SAME file
    └── (stretch) ors_proxy.py           ← Harbor BaseAgent that proxies to localhost:8080 ORS
```

**What is NOT duplicated:** `kernel_server.py`, `run_cell.py`, the Dockerfile, `pull_bucket.py`, `grader.py`. Stage 2 vendors all of these by importing or uploading from Stage 1's locations.

## Implementation order

1. **Pin `harbor==0.6.6`** in `pyproject.toml` — Harbor doesn't declare API stability and we want a reproducible base.

2. **Write `env/list_tasks_helper.py`** — walk `HARBOR_SUITE_DIR`, parse each `task.toml` via `harbor.models.task.task.Task(task_dir)`, return:
   ```python
   [{
       "id": task_dir.name,
       "task_dir": str(task_dir),
       "instruction": (task_dir / "instruction.md").read_text(),
       "gold_answer": t.config.verifier.env.get("EXPECTED_ANSWER"),
       "question":    t.config.verifier.env.get("QUESTION"),
   }, ...]
   ```

3. **Write `env/verdict.py`** — Pydantic Verdict + Enum schema for the LLM-judge structured output:
   ```python
   from enum import Enum
   from pydantic import BaseModel
   class JudgeVerdict(str, Enum):
       CORRECT = "CORRECT"; INCORRECT = "INCORRECT"; NOT_ATTEMPTED = "NOT_ATTEMPTED"
   class JudgeOut(BaseModel):
       verdict: JudgeVerdict
       reasoning: str
   ```

3. **Write `env/server.py`** — `class JupyterAgentEnv(Environment)`. Mirror Harbor's `Trial.__init__`:
   ```python
   def __init__(self, task_spec, secrets):
       super().__init__(task_spec)
       task_dir = Path(task_spec["task_dir"])
       self._task = Task(task_dir)                                          # parses task.toml
       self._session_id = uuid.uuid4().hex[:12]                             # container-unique

       # Per-session scratch dir (Harbor bind-mounts subdirs of this)
       trial_dir = Path(tempfile.mkdtemp(prefix=f"ors-{self._session_id}-"))
       self._trial_paths = TrialPaths(trial_dir=trial_dir)
       self._trial_paths.mkdir()                                            # ← REQUIRED

       # Mirror Trial.__init__ exactly
       self._henv = EnvironmentFactory.create_environment_from_config(
           config=self._task.config.environment,
           environment_dir=task_dir / "environment",
           environment_name=f"ors-{task_spec['id']}-{self._session_id[:8]}",
           session_id=self._session_id,
           trial_paths=self._trial_paths,
           task_env_config=self._task.config.environment,
           logger=logging.getLogger(__name__),
       )

       self.grader_client = AsyncOpenAI(api_key=secrets["OPENAI_API_KEY"])
       self._kernel_started = False

   @classmethod
   def list_splits(cls):
       return [Split(name="train"), Split(name="eval")]

   @classmethod
   def list_tasks(cls, split):
       suite_dir = Path(os.environ["HARBOR_SUITE_DIR"])
       return [...read each task_dir...]                                    # see step 2

   def get_prompt(self):
       return [TextBlock(text=self.task_spec["instruction"])]

   async def setup(self):
       await self._henv.start(force_build=False)
       await self._henv.run_healthcheck()                                   # bucket lands HERE
       if self._henv.capabilities.mounted:
           self._trial_paths.chmod_dir()                                    # ← REQUIRED if mounted
       await self._henv.upload_file(KERNEL_SERVER_PATH, "/opt/kernel_server.py")
       await self._henv.upload_file(RUN_CELL_PATH, "/opt/run_cell.py")
       await self._henv.exec("nohup setsid python3 /opt/kernel_server.py >/tmp/k.log 2>&1 &")
       # poll http://127.0.0.1:8765/ until 200
       self._kernel_started = True

   async def teardown(self):
       await self._henv.stop(delete=True)
       shutil.rmtree(self._trial_paths.trial_dir, ignore_errors=True)
   ```
   Then 5 `@tool async def` methods (canonical pattern documented in `using-llm-graders`).

4. **Tool implementations**:
   ```python
   @tool
   async def add_and_execute_code_cell(self, params: CodeCellParams) -> ToolOutput:
       b64 = base64.b64encode(params.code.encode()).decode()
       r = await self._henv.exec(f"python3 /opt/run_cell.py --code-b64 {b64}", timeout_sec=180)
       payload = json.loads(r.stdout or '{"output":"","ok":false}')
       return ToolOutput(blocks=[TextBlock(text=payload["output"])], reward=0.0, finished=False)

   @tool
   async def final_answer(self, params: FinalAnswerParams) -> ToolOutput:
       # Tier 1+2: deterministic
       res = grade(self.task_spec["gold_answer"], params.answer, judge=False)
       if res.reward == 0.0 and self.task_spec.get("question"):
           # Tier 3: structured-output LLM judge
           resp = await self.grader_client.beta.chat.completions.parse(
               model="gpt-4o-mini",
               messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                   question=self.task_spec["question"],
                   gold=self.task_spec["gold_answer"],
                   pred=params.answer,
               )}],
               response_format=JudgeOut,                                    # Pydantic schema
               temperature=0,
           )
           verdict = resp.choices[0].message.parsed.verdict
           reward = 1.0 if verdict == JudgeVerdict.CORRECT else 0.0
       else:
           reward = res.reward
       return ToolOutput(blocks=[TextBlock(text="ok")], reward=reward, finished=True)
   ```

5. **Smoke-test the server**: `HARBOR_SUITE_DIR=harbor/tasks/jupyter-agent-v1 python -m env.server &` then `curl http://localhost:8080/list_environments` → should return `["jupyteragentenv"]`. Then `curl /environments/jupyteragentenv/tasks/train`.

6. **Write `rl/rollouts/rollout_openai.py`** — modeled on `references/RL_Envs_101/envs/jupyter_env/ors/rollout.py`:
   - args: `--task-id` or `--task-index`, `--model`, `--max-turns`
   - reads `ORS_URL` (default `http://localhost:8080`), `ORS_ENV_NAME` (default `jupyteragentenv`)
   - opens `with env.session(task=task, secrets={"OPENAI_API_KEY": ...}) as s`
   - multi-turn OpenAI tool-calling loop using `s.call_tool(name, args)`
   - prints cumulative reward + final answer

7. **Verify reward parity with Stage 1** — run `rollouts/rollout_openai.py --task-id 0082_302_82302927_qa_3 --model openai/gpt-5`. Must produce `reward=1.0` and a predicted value in the same numeric band (~0.749x) as Stage 1's Harbor run. Exact match impossible (LLM stochasticity even at temperature=0), pass-rate parity is the right check.

8. **(Stretch) `harbor_agents/ors_proxy.py`** — Harbor `BaseAgent` that opens an `EnvironmentsAPI` session against `localhost:8080` and proxies tool calls. Reuses the Stage 1 task suite + verifier. Pure thin shim (~80 LOC).

9. **(Future) Push the ORS env to HF Spaces** — `env/Dockerfile` + GHA workflow. Mirror `references/RL_Envs_101/envs/jupyter_env/ors/` (deployed at `AdithyaSK/jupyter-agent-ors`). Then anyone runs `EnvironmentsAPI(base_url="https://<user>-jupyter-agent.hf.space")`.

## Locked decisions

| | |
|--|--|
| **Tools** | Exactly the 5 from `references/RL_Envs_101/envs/jupyter_env/ors/server.py` and `harbor_agents/jupyter.py`. Same names, same input schemas. |
| **Reward shape** | `final_answer` returns `ToolOutput(reward=grader_result, finished=True)`. Other tools return `reward=0.0, finished=False`. No per-call shaping in v1; revisit when training. |
| **Sandbox backend** | **`harbor.environments.factory.EnvironmentFactory.create_environment(...)` — same code-path as `harbor run`.** No re-implementation. Sandbox type picked via `HARBOR_ENV_TYPE` env var (`docker | e2b | modal | …`). |
| **Bucket pull** | Via Harbor's `[environment.healthcheck]` calling `pull_bucket.py` — explicitly invoked with `await self._henv.run_healthcheck()`. Same script Stage 1 ships. |
| **Kernel** | `harbor_agents/kernel_server.py` uploaded into the Harbor container at session setup. Stateful Python via persistent globals over HTTP. Shared infra with `JupyterToolAgent`. |
| **Task source** | `harbor/tasks/jupyter-agent-<slug>/` folders on disk. `HARBOR_SUITE_DIR` env var controls the slug. Parsed via `harbor.models.task.task.Task(task_dir)`. |
| **Grader** | `rl/grader.py` shared with Stage 1's `tests/grader.py`. Exact + numeric tiers same. **LLM-judge tier upgraded to structured output via `AsyncOpenAI.beta.chat.completions.parse(response_format=JudgeOut)`** — no regex. |
| **Secrets** | OpenReward injects per-session via the `secrets` dict in `Environment.__init__`. Client side passes `secrets={"OPENAI_API_KEY": ..., "HF_TOKEN": ..., "E2B_API_KEY": ...}` when opening a session. |
| **Async** | All Harbor I/O is async (`start`, `stop`, `exec`, `upload_file`, `run_healthcheck`). Our `@tool` methods are `async def` accordingly. |
| **Harbor version** | `harbor==0.6.6` pinned. No declared API stability — bump deliberately, not auto. |

## What this unlocks once shipped

- **TRL/SkyRL training** — point the trainer at the ORS HTTP endpoint, treat each rollout as one episode against the env. Per-call rewards mean GRPO/PPO can shape behavior across tool calls, not just on the final answer.
- **Cross-platform eval** — same 5-tool surface as Harbor + the JupyterToolAgent. Numbers from `rollout_openai.py` (ORS) and `harbor run -a opencode` (Harbor) should match on the same task + model + grader, since they share `grader.py`.
- **Public env** — push the ORS server to HF Spaces and the eval becomes a single-URL service anyone can run rollouts against. (Out of scope this stage, but the design is the same.)

## Risks called out by research

These came out of the audit pass against `openrewardstandard/python-sdk` and `harbor-framework/harbor` source:

1. **No public adapter exists.** OpenReward's "Harbor mode" is a closed-source server-side feature on openreward.ai. We are writing the first public `Environment` subclass that imports `harbor.environments`. Building blocks are all public.
2. **Healthcheck is not auto-run by `start()`.** Easy to miss. Must explicitly `await env.run_healthcheck()` after `start()`, otherwise the agent runs before `pull_bucket.py` lands files.
3. **`TrialPaths` requires side-effects.** `TrialPaths(trial_dir=...)` is a frozen dataclass with only `trial_dir` required, BUT you MUST call:
   - `trial_paths.mkdir()` — creates `agent/`, `verifier/`, `artifacts/` subdirs (Docker bind-mounts fail without them).
   - `trial_paths.chmod_dir()` — only if `env.capabilities.mounted` is True (non-root users need to write to mounted dirs).
4. **`environment_dir` must already contain the Dockerfile + healthcheck files**. Use `task_dir/"environment"` from our Stage 1 layout — Harbor doesn't synthesize.
5. **`session_id` must be unique per concurrent session** — used as the container name. Generate via `uuid.uuid4().hex[:12]` per session.
6. **Use `EnvironmentFactory.create_environment_from_config`, not the raw `create_environment`** — `Trial.__init__` (Harbor's own caller) uses the `_from_config` variant. Mirror it exactly.
7. **All Harbor methods are async** — `@tool async def`, `await env.exec(...)`. OpenReward's `@tool` decorator transparently handles both sync and async via `await maybe_await(fn(inp))` (verified in `src/ors/environment.py`).
8. **Harbor version drift.** Pin `harbor==0.6.6`. Their pydantic models for `EnvironmentConfig` / `Task` / `TrialPaths` may break on minor releases.
9. **Reference repo `references/RL_Envs_101/envs/jupyter_env/ors/` uses the E2B Code Interpreter SDK directly.** We're *not* following that path — Harbor's factory abstracts E2B, Docker, Modal, Daytona behind one interface. Reference is for tool-shape only.

## The Trial.__init__ pattern we mirror

Verbatim from `harbor.trial.trial.Trial.__init__` (the canonical Harbor caller for `EnvironmentFactory`):

```python
# Harbor's own code — this is what `harbor run` does internally
self._trial_paths = TrialPaths(trial_dir=self.trial_dir)
self._trial_paths.mkdir()

self._environment = EnvironmentFactory.create_environment_from_config(
    config=config.environment,                       # trial-level EnvironmentConfig
    environment_dir=self._task.paths.environment_dir,  # <task_dir>/environment/
    environment_name=self._task.name,                # used as container name
    session_id=self.config.trial_name,               # unique per trial
    trial_paths=self._trial_paths,
    task_env_config=self._task.config.environment,   # the [environment] block from task.toml
    logger=self._logger,
)

if self._environment.capabilities.mounted:
    self._trial_paths.chmod_dir()
```

Our `env/server.py` does exactly this inside `Environment.__init__`. Plus `await self._henv.start(force_build=False)` + `await self._henv.run_healthcheck()` in `setup()`.

## Open questions

1. **Single ORS server per slug, or one server multiplexing all slugs?** Simplest: one server, slug selected by `HARBOR_SUITE_DIR` at startup. If we want a permanent service, multiplex via the `Split` API (`v1-train`, `v1-eval`, `v2-train`, ...).
2. **Sandbox type at runtime: `docker` or `e2b`?** Both work via `HARBOR_ENV_TYPE`. **Default: `docker`** for local dev (fast cold-start, free); **deploy mode: `e2b`** (scales horizontally for many parallel rollouts).
3. **One sandbox per session, or sandbox pool?** Per-session is simplest; pool is the optimization once we hit RL rollout throughput. Start per-session.
4. **Hosted deploy now or later?** Local-only first. Hosted is a 2-file diff (`Dockerfile` + GHA workflow) once the local server works.

## Sources

- [Deploying Harbor Environments — OpenReward](https://docs.openreward.ai/environments/deploying-harbor-environments)
- [Using LLM Graders — OpenReward](https://docs.openreward.ai/environments/using-llm-graders)
- [Your First Environment — OpenReward](https://docs.openreward.ai/environments/your-first-environment)
- [Harbor source — github.com/harbor-framework/harbor](https://github.com/harbor-framework/harbor)
- [`harbor==0.6.6` on PyPI](https://pypi.org/project/harbor/)
- [`openreward==0.1.81` on PyPI](https://pypi.org/project/openreward/)
- [Reference repo: RL_Envs_101 jupyter_env/ors](https://github.com/adithya-s-k/RL_Envs_101/tree/main/envs/jupyter_env/ors)
