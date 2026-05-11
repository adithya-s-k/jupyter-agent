# Jupyter Agent — Current Pipeline (iterate here)

Snapshot of where we are right now, what each script does, and what we've verified working.

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

## Open questions for you

1. Drop the local `data/<prefix>/` mirror? Saves ~150 MB local for the 5-task slice; bigger savings at N=1000.
2. Standardize on `--env e2b` for everything (consistency across local + cloud) or keep `--env docker` as the default for fast iteration?
3. Default model for our pass-rate baseline: `openai/gpt-5` or `anthropic/claude-sonnet-4-5`? (Either works; gpt-5 is cheaper, sonnet may be more reliable.)
4. Build the OpenReward server next (RL path), or do a wider Harbor eval first (5 tasks × N models)?
