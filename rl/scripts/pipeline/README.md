# Verification pipeline — what it does, end to end

A reference for the staged-verification CLI that lives under `rl/scripts/pipeline/`.
This document describes **what the pipeline does** — the phases, the tools, the
state model, the file artifacts. If you're trying to **run** it, see [`HF dataset
README`](https://huggingface.co/datasets/AdithyaSK/data_agent_rl_environment_eval) and
`data/verification/eval/REPORT.md` for an example run-through.

---

## Mental model in one paragraph

The pipeline takes a task — a `(question, gold_answer, kaggle_dataset, reward_mode)` row from
the manifest — and tries to produce a *verified verdict*: did the agent solve it, and if
so, how hard was it? It runs that task through up to four phases (build the scaffold,
have a Sonnet agent attempt it inside a Docker container, send the failure to a
Sonnet "doctor" if needed, assign a difficulty label on success). Every artifact
(LLM trajectory, grader output, cost, verdict) is logged as JSONL events; the
scaffold folder moves between `pending/ → verified/ | dropped/ | phase_b_failed/`
as the verdict crystallises.

```
manifest row → Phase A: build  → Phase B: anchor run    → Phase C: doctor (if B fails)
                                       ↓                          ↓
                                       ├── Phase B2 (after rewrite)
                                       ↓
                                       Phase D: categorize (if verified)
                                       ↓
                                       move scaffold to final bucket + write decisions row
```

The whole thing is invoked as **one CLI call** per batch of task IDs:

```bash
python -m scripts.pipeline run --stage 1 --ids-from <ids.json> ...
```

The `--stage` flag is a preset that bundles which phases are enabled.

---

## The four phases

### Phase A — `build_spec` (`build.py`)
- Reads the manifest row
- Renders a fresh Harbor task scaffold under `harbor/tasks/<suite>/pending/<task_id_safe>/`:
  - `task.toml` — Harbor task spec with `gold_answer`, `reward_mode_initial`, `package_tier`, `difficulty_level=0` (filled later)
  - `instruction.md` — the natural-language question
  - `environment/Dockerfile` + `pull_bucket.py`
  - `tests/test.sh` + `grader.py` (mode-aware)
- **Idempotent.** If the scaffold already exists in *any* bucket of the same suite
  (`pending/`/`verified/`/`dropped/`/`phase_b_failed/`), it's reused. Pass
  `--rewrite-spec` to force regeneration.

### Phase B — anchor trials (`_run_phase_b` in `orchestrator.py`)
- Up to `--k-max` trials of the **anchor model** (default `anthropic/claude-sonnet-4-6`).
  Adaptive: stops on the first pass. k=2 default catches single-attempt flakes.
- Each trial calls `verify.run_trial()` which shells out to `harbor run ... --env docker ...`.
- The agent runs inside a container, writes its final answer to `/workdir/answer.txt`,
  the verifier container runs `tests/test.sh` → `grader.py` → emits a reward.
- **Regrade fallback** (`regrade.py`): when a trial fails under the current reward_mode,
  the orchestrator re-grades the SAME trajectory under a looser mode (e.g. `numeric` →
  `flexible`, `flexible` → llm_judge via `--regrade-judge-model`). If the answer passes
  under the looser mode, the trial is **promoted** to passing (and `reward_mode_initial`
  in the task.toml is updated). No new LLM call to the agent.
- **Retries on transient errors** (`--max-retries-per-trial`, default 1): SandboxException,
  TimeoutException, 5xx — exponential backoff.

### Phase C — doctor (`doctor.py`, only if Phase B fails)
- A second LLM tool-loop, driven by the **doctor model** (default Sonnet).
- The doctor reads the failing trial(s) + spec + gold, then emits tool calls until
  it calls `finalize(verdict=..., reasoning=...)`. Available tools:
  | Tool | What it does |
  |---|---|
  | `read_file` | inspect any file in the task scaffold |
  | `preview_dataset` | quick sample of the Kaggle CSV/parquet |
  | `list_files` | list `/home/user/input/` inside the container |
  | `read_trajectory` | read the failing Phase B trial's trajectory.json |
  | `probe_with_model` | run a **mini Phase-B trial** with a different LLM (nano / opus / gpt-5.5 / gpt-5.5-codex / qwen / glm / kimi / deepseek). Returns the alt-answer for cross-check. |
  | `edit_task_toml` | rewrite spec fields (gold_answer, reward_mode, ATOL/RTOL, …) |
  | `edit_instruction` | rewrite `instruction.md` (rare — only when the prompt is misleading) |
  | `finalize` | declare verdict and end the session |
- Three budgets bound the doctor: `--doctor-max-calls` (default 20 tool calls),
  `--doctor-budget` (default $0.50 LLM spend), `--max-rewrites` (default 1, so only
  one spec rewrite + re-run is allowed).
- The doctor's possible `finalize` verdicts:
  - `spec_fixed` — used `edit_task_toml` / `edit_instruction`; orchestrator triggers Phase B2
  - `gold_corrected` — overwrote `gold_answer` with cross-model consensus; orchestrator
    triggers Phase B2 with the new gold
  - `verifiable_judge` — anchor's answer is semantically equivalent to gold (probes
    converged on the same answer; grader's strict mode was the issue). No Phase B2.
  - `unverifiable` — probes disagree → genuinely ambiguous question. Drop.

### Phase B2 — re-run after rewrite (only if doctor called spec_fixed / gold_corrected)
- Same loop as Phase B but with the edited `task.toml`. `rewrite_idx=1` in events.

### Phase D — categorize (`categorize.py`, only on verified-class verdicts)
- A single Sonnet rubric call (D1): reads the passing trajectory + question + gold,
  returns a JSON `{level: 1-5, reasoning, confidence, signal}` validated via Pydantic.
- Optional D2 empirical probe (`--empirical-probe`): runs `gpt-4o + seta + k=1` as a
  cross-check on intrinsic ease. Off by default.
- The level lands in `decisions.csv:difficulty_level` AND (when configured) is written
  back into the scaffold's `task.toml:[metadata].difficulty_level`.

---

## Verdict semantics

Final verdicts written to `decisions.csv:verdict`:

| Verdict | What happened |
|---|---|
| `verified` | Phase B passed (possibly via k=2 retry or regrade promotion) |
| `verified_gold_corrected` | Doctor rewrote `gold_answer`; Phase B2 then passed |
| `verified_after_rewrite` | Doctor rewrote `reward_mode` / spec; Phase B2 then passed |
| `verifiable_judge` | Doctor's probes confirmed anchor's answer ≡ gold via LLM judge |
| `dropped` | Doctor finalised `unverifiable`, OR task_timeout/budget exhausted |
| `phase_b_failed` | `--skip-doctor` was set and Phase B failed (Stage-1 cohort) |

The scaffold folder moves to match: `verified/` for the first 4, `dropped/` for the
5th, `phase_b_failed/` for the 6th.

---

## Stages — what `--stage 1` vs `--stage 2` does

`--stage` is a **preset** that toggles several existing flags in one go.

### `--stage 1` — anchor screen
```python
args.skip_doctor    = True       # no Phase C
args.skip_categorize = False     # Phase D fires on passes
```
- One Sonnet anchor pass (k=2 retries inside Phase B)
- Pass → categorize → moves to `verified/` with `difficulty_level` set
- Fail → moves to `phase_b_failed/` (Stage 2 will pick these up)

Use this on **fresh task IDs**. Cheap (~$0.05–0.10/task).

### `--stage 2` — doctor pass
```python
args.skip_doctor    = False      # Phase C enabled
args.skip_categorize = False     # Phase D on recovery
args.probe_aliases  = ['nano','gpt-5.5']   # ←  if not already set
```
- Reads `--ids-from <phase_b_failed_ids>` (you pass the list of Stage-1 failures)
- Does another Phase B (catches additional flake-rescues beyond Stage 1's k=2 — see
  `data/verification/eval/REPORT.md §7` for why this is often a net win)
- On fail: doctor diagnoses, may rewrite/correct/drop
- On recovery (incl. doctor's `spec_fixed`/`gold_corrected` paths): Phase B2 + categorize
- Lean probe roster: nano + gpt-5.5 (Opus + deepseek excluded — they cost 5-10 min/probe
  and don't materially change recovery rate). Per-probe wall capped via `--probe-timeout-sec`
  (default 180s).

Cost roughly $0.15–0.30/task depending on how often the doctor fires.

---

## CLI flag reference

```bash
python -m scripts.pipeline run [...]
```

### Input selection
| Flag | Default | Notes |
|---|---|---|
| `--manifest` | `data/splits/eval_manifest.parquet` | source of `(task_id, question, answer, kaggle, reward_mode, ...)` |
| `--ids` | none | space-separated task IDs (overrides --from/--to/--limit) |
| `--ids-from` | none | file with one id per line OR JSON array |
| `--from` / `--to` / `--limit` | 0 / end / none | row-range slice of the manifest |
| `--suite` | `data-agent-eval-v1` | scaffold subfolder under `harbor/tasks/` |
| `--state-dir` | `data/verification/eval` | where this run's `state.jsonl` + `decisions.csv` land |
| `--resume` | off | skip IDs already terminal in the cross-run rollup decisions.csv |
| `--rewrite-spec` | off | force Phase A to regenerate scaffolds even when they exist |

### Anchor / Phase B
| Flag | Default | |
|---|---|---|
| `--model` | `anthropic/claude-sonnet-4-6` | anchor LLM |
| `--k-max` | 2 | retries on flaky failure before doctor |
| `--sandbox` | `docker` | `docker` \| `e2b` |
| `--subprocess-timeout-sec` | 900 | outer wall-cap on each `harbor run` call |
| `--max-retries-per-trial` | 1 | retries on transient SandboxException / 5xx |
| `--regrade-judge-model` | `openai/gpt-5.4-nano` | LLM judge for the auto-regrade fallback |

### Doctor / Phase C
| Flag | Default | |
|---|---|---|
| `--skip-doctor` | off | hard-disable Phase C |
| `--doctor-model` | `anthropic/claude-sonnet-4-6` | doctor's brain |
| `--doctor-budget` | 0.50 | $ cap per doctor session |
| `--doctor-max-calls` | 20 | tool-call cap per doctor session |
| `--probe-aliases` | none = all 8 | restrict probe roster (e.g. `nano gpt-5.5`) |
| `--probe-timeout-sec` | 180 | per-probe wall cap (independent of `--subprocess-timeout-sec`) |
| `--max-rewrites` | 1 | how many times doctor may trigger Phase B2 |

### Categorize / Phase D
| Flag | Default | |
|---|---|---|
| `--skip-categorize` | off | hard-disable Phase D |
| `--categorize-model` | `anthropic/claude-sonnet-4-6` | rubric judge |
| `--empirical-probe` | off | run gpt-4o + seta + k=1 as a cheap intrinsic-difficulty cross-check |

### Run control
| Flag | Default | |
|---|---|---|
| `--concurrent` | 1 | parallel task workers (ThreadPoolExecutor) |
| `--task-timeout-sec` | 1500 | orchestrator-level soft cap per task (includes all phases) |
| `--stagger-sec` | 0.5 | sleep between submitting initial workers — smooths sandbox-creation pressure |
| `--total-cost-cap` | none | $ cumulative cap; halts new tasks (does not interrupt running ones) |

### Presets
| Flag | Default | |
|---|---|---|
| `--stage {1,2}` | none | bundle of the above (see Stages section) |
| `--all-oss` | off | swap every closed-source default to `Qwen3-235B`, probes to `{qwen,glm,kimi,deepseek}` |

---

## Input format

### Manifest parquet schema (one row per task)
| Column | Type | |
|---|---|---|
| `id` | string | canonical task_id (e.g. `0000/419/419825.ipynb_qa_1`) |
| `question` | string | natural-language question for the agent |
| `answer` | string | gold answer |
| `kaggle_dataset_name` | string | e.g. `nolanbconaway/whatcd-hiphop` — used by `pull_bucket.py` |
| `executor_type` | string | always `e2b` post-filter |
| `files_used` | list[str] | which CSVs the original notebook used |
| `packages_used` | list[str] | numpy/pandas/sklearn/etc. (informational) |
| `reward_mode_initial` | string | `exact_short` \| `numeric` \| `exact_bool` \| `flexible` \| `list` \| `list_csv` \| `llm_judge_long` |
| `package_tier` | int | 0-3 difficulty proxy (0 = stdlib, 3 = ML libs) |
| (others) | | edu_score, answer_len, q_word_count, … informational |

### IDs file (`--ids-from`)
- One ID per line, OR a JSON array of strings
- IDs are matched against `manifest.id` — rows not in the manifest are silently skipped

---

## Output artifacts

Per-run directory: `<state-dir>/runs/<UTC_timestamp>/`

| Path | What it is |
|---|---|
| `state.jsonl` | append-only event log — **source of truth** for in-flight monitoring |
| `decisions.csv` | one row per task in THIS run with the final verdict + difficulty + cost rollup |
| `cost.jsonl` | per-LLM-call cost events (for fine-grained accounting) |
| `trials/<job_name>/<task_id_safe>__<random>/` | one dir per Phase B / probe / Phase B2 trial — contains `agent/trajectory.json`, `verifier/reward.txt`, `verifier/test-stdout.txt`, `result.json` |
| `cli_args.json` | snapshot of the CLI flags this run was invoked with |
| `summary.json` | end-of-run aggregate (verified count, total cost, etc.) |

Cross-run rollup at `<state-dir>/decisions.csv` — pipeline rebuilds it from all `runs/*/state.jsonl` files at the end of each invocation. **Manual edits to this file get overwritten on the next run.**

### `state.jsonl` event vocabulary
| Event | When | Notable fields |
|---|---|---|
| `run_start` | once per CLI invocation | cli_args |
| `task_start` | per task | task_id |
| `spec_built` | Phase A | spec_dir |
| `trial_start` | each trial submission | task_id, model, k_attempt, rewrite_idx, job_name |
| `trial_finish` | each trial completion | reward, predicted, error_kind, cost_usd, prompt_tokens, completion_tokens, cached_tokens, trial_dir |
| `regrade_promoted` | regrade fallback fired | regrade_mode, new_reward |
| `doctor_start` / `doctor_turn` / `doctor_tool` / `doctor_finish` | Phase C lifecycle | tool_name, verdict, reasoning |
| `probe_start` / `probe_finish` | each probe spawned by doctor | model, reward, predicted, cost_usd |
| `categorize_finish` | Phase D | level, confidence, reasoning, signal |
| `task_finish` | per task | verdict, total_cost_usd, total_trials |

### `decisions.csv` schema (one row per task)
| Column | What |
|---|---|
| `task_id`, `verdict` | identity + final result |
| `passing_round` | `B` / `B2` / `C-judge` — which path produced the pass |
| `passing_model`, `passing_k`, `passing_predicted` | which trial passed |
| `gold_corrected`, `gold_original` | did doctor overwrite gold? |
| `spec_rewrite_count`, `doctor_verdict`, `doctor_reasoning` | doctor's actions + outcome |
| `difficulty_level`, `difficulty_confidence`, `difficulty_reasoning`, `difficulty_signal` | Phase D output |
| `empirical_easy` | optional D2 cross-check result |
| `total_cost_usd`, `phase_b_cost_usd`, `doctor_cost_usd`, `probe_cost_usd`, `categorize_cost_usd` | per-phase $ split |
| `total_trials`, `prompt_tokens`, `completion_tokens`, `cached_tokens` | usage |
| `run_id`, `ts_updated` | provenance |

---

## Scaffold state machine

Every task lives in exactly one of four subfolders under `harbor/tasks/<suite>/`:

```
            ┌─────────────┐
            │  pending/   │  ← new scaffolds land here (Phase A)
            └──────┬──────┘
                   │
        ┌──────────┼───────────────────────────────┐
        │          │                               │
    passing     phase_b_failed                doctor → dropped
        ▼          ▼                               ▼
   ┌──────────┐  ┌──────────────────┐         ┌──────────┐
   │verified/ │  │ phase_b_failed/  │ ─────→  │ dropped/ │
   └──────────┘  │ (Stage-2 queue)  │         └──────────┘
                 └──────────────────┘
                          │
                  Stage 2 doctor recovery
                          ▼
                     verified/ (verdict carries: gold_corrected,
                                after_rewrite, or verifiable_judge)
```

`build.py`'s `_resolve_task_dir()` searches all four subfolders, so re-running Stage 1
on the same task ID is idempotent — the existing scaffold gets reused, and only
Phase B/C/D run again (and even Phase B can be skipped via `--resume` if the row is
already terminal in the cross-run decisions.csv).

---

## Concurrency model

- `cli.py:cmd_run` uses `concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent)`.
- Each worker calls `orchestrator.process_task(row, cfg, ...)` synchronously.
- Inside a task, `verify.run_trial()` is a blocking `subprocess.run(['harbor','run',...])`.
- `--stagger-sec` puts a sleep between the first N submissions to avoid sandbox-creation
  thundering-herd issues on E2B. Not needed for Docker (but harmless).
- `--total-cost-cap` is enforced via a shared `threading.Lock`-protected counter; new
  task submissions stop when the cumulative spend crosses the cap (running tasks
  finish normally).

**Probes inside the doctor are sequential** — the doctor LLM emits one `probe_with_model`
call per turn, the orchestrator runs it, then the LLM picks the next action. To parallelise
probes you'd need to (a) prompt the doctor to emit multiple probe calls per turn and
(b) dispatch `asyncio.gather()` in the tool-call executor. Not yet implemented.

---

## Cost accounting

Every LLM call routes through `llm_client.call()` which:
- Computes per-call cost from the model's known pricing
- Logs the call to `cost.jsonl` with `task_id`, `phase`, `call_kind`, `prompt_tokens`,
  `completion_tokens`, `cached_tokens`, `cost_usd`
- Increments the StateStore's running totals

End-of-run rollup splits cost into:
- `phase_b_cost_usd` — anchor seta-loop trials (the dominant line item, ~⅔ of total)
- `doctor_cost_usd` — doctor brain calls (cheap; high cache-hit rate)
- `probe_cost_usd` — probes (nano + gpt-5.5 + opus + deepseek as configured)
- `categorize_cost_usd` — Phase D rubric (one cheap call per verified task)

The eval cost benchmark from `data/verification/eval/REPORT.md`: **~$0.20 per verified task**
end-to-end (Stage 1 + Stage 2 combined on the 500-task pool).

---

## Subcommands

```bash
python -m scripts.pipeline {run, ping, migrate-buckets} [...]
```

| Subcmd | What it does |
|---|---|
| `run` | the main A→D pipeline described above |
| `ping` | sanity-check connectivity + tool-calling for one or more models (no Harbor, no Docker) |
| `migrate-buckets` | one-shot to convert a flat suite (`<suite>/<task_id>/`) into the modern `pending/verified/dropped/phase_b_failed/` layout |

---

## Common invocations

```bash
# Smoke-test the CLI on a single task
python -m scripts.pipeline run --stage 1 --sandbox docker --concurrent 1 \
  --ids 0000/419/419825.ipynb_qa_1 \
  --suite data-agent-eval-v1 --state-dir data/verification/eval

# Stage 1 batch — Sonnet anchor + categorize-on-pass
python -m scripts.pipeline run --stage 1 --sandbox docker --concurrent 30 \
  --ids-from cache/batch_ids.json \
  --suite data-agent-eval-v1 --state-dir data/verification/eval \
  --subprocess-timeout-sec 900 --task-timeout-sec 1200 \
  --stagger-sec 2.0 --max-retries-per-trial 1

# Stage 2 — doctor on Stage-1 failures (filter the failures from the prior run's state.jsonl)
python -m scripts.pipeline run --stage 2 --sandbox docker --concurrent 30 \
  --ids-from cache/stage2_ids.json \
  --suite data-agent-eval-v1 --state-dir data/verification/eval \
  --subprocess-timeout-sec 900 --task-timeout-sec 1800

# Same task, full A→D + categorize, OSS-only models (Qwen3-235B everywhere)
python -m scripts.pipeline run --all-oss --concurrent 30 \
  --ids-from cache/batch_ids.json --suite data-agent-eval-v1

# Sanity ping
python -m scripts.pipeline ping \
  --models anthropic/claude-sonnet-4-6 openai/gpt-5.4-nano
```

---

## Layout — files and what they own

```
scripts/pipeline/
├── cli.py             — argparse + ThreadPoolExecutor + `--stage` preset
├── orchestrator.py    — RunConfig + process_task() = Phase A→D state machine
├── build.py           — Phase A: scaffold generator with task.toml template
├── verify.py          — wraps `harbor run` subprocess; transient-error retry
├── regrade.py         — re-grade a failed trial under a looser reward_mode
├── doctor.py          — Phase C: LLM tool-loop + ALLOWED_PROBE_MODELS + 8 tools
├── categorize.py      — Phase D: rubric (D1) + optional empirical probe (D2)
├── llm_client.py      — multi-provider LLM call + cost accounting
├── state.py           — StateStore: state.jsonl append + decisions.csv upsert
└── prompts/
    ├── doctor.md      — doctor's system prompt
    └── categorize.md  — rubric system prompt
```
