# Stage 2 — Verification & Difficulty Pipeline

> The most important stage. We iterate here on a single task → 10 → 100 → 1000 until the pipeline is robust enough to run unattended on 28.5k train rows. Designed for fast iteration: one entry point, locally-stored state, resumable, tight cost loop.

---

## Goal

Take a `(question, gold_answer, kaggle_dataset)` row → produce a Harbor spec → **verify it end-to-end** with Sonnet 4.6 + seta → if it fails, an agentic doctor (also Sonnet) with file-editing tools tries to fix the spec or correct the gold via cross-model consensus → categorize as a 1-5 difficulty level. Output per row: `{ verifiable, gold_corrected, spec_rewrite_count, dropped, difficulty_1_5, total_cost }`.

Three things must work consistently before we declare the pipeline robust:

1. **Spec generation** — every row produces a Harbor spec that builds, healthchecks, and grades correctly when the right answer is written to `/workdir/answer.txt`.
2. **Verification** — Sonnet 4.6 + seta passes within ≤3 attempts, OR the agentic doctor recovers via spec-fix / gold-correction, OR the row is explicitly dropped with a reason.
3. **Categorization** — same task assigned the same 1-5 level across two independent runs (≥85% exact-match on a 50-task smoke test).

---

## Locked decisions

### Frontier roster (no harness sweep)

| Role | Model ID | $/M tok in | $/M tok out | When used |
|---|---|---|---|---|
| Anchor (verify + doctor + categorize) | `anthropic/claude-sonnet-4-6` | $3 | $15 | Every task; also runs the doctor loop |
| Escalation #1 | `openai/gpt-5.5-2026-04-23` | ~$5 | ~$30 | Doctor's `probe_with_model` when Sonnet's diagnosis is inconclusive |
| Escalation #2 | `anthropic/claude-opus-4-7` | $5 | $25 | Doctor's `probe_with_model` for "Sonnet still failed after rewrite" |
| Code specialist | `openai/gpt-5.5-codex` | ~$5 | ~$30 | Doctor's `probe_with_model` for code-heavy tasks (>15 turns failure pattern). Confirmed via LiteLLM Day-0 support listing + OpenAI Codex models page. |
| Cheap probe | `openai/gpt-4o` | ~$2.50 | ~$10 | Difficulty empirical signal (D2) |

Doctor's escalation order on `probe_with_model`: **Sonnet → GPT-5.5 → Opus → Codex**. Sonnet is also the doctor itself. The doctor decides which escalation step is appropriate per task (it has visibility into the failing trajectory).

### Harness

**`seta` only.** No harness sweep — the v2 data showed seta has the highest pass rate (21% vs jupy 14% vs bash 9%) and a Sonnet failure on seta is unlikely to be a tool-design issue.

### Sandbox

`e2b` default. `--sandbox local` (docker) for offline single-task iteration.

### Storage policy

**Local-only.** Everything lives under `data/verification/<split>/`. No external publishing during this stage — we'll figure out where the verified set goes once we have one.

### LiteLLM gateway

All model calls go through one wrapper that:
- Picks provider from the model string (`anthropic/...`, `openai/...`)
- For Anthropic models, structures the system message as a list of text content blocks and sets `cache_control: {"type": "ephemeral"}` on the stable prefix block (the system prompt + rubric/checklist). Per-task content goes in a separate, non-cached user block. Optionally use LiteLLM's `cache_control_injection_points` to auto-inject.
- Records per-call cost + token counts via a `CustomLogger` callback to `data/verification/<split>/cost.jsonl`

---

## Per-task flow (the unit of work)

```
┌─ PHASE A: BUILD SPEC ───────────────────────────────────────┐
│ row → harbor/tasks/data-agent-eval-v1/<id_safe>/             │
│   task.toml, instruction.md, tests/test.sh,                  │
│   environment/Dockerfile + pull_bucket.py                   │
│ Validate: docker build, healthcheck, grader-with-gold=1.0   │
│ Fail here → log spec_build_error, skip                      │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─ PHASE B: VERIFY (Sonnet 4.6 + seta, adaptive K up to 3) ──┐
│ trial 1 → pass → DONE → Phase D                             │
│        → fail → trial 2 (prompt-cached)                     │
│              → fail → trial 3 (prompt-cached)               │
│                    → fail → Phase C                         │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─ PHASE C: AGENTIC DOCTOR (tool-using LLM) ──────────────────┐
│ Sonnet 4.6 (same as anchor) with these tools, run in a loop │
│ until it calls `finalize`:                                  │
│   read_file(path)         peek at task.toml / instruction.md │
│   preview_dataset(file)   head N rows of a CSV/SQLite        │
│   list_files()            see the bucket contents            │
│   read_trajectory(trial)  see what the failing agent did     │
│   probe_with_model(model) run a single seta trial w/ another │
│                          model to gather alt-answers         │
│   edit_task_toml(field, value)   change REWARD_MODE / ATOL /  │
│                                  EXPECTED_ANSWER / QUESTION  │
│   edit_instruction(old, new)     fix the instruction prompt  │
│   finalize(verdict, reasoning)   end the loop                │
│                                                              │
│ Verdicts (terminal):                                         │
│   spec_fixed       — doctor edited spec; pipeline re-runs   │
│                      Phase B; max 1 rewrite per task        │
│   gold_corrected   — doctor verified via consensus that     │
│                      gold was wrong, replaced it; re-runs   │
│                      Phase B (existing trials re-graded)    │
│   verifiable_judge — Sonnet kept failing, but ≥2 probe-    │
│                      models (from gpt-5.5 / opus / codex)   │
│                      passed with answers matching the       │
│                      existing gold. Mark verified.          │
│   unverifiable     — doctor concludes task is ambiguous /   │
│                      dataset mismatch / answer unreachable. │
│                      Drop with reason.                      │
│                                                              │
│ Sandbox: doctor's edits scoped to the task's own folder.    │
│ A v0 snapshot is taken before doctor runs.                  │
│ Budget: max 20 tool calls per task, $0.50 hard cap.         │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─ PHASE D: CATEGORIZE (only if verified) ────────────────────┐
│ D1. RUBRIC JUDGE (Sonnet 4.6, 1 call) — 1-5 level w/ rubric │
│     Input: question + gold + 1 passing trajectory excerpt    │
│     Output: {level: 1-5, reasoning, confidence}              │
│                                                              │
│ D2. EMPIRICAL PROBE (gpt-4o + seta, K=1) — cheap            │
│     If gpt-4o solves the task → empirical_easy = true       │
│                                                              │
│ Final level = D1. Log D2 as `empirical_easy_consistent`     │
│ if (level ≤ 2 and empirical_easy=true) or (level ≥ 3 and    │
│ empirical_easy=false). Flag inconsistencies for spot-check. │
└─────────────────────────────────────────────────────────────┘
```

---

## Difficulty levels (1-5) — locked rubric

Stored at `rl/scripts/pipeline/prompts/categorize.md`. Every judge call uses this exact rubric to keep classification consistent across runs.

```
1 — Trivial
    • Single column lookup, simple aggregation (min/max/mean/count)
    • ≤2 meaningful cells of code
    • No groupby, no joins, no transformation beyond .max() / .count()
    Examples: "What is the highest votes?" "How many rows are NaN?"

2 — Simple
    • Single-file dataframe operation with a filter and an aggregation
    • Boolean / percentage computation with a single condition
    • ≤4 cells
    • No multi-step transformation
    Examples: "What % of users are aged > 30?" "Most common category?"

3 — Moderate
    • Groupby / pivot / sort + aggregation
    • Two-step transformation (filter → group → top-k)
    • Up to 1 join across at most 2 files
    • Simple statistics (correlation, basic descriptives)
    • ≤8 cells
    Examples: "Which year had the highest mean revenue per group?"

4 — Complex
    • Multi-file join with cleaning
    • Non-trivial feature engineering (encoding, scaling, binning)
    • Basic ML (one model train, default hyperparams)
    • Statistical inference (CI, p-value, hypothesis test)
    • Time-series ops (resample, rolling)
    • 8-15 cells
    Examples: "Train logistic regression and report test accuracy."

5 — Hard
    • Multi-step ML pipeline (feature eng + multiple model train + comparison)
    • Deep-learning training or fine-tuning
    • Multi-file research-grade analysis with non-trivial cleaning
    • Time-series forecasting
    • >15 cells, or requires non-trivial library usage (transformers, torch, etc.)
    Examples: "Train a CNN and report top-5 accuracy on the test set."

Output JSON:
  {"level": <1-5>, "reasoning": "<one sentence>",
   "confidence": <0.0-1.0>,
   "signal": "what tipped you off"}
```

---

## Pipeline folder structure

```
rl/scripts/pipeline/
├── __init__.py
├── __main__.py              # CLI entry: `python -m scripts.pipeline ...`
├── cli.py                   # argparse, subcommand dispatch
├── orchestrator.py          # per-task flow: Phase A→B→C→D
├── state.py                 # state.jsonl + decisions.parquet I/O
├── select.py                # --ids / --from / --to / --limit / --resume logic
├── build.py                 # Phase A
├── verify.py                # Phase B (Sonnet+seta loop, adaptive K)
├── doctor.py                # Phase C — tool-using diagnostician + fixer
├── categorize.py            # Phase D
├── llm_client.py            # LiteLLM wrapper + cost callback + cache_control
├── viz.py                   # builds dashboard from state.jsonl
└── prompts/
    ├── doctor_system.md
    ├── categorize.md        # the 1-5 rubric above
    └── grader_judge.md      # llm-judge grader prompt

rl/harbor/
├── grader/                  # NEW — runs inside Harbor container at /opt/grader/
│   ├── grader.py            # multi-mode dispatcher reading REWARD_MODE env
│   ├── modes.py             # numeric / exact / list / flexible / judge etc.
│   └── prompts/
│       └── judge.md
├── agents/                  # renamed from harbor_agents/
│   ├── _shared/
│   │   ├── llm_client.py    # the same wrapper, reused inside agents
│   │   └── ...
│   ├── seta/                # primary harness
│   ├── jupy/                # kept for ablation
│   └── bash/                # kept for ablation
└── tasks/
    └── data-agent-eval-v1/  # generated by Phase A
```

---

## CLI surface

```
python -m scripts.pipeline <cmd> [flags]

Commands
  run          full pipeline per task: A → B → C → D
  build        Phase A only — generate Harbor specs
  verify       Phase B only — re-run verification on existing specs
  doctor       Phase C only — invoke doctor on failed tasks
  categorize   Phase D only
  show         inspect state.jsonl / decisions.parquet for given IDs
  viz          regenerate dashboard
  smoke        categorization-consistency check (50 IDs run 2x)

Selection (every cmd)
  --dataset DATASET           HF repo id OR local parquet (default: local manifest)
  --split eval|train          default eval
  --from N                    start index (0-based, default 0)
  --to N                      end index exclusive
  --limit N                   alias for --from N --to N+limit
  --ids ID [ID ...]           specific IDs; overrides --from/--to/--limit

State
  --state-dir PATH            default data/verification/<split>/
  --resume                    skip IDs whose decisions.parquet row is terminal
  --rewrite                   force redo; snapshots state.jsonl → .bak.<ts>

Execution
  --concurrent N              parallel task workers (default 4)
  --sandbox e2b|local         default e2b
  --k-max 3                   adaptive K upper bound
  --grader-mode auto|MODE     auto = classifier's choice
  --max-rewrites N            doctor's per-task spec rewrite budget (default 1)
  --doctor-budget USD         doctor's hard cost cap per task (default 0.50)
  --doctor-max-calls N        doctor's per-task tool-call cap (default 20)
  --doctor-temperature F      doctor's LLM temperature (default 0 for reproducibility)
  --total-cost-cap USD        circuit-breaker on cumulative LLM spend across
                              the whole sweep (default unset). Hits only
                              count LLM API spend — sandbox-time / E2B credit
                              is NOT capped here (those are per-trial budgets
                              enforced by Harbor's timeout_sec).
  --skip-categorize           skip Phase D
  --doctor-model MODEL        default anthropic/claude-sonnet-4-6
  --judge-model MODEL         for grader llm-judge mode; default gpt-4o
  --dry-run                   print plan, run nothing
  --log-level info|debug
```

### Iteration ladder

```bash
# M1 — one task
python -m scripts.pipeline run --limit 1

# M2 — small batch, debug
python -m scripts.pipeline run --from 0 --to 10 --concurrent 2 --log-level debug

# M3 — 100, resume-safe
python -m scripts.pipeline run --limit 100 --resume --concurrent 4

# inspect a failure
python -m scripts.pipeline show --ids 0001/239/1239804.ipynb_qa_4

# rerun verification only (e.g., after grader-mode fix)
python -m scripts.pipeline verify --limit 100 --rewrite

# full sweep
python -m scripts.pipeline run --limit 1000 --resume --concurrent 8

# categorization consistency
python -m scripts.pipeline smoke --n 50
```

---

## State & storage

```
data/verification/eval/
├── state.jsonl              # APPEND-ONLY single source of truth
│   # {ts, event, task_id, phase, model, harness, k_attempt,
│   #  reward, predicted, error_kind, elapsed_sec,
│   #  trial_dir, cost_usd, prompt_tokens, completion_tokens, cached_tokens}
│
├── cost.jsonl               # per-call cost from LiteLLM callback
│   # {ts, task_id, phase, model, cost_usd, prompt_tokens,
│   #  cached_tokens, completion_tokens, elapsed_sec, call_kind}
│
├── decisions.parquet        # FINAL per-task summary. One row per task_id.
│   # Maintained as an in-process dict during a run; written to parquet
│   # on every flush (every N tasks, default 25) AND on shutdown. Cheap
│   # at our scale (≤30k rows × ~20 cols ≈ a few MB).
│   # Columns: {task_id, verdict, reward_mode_final, gold_corrected,
│   #  gold_original, spec_rewrite_count, doctor_verdict, doctor_tool_calls,
│   #  difficulty_level (1-5), difficulty_confidence, difficulty_reasoning,
│   #  empirical_easy, total_cost_usd, total_trials, ts_final}
│
├── trials/<id_safe>__<phase>__<model>__<k>/
│   # Harbor's per-trial dirs
│
├── specs/<task_id>/
│   ├── v0.toml              # original (snapshot before doctor)
│   ├── v1.toml              # after first rewrite (if any)
│   └── doctor_log.jsonl     # doctor's tool-call trajectory
│
├── corrections/<task_id>.json   # {original_gold, new_gold, agreeing_models}
│
└── reports/
    ├── verified_ids.txt
    ├── dropped_ids.txt
    └── viz.html
```

**Resume semantics.** On `--resume`: read `decisions.parquet`, build the set of task_ids with terminal verdicts, skip them.

**Rewrite semantics.** `--rewrite` for a task: snapshot affected state.jsonl lines to `state.jsonl.bak.<ts>`, drop the row from decisions.parquet, restart Phase A (which preserves trials/ as evidence but writes new ones).

---

## Doctor design (Phase C deep-dive)

The doctor is a tool-using agent that runs on Sonnet 4.6 (same model as the anchor — keeps the system simple and Sonnet is plenty capable for diagnostic work; we only escalate when *it* explicitly chooses to via `probe_with_model`). Its system prompt (locked in `prompts/doctor_system.md`):

```
You are a task-spec doctor. A frontier model (Sonnet 4.6 + seta harness)
failed 3 times on this task. Your job: figure out why, fix it if possible,
or decide it's unrecoverable.

You have read access to: task.toml, instruction.md, the dataset files (head
only), and the failing agent's trajectory. You have write access ONLY to
this task's spec directory.

Common patterns to check (in order):
1. Is REWARD_MODE too strict? If the agent's predicted answer is
   semantically right but the grader rejected it (e.g., "12.5%" vs "12.5"),
   call edit_task_toml to change REWARD_MODE to "flexible" or "llm-judge",
   then finalize with verdict=spec_fixed.
2. Is the EXPECTED_ANSWER actually derivable from the dataset?
   Use preview_dataset to sanity-check. If unclear, use probe_with_model
   to have a second model attempt the task.
3. Is the gold simply wrong? Use probe_with_model — start with GPT-5.5;
   if GPT-5.5 also disagrees with the gold, run Opus too; if the task
   looks code-heavy (the failing trajectory has >15 turns of code/debug),
   also probe with Codex. If 2+ models converge on a different answer
   that is reproducibly derivable, call edit_task_toml to update
   EXPECTED_ANSWER, then finalize with verdict=gold_corrected.
4. Is the question ambiguous? If the failing trajectory's interpretations
   diverge in plausible ways, either edit_instruction to disambiguate,
   or finalize with verdict=unverifiable and reason=ambiguous_question.
5. Is the dataset wrong? Files don't contain the required columns/data.
   Mark unverifiable with reason=dataset_mismatch.

Escalation policy for probe_with_model:
  - default: gpt-5.5  (one model is usually enough to confirm/deny gold)
  - if gpt-5.5 disagrees with gold but you're unsure: try opus
  - if the failure pattern is code-heavy (long trajectory, lots of
    print/debug, tool failures): try gpt-5.5-codex
  - do not probe with sonnet (we already know it failed)

Budget: 20 tool calls and $0.50 maximum. Be efficient. Call finalize as
soon as you have enough information.
```

### Doctor tool set

### What the doctor sees up-front (turn 0)

Before the doctor makes its first tool call, the user message contains a **full run dossier** so it has the entire context without having to ask:

```
TASK ID:          <id>
QUESTION:         <question, verbatim>
GOLD ANSWER:      <answer from manifest>
KAGGLE DATASET:   <name + bucket prefix + total file count>
FILES USED:       <list from manifest>
REWARD_MODE:      <current value in task.toml>
CLASSIFIER:       <reward_mode_initial, answer_type, package_tier>

SPEC FILES:
  task.toml:                   <inline, full content>
  instruction.md:              <inline, full content>
  tests/test.sh:               <inline if non-default>

DATASET PREVIEW (head 5 of each file):
  <file1>: <first 5 rows or 200 chars>
  <file2>: ...

PHASE B FAILURES (all 3 trials, oldest first):
  TRIAL 1 [model=sonnet, k=1, elapsed=Xs, cost=$Y]
    PREDICTED:    <what the agent wrote to /workdir/answer.txt>
    GRADER:       <reward.txt + test-stdout.txt verbatim>
    LAST 4 TURNS: <abridged agent trajectory>
    FULL TRAJECTORY available via read_trajectory(1).
  TRIAL 2 [...]: ...
  TRIAL 3 [...]: ...
```

This avoids the first 5 read_file / read_trajectory calls being mechanical — the doctor jumps straight to diagnosis.

### Doctor tool set (for follow-up exploration)

| Tool | Purpose | Side effects |
|---|---|---|
| `read_file(path)` | Re-read task spec file at any point (e.g., after an edit) | none |
| `preview_dataset(file, n=20)` | Show more rows than the dossier preview, or specific columns | none |
| `list_files()` | List files under bucket prefix | none |
| `read_trajectory(trial_idx)` | Full per-trial dir contents: complete agent trajectory + `verifier/reward.txt` + `verifier/test-stdout.txt` + `verifier/reward.json` + `agent/<X>.usage.json`. `trial_idx` ∈ {0, 1, 2} for Phase B trials, or returned by `probe_with_model`. | none |
| `probe_with_model(model)` | Run **synchronously** — one seta trial with the named model (allowed: gpt-5.5, opus, gpt-5.5-codex; not sonnet — already tried). Returns trial_idx + predicted answer + reward. Blocks doctor's tool loop for ~30-90s while the sandbox runs. | spawns a Harbor trial, costs $$ |
| `edit_task_toml(field, value)` | Edit one field in `[verifier.env]` (REWARD_MODE / EXPECTED_ANSWER / ATOL / RTOL / ANSWER_TYPE). Parses the file after edit; reverts if invalid TOML. | writes v1.toml |
| `edit_instruction(old_str, new_str)` | Edit instruction.md | writes v1 instruction |
| `finalize(verdict, reasoning)` | End the loop | terminal |

After `finalize`:
- If `verdict in {spec_fixed, gold_corrected}` → re-run Phase B once (max 1 rewrite per task)
- If `verdict == verifiable_judge` → mark verified directly (used when 2+ models converge)
- If `verdict == unverifiable` → mark dropped with the reasoning as the reason

Doctor's tool-call trajectory persisted at `specs/<task_id>/doctor_log.jsonl`. Easy to spot-check what the doctor actually did.

### Doctor cost containment

- Per-task hard cap: $0.50 (configurable via `--doctor-budget`)
- Per-task call cap: 20 tool calls (configurable via `--doctor-max-calls`)
- If cap hit, doctor is force-finalized with verdict=unverifiable, reason=doctor_budget_exhausted
- Cached system prompt across doctors (Sonnet, 1h TTL) — first 5 tasks pay, rest get ~95% cache hit
- `probe_with_model` is the doctor's expensive lever; budget governs how many escalations it can afford. Typical doctor session: 0-2 probes ≈ $0.05-0.30. Hard cap stops runaway escalations.

---

## Prompt caching plan

| Site | Cache | Expected hit rate |
|---|---|---|
| Sonnet seta agent (Phase B trials 2, 3) | system + instruction | 95% on retries (same task) |
| Sonnet seta agent (across tasks) | system only (instruction differs) | 80%+ on system portion |
| Sonnet doctor (across tasks) | system + diagnostic checklist | 95% after first 5 |
| Sonnet categorize (across tasks) | system + 1-5 rubric | 99% after first 5 |
| gpt-4o empirical probe (across tasks) | OpenAI auto-cache if prompt ≥1024 tok | passive |
| LLM-judge grader (across tasks) | system | 90%+ |

`llm_client.py` structures Anthropic calls so the **stable prefix is its own content block inside the system message**, marked with `cache_control: {"type": "ephemeral", "ttl": "1h"}`. Per-task variable content (task ID, instruction, trajectory) is a separate user block that is NOT cached. We verify cache is firing by logging `cached_tokens` per call to `cost.jsonl` and checking the sum > 0 after the first 5 tasks. LiteLLM's `cache_control_injection_points` can auto-add the marker if we prefer not to hand-structure messages.

---

## Cost ballpark (1000 eval, optimizations applied)

| Bucket | % of 1000 | $/task | Sub-total |
|---|---|---|---|
| Phase A build (every task) | 100% | $0 | $0 |
| Phase B pass on trial 1 (Sonnet) | 65% | $0.020 | $13 |
| Phase B pass on trial 2-3 (cached) | 10% | $0.022 | $2 |
| Phase C reaches doctor (Sonnet + 0-2 probes) | 25% | $0.18 avg | $45 |
| Phase D for verified ~85% (Sonnet rubric + gpt-4o probe) | 850 | $0.018 | $15 |
| **Total** | | | **~$75** |

Doctor + Phase D are the swing factors. Phase D went up a bit (gpt-4o vs the older mini estimate) but Phase C went down (Sonnet doctor cheaper than Opus would have been). Net roughly flat.

---

## Smoke tests (gating criteria)

`scripts/pipeline/_smoke_tests.py` — runs in <2 min:

1. **`spec_builds(task_id)`** — `docker build` succeeds for a generated Dockerfile.
2. **`grader_oracle(task_id)`** — feeding the gold answer to the grader produces reward=1.0.
3. **`grader_negative(task_id)`** — feeding "obviously wrong" produces reward=0.0.
4. **`litellm_cost_log_grows(model)`** — one call → cost.jsonl gains one line with `cost_usd > 0`.
5. **`cache_hit_seen(model)`** — second call to same prompt → `cached_tokens > 0`.
6. **`doctor_tool_loop()`** — synthetic broken spec → doctor runs ≤20 turns → emits finalize.
7. **`resume_round_trips()`** — kill mid-sweep, restart with `--resume`, no duplicate trials.
8. **`rewrite_archives_state()`** — `--rewrite` produces `state.jsonl.bak.*` and clears that ID from decisions.

### Categorization-consistency gate

`pipeline smoke --n 50` runs categorize on 50 verified tasks twice with different seeds. Required: ≥85% exact-match on the 1-5 level. Below that, refine the rubric prompt before running full sweep.

---

## Iteration ladder

| Milestone | What | Gate |
|---|---|---|
| **M1** — one task | Pick one row, run `pipeline run --limit 1` | Phase A-D all complete; state.jsonl + decisions.parquet have 1 row |
| **M2** — ten tasks | `pipeline run --limit 10 --concurrent 2 --log-level debug` | ≥7 verified, no concurrency bugs, viz renders |
| **M3** — one hundred | `pipeline run --limit 100 --resume --concurrent 4` | ≥70 verified, doctor fire-rate ≤25%, prompt cache hit ≥80% |
| **M4** — categorize gate | `pipeline smoke --n 50` | ≥85% consistency on the 1-5 level |
| **M5** — full 1000 | `pipeline run --limit 1000 --resume --concurrent 8` | ≥70% verified end-to-end, cost <$100 |

Only after M5 do we move to Stage 3 (train pool). External publishing of the verified set is handled separately — not part of this stage.

---

## Concrete execution order

### Pre-work (do once before any pipeline code)

P0. **Deferred restructure** — rename `harbor_agents/` → `harbor/agents/`, update sweep.py's `--agent-import-path` strings, fix import path in `harbor/agents/seta/agent.py` etc. Single PR.
P1. **LiteLLM client** — `harbor/agents/_shared/llm_client.py` replaces our cost.py + providers.py. Update the three agents to call it.
P2. **Multi-mode grader** — `harbor/grader/grader.py` + `modes.py`, mounted into Harbor container at `/opt/grader/`. Update task.toml template to reference it.
P3. **Doctor system prompt + categorize rubric** — locked at `scripts/pipeline/prompts/{doctor_system.md, categorize.md}`. These are *the* IP of this stage; review carefully before code.

### Pipeline build (in this order)

S0. `scripts/pipeline/state.py` + `select.py` — boring foundation. Unit-test with `_smoke_tests.py:resume_round_trips`.
S1. `scripts/pipeline/build.py` — Phase A. Smoke-test against the eval-v1 template.
S2. `scripts/pipeline/verify.py` — Phase B. Single-trial run, then K=3 adaptive.
S3. `scripts/pipeline/orchestrator.py` + `cli.py` + `__main__.py` — stitch A+B together, no doctor yet. Run M1, M2.
S4. `scripts/pipeline/doctor.py` — Phase C with tool loop. Add to orchestrator. Run M3.
S5. `scripts/pipeline/categorize.py` — Phase D. Run M3 again with categorize.
S6. `scripts/pipeline/viz.py` — dashboard.
S7. `scripts/pipeline/smoke.py` — categorize-consistency. Run M4.
S8. M5 (full 1000).

---

## Out of scope (deliberately)

- **Qwen ladder difficulty bucketing.** Difficulty levels here come from the LLM rubric, not from "which tier of model first solves." (Empirical signal D2 uses gpt-4o only as a cross-check, not as the source of truth.)
- **Harness sweep.** Seta only.
- **Open-weights frontier.** Anthropic + OpenAI only.
- **Auto-rewriting questions without bound.** Doctor's rewrite budget is hard-capped at 1 per task.
- **Process reward.** Not used. RL training stage decides this separately.
- **External publishing.** Local-only; we'll address publication separately once the verified set exists.

Sources used to lock model versions, pricing, and caching mechanics:
- [Claude Opus 4.7 in Bedrock](https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/)
- [Introducing GPT-5.5 — OpenAI](https://openai.com/index/introducing-gpt-5-5/)
- [OpenAI Codex Models (GPT-5.5-Codex)](https://developers.openai.com/codex/models)
- [LiteLLM — Anthropic provider docs (model strings)](https://docs.litellm.ai/docs/providers/anthropic)
- [LiteLLM Day-0 Sonnet 4.6 / Opus 4.7](https://docs.litellm.ai/blog/claude_sonnet_4_6)
- [LiteLLM Prompt Caching reference (cache_control on content blocks)](https://docs.litellm.ai/docs/completion/prompt_caching)
- [LiteLLM Cost Tracking](https://docs.litellm.ai/docs/proxy/cost_tracking)
- [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
