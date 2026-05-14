# Data-Agent RL — Pipeline Plan

> Renaming the project from "jupyter-agent" → **data-agent**. The repo lives in `rl/` (legacy name) but everything we generate from this point forward uses `data-agent` / `data_agent` naming so it can be told apart from the prior eval-v1 / sweep-v2 work.

---

## Context (one-pager)

- The cached dataset at `data/raw/{thinking,non_thinking}.parquet` contains **51,388 unique rows**.
- The two splits share `(id, question, answer, kaggle_dataset_name)` 100% — only the `messages` trace differs. So we read from `thinking.parquet` only and ignore the other.
- Of those, **29,561 rows have `executor_type = e2b`** and a coherent Kaggle dataset name. The remaining 21,827 `executor_type = llm` rows have mismatched Kaggle metadata (the `kaggle_dataset_name` field doesn't match the file paths in the notebook) and LLM-hallucinated answers — **drop entirely**.
- **Gold answers are not 100% trustworthy.** A non-trivial fraction were produced by the original simulation pipeline and may be stale, formatted oddly, or computed against a different snapshot of the data. The verification pipeline must handle this gracefully (frontier consensus, gold auto-correction).

**Goal:** from 29,561 e2b rows, produce a HF-Hub-published `data_agent_rl` dataset with:
- `eval` split: 1,000 *candidate* rows (the "eval pool") of which some subset survives verification → final eval set.
- `train` split: the remaining ~28,500 rows, run through a cheaper single-pass verification.

We perfect the pipeline on the 1,000 eval pool first, then scale to the train pool.

---

## Naming conventions (lock these everywhere)

| Concept | Name |
|---|---|
| Project (going forward) | `data-agent` |
| Python package / cached dirs (snake_case) | `data_agent` |
| HF Hub dataset | `AdithyaSK/data_agent_rl` (splits: `train`, `eval`) |
| Harbor task suite (eval pool) | `data-agent-eval-v1` |
| Harbor task suite (train pool) | `data-agent-train-v1` |
| HF Hub kaggle bucket (reused) | `AdithyaSK/jupyter-agent-kaggle-all` (keep the old name — only the source-data bucket, no rebrand needed) |
| Local file names | snake_case: `eval_ids.txt`, `train_manifest.parquet`, etc. |
| Old artifacts | All under `_archive/` — never delete, just isolate |

---

## Target folder structure (the restructure)

Goal: at the top level of `rl/`, three obvious folders — **`data/`** (outputs), **`harbor/`** (specs + agents + run artifacts), **`scripts/`** (executable pipeline steps). Everything else gets archived.

```
rl/
├── PLAN.md                              # THIS file — the only active plan
├── README.md                            # one-page quickstart pointing to PLAN.md
├── pyproject.toml + uv.lock + .venv     # python deps
├── .gitignore
│
├── data/                                # all data outputs (gitignored except splits/)
│   ├── raw/                             # HF dataset cache (current rl/cache/raw/)
│   ├── classified.parquet               # Stage 1.1 output
│   ├── splits/                          # Stage 1.2 outputs (committed to git)
│   │   ├── eval_ids.txt                 # 1000 IDs
│   │   ├── train_ids.txt                # ~28,500 IDs
│   │   ├── eval_manifest.parquet
│   │   ├── train_manifest.parquet
│   │   └── splits.yaml                  # seed, strata, filter rules, hashes
│   └── verification/                    # Stages 2 & 3 outputs
│       ├── eval/
│       │   ├── state.jsonl              # append-only event log (per-trial)
│       │   ├── pass_at_k.parquet        # per (task,model): pass@1, pass@3
│       │   ├── decisions.parquet        # final per-task verdict
│       │   ├── gold_corrections.parquet # auto-corrected gold answers
│       │   ├── verifiable_ids.txt       # final eval set
│       │   ├── unverifiable_ids.txt     # dropped IDs + reasons
│       │   ├── trials/                  # per-trial dirs (was rl/jobs/sweep-*/)
│       │   ├── logs/                    # per-run stdout/stderr
│       │   └── viz.html
│       └── train/                       # same shape, Stage 3
│
├── harbor/                              # all Harbor-related
│   ├── tasks/
│   │   ├── data-agent-eval-v1/          # 1000 generated eval-pool tasks
│   │   └── data-agent-train-v1/         # 28k+ train tasks (Stage 3)
│   ├── agents/                          # renamed from harbor_agents/
│   │   ├── _shared/                     # cost.py, providers.py
│   │   ├── bash/
│   │   ├── jupyter/
│   │   └── seta/
│   └── grader/                          # NEW: shared grader code that tasks reference
│       ├── grader.py                    # multi-mode dispatcher (REWARD_MODE env)
│       ├── modes.py                     # numeric / exact / list / judge implementations
│       └── rubrics/                     # task-specific rubric YAMLs go here
│
└── scripts/                             # all pipeline scripts (was prepare/)
    ├── 01_cache_dataset.py              # already exists, kept
    ├── 02_classify.py                   # NEW
    ├── 03_build_splits.py               # NEW
    ├── 04_publish_splits.py             # NEW — pushes to HF Hub
    ├── 05_build_tasks.py                # NEW (subsumes build_harbor_tasks.py)
    ├── 06_verify_pool.py                # NEW — frontier sweep w/ pass@K
    ├── 07_diagnose.py                   # NEW — recovery loop
    ├── 08_visualize.py                  # was build_visualization.py
    ├── fetch_kaggle.py                  # utility, kept
    └── sweep.py                         # Qwen ladder, deferred to Stage 4
```

Everything not in those four top-level dirs goes into:

```
rl/_archive/                             # never touched after move
├── plans/
│   ├── PLAN_v0.md                       # the old big plan
│   ├── SWEEP_PLAN.md
│   └── PIPELINE_PLAN.md                 # previous draft of this plan
├── jupyter-agent-eval-v1/               # the frozen 100-task suite (was harbor/tasks/)
├── jobs-v2/                             # the 1106-trial sweep-v2 run dirs
├── cache-old/                           # old cache/{eval,sweep,hf-datasets} subdirs
├── prepare-legacy/                      # one-shot scripts no longer used
│   ├── build_eval_set.py
│   ├── pick_three.py
│   ├── replace_flagged.py
│   ├── llm_audit.py
│   └── stage_data.py
├── rollouts/                            # the OpenAI/Anthropic rollout scripts from stage 2
├── env/ + ors/                          # the OpenReward env experiment
└── logs/                                # one-off log dumps
```

This restructure happens **once**, mechanically, before Stage 1 code lands. It's a `git mv` exercise.

---

## Stage 0 — Lock decisions

| Decision | Value | Rationale |
|---|---|---|
| Usable pool | e2b only (29,561 rows) | llm rows have broken Kaggle↔file metadata |
| Eval pool size | 1,000 candidates | "Pool" — actual eval is whatever survives verification |
| Train pool size | rest (~28,500) | One-shot verification at scale |
| Eval stratification | (reward_mode × package_tier), max-K-per-Kaggle (K=4) | Diversity, no Kaggle dataset leakage |
| Frontier models for eval | Sonnet 4.6, GPT-5.5, Opus 4.7 | 3 independent vendors for true consensus |
| Frontier model for train | Sonnet 4.6 (+ GPT-5.5 on failure) | Cheaper, single-pass |
| Verification harness | `seta` (highest pass rate from v2) | One harness keeps cost predictable |
| Pass@K | K=3 for eval, K=1 for train | Eval is exhaustive, train is one-shot |
| Verifiability rule | ≥2 of 3 frontier models hit pass@K ≥ 1 | Robust to one model's blind spots |
| Gold auto-correction | Enabled (see Stage 2.5) | Handles "the gold answer might be wrong" |
| RL training reward | `flexible` (outcome-only) | Avoids PRM-exploitation pathology |
| HF Hub dataset | `AdithyaSK/data_agent_rl` (private to start) | Source of truth for IDs |

---

## Stage 1 — Splits and Hub publishing

### 1.1 Classify all 29,561 rows (`scripts/02_classify.py`)

Inputs: `data/raw/thinking.parquet` (we ignore non_thinking; answers are 100% identical).

For each row, attach:
- `reward_mode_initial` ∈ {`numeric`, `exact_short`, `exact_bool`, `list`, `list_csv`, `flexible`, `llm_judge_long`, `missing`} — deterministic rules (same as the prototype we ran).
- `answer_norm` — strip `%`, parentheticals, trailing units.
- `q_word_count`, `answer_len`, `n_files`, `n_packages`.
- `package_tier`: `0` (pandas/numpy/matplotlib only), `1` (+ sklearn/scipy/seaborn), `2` (deep-learning), `3` (other).
- `kaggle_dataset_name` — passthrough, used for stratification.

Output: `data/classified.parquet` (~29,561 rows × ~20 cols).

### 1.2 Sample splits (`scripts/03_build_splits.py`)

Algorithm:
1. Drop classifier output rows where `reward_mode_initial == missing`.
2. Stratify by `(reward_mode_initial, package_tier)`.
3. Within each stratum, sort by Kaggle dataset name, cap at **K=4 rows per Kaggle dataset** in eval.
4. Sample 1,000 eval IDs proportional to natural distribution (≈45% numeric, 27% exact, 23% flexible, etc.).
5. Train pool = `classified_pool \ eval_pool`, with `K=8` cap per Kaggle for diversity.

Outputs:
```
data/splits/
├── eval_ids.txt           # 1000 lines
├── train_ids.txt          # ~28,500 lines
├── eval_manifest.parquet  # full row data + classifier output for eval
├── train_manifest.parquet
└── splits.yaml            # seed=42, strata config, content hashes
```

### 1.3 Publish to HF Hub (`scripts/04_publish_splits.py`)

Create a new private HF dataset `AdithyaSK/data_agent_rl` with two configs (splits):
- `eval` — 1000 rows from `eval_manifest.parquet`
- `train` — ~28,500 rows from `train_manifest.parquet`

Per-row schema:
```
id                    str
question              str
answer                str (original gold)
answer_norm           str (classifier-normalized)
kaggle_dataset_name   str
files_used            list[str]
packages_used         list[str]
edu_score             int
executor_type         str ("e2b")
reward_mode_initial   str
package_tier          int
# verification columns added in Stage 2 (republished):
verifiable            bool       # None on first push, True/False after verify
reward_mode_final     str        # the mode the verifier ended up using
gold_corrected        bool       # True if frontier consensus overrode the gold
gold_original         str        # original gold (only set when gold_corrected=True)
pass_rate             float      # pass@K across all frontier rollouts
```

Also upload `splits.yaml` and a `README.md` describing the dataset, classifier rules, and verification methodology.

**Usage from anywhere:**
```python
from datasets import load_dataset
eval = load_dataset("AdithyaSK/data_agent_rl", split="eval")
```

Republished after Stage 2 (with verification columns populated) and again after Stage 3.

---

## Stage 2 — Verification pipeline (iterate on the 1000 eval pool)

### 2.1 Multi-mode grader (`harbor/grader/grader.py`)

One file dispatches on `REWARD_MODE` env var:

| Mode | Implementation |
|---|---|
| `exact` / `exact_short` | case-insensitive string equality |
| `exact_bool` | normalize to bool, compare |
| `numeric` | parse both as float, return 1.0 if `|p−g| ≤ atol` or `|p−g|/|g| ≤ rtol` |
| `list` | parse as Python literal, set/order compare |
| `list_csv` | split on commas, set compare |
| `flexible` | exact → numeric → llm-judge fallback (eval-v1 behavior) |
| `llm-judge` | one GPT-4o-mini call: "is `pred` equivalent to `gold`?" |
| `vote-judge` | 3-vote majority judge |
| `rubric` | per-task `rubric.yaml`, weighted-sum of per-criterion judge calls |
| `hybrid` | `0.7 * flexible + 0.3 * rubric` |

Outputs:
- `/logs/verifier/reward.txt` — scalar (Harbor's required signal)
- `/logs/verifier/reward.json` — `{reward, mode, components, judge_calls, predicted, gold}` for analysis

### 2.2 Build Harbor task specs (`scripts/05_build_tasks.py`)

For each ID in `eval_ids.txt`:
- Read row from `eval_manifest.parquet`.
- Generate `harbor/tasks/data-agent-eval-v1/<id>/`:
  - `task.toml` with `[verifier.env]` set from classifier:
    ```toml
    [verifier.env]
    REWARD_MODE = "numeric"
    EXPECTED_ANSWER = "{answer}"
    EXPECTED_ANSWER_NORM = "{answer_norm}"
    ANSWER_TYPE = "numeric"
    QUESTION = "{question}"
    ATOL = "0.001"
    RTOL = "0.01"
    JUDGE_MODEL = "openai/gpt-4o-mini"
    OPENAI_API_KEY = "${OPENAI_API_KEY}"
    ```
  - `instruction.md` — reused from the eval-v1 template (same prompt the SFT model saw).
  - `tests/test.sh` — invokes `harbor/grader/grader.py` (mounted at runtime).
  - `environment/Dockerfile` + `pull_bucket.py` — unchanged.
  - `rubric.yaml` only when classifier picks `rubric` mode.

Parallel generation with `ThreadPoolExecutor` — ~1000 folders in <60s.

### 2.3 Frontier verification with pass@K (`scripts/06_verify_pool.py`)

For each task in `data-agent-eval-v1`:
- For each model in `[sonnet-4-6, gpt-5.5, opus-4-7]`:
  - For `k in 1..3`: run 1 trial with `seta-tool` harness.
- 9 trials per task × 1,000 tasks = **9,000 trials, ~$270 estimated**.

State written append-only to `data/verification/eval/state.jsonl`:
```jsonc
{"ts":"...","event":"finish","task_id":"...","model":"sonnet-4-6","k":1,
 "reward":1.0,"cost_usd":0.03,"prompt_tokens":12000,"completion_tokens":800,
 "predicted_answer":"645","error_kind":"ok"}
```

After all trials, compute per (task, model):
- `pass@1` = trial 1 passed (reward ≥ 1)
- `pass@3` = any of 3 trials passed
- And per task: how many distinct models hit pass@3.

### 2.4 Verifiability decision (per task)

After the sweep:
- **Verifiable**: ≥2 of 3 frontier models hit pass@3 ≥ 1 under the *classifier-assigned* mode.
- **Borderline**: exactly 1 of 3 hits pass@3.
- **Unverifiable**: 0 of 3 hit pass@3.

### 2.5 Recovery loop (`scripts/07_diagnose.py`) — three paths

For Borderline / Unverifiable tasks, **before dropping**, try three paths in order:

**Path A: Grader-too-strict → escalate mode.**
- Read each failed trial's predicted answer from `state.jsonl`.
- Re-grade with progressively looser modes: current → `flexible` → `llm-judge` → `vote-judge`.
- If a looser mode finds at least 2 frontier models passing, **promote the mode** in `task.toml` and mark verifiable. Log `mode_upgrade` event.

**Path B: Gold answer wrong → frontier consensus auto-correction.**
- If all 3 frontier models *agree with each other* but disagree with the gold (all 3 produce the same alternative answer across pass@3), this is strong evidence the gold is wrong.
- Auto-replace `EXPECTED_ANSWER` in `task.toml` with the frontier-consensus answer, set `gold_corrected=True`, save `gold_original` to the manifest.
- Re-grade existing trials against the new gold. If now verifiable, keep with `gold_corrected=True` flag.
- *Conservative threshold*: require all 3 frontier models AND ≥2/3 trials per model to give the same alt-answer (so 6+ identical alt-answers).

**Path C: Task is genuinely broken → drop with reason.**
- Run a one-shot LLM diagnostic (1 GPT-4o-mini call, ~$0.0005 each):
  ```
  Categorize: {
    ambiguous_question, dataset_mismatch, unreachable_answer,
    code_runtime_error, kaggle_unreachable, genuinely_hard
  }
  ```
- Drop and log reason to `unverifiable_ids.txt`.

Outcomes are appended to `decisions.parquet` and the manifest:

```
decisions.parquet
├── task_id
├── verdict          ∈ {verifiable, verifiable_after_upgrade, verifiable_gold_corrected, dropped}
├── reward_mode_final
├── gold_corrected   (bool)
├── gold_original    (only set when gold_corrected)
├── frontier_pass    (count of frontier models passing)
└── drop_reason      (one of the LLM-diag categories; only set when dropped)
```

### 2.6 Visualization (`scripts/08_visualize.py`)

Reuse the existing `build_visualization.py` but pointed at `data/verification/eval/state.jsonl`. Adds two new views over the v2 dashboard:
- "Gold corrections" — table of tasks where frontier overrode the gold answer, sortable, with original vs new gold.
- "Drop reasons" — pie/bar of drop categories.

### 2.7 Stage-2 outputs

```
data/verification/eval/
├── state.jsonl
├── pass_at_k.parquet
├── decisions.parquet
├── gold_corrections.parquet
├── verifiable_ids.txt              # subset of 1000 — the actual eval set
├── unverifiable_ids.txt
├── trials/<task>__<rand>/...       # per-trial Harbor artifacts
├── logs/                           # per-sweep stdout/stderr
└── viz.html
```

Republish `AdithyaSK/data_agent_rl` with verification columns populated for the eval split.

---

## Stage 3 — Scale to train (~28.5k)

Same pipeline, cheaper config:
- **1 model × pass@1** (Sonnet 4.6 only).
- On failure, retry once with GPT-5.5.
- Skip the diagnosis loop — just drop failures with their pass/fail status.
- No gold auto-correction at this scale (it requires multi-model consensus; we only run 1-2 models).

Estimated cost: ~$700-1,000 for the 28.5k.

Outputs mirror `data/verification/eval/` shape under `data/verification/train/`.

Republish `AdithyaSK/data_agent_rl` with verification columns populated for the train split.

---

## Stage 4 — Difficulty bucketing (deferred)

Once the verifiable pool exists, run the Qwen ladder sweep (`scripts/sweep.py`, the existing v2 code) on the verifiable eval to attach easy/medium/hard. Not blocking for the rest of the pipeline.

---

## Ablations (post-hoc analysis on Stage 2 results)

All recompute from `state.jsonl` — no extra rollouts:

| Ablation | What it answers |
|---|---|
| pass@1 vs pass@3 vs pass@5 | Marginal benefit of more rollouts |
| 1 frontier model vs 2 vs 3 | Is Opus worth the cost vs Sonnet alone |
| Grader mode (strict→loose) | False-positive / false-negative rate of each mode |
| Cross-harness re-grade | Re-grade seta trajectories against bash/jupy graders to confirm verifiability isn't harness-biased |
| Gold-correction rate by category | What % of verifiable tasks needed correction |

One opt-in extra rollout: **rerun verify_pool with `jupy` harness** to confirm verifiability isn't seta-specific.

---

## Verification criteria for "pipeline works" (gate before Stage 3)

1. **Verifiable rate ≥ 70%** of the 1000 eval pool (including post-correction). Lower → classifier or grader needs work.
2. **Mode-upgrade rate ≤ 15%**. Higher → classifier rules need refining.
3. **Gold-correction rate ≤ 10%** (sanity — if we're "correcting" >10% of gold answers, something's off; either our threshold is too loose or the dataset is more broken than we thought).
4. **Per-frontier pass-rate gap ≤ 20pp**. Outlier model = integration issue.
5. **Drop-reason distribution looks plausible.** Most drops should be `ambiguous_question` or `code_runtime_error`. Anomalies trigger investigation.

---

## Cost ballpark

| Stage | Trials | $/trial | Total |
|---|---|---|---|
| Stage 1 (sample + publish) | 0 | — | $0 |
| Stage 2 (1000 × 3 models × pass@3, seta) | 9,000 | $0.03 | **~$270** |
| Stage 2 LLM diagnosis | ~300 calls | $0.0005 | < $0.50 |
| Stage 3 (28.5k × 1-1.5 models × pass@1, seta) | ~42,000 | $0.025 | **~$1,050** |
| Stage 4 (Qwen ladder, deferred) | — | — | — |
| **Total** | | | **~$1,300** |

---

## Concrete execution order

### Pre-work (do once)

0. **Restructure**: move legacy stuff to `_archive/`, rename `harbor_agents/` → `harbor/agents/`, `prepare/` → `scripts/`. Single PR. Update import paths.

### Stage 1 (sample + publish)

1. `scripts/02_classify.py` — classify 29,561 rows → `data/classified.parquet`.
2. `scripts/03_build_splits.py` — sample 1000 eval + 28.5k train → `data/splits/`.
3. `scripts/04_publish_splits.py` — push to `AdithyaSK/data_agent_rl` on HF Hub.

### Stage 2 (verify the eval pool — iterate here)

4. `harbor/grader/grader.py` + `modes.py` — implement multi-mode dispatcher. Unit tests with hand-picked rows for each mode.
5. `scripts/05_build_tasks.py` — generate `harbor/tasks/data-agent-eval-v1/` (1000 task folders).
6. `scripts/06_verify_pool.py` — pilot run on 50 tasks first; check artifacts; then full 1000.
7. `scripts/07_diagnose.py` — run on borderline/unverifiable.
8. `scripts/08_visualize.py` — generate dashboard.
9. **REVIEW gate**: check the 5 criteria above. Iterate on classifier/grader/generator until pass.
10. Republish HF dataset with verification columns.

### Stage 3 (scale to train)

11. `scripts/06_verify_pool.py --pool train --models sonnet-4-6 --k 1` on the 28.5k.
12. Republish HF dataset with train-split verification columns.

### Stage 4 (deferred)

13. Difficulty bucketing via Qwen ladder, only if we want a stratified eval card.

---

## What this plan does NOT cover

- **RL training itself.** Producing the task suite only. Training is a separate plan.
- **Process reward (PRM).** Listed as a `REWARD_MODE` but not used in verification. Reserved for a future ablation.
- **Recovering the 21k llm-executor rows.** Discarded; not worth the engineering cost.
- **Cross-language tasks.** Everything assumes Python + standard data-science stack.
