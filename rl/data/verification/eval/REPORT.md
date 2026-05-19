# data-agent eval — 500-task verification report

_Generated 2026-05-18 from `rl/data/verification/eval/decisions.csv` + suite folder state._

This document is the complete record of how the **500-task data-agent eval** was constructed end-to-end: the pool, the staged-verification pipeline, the optimizations applied during the run, the final verdicts + difficulty labels, and the open issues.

---

## 1. TL;DR

| | |
|---|---:|
| Eval pool size | **500 tasks** (reduced from 1000 — see §2) |
| Verified (any verified-class verdict) | **366 (73%)** |
| Dropped (genuinely unverifiable) | **127 (25%)** |
| Stuck `phase_b_failed` residue | 7 (1%) |
| Final difficulty coverage | **100% of verified labelled L1-L5** |
| Cumulative LLM spend | **~$72** end-to-end |
| Suite root | `rl/harbor/tasks/data-agent-eval-v1/` |
| HF dataset | [`AdithyaSK/data_agent_rl`](https://huggingface.co/datasets/AdithyaSK/data_agent_rl) — splits committed |

---

## 2. Pool reduction: 1000 → 500

The original pool (`data/splits/eval_manifest.parquet`, v1) held 1000 task IDs sampled from `jupyter-agent/jupyter-agent-dataset`, stratified by `(reward_mode_initial × package_tier)` with up to 4 questions per Kaggle dataset.

On **2026-05-18** we cut the eval pool to 500 to keep the verification budget bounded:

| | Before | After |
|---|---:|---:|
| eval pool | 1000 | **500** |
| train pool | 28,555 | **29,055** (+500 moved from eval) |

**How the 500 were picked:**
- **327 already-touched** in some prior run (decisions.csv ∪ scaffolded in any status folder) — kept in eval mandatorily so prior $$ wasn't wasted.
- **173 of the 673 untouched** — stratified random pick by `(reward_mode_initial × package_tier)`, seed=42, to preserve the eval distribution.
- The other **500 untouched** moved into the train pool.

**Disjointness verified:** final eval ∩ final train = 0 task_ids overlap.

**Backup:** `rl/data/splits/_backup/20260518T194157Z/` holds the originals of all 5 files before this reduction.

---

## 3. Staged-verification pipeline

The eval is processed in two stages per task. Each stage is invoked via the same `python -m scripts.pipeline run` CLI; the `--stage` flag toggles a small bundle of presets.

### Stage 1 — Sonnet anchor + categorize-on-pass

```bash
uv run python -m scripts.pipeline run --stage 1 \
  --sandbox docker --concurrent 30 \
  --ids-from <list> --suite data-agent-eval-v1 \
  --state-dir data/verification/eval
```

What happens per task:
1. **Phase A** — `build_spec()` ensures the Harbor scaffold exists in `pending/` (idempotent).
2. **Phase B** — Sonnet 4.6 + `seta` agent runs a trial in a Docker container, up to `--k-max 2` retries to catch flakes. Output is graded against gold via `verifier/grader.py` using the row's `reward_mode_initial` (exact / numeric / flexible / llm_judge).
3. **Phase D categorize (only on pass)** — Sonnet rubric assigns `difficulty_level` 1-5 with confidence + reasoning. Cost ~$0.005/task.
4. Move scaffold `pending/ → verified/` or `pending/ → phase_b_failed/`.

### Stage 2 — Doctor on Stage-1 failures

```bash
uv run python -m scripts.pipeline run --stage 2 \
  --sandbox docker --concurrent 30 \
  --ids-from <phase_b_failed list> --suite data-agent-eval-v1 \
  --state-dir data/verification/eval
```

What happens per task:
1. **Phase B retry** — same Sonnet+seta as Stage 1 (catches additional flake-recoveries beyond Stage 1's k=2).
2. **Phase C doctor (Sonnet brain)** — reads the failing trajectory + spec + gold; emits tool calls:
   - `probe(model="nano")` — cheap cross-model trial
   - `probe(model="gpt-5.5")` — frontier confirmation if probes disagree
   - `rewrite_spec()` — relax reward_mode (e.g. numeric → flexible)
   - `correct_gold()` — overwrite gold with cross-model consensus
   - `drop()` — declare unverifiable
3. **Phase B2** — re-run Phase B if doctor rewrote spec or corrected gold.
4. **Phase D categorize** on recoveries.

**Two-stage rationale** (vs. doing everything in one run): Stage 2 only fires on the residual that Stage 1 couldn't crack, so we don't pay for doctor probes / spec rewrites on the ~40-60% of tasks Sonnet handles cleanly.

---

## 4. Optimizations applied during this run

| Patch | File | Effect |
|---|---|---|
| **`--stage` preset flag** | `scripts/pipeline/cli.py` | One CLI knob bundles model + doctor + categorize defaults per stage |
| **Categorize gate accepts `verifiable_judge`** | `scripts/pipeline/orchestrator.py:253-269, 304` | `verifiable_judge` verdicts now trigger Phase D categorize; previously they fell through as L0 |
| **Stage-2 probe roster lean default** | `scripts/pipeline/cli.py` | `--stage 2` defaults to `--probe-aliases nano gpt-5.5` (drops Opus + deepseek which run 5-10 min/probe). Override with explicit `--probe-aliases`. |
| **`--probe-timeout-sec` flag** | `scripts/pipeline/{cli,orchestrator,doctor}.py` | Per-probe wall cap, default 180s. Independent of `--subprocess-timeout-sec` (which governs anchor trials). Kills long-tail probe outliers. |
| **Lighter Docker config** | `scripts/pipeline/build.py` | New scaffolds: `cpus=1, memory_mb=1024, storage_mb=5120` (was 2/4096/10240). Allows 30+ concurrent on Docker Desktop 33 GiB VM. |
| **`difficulty_level` in scaffold template** | `scripts/pipeline/build.py` | New scaffolds carry `difficulty_level = 0` in `[metadata]` from the start; orchestrator overwrites on Phase D. |

---

## 5. Final verdict breakdown (500 tasks)

| Verdict | Count | % | Comes from |
|---|---:|---:|---|
| `verified` | **273** | 55% | Phase B passed in Stage 1 or Stage 2 |
| `verified_gold_corrected` | 57 | 11% | Doctor (Stage 2): probes converged on a NEW answer; gold was wrong |
| `verifiable_judge` | 20 | 4% | Doctor (Stage 2): LLM judge agreed agent's answer ≡ gold |
| `verified_after_rewrite` | 16 | 3% | Doctor (Stage 2): reward_mode relaxed (e.g. numeric → flexible); Phase B2 passed |
| **verified-class TOTAL** | **366** | **73%** | |
| `dropped` | 127 | 25% | Doctor declared genuinely unverifiable (probes disagree) OR doctor budget exhausted |
| `phase_b_failed` | 7 | 1% | Manual force-kill / task_timeout residue; doctor never finalized |

### Pass-rate by `reward_mode_initial`

| Mode | n | Verified | Pass-rate |
|---|---:|---:|---:|
| `exact_bool` | 23 | 21 | **91%** |
| `exact_short` | 131 | 107 | 82% |
| `list_csv` | 10 | 8 | 80% |
| `flexible` | 112 | 79 | 71% |
| `numeric` | 213 | 145 | 68% |
| `list` | 6 | 4 | 67% |
| `llm_judge_long` | 5 | 2 | 40% |

`numeric` has the largest drop rate by absolute count (67 dropped) — driven by tolerance-vs-rounding edge cases that the doctor's probes couldn't unambiguously resolve. `llm_judge_long` has the worst rate but the smallest n.

---

## 6. Difficulty distribution across the 366 verified

```
L1   75 ███████████████████████████████████████████████████████████████████████████ (20%)
L2  151 ███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ (41%)  ← mode
L3   71 ███████████████████████████████████████████████████████████████████████ (19%)
L4   68 █████████████████████████████████████████████████████████████████████ (19%)
L5    1 █ (<1%)
```

| Level | Count | Typical pattern |
|---|---:|---|
| L1 | 75 | one-line filter / aggregation |
| L2 | 151 | filter + groupby + aggregate (2-4 turns) |
| L3 | 71 | multi-step pandas, joins, light feature work |
| L4 | 68 | ML training / non-trivial pipelines / complex statistical reasoning |
| L5 | 1 | extreme complexity (single outlier task) |

**Coverage: 100% of verified-class tasks now have a non-zero difficulty label.** Pre-fix the categorize step skipped `verifiable_judge` verdicts (gap captured in §4 patch table); after the fix + a 27-task backfill, no L0 remain.

---

## 7. Cost breakdown — $71.93 cumulative on the 500-task eval

| Bucket | $ | % |
|---|---:|---:|
| **Phase B** (anchor seta trials, all stages combined) | $48.82 | **68%** |
| Doctor LLM (Sonnet brain) | $7.56 | 11% |
| Doctor probes (nano + gpt-5.5 + opus + deepseek on some pre-optimization runs) | $13.86 | 19% |
| Phase D categorize (Sonnet rubric) | $1.83 | 3% |
| **Total** | **$71.93** | 100% |

**$/verified-task: $0.20** (= $71.93 / 366).

Phase B is the dominant cost line — the agent's tool-loop inside Harbor accounts for ~⅔ of total spend. This matches the closed-only ablation finding. **Per-task cost will go up at scale if pass-rate drops** (more tasks need doctor + Phase B2 re-runs).

The Stage 2 probe optimization (drop Opus + deepseek defaults, cap probes at 180s) materially shifted Stage-2 wall time — anecdotally ~25-30 min instead of 30-45.

---

## 8. Operational issues observed

1. **Docker container leaks (zombies).** Harbor's container cleanup is best-effort; if the host harbor subprocess dies (timeout, sigkill, host crash) the container stays up until manually killed. Saw ~10-15 zombies over the run, max ~2 GiB resident. Same shape as the prior E2B ghost-sandbox issue, smaller scale. Mitigation: `docker container prune -f` post-run; consider an `atexit` hook in the Harbor docker provider.
2. **Block-buffered stdout** when the pipeline runs with `> file.log`. Verdict lines flush in bursts, not real-time. State.jsonl is the actual source of truth for in-flight monitoring; the textual log lags. Worth a one-liner `sys.stdout.reconfigure(line_buffering=True)` somewhere in `cli.py:cmd_run`.
3. **Doctor probes are sequential**, not parallel. Within one task, doctor emits one `probe()` call per turn, waits for it, decides next. Confirmed via state.jsonl event timelines. A future patch could (a) prompt the doctor to emit multiple probes per turn and (b) execute parallel-safe tool calls via `asyncio.gather`. ~3× wall-time win at the cost of higher peak container count.
4. **`docker stop` 404 errors** during force-kill of leftover containers — appears to be a Docker API hiccup. `docker kill <id>` worked where `docker stop` 404'd.

---

## 9. Where everything lives

```
rl/
├── data/splits/                                 # source-of-truth splits
│   ├── eval_manifest.parquet                    # 500 rows (new)
│   ├── train_manifest.parquet                   # 29,055 rows (new)
│   ├── eval_ids.txt / train_ids.txt
│   ├── splits.yaml                              # versioning + distributions + sha256s
│   └── _backup/20260518T194157Z/                # pre-reduction originals
│
├── data/verification/eval/                      # verification working dir
│   ├── decisions.csv                            # 500 rows — final verdicts + difficulty + costs
│   ├── runs/<timestamp>/                        # per-invocation event logs
│   │   ├── state.jsonl                          # SOURCE OF TRUTH for events
│   │   ├── trials/                              # Harbor trial dirs (trajectories, reward.txt)
│   │   ├── cost.jsonl
│   │   └── decisions.csv
│   └── REPORT.md                                # ← this file
│
├── harbor/tasks/data-agent-eval-v1/             # the Harbor task suite
│   ├── pending/      (0 — all processed)
│   ├── verified/     (366 scaffolds with L1-L5 labels)
│   ├── dropped/      (127)
│   └── phase_b_failed/ (7 residue)
│
├── scripts/pipeline/                            # the verify CLI
│   ├── cli.py          — entry point with --stage flag
│   ├── orchestrator.py — process_task() driving Phases A-D
│   ├── build.py        — Harbor scaffold generator (light Docker defaults)
│   ├── doctor.py       — Sonnet doctor tool-loop with probe/rewrite/correct/drop tools
│   ├── categorize.py   — Phase D rubric judge
│   └── verify.py       — run_trial() — the `harbor run ...` subprocess wrapper
│
├── cache/
│   ├── eval_remaining_278.json                  — the 278 IDs run in this batch
│   ├── stage2_eval_remaining.json               — the 168 phase_b_failed → Stage 2 input
│   ├── difficulty_lookup.json                   — task_id → difficulty across all sources
│   └── batch_2_ids.json                         — earlier 100-task batch list
│
└── data/ablation/                               # FROZEN — earlier 30-task closed/mixed/oss-only ablation
    └── REPORT.md
```

---

## 10. HuggingFace Hub state

Splits live at [`AdithyaSK/data_agent_rl`](https://huggingface.co/datasets/AdithyaSK/data_agent_rl).

**Last commit** ([`1ff7a053`](https://huggingface.co/datasets/AdithyaSK/data_agent_rl/commit/1ff7a053e8da54cfa17d344475b0484e75a16083), 2026-05-18):
- `data/eval-00000-of-00001.parquet` — 500 rows (was 1000)
- `data/train-00000-of-00001.parquet` — 29,055 rows (was 28,555)
- `splits.yaml` — updated sizes + sha256 hashes

The README prose on HF still references the v1 numbers (1000 eval, 28,555 train); the parquet data itself is authoritative.

---

## 11. Future work

- **Push the verified Harbor task suite to HF** as a separate dataset (`AdithyaSK/data-agent-eval-v1-harbor` or similar). 366 task scaffolds with their `difficulty_level` metadata embedded in `task.toml` makes for a publishable, runnable benchmark.
- **Parallel probes in the doctor** (item 3 above) — biggest remaining wall-time win.
- **Line-buffered stdout** — quick fix for monitoring during long runs.
- **Doctor cleanup hook** to delete its probe containers on host-process exit (avoid zombie accumulation).
- **Split-3: 500 unused** untouched IDs are now in `train_manifest.parquet` waiting for SFT/RL training. Build SFT data with the existing `finetuning/` recipe pointed at the new train manifest.

---

_End of report._
