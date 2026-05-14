# Model-size sweep plan — one task at a time, escalating bottom-up

Goal: for each eval task, find the **smallest Qwen model** that can solve it
(reward = 1.0) on each of our 4 agent harnesses. Run small → big, stop
escalating once enough wins land. Track cost, throughput, and timing
end-to-end. Resume cleanly on any interruption.

---

## 1. Qwen model list (smallest → biggest, served by HF Inference Providers)

Probed live from `https://huggingface.co/models?inference_provider=…&search=qwen`.
All are **Instruct-tuned** (skip Coder unless you want a code-domain sub-sweep)
and verified to support **OpenAI-compatible tool calling** through HF's
router at `https://router.huggingface.co/v1/chat/completions`.

| Tier | Model ID (HF) | Params | Provider | Notes |
|---:|---|---:|---|---|
| **T1** | `Qwen/Qwen3-4B-Instruct-2507`        |   4 B | `nscale`           | Student-size baseline. Same gen as our SFT student. |
| **T1** | `Qwen/Qwen3-4B-Thinking-2507`        |   4 B | `nscale`           | Thinking variant — emits `<think>…</think>`. Useful comparison. |
| **T2** | `Qwen/Qwen3-8B`                      |   8 B | `nscale`           | Mid-small, first to solve in the earlier 12-cell smoke. |
| **T2** | `Qwen/Qwen2.5-7B-Instruct`           |   7 B | `together`         | Older-gen reference (different post-training). |
| **T3** | `Qwen/Qwen3-14B`                     |  14 B | `nscale`           | Best non-MoE size for the SFT student gap. |
| **T4** | `Qwen/Qwen3-32B`                     |  32 B | `nscale`           | Largest dense before MoE flagships. |
| **T4** | `Qwen/QwQ-32B`                       |  32 B | `nscale`           | Reasoning-tuned 32B — separate post-training. |
| **T5** | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235 B (22 B active) | `nscale` / `scaleway` | MoE flagship. Already smoked. |
| **T5** | `Qwen/Qwen3-235B-A22B`               | 235 B               | `nscale`           | Non-Instruct flagship (less tool-cal friendly — backstop). |

**Locked spine** (7 Qwen3 models, Instruct + 1 Thinking + 1 Coder, no Qwen2.5):

```
T1: Qwen3-4B-Instruct-2507         (4B  dense,   nscale)
T1: Qwen3-4B-Thinking-2507         (4B  thinking, nscale)
T2: Qwen3-8B                       (8B  dense,    nscale)   ← "6b" in user spec → snapped to 8B (no Qwen3-6B exists)
T3: Qwen3-14B                      (14B dense,    nscale)
T4: Qwen3-Coder-30B-A3B-Instruct   (30B MoE, 3B active, scaleway)
T4: Qwen3-32B                      (32B dense,    nscale)
T5: Qwen3-235B-A22B-Instruct-2507  (235B MoE, 22B active, nscale)
```

Tool calling: every model above is exposed via the router's `:nscale` /
`:together` / `:scaleway` suffix and accepts `tools=[…]` on chat completions.
Earlier smoke proved this for 4B, 8B, 14B, 235B. 32B is the only untested
one — we'll smoke-it on the easy tesla task before kicking off the full
sweep.

Pricing (per provider — list pricing as of 2026-05; will pull live):

| Model | Input $/M | Output $/M | Indicative $/run* |
|---|---:|---:|---:|
| Qwen3-4B | 0.05 | 0.15 | $0.0003 |
| Qwen3-8B | 0.10 | 0.30 | $0.0008 |
| Qwen3-14B | 0.20 | 0.60 | $0.002 |
| Qwen3-32B | 0.50 | 1.50 | $0.006 |
| Qwen3-235B-A22B | 0.80 | 2.40 | $0.015 |

\* Indicative = average of ~3k prompt + ~2k completion per task across 10 turns.
Real numbers will be tracked per-run.

---

## 2. Agents (locked) — 4 harnesses, all multi-provider

| Slug | Tools | Notes |
|---|---|---|
| `jupy` | 4 (stateful kernel + shell + state-inspect)        | `rl.harbor_agents.jupyter:JupyterToolAgent` — matches the SFT-training shape. |
| `bash` | 1 (just `bash`)                                    | `rl.harbor_agents.bash:BashOnlyAgent` — our new minimal one (replaces `mini-swe-agent` which fails on `hf/…:nscale` model strings). |
| `seta` | 10 (6 shell + 4 notes auto-injected into context)  | `rl.harbor_agents.seta:SetaToolAgent` — best for weak models per the smoke. |
| `oc`   | bash / read / write / edit / grep / glob / ls       | Harbor's built-in `opencode`. Different model-string format (`huggingface/…:nscale`). |

All four go through the same provider routing in `_shared/providers.py`
and write submissions to `/workdir/answer.txt`.

---

## 3. Sweep algorithm

**Two phases run in sequence:**

### Phase 3A — Qwen sweep (cheap models, bucket-ordered)

Tasks are processed **by difficulty bucket in order: easy → medium → hard**.
This way we get fast pass-rate signal on easy first, then commit to the
harder tiers only once that's complete.

For each task `T` in `bucket_order(easy, medium, hard)`:

```
passes = 0
for tier in (T1, T2, T3, T4, T5):
    pending = [(T, m, a) for m in tier for a in AGENTS
               if not already_completed(T, m, a)]
    launch_parallel(pending)        # bounded by --concurrency (32)
    passes += count_passes_in(T, tier)
    if passes >= STOP_AT:           # default 2
        break                       # task graduated at this tier
record_task_outcome(T, smallest_passing_tier, agents_that_passed)
```

**Stop condition** (`--stop-at 2` default): ≥2 (agent × model) combos passed.

Per-task tag:
- `EASY-WIN`: graduated at T1 (4B).
- `MID-WIN`: graduated at T2/T3 (8B–14B).
- `HARD-WIN`: only graduated at T4/T5 (32B / 235B).
- **`QWEN-FAIL`**: zero passes across all 7 models × 4 agents (28 trials). → goes to Phase 3B.

### Phase 3B — frontier fallback (only for QWEN-FAIL tasks)

After Phase 3A completes for all 100 tasks, identify the QWEN-FAIL set
(tasks where zero Qwen × agent combo passed) and run them through two
frontier models, also bottom-up:

| Order | Model | Why |
|---|---|---|
| F1 | `claude-sonnet-4-6` (anthropic OpenAI-compat shim) | Cheaper, strong tool-calling, our existing baseline |
| F2 | `gpt-5.5` (openai native) | Latest frontier, recommended for Codex/coding-heavy tasks (released 2026-04-24) |

Per QWEN-FAIL task:
```
passes = 0
for m in (claude-sonnet-4-6, gpt-5.5):
    pending = [(T, m, a) for a in AGENTS if not already_completed(T, m, a)]
    launch_parallel(pending)
    passes += count_passes_in(T, m)
    if passes >= STOP_AT:
        break
```

Final per-task tag:
- `FRONTIER-WIN`: solved by sonnet or gpt-5.5 when Qwen couldn't.
- **`UNSOLVED`**: zero passes ever — task is either inherently un-graderable or our agent/grader has a bug. Surfaces as a follow-up audit list.

**Per task budget**: max 5 Qwen tiers × 4 agents + 2 frontier × 4 = 28 + 8 = 36 runs. Typical task that's `EASY-WIN`: 4 runs.

---

## 4. Concurrency strategy

- E2B max concurrent: **100** (user's quota).
- Each `harbor run` invocation = 1 E2B sandbox.
- **Default: 32 concurrent runs** (4× safety margin under the 100 cap, leaves
  room for `harbor run` internal slack + other jobs).
- Adjustable via `--concurrency N`.

**Scheduling**: a single Python orchestrator (`rl/prepare/sweep.py`) holds the
queue of pending `(task, model, agent)` triples. Workers (subprocess-per-job)
pull from the queue; cap = `--concurrency`. When `count_passes_in(T, tier) >= 2`
for some task, all *future* triples for `T` are dropped from the queue (we
already have enough signal).

This is more efficient than "wait for tier to finish, then escalate":
- Tier-1 + tier-2 can overlap if tier-1 hasn't finished and we already know
  it's not going to pass.
- Concurrency stays saturated.

---

## 5. State machine + resumability

**Single source of truth**: `rl/cache/sweep/v1/state.jsonl` (append-only).
Each line:

```json
{
  "ts": "2026-05-13T01:23:45Z",
  "task_id": "0065_794_65794937_qa_1",
  "model": "Qwen/Qwen3-8B:nscale",
  "agent": "jupy",
  "tier": "T2",
  "job_name": "sweep-v1-0065_794_65794937_qa_1-qwen3-8b-jupy",
  "status": "completed" | "errored" | "skipped",
  "reward": 1.0,
  "elapsed_sec": 47.2,
  "prompt_tokens": 3120,
  "completion_tokens": 1840,
  "cached_tokens": 0,
  "cost_usd": 0.000812,
  "trajectory_path": "rl/jobs/sweep-v1-…/<trial>/agent/jupyter_agent.trajectory.json",
  "answer_pred": "37163900",
  "answer_gold": "37163900"
}
```

**On startup**, sweep reads state.jsonl and builds the "already done" set.
Any `(task, model, agent)` triple present skips the launch.

**Crash safety**: state.jsonl is line-appended after each run completes;
mid-flight runs that crash leave no state line and get re-launched next
invocation. Idempotent.

**Tier-skip**: if state shows `count_passes(T) >= STOP_AT` already, we don't
re-launch higher tiers either.

**Concrete files**:

```
rl/cache/sweep/v1/
├── config.yaml            # model list, agents, pricing, stop-condition
├── tasks.txt              # source-row ids to sweep (copy of eval_ids.txt at run start)
├── state.jsonl            # append-only run log (above schema)
├── cost_running.json      # rolling totals — updated after each run
└── README.md              # what was run, when, with what config

rl/jobs/sweep-v1-<task>-<model_slug>-<agent>/
└── …                      # per-Harbor-run dir (trajectory, result.json, reward, logs)
```

---

## 6. Cost + throughput instrumentation

**Per-run** — every agent's `run()` accumulates `resp.usage.{prompt_tokens,
completion_tokens, prompt_tokens_details.cached_tokens}` across all LLM calls,
multiplies by the model's pricing entry, and writes to `context.cost_usd /
n_input_tokens / n_output_tokens / n_cache_tokens` at the end of `run()`.
Harbor surfaces these to `result.json`.

The 10-line snippet is already drafted (cf. the "What to add" section of the
cost-monitoring writeup earlier). We'll wire it into all 3 custom agents
(`jupyter/`, `bash/`, `seta/`) before kickoff.

For `opencode`, Harbor already extracts tokens from its JSON-line stdout —
the cost calculation works for-free as long as we register the model price.

**Per-sweep** — after each run completes, the orchestrator:
1. Reads `rl/jobs/<job>/result.json`
2. Extracts cost + tokens
3. Appends to `state.jsonl`
4. Updates `cost_running.json`:
   ```json
   {
     "total_runs": 247, "total_cost_usd": 12.43,
     "by_model": {"Qwen3-4B": {"runs": 80, "cost": 0.18, "passes": 12}, ...},
     "by_agent": {"jupy": {...}, "bash": {...}, ...},
     "by_tier_first_pass": {"T1": 12, "T2": 18, "T3": 8, ...},
     "tokens": {"input": 1_234_567, "output": 845_300, "cached": 30_000},
     "throughput": {"reqs_per_min": 18.4, "tok_per_sec": 2310},
     "wall_time_sec": 2840
   }
   ```

**Stop-on-budget** (`--max-cost-usd 50`): orchestrator stops launching new
runs (in-flight ones finish) when running cost crosses the cap. State is
preserved — re-run continues where it left off.

**Stop-on-throughput** (`--min-tok-per-sec 100`): if rolling throughput
drops below a floor for >5 min, pause and surface for inspection (likely
HF/Nscale rate-limiting).

---

## 7. Trace storage — where everything lives

| What | Where | Why |
|---|---|---|
| Per-run trajectory (full conversation + tool calls) | `rl/jobs/<job_name>/<trial>/agent/<agent>.trajectory.json` | Harbor writes this; agent code does it best-effort. |
| Per-run notes (SETA) | `rl/jobs/<job_name>/<trial>/agent/seta_agent.notes.json` | Surface the "Plan" note that drove the run. |
| Per-run reward + verdict | `rl/jobs/<job_name>/<trial>/verifier/reward.txt` + `test-stdout.txt` | What the grader saw. |
| Per-run Harbor result | `rl/jobs/<job_name>/result.json` | Harbor's aggregate (cost, tokens, runtime, reward). |
| Sweep-level state log | `rl/cache/sweep/v1/state.jsonl` | Single tail-able file for "what happened so far". |
| Cost dashboard | `rl/cache/sweep/v1/cost_running.json` | Live totals; pretty-printable. |

**E2B sandboxes themselves are ephemeral** — nothing inside the container
survives. Harbor copies the necessary artifacts (trajectory, answer.txt,
verifier output) back to the host before destroying the sandbox. Confirmed
working in our smoke runs.

---

## 8. Implementation phases

Three phases, each independently testable:

### Phase A — instrumentation (deterministic, single-trial)
1. Add cost/token tracking to the 3 custom agents
   (`jupyter/`, `bash/`, `seta/`) — populate `context.{cost_usd, n_input_tokens,
   n_output_tokens, n_cache_tokens}` from `resp.usage`.
2. Add `cost_table.yaml` keyed by `<HF_model_id>:provider` with input/output/cached prices.
3. Smoke: re-run easy task × Qwen3-8B × jupyter — confirm `result.json` has non-null `cost_usd`.

### Phase B — sweep orchestrator (`rl/prepare/sweep.py`)
1. CLI: `uv run python -m prepare.sweep --sweep v1 --tasks cache/eval/eval_ids.txt --concurrency 32 --stop-at 2`.
2. State machine: read state.jsonl, build pending queue, run with bounded
   concurrency (asyncio.Semaphore), append to state.jsonl as each finishes.
3. Tier-skip: after each pass detected, drop higher tiers for that task from
   the queue.
4. Cost-cap: stop launching when running total ≥ `--max-cost-usd`.

### Phase C — analysis + summary
1. `rl/prepare/sweep_report.py`: walk state.jsonl, produce:
   - Per-task: smallest passing tier, agents that passed
   - Per-model: pass-rate, mean cost, mean tokens, median latency
   - Per-agent: same breakdown
   - Histogram of "first-pass tier" across tasks
   - Cost breakdown table
2. Markdown report → `rl/cache/sweep/v1/REPORT.md`.

---

## 9. Budget + wall-time guesstimate

100 eval tasks (50 easy + 25 medium + 25 hard), stop-at-2-passes, 32-way concurrency, `:fastest` provider policy.

**Wall-time math** — the tail is gated by 235B + opencode (~3-5 min/run). At 32-concurrent, peak T5 concurrency is ~6-8 runs. With easy→medium→hard ordering:
- Easy bucket (50 tasks, most graduating at T1/T2): ~25-40 min
- Medium bucket (25 tasks): ~20-30 min
- Hard bucket (25 tasks, more escalation): ~25-40 min
- Frontier fallback (probably 10-20 tasks × 8 runs): ~15-25 min
- **Total: ~90-130 min wall time**

**Cost** (assumes 40% of tasks reach T3+, 15% reach T5, 15% become QWEN-FAIL):

| Phase | Tier / Model | Avg tasks here | Runs (× 4 agents) | $/run | Subtotal |
|---|---|---:|---:|---:|---:|
| 3A | T1 (4B Inst + Thinking) | 100 × 2 = 200 | 800 | $0.0003 | $0.24 |
| 3A | T2 (8B) | 60 | 240 | $0.0008 | $0.19 |
| 3A | T3 (14B) | 40 | 160 | $0.002 | $0.32 |
| 3A | T4 (32B + 30B-Coder) | 25 × 2 = 50 | 200 | $0.006 | $1.20 |
| 3A | T5 (235B) | 15 | 60 | $0.015 | $0.90 |
| 3B | Claude Sonnet 4.6 | 15 | 60 | $0.20 | $12 |
| 3B | GPT-5.5 (only if Sonnet fails) | 7 | 28 | $0.50 | $14 |
| Misc | LLM-judge calls in verifier | – | ~1500 | $0.0005 | $0.75 |
| | | | **~1550** | | **≈ $30** |

**Net: ~$25–50 for the full 100-task end-to-end.** Mostly the frontier fallback. If QWEN-FAIL rate is higher than expected (e.g. 30 tasks), bump to ~$60-90.

---

## 10. Locked decisions

- **Stop threshold**: `--stop-at 2` wins per task (per Qwen tier; also per frontier model in Phase 3B).
- **Model lineup**: 7-model Qwen3 spine + frontier fallback (claude-sonnet-4-6, gpt-5.5).
- **Concurrency**: **32** (sweet spot per math in section 9 — saves ~25 min vs. 64 but with cleaner state.jsonl and zero provider-side rate-limit risk).
- **Task order**: easy bucket first (50 tasks), then medium (25), then hard (25). Bucket transitions are natural checkpoints.
- **Cost cap**: **none** — let it run. Estimate: $5–15 Qwen + $30–80 frontier fallback = $35–95 total.
- **Per-run timeout**: 8 min/run default (handles 235B's slow decode).
- **HF org tier**: thread `X-HF-Bill-To: huggingface` header on the router call → Enterprise Plus quota (10k API req/5min ≈ 33 req/sec sustained) and bills the org, not the personal account.
- **Adaptive throttle**: if ≥3 `429 Too Many Requests` arrive in 60s, drop concurrency by 25% until window clears.
- **Provider failover**: use `:fastest` policy (server auto-routes to fastest available provider) instead of pinning `:nscale`. Auto-fails over on provider-side rate-limits.
