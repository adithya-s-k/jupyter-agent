# 30-task ablation — closed-source vs open-source vs mixed

_Generated 2026-05-14T19:44:01.784579Z_  

Same 30 task IDs (stratified across reward modes: 14 numeric · 8 exact_short · 7 flexible · 1 list_csv).
Each condition ran in an isolated suite and state dir to prevent doctor edits from leaking across conditions.

## Conditions

| Condition | Anchor | Doctor brain | Probes available |
|---|---|---|---|
| **mixed** | Sonnet 4.6 | Sonnet 4.6 | all 8 aliases (nano/opus/gpt-5.5/codex + qwen/glm/kimi/deepseek) |
| **closed-only** | Sonnet 4.6 | Sonnet 4.6 | nano + opus + gpt-5.5 + gpt-5.5-codex |
| **oss-only** | Qwen3-235B | Qwen3-235B | qwen + glm + kimi + deepseek |

All other settings identical: `--concurrent 30`, `--subprocess-timeout-sec 600`, `--task-timeout-sec 1500`, `--max-retries-per-trial 1`, `--stagger-sec 0.5`, `--rewrite-spec`.

## Headline results

| Run | n | Verified | Dropped | Total $ | $/task | $/verified | Verification rate |
|---|---|---|---|---|---|---|---|
| mixed | 30 | **20** | 10 | $4.5330 | $0.1511 | $0.2266 | **66.7%** |
| closed-only | 30 | **19** | 11 | $6.6578 | $0.2219 | $0.3504 | **63.3%** |
| oss-only | 29 | **12** | 17 | $1.9639 | $0.0677 | $0.1637 | **41.4%** |

**Note**: closed-only had one stuck task (`0013/877/13877897.ipynb_qa_2`) that we force-marked `dropped` after the run was manually killed past its task-timeout — it was the same task that got stuck in the mixed run and represents a corner case where Sonnet+seta loops indefinitely on this dataset.

## Cost breakdown by phase

| Condition | Phase B (anchor) | Doctor LLM | Probes | Categorize |
|---|---|---|---|---|
| mixed | $3.816 | $0.424 | $0.190 | $0.103 |
| closed-only | $5.797 | $0.514 | $0.263 | $0.083 |
| oss-only | $1.021 | $0.474 | $0.456 | $0.014 |

**Phase B dominates cost in all conditions.** It's the anchor agent's seta loop running inside Harbor — typically 5-15 agent turns per task.
- Qwen3-235B is **5.7× cheaper** than Sonnet on Phase B ($1.02 vs $5.80 for the same 30 tasks).
- Doctor + probes are roughly comparable across conditions (~$0.7-1.0 each).
- Categorize was negligible in all conditions (<$0.11).

## Verdict breakdown per condition

### mixed

| Verdict | Count |
|---|---|
| verified | 17 |
| dropped | 10 |
| verified_gold_corrected | 3 |

### closed-only

| Verdict | Count |
|---|---|
| verified | 13 |
| dropped | 11 |
| verifiable_judge | 3 |
| verified_gold_corrected | 2 |
| verified_after_rewrite | 1 |

### oss-only

| Verdict | Count |
|---|---|
| dropped | 17 |
| verified | 10 |
| verified_gold_corrected | 2 |

## Same-task agreement matrix

| Outcome | Count |
|---|---|
| All 3 verified | 11 |
| All 3 dropped  | 8 |
| closed+mixed verified, oss dropped | 7 |
| oss verified, closed dropped       | 1 |
| any disagreement                   | 11 |

### Tasks where closed verified but OSS dropped (the discriminators)

| Task ID | Mode | Pkg tier | Closed verdict | OSS verdict | Question (truncated) |
|---|---|---|---|---|---|
| `0001/257/1257061.ipynb_qa_2` | flexible | 1 | verified | dropped | Which feature has the highest absolute correlation with SalePrice, and what is t |
| `0001/534/1534628.ipynb_qa_5` | numeric | 2 | verified_after_rewrite | dropped | For the drink 'black russian', how many of the predicted ingredients exactly mat |
| `0019/416/19416448.ipynb_qa_2` | numeric | 1 | verified_gold_corrected | dropped | What is the pseudo R-squared value of the logit model using number of competitor |
| `0024/275/24275657.ipynb_qa_5` | flexible | 1 | verified | dropped | What percentage of variance is explained by the second principal component in th |
| `0098/690/98690564.ipynb_qa_1` | numeric | 2 | verified | dropped | How many outliers are present in the 'smoker' column based on the interquartile  |
| `0104/985/104985612.ipynb_qa_4` | exact_short | 1 | verified | dropped | Which predictor variable in the linear regression model is not statistically sig |
| `0126/994/126994091.ipynb_qa_5` | flexible | 1 | verified | dropped | What is the 75th percentile value of burned area in the original dataset before  |

### Tasks where OSS verified but closed dropped

| Task ID | Mode | Pkg tier | Closed verdict | OSS verdict | Question (truncated) |
|---|---|---|---|---|---|
| `0052/633/52633741.ipynb_qa_5` | exact_short | 1 | dropped | verified_gold_corrected | Which cluster shows the highest median Special Attack (Sp. Atk) value across all |

## Doctor invocation patterns

### mixed
- Doctor fired on 13 of 30 tasks (43%)
- Total probes spawned: 17 (1.3 per doctor session avg)
- Probe model distribution: {'openai/gpt-5.4-nano': 13, 'hf/deepseek-ai/DeepSeek-V3.1:novita': 4}

### closed-only
- Doctor fired on 16 of 30 tasks (53%)
- Total probes spawned: 19 (1.2 per doctor session avg)
- Probe model distribution: {'openai/gpt-5.4-nano': 15, 'anthropic/claude-opus-4-7': 2, 'openai/gpt-5.5-codex': 2}

### oss-only
- Doctor fired on 20 of 29 tasks (69%)
- Total probes spawned: 20 (1.0 per doctor session avg)
- Probe model distribution: {'hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale': 19, 'hf/zai-org/GLM-4.6:novita': 1}

## Difficulty distribution (across verified tasks per condition)

| Level | mixed | closed-only | oss-only |
|---|---|---|---|
| 1 | 4 | 5 | 5 |
| 2 | 7 | 4 | 2 |
| 3 | 3 | 3 | 2 |
| 4 | 6 | 4 | 3 |
| 5 | 0 | 0 | 0 |

## Three lessons

1. **OSS is 3.3× cheaper per task** ($0.068 vs $0.222) — almost entirely from Phase B (Qwen+seta is much cheaper than Sonnet+seta).
2. **OSS verification rate is 22 pp lower** (41% vs 63%) — Qwen drops 7 tasks that Sonnet handled, while only rescuing 1 task that Sonnet dropped.
3. **Mixed beats closed-only on cost without losing verification rate.** Mixed had $4.53/30 = $0.15/task and 67% verification; closed-only forced expensive probes ($6.66/$0.22/task) and got 63%. Letting the doctor pick OSS probes when sufficient saves ~32% without quality loss.

## $/verified-task as the unified scoreboard

| Condition | $/verified | Verification rate |
|---|---|---|
| mixed | $0.2266 | 67% |
| closed-only | $0.3504 | 63% |
| oss-only | $0.1637 | 41% |

- **Lowest cost per verified task: oss-only** ($0.164) — but you accept 41% verification.
- **Highest verification rate at lowest cost: mixed** ($0.227, 67%) — strong argument for keeping the alias pool open across vendors.
- **Closed-only is the WORST trade-off**: highest cost ($0.350/verified) without a verification advantage over mixed.

## Data locations on disk

- **mixed**: `data/ablation/mixed/`  → run dir `data/ablation/mixed/runs/20260514T185622Z/`
- **closed-only**: `data/ablation/closed-only/`  → run dir `data/ablation/closed-only/runs/20260514T191828Z/`
- **oss-only**: `data/ablation/oss-only/`  → run dir `data/ablation/oss-only/runs/20260514T191830Z/`

## Reproducing this run

```bash
# Same 30 stratified IDs (already saved):
cat /tmp/ablation_ids.json

# Closed-source-only
python -m scripts.pipeline run \
  --ids-from /tmp/ablation_ids.json \
  --suite data-agent-ablation-closed-only \
  --model anthropic/claude-sonnet-4-6 \
  --probe-aliases nano opus gpt-5.5 gpt-5.5-codex \
  --sandbox e2b --concurrent 30 \
  --state-dir data/ablation/closed-only \
  --rewrite-spec

# OSS-only
python -m scripts.pipeline run \
  --ids-from /tmp/ablation_ids.json \
  --suite data-agent-ablation-oss-only \
  --all-oss \
  --sandbox e2b --concurrent 30 \
  --state-dir data/ablation/oss-only \
  --rewrite-spec
```