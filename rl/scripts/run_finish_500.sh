#!/bin/bash
# Drive Stage 1 + Stage 2 to completion on the remaining-eval set.
# Emits clearly-marked phase boundaries so the monitor can track.
set -e
cd /Users/adithya/projects/jupyter-agent/rl

mkdir -p /tmp/finish500
echo "=================================================="
echo "=== STAGE 1: Sonnet anchor + categorize on pass ==="
echo "=================================================="
date -u
echo "tasks: $(jq length cache/eval_remaining_278.json) (from cache/eval_remaining_278.json)"
echo

uv run python -m scripts.pipeline run \
  --stage 1 \
  --sandbox docker \
  --concurrent 30 \
  --ids-from cache/eval_remaining_278.json \
  --suite data-agent-eval-v1 \
  --state-dir data/verification/eval \
  --subprocess-timeout-sec 900 \
  --task-timeout-sec 1200 \
  --stagger-sec 2.0 \
  --max-retries-per-trial 1 \
  --total-cost-cap 50.0

S1_RUN_ID=$(ls -t data/verification/eval/runs/ | head -1)
echo
echo "=================================================="
echo "=== STAGE 1 COMPLETE — preparing Stage 2 input ==="
echo "=================================================="
echo "stage1 run_id: $S1_RUN_ID"

# Extract phase_b_failed from THIS stage 1 run only (don't pick up legacy fails)
uv run python -c "
import json
path = 'data/verification/eval/runs/$S1_RUN_ID/state.jsonl'
events = [json.loads(l) for l in open(path) if l.strip()]
fails = sorted({e['task_id'] for e in events
                if e.get('event')=='task_finish' and e.get('verdict')=='phase_b_failed'})
with open('cache/stage2_eval_remaining.json','w') as f: json.dump(fails, f)
print(f'phase_b_failed from stage 1: {len(fails)}')
"

S2_COUNT=$(jq length cache/stage2_eval_remaining.json)
if [ "$S2_COUNT" -eq 0 ]; then
  echo "no Stage 2 work. STAGE 2 SKIPPED."
  echo "=== FULL PIPELINE DONE (no stage 2 needed) ==="
  date -u
  exit 0
fi

echo
echo "=================================================="
echo "=== STAGE 2: Doctor + categorize on $S2_COUNT failures ==="
echo "=================================================="
date -u

uv run python -m scripts.pipeline run \
  --stage 2 \
  --sandbox docker \
  --concurrent 30 \
  --ids-from cache/stage2_eval_remaining.json \
  --suite data-agent-eval-v1 \
  --state-dir data/verification/eval \
  --subprocess-timeout-sec 900 \
  --task-timeout-sec 1800 \
  --stagger-sec 2.0 \
  --max-retries-per-trial 1 \
  --total-cost-cap 50.0

echo
echo "=================================================="
echo "=== FULL PIPELINE DONE ==="
echo "=================================================="
date -u
