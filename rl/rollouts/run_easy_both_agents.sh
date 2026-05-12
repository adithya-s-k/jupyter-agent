#!/usr/bin/env bash
# Run the easy task through BOTH agents (jupyter-tool, opencode) × 3 models,
# now that final_answer is gone and the instruction always points at
# /workdir/answer.txt. Validates the agent-agnostic submission protocol.
#
# Usage: bash rl/rollouts/run_easy_both_agents.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

get_key() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }
OPENAI_KEY=$(get_key OPENAI_API_KEY)
ANTHROPIC_KEY=$(get_key ANTHROPIC_API_KEY)
HF_KEY=$(get_key HF_TOKEN)

SUITE=rl/harbor/tasks/jupyter-agent-eval-smoke3
JOBS_DIR=rl/jobs
EASY=0065_794_65794937_qa_1

# (agent_label, model, ae)
declare -a MATRIX=(
  "jupy-gpt5|--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent|openai/gpt-5"
  "jupy-sonnet|--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent|anthropic/claude-sonnet-4-6"
  "jupy-qwen|--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent|hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale"
  "oc-gpt5|--agent opencode|openai/gpt-5"
  "oc-sonnet|--agent opencode|anthropic/claude-sonnet-4-6"
  "oc-qwen|--agent opencode|huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale"
)

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r label agent_flag model <<<"$entry"
  echo
  echo "============================================================"
  echo " $label  agent=$agent_flag  model=$model"
  echo "============================================================"
  rm -rf "$JOBS_DIR/easy2-$label"
  start=$(date +%s)
  # shellcheck disable=SC2086
  harbor run \
    -p "$SUITE" \
    $agent_flag \
    --model "$model" \
    --ae OPENAI_API_KEY="$OPENAI_KEY" \
    --ae ANTHROPIC_API_KEY="$ANTHROPIC_KEY" \
    --ae HF_TOKEN="$HF_KEY" \
    --ve OPENAI_API_KEY="$OPENAI_KEY" \
    --env e2b --env-file .env --yes \
    --job-name "easy2-$label" --jobs-dir "$JOBS_DIR" \
    -i "$EASY" -n 1 2>&1 | tail -12
  echo "[$label] elapsed: $(( $(date +%s) - start ))s"
done

echo
echo "============================================================"
echo " Easy task — 6-cell matrix"
echo "============================================================"
python3 - <<'PY'
import json, glob
print(f"{'job':<28s} {'easy':>8s}")
print("-" * 38)
for path in sorted(glob.glob("rl/jobs/easy2-*/result.json")):
    job = path.split("/")[-2]
    with open(path) as f:
        d = json.load(f)
    e = list(d["stats"]["evals"].values())[0]
    rs = e.get("reward_stats", {}).get("reward", {})
    out = "-"
    for r, ids in rs.items():
        if ids:
            out = r
    print(f"{job:<28s} {out:>8s}")
PY
