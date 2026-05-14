#!/usr/bin/env bash
# Run the easy tesla task on gpt-5 across 4 agents:
#   1. jupyter-tool      — custom 4-tool jupyter abstraction
#   2. mini-swe-agent    — Harbor built-in, single bash tool
#   3. seta-tool         — our SETA-style 10-tool agent (6 shell + 4 notes)
#   4. opencode          — Harbor built-in TUI agent (bash/edit/read)
#
# Validates that all 4 tool abstractions can solve the same task with the same
# agent-agnostic submission protocol (write to /workdir/answer.txt).
#
# Usage: bash rl/rollouts/run_easy_4agents.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)
SUITE=rl/harbor/tasks/jupyter-agent-eval-smoke3
JOBS_DIR=rl/jobs
EASY=0065_794_65794937_qa_1

# (label, agent_flag, optional --ak)
declare -a MATRIX=(
  "jupyter|--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent"
  "bash|--agent mini-swe-agent"
  "seta|--agent-import-path rl.harbor_agents.seta:SetaToolAgent"
  "opencode|--agent opencode"
)

for entry in "${MATRIX[@]}"; do
  IFS='|' read -r label agent_flag <<<"$entry"
  echo
  echo "============================================================"
  echo " gpt-5 × $label"
  echo "============================================================"
  rm -rf "$JOBS_DIR/easy4-$label"
  start=$(date +%s)
  # shellcheck disable=SC2086
  harbor run \
    -p "$SUITE" \
    $agent_flag \
    --model openai/gpt-5 \
    --ae OPENAI_API_KEY="$OPENAI_KEY" \
    --ve OPENAI_API_KEY="$OPENAI_KEY" \
    --env e2b --env-file .env --yes \
    --job-name "easy4-$label" --jobs-dir "$JOBS_DIR" \
    -i "$EASY" -n 1 2>&1 | tail -12
  echo "[$label] elapsed: $(( $(date +%s) - start ))s"
done

echo
echo "============================================================"
echo " Easy task × gpt-5 × 4 agents"
echo "============================================================"
python3 - <<'PY'
import json, glob
print(f"{'agent':<14s} {'reward':>7s} {'elapsed':>10s}")
print("-" * 36)
for path in sorted(glob.glob("rl/jobs/easy4-*/result.json")):
    job = path.split("/")[-2]
    label = job.replace("easy4-", "")
    with open(path) as f:
        d = json.load(f)
    e = list(d["stats"]["evals"].values())[0]
    rs = e.get("reward_stats", {}).get("reward", {})
    out = "-"
    for r, ids in rs.items():
        if ids:
            out = r
    started = d.get("started_at", "")
    finished = d.get("finished_at", "")
    try:
        from datetime import datetime
        elapsed = (datetime.fromisoformat(finished.rstrip("Z").replace("Z","")) - datetime.fromisoformat(started.rstrip("Z").replace("Z",""))).total_seconds()
        elapsed_s = f"{elapsed:.0f}s"
    except Exception:
        elapsed_s = "-"
    print(f"{label:<14s} {out:>7s} {elapsed_s:>10s}")
PY
