#!/usr/bin/env bash
# Parallel Qwen scaling matrix — 3 sizes × 4 agents = 12 runs, all in parallel.
# Each `harbor run` gets its own E2B sandbox, so they don't fight for resources
# (subject to E2B/Nscale rate limits — should be fine for 12).
#
# Usage: bash rl/rollouts/run_qwen_scaling_parallel.sh

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)
HF_KEY=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2-)
SUITE=rl/harbor/tasks/jupyter-agent-eval-smoke3
JOBS_DIR=rl/jobs
LOGS_DIR=rl/jobs/_qwen_logs
EASY=0065_794_65794937_qa_1
mkdir -p "$LOGS_DIR"

declare -a MODELS=(
  "4b|hf/Qwen/Qwen3-4B-Instruct-2507:nscale|huggingface/Qwen/Qwen3-4B-Instruct-2507:nscale"
  "8b|hf/Qwen/Qwen3-8B:nscale|huggingface/Qwen/Qwen3-8B:nscale"
  "14b|hf/Qwen/Qwen3-14B:nscale|huggingface/Qwen/Qwen3-14B:nscale"
)
declare -a AGENTS=(
  "jupy|--agent-import-path rl.harbor_agents.jupyter:JupyterToolAgent|0"
  "bash|--agent-import-path rl.harbor_agents.bash:BashOnlyAgent|0"
  "seta|--agent-import-path rl.harbor_agents.seta:SetaToolAgent|0"
  "oc|--agent opencode|1"
)

pids=()
labels=()
for mentry in "${MODELS[@]}"; do
  IFS='|' read -r size our_model oc_model <<<"$mentry"
  for aentry in "${AGENTS[@]}"; do
    IFS='|' read -r agent agent_flag use_oc_fmt <<<"$aentry"
    model="$our_model"
    [ "$use_oc_fmt" = "1" ] && model="$oc_model"
    label="qwen-${size}-${agent}"
    rm -rf "$JOBS_DIR/$label"
    echo "[launch] $label  model=$model"
    (
      # shellcheck disable=SC2086
      harbor run \
        -p "$SUITE" \
        $agent_flag \
        --model "$model" \
        --ae OPENAI_API_KEY="$OPENAI_KEY" \
        --ae HF_TOKEN="$HF_KEY" \
        --ve OPENAI_API_KEY="$OPENAI_KEY" \
        --env e2b --env-file .env --yes \
        --job-name "$label" --jobs-dir "$JOBS_DIR" \
        -i "$EASY" -n 1 > "$LOGS_DIR/$label.log" 2>&1
    ) &
    pids+=($!)
    labels+=("$label")
  done
done

echo
echo "[wait] launched ${#pids[@]} parallel harbor runs (PIDs: ${pids[*]})"
echo "[wait] monitoring; per-run logs in $LOGS_DIR/<label>.log"
echo

# Wait for all
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  lbl="${labels[$i]}"
  if wait "$pid"; then
    echo "[done] $lbl  exit=0"
  else
    rc=$?
    echo "[fail] $lbl  exit=$rc"
  fi
done

echo
echo "============================================================"
echo " Qwen scaling matrix — easy task (gold='37163900')"
echo "============================================================"
python3 - <<'PY'
import json, glob
rows = []
for path in sorted(glob.glob("rl/jobs/qwen-*/result.json")):
    job = path.split("/")[-2]
    parts = job.split("-")
    if len(parts) < 3: continue
    size, agent = parts[1], parts[2]
    try:
        d = json.load(open(path))
        evs = list(d["stats"]["evals"].values())
        if not evs:
            rows.append((size, agent, "noeval")); continue
        e = evs[0]
        rs = e.get("reward_stats", {}).get("reward", {})
        excs = e.get("exception_stats", {})
        if excs:
            outcome = list(excs.keys())[0][:14]
        else:
            out = next((r for r, ids in rs.items() if ids), "-")
            outcome = out
        rows.append((size, agent, outcome))
    except Exception as ex:
        rows.append((size, agent, f"parse-err"))

agents = ["jupy", "bash", "seta", "oc"]
sizes = ["4b", "8b", "14b"]
print(f"{'size':<6s} | " + " | ".join(f"{a:^14s}" for a in agents))
print("-" * (8 + 17 * len(agents)))
for sz in sizes:
    cells = []
    for ag in agents:
        cell = next((r for s, a, r in rows if s == sz and a == ag), "-")
        cells.append(f"{cell:^14s}")
    print(f"{sz:<6s} | " + " | ".join(cells))
PY
