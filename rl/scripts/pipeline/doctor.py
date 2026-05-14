"""Phase C — agentic doctor with file-editing tools.

Invoked after Phase B exhausts all K trials without a pass. Walks a tool
loop on Sonnet 4.6 to:
  - Diagnose why all trials failed.
  - Apply a fix if reasonable (REWARD_MODE change, EXPECTED_ANSWER update,
    instruction tweak).
  - Or declare unverifiable.

Hard caps: 20 tool calls, $0.50 doctor-side spend (NOT counting any
probe_with_model trials, which are tracked separately in state.jsonl).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import llm_client
from .build import TASKS_DIR, id_safe, DEFAULT_SUITE
from .verify import run_trial, TrialResult


DOCTOR_MODEL = "anthropic/claude-sonnet-4-6"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "doctor_system.md"
DOCTOR_SYSTEM_PROMPT = _PROMPT_PATH.read_text()

ALLOWED_PROBE_MODELS = {
    "gpt-5.5":       "openai/gpt-5.5",
    "opus":          "anthropic/claude-opus-4-7",
    "gpt-5.5-codex": "openai/gpt-5.5-codex",
}

ALLOWED_TOML_FIELDS = {"REWARD_MODE", "EXPECTED_ANSWER", "ATOL", "RTOL", "ANSWER_TYPE"}


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI tool-calling schema)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the task spec directory. Paths are relative to the task dir (e.g. 'task.toml', 'instruction.md').",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path inside the task dir."}
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "preview_dataset",
        "description": "Show the first N lines of a file from the kaggle bucket for this task. Fetches lazily from HF Hub on first call.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string", "description": "Filename from the bucket (no path prefix)."},
            "n": {"type": "integer", "description": "Number of lines to show (default 10).", "default": 10},
        }, "required": ["file"]},
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List all files in the kaggle bucket for this task.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "read_trajectory",
        "description": "Read the full trajectory + grader output + agent usage of one failed Phase-B trial. trial_idx ∈ {0,1,2}.",
        "parameters": {"type": "object", "properties": {
            "trial_idx": {"type": "integer", "description": "0-indexed Phase-B trial. The dossier already shows last 4 turns; use this to see the whole thing."},
        }, "required": ["trial_idx"]},
    }},
    {"type": "function", "function": {
        "name": "probe_with_model",
        "description": "Run ONE additional seta trial with a different model to gather an alt-answer. Synchronous; takes 30-90s. Costs $0.05-0.15. Allowed models: 'gpt-5.5', 'opus', 'gpt-5.5-codex'.",
        "parameters": {"type": "object", "properties": {
            "model": {"type": "string", "enum": list(ALLOWED_PROBE_MODELS.keys())},
        }, "required": ["model"]},
    }},
    {"type": "function", "function": {
        "name": "edit_task_toml",
        "description": "Set a field in [verifier.env] of task.toml. Allowed fields: REWARD_MODE, EXPECTED_ANSWER, ATOL, RTOL, ANSWER_TYPE. The pre-edit task.toml is snapshotted to specs/<id>/v0.toml automatically.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "enum": list(ALLOWED_TOML_FIELDS)},
            "value": {"type": "string"},
        }, "required": ["field", "value"]},
    }},
    {"type": "function", "function": {
        "name": "edit_instruction",
        "description": "Replace `old_string` with `new_string` in instruction.md. The match must be unique. Use this to clarify ambiguous questions.",
        "parameters": {"type": "object", "properties": {
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        }, "required": ["old_string", "new_string"]},
    }},
    {"type": "function", "function": {
        "name": "finalize",
        "description": "Terminate the doctor loop with a verdict.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["spec_fixed", "gold_corrected", "verifiable_judge", "unverifiable"]},
            "reasoning": {"type": "string", "description": "One sentence summary of what you found."},
        }, "required": ["verdict", "reasoning"]},
    }},
]


# ---------------------------------------------------------------------------
# Dossier — what the doctor sees on turn 0
# ---------------------------------------------------------------------------

def _format_trajectory_tail(messages: list, n_turns: int = 4) -> str:
    """Last N assistant turns + their tool results, truncated."""
    out: list[str] = []
    # Walk from the end, collect last n assistant turns
    asst_idxs = [i for i, m in enumerate(messages) if (m.get("role") == "assistant")]
    if not asst_idxs:
        return "(no assistant turns)"
    keep_from = asst_idxs[-n_turns] if len(asst_idxs) >= n_turns else 0
    for m in messages[keep_from:]:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " | ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        content = str(content)[:600]
        line = f"  [{role}] {content}"
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = str(fn.get("arguments", ""))[:400]
                line += f"\n    → CALL {fn.get('name')}({args})"
        out.append(line)
    return "\n".join(out)


def _read_trial_artifacts(trial_dir: Path) -> dict:
    """Return predicted answer, reward, grader output, and full trajectory for a trial dir."""
    out: dict = {}
    try:
        out["reward"] = (trial_dir / "verifier" / "reward.txt").read_text().strip()
    except Exception:
        out["reward"] = ""
    try:
        out["grader_stdout"] = (trial_dir / "verifier" / "test-stdout.txt").read_text()
    except Exception:
        out["grader_stdout"] = ""
    m = re.search(r"pred=(['\"])(.*?)\1", out.get("grader_stdout", ""))
    out["predicted"] = m.group(2) if m else ""
    # trajectory file naming: seta_agent.trajectory.json (or similar)
    for f in (trial_dir / "agent").glob("*.trajectory.json"):
        try:
            out["trajectory"] = json.loads(f.read_text())
            out["trajectory_path"] = str(f)
        except Exception:
            pass
        break
    return out


def build_dossier(row: Mapping, spec_dir: Path, trial_dirs: list[Path]) -> str:
    """Construct the up-front user message: full run context for the doctor."""
    task_toml = (spec_dir / "task.toml").read_text() if (spec_dir / "task.toml").exists() else "(missing)"
    instruction = (spec_dir / "instruction.md").read_text() if (spec_dir / "instruction.md").exists() else "(missing)"

    trial_blocks: list[str] = []
    for idx, td in enumerate(trial_dirs):
        if td is None:
            trial_blocks.append(f"TRIAL {idx} [no trial dir captured]")
            continue
        a = _read_trial_artifacts(td)
        tail = ""
        if a.get("trajectory"):
            tail = _format_trajectory_tail(a["trajectory"], n_turns=4)
        trial_blocks.append(
            f"TRIAL {idx} [trial_dir={td.name}]\n"
            f"  REWARD:    {a.get('reward', '?')}\n"
            f"  PREDICTED: {a.get('predicted', '(none)')}\n"
            f"  GRADER:    {a.get('grader_stdout', '').strip()[:500]}\n"
            f"  LAST TURNS:\n{tail}\n"
            f"  (full trajectory available via read_trajectory({idx}))"
        )

    files_used = list(row.get("files_used") or [])
    return (
        f"TASK ID:          {row['id']}\n"
        f"QUESTION:         {row['question']}\n"
        f"GOLD ANSWER:      {row['answer']}\n"
        f"KAGGLE DATASET:   {row['kaggle_dataset_name']}\n"
        f"FILES USED:       {files_used}\n"
        f"REWARD_MODE:      {row.get('reward_mode_initial', '?')}\n"
        f"ANSWER_TYPE:      {row.get('answer_type', '?')}  (heuristic from classifier)\n"
        f"PACKAGE_TIER:     {row.get('package_tier', '?')}\n\n"
        f"SPEC FILES:\n"
        f"  task.toml:\n{task_toml}\n\n"
        f"  instruction.md:\n{instruction}\n\n"
        f"PHASE B FAILURES ({len(trial_blocks)} trials):\n"
        + "\n\n".join(trial_blocks)
        + "\n\n"
        "Diagnose. Call tools as needed. End with finalize."
    )


# ---------------------------------------------------------------------------
# Tool dispatchers
# ---------------------------------------------------------------------------

class DoctorCtx:
    """All the runtime state a tool dispatcher might need."""
    def __init__(self, *, row: Mapping, spec_dir: Path, trial_dirs: list[Path],
                 specs_archive_dir: Path, state, jobs_dir: Path):
        self.row = row
        self.spec_dir = spec_dir
        self.trial_dirs = trial_dirs        # paths to Phase B trial dirs by k_idx (0-2)
        self.probe_trial_dirs: list[Path] = []
        self.specs_archive_dir = specs_archive_dir
        self.state = state
        self.jobs_dir = jobs_dir
        self.bucket_files: list[str] | None = None
        self.v0_snapshotted = False

    def _snapshot_v0(self) -> None:
        if self.v0_snapshotted:
            return
        self.specs_archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.spec_dir / "task.toml", self.specs_archive_dir / "v0.toml")
        shutil.copy(self.spec_dir / "instruction.md", self.specs_archive_dir / "v0.instruction.md")
        self.v0_snapshotted = True


def _bucket_files(ctx: DoctorCtx) -> list[str]:
    if ctx.bucket_files is None:
        from huggingface_hub import list_bucket_tree
        kaggle = ctx.row["kaggle_dataset_name"]
        bucket = "AdithyaSK/jupyter-agent-kaggle-all"
        prefix = str(kaggle).replace("/", "__") + "/"
        try:
            ctx.bucket_files = [
                Path(it.path).name
                for it in list_bucket_tree(bucket, prefix=prefix, recursive=True)
                if getattr(it, "type", None) == "file"
            ]
        except Exception as e:
            ctx.bucket_files = []
            return [f"(bucket listing failed: {e})"]
    return ctx.bucket_files or []


def _dispatch_tool(name: str, args: dict, ctx: DoctorCtx) -> tuple[str, dict]:
    """Execute one tool call. Returns (string_result_for_llm, side_effects_dict)."""
    side: dict = {}

    if name == "read_file":
        path = (ctx.spec_dir / args["path"]).resolve()
        if ctx.spec_dir.resolve() not in path.parents and path != ctx.spec_dir / args["path"]:
            return f"ERROR: refused — {args['path']} escapes the task dir.", side
        try:
            return path.read_text()[:8000], side
        except FileNotFoundError:
            return f"ERROR: file not found: {args['path']}", side

    if name == "list_files":
        files = _bucket_files(ctx)
        return "Files in bucket:\n  " + "\n  ".join(files), side

    if name == "preview_dataset":
        from huggingface_hub import download_bucket_files
        kaggle = ctx.row["kaggle_dataset_name"]
        prefix = str(kaggle).replace("/", "__") + "/"
        local_cache = Path("/tmp/doctor_bucket_cache") / prefix
        local_cache.mkdir(parents=True, exist_ok=True)
        target = local_cache / args["file"]
        if not target.exists():
            try:
                download_bucket_files(
                    "AdithyaSK/jupyter-agent-kaggle-all",
                    files=[(prefix + args["file"], str(target))],
                )
            except Exception as e:
                return f"ERROR: could not fetch {args['file']}: {e}", side
        n = int(args.get("n", 10))
        # CSV → head; SQLite → tables; else read first chunk of bytes
        try:
            if args["file"].endswith(".csv"):
                import csv
                lines = []
                with target.open() as f:
                    rdr = csv.reader(f)
                    for i, row in enumerate(rdr):
                        if i >= n + 1: break
                        lines.append(",".join(str(c) for c in row))
                return f"--- {args['file']} (first {n} rows) ---\n" + "\n".join(lines), side
            else:
                return f"--- {args['file']} (first {n * 80} bytes) ---\n" + target.read_text(errors="replace")[: n * 80], side
        except Exception as e:
            return f"ERROR reading {args['file']}: {e}", side

    if name == "read_trajectory":
        idx = int(args["trial_idx"])
        if idx < 0 or idx >= len(ctx.trial_dirs):
            return f"ERROR: trial_idx {idx} out of range (0..{len(ctx.trial_dirs)-1})", side
        td = ctx.trial_dirs[idx]
        if td is None:
            return f"ERROR: trial {idx} has no captured dir.", side
        a = _read_trial_artifacts(td)
        out = (f"TRIAL {idx} full artifacts:\n"
               f"REWARD: {a.get('reward')}\n"
               f"PREDICTED: {a.get('predicted')}\n\n"
               f"GRADER STDOUT:\n{a.get('grader_stdout', '')}\n\n"
               f"TRAJECTORY:\n")
        if a.get("trajectory"):
            out += json.dumps(a["trajectory"], indent=2)[:20000]
        return out, side

    if name == "probe_with_model":
        m = args["model"]
        if m not in ALLOWED_PROBE_MODELS:
            return f"ERROR: model {m} not allowed.", side
        full_model = ALLOWED_PROBE_MODELS[m]
        suite_name = ctx.spec_dir.parent.name
        suite_path = ctx.spec_dir.parent
        task_id = ctx.row["id"]
        probe_idx = len(ctx.probe_trial_dirs)
        slug = full_model.replace("/", "-").replace(".", "-")
        job_name = f"doctor-probe-{id_safe(task_id)}-{slug}-{probe_idx}"
        ctx.state.append_event(event="probe_start", task_id=task_id, phase="C",
                               model=full_model, job_name=job_name)
        t0 = time.time()
        result: TrialResult = run_trial(
            suite_path=suite_path, task_id=task_id, model=full_model,
            job_name=job_name, jobs_dir=ctx.jobs_dir,
            sandbox="docker", log_dir=ctx.state.state_dir / "logs",
        )
        ctx.probe_trial_dirs.append(result.trial_dir)
        ctx.state.append_event(
            event="probe_finish", task_id=task_id, phase="C",
            model=full_model, job_name=job_name,
            reward=result.reward, predicted=result.predicted_answer,
            error_kind=result.error_kind, elapsed_sec=time.time() - t0,
            cost_usd=result.cost_usd,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            trial_dir=str(result.trial_dir) if result.trial_dir else None,
        )
        side["probe_cost"] = result.cost_usd
        return (f"PROBE [{full_model}] reward={result.reward:.2f} "
                f"predicted={result.predicted_answer!r} cost=${result.cost_usd:.4f}"), side

    if name == "edit_task_toml":
        if args["field"] not in ALLOWED_TOML_FIELDS:
            return f"ERROR: field {args['field']} not allowed.", side
        ctx._snapshot_v0()
        toml_path = ctx.spec_dir / "task.toml"
        text = toml_path.read_text()
        # Find the value inside [verifier.env]
        pattern = rf'^({re.escape(args["field"])}\s*=\s*)("[^"]*"|\S+)\s*$'
        new_line = f'{args["field"]} = "{args["value"]}"'
        if re.search(pattern, text, flags=re.MULTILINE):
            text = re.sub(pattern, new_line, text, count=1, flags=re.MULTILINE)
        else:
            # Field missing in [verifier.env] — insert at end of that section
            ve_pattern = r'(\[verifier\.env\][^\[]*)'
            if re.search(ve_pattern, text):
                text = re.sub(ve_pattern, lambda m: m.group(1).rstrip() + f"\n{new_line}\n", text, count=1)
            else:
                return "ERROR: [verifier.env] section not found in task.toml", side
        # Validate it parses
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        try:
            tomllib.loads(text)
        except Exception as e:
            return f"ERROR: edit produced invalid TOML; reverted. {e}", side
        toml_path.write_text(text)
        side["spec_edited"] = True
        return f"OK: set {args['field']} = {args['value']!r} in [verifier.env]", side

    if name == "edit_instruction":
        ctx._snapshot_v0()
        path = ctx.spec_dir / "instruction.md"
        text = path.read_text()
        if text.count(args["old_string"]) != 1:
            return f"ERROR: old_string not unique (or not found): occurs {text.count(args['old_string'])}×.", side
        path.write_text(text.replace(args["old_string"], args["new_string"], 1))
        side["spec_edited"] = True
        return "OK: instruction edited.", side

    if name == "finalize":
        side["finalize"] = {
            "verdict": args["verdict"],
            "reasoning": args.get("reasoning", ""),
        }
        return f"FINALIZED verdict={args['verdict']}", side

    return f"ERROR: unknown tool {name}", side


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class DoctorResult:
    verdict: str           # "spec_fixed" | "gold_corrected" | "verifiable_judge" | "unverifiable"
    reasoning: str
    spec_edited: bool
    n_tool_calls: int
    total_cost_usd: float  # LLM doctor turns; probe costs are tracked separately in state.jsonl
    probe_cost_usd: float


def run_doctor(*, row: Mapping, spec_dir: Path, trial_dirs: list[Path],
               state, jobs_dir: Path,
               specs_archive_dir: Path,
               max_calls: int = 20,
               max_budget: float = 0.50,
               temperature: float = 0.0,
               model: str = DOCTOR_MODEL) -> DoctorResult:
    task_id = str(row["id"])
    ctx = DoctorCtx(row=row, spec_dir=spec_dir, trial_dirs=trial_dirs,
                    specs_archive_dir=specs_archive_dir, state=state, jobs_dir=jobs_dir)

    user_dossier = build_dossier(row, spec_dir, trial_dirs)
    messages: list[dict] = [
        {"role": "system", "content": DOCTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_dossier},
    ]
    state.append_event(event="doctor_start", task_id=task_id, phase="C",
                       model=model, dossier_chars=len(user_dossier))

    total_doctor_cost = 0.0
    total_probe_cost = 0.0
    spec_edited = False
    verdict = "unverifiable"
    reasoning = "doctor terminated without finalize"
    n_calls = 0

    for turn in range(max_calls + 4):  # extra slack for tool-result roundtrips
        resp = llm_client.call(model=model, messages=messages, tools=TOOLS,
                               temperature=temperature)
        total_doctor_cost += resp.cost_usd
        state.append_event(
            event="doctor_turn", task_id=task_id, phase="C",
            turn=turn, cost_usd=resp.cost_usd,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached_tokens=resp.cached_tokens,
            n_tool_calls=len(resp.tool_calls),
            finish_reason=resp.finish_reason,
        )

        # Append the assistant turn (even if no content, it may carry tool_calls)
        assistant_entry: dict = {"role": "assistant", "content": resp.content or ""}
        if resp.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ]
        messages.append(assistant_entry)

        if not resp.tool_calls:
            reasoning = "doctor produced no tool call; treating as no-op finalize"
            verdict = "unverifiable"
            break

        finished = False
        for tc in resp.tool_calls:
            n_calls += 1
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_result_text, side = _dispatch_tool(tc["name"], args, ctx)
            state.append_event(
                event="doctor_tool", task_id=task_id, phase="C",
                turn=turn, n_call=n_calls,
                tool=tc["name"], arguments=tc["arguments"][:400],
                result_preview=tool_result_text[:200],
            )
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": tool_result_text})

            if side.get("spec_edited"):
                spec_edited = True
            if side.get("probe_cost"):
                total_probe_cost += float(side["probe_cost"])

            if side.get("finalize"):
                verdict = side["finalize"]["verdict"]
                reasoning = side["finalize"]["reasoning"]
                finished = True
                break

        if finished:
            break

        if n_calls >= max_calls:
            verdict = "unverifiable"
            reasoning = f"doctor_budget_exhausted (n_calls={n_calls})"
            break
        if total_doctor_cost + total_probe_cost >= max_budget:
            verdict = "unverifiable"
            reasoning = f"doctor_budget_exhausted (cost=${total_doctor_cost+total_probe_cost:.4f})"
            break

    state.append_event(event="doctor_finish", task_id=task_id, phase="C",
                       verdict=verdict, reasoning=reasoning,
                       n_tool_calls=n_calls,
                       doctor_cost_usd=total_doctor_cost,
                       probe_cost_usd=total_probe_cost,
                       spec_edited=spec_edited)

    return DoctorResult(verdict=verdict, reasoning=reasoning,
                        spec_edited=spec_edited, n_tool_calls=n_calls,
                        total_cost_usd=total_doctor_cost,
                        probe_cost_usd=total_probe_cost)
