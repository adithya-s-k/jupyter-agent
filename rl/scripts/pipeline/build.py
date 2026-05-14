"""Phase A — generate a Harbor task spec for one row from the manifest.

Inputs: a row dict from data/splits/{eval,train}_manifest.parquet
Outputs: rl/harbor/tasks/data-agent-eval-v1/<id_safe>/
   ├── task.toml
   ├── instruction.md
   ├── tests/test.sh
   ├── tests/grader.py
   └── environment/
       ├── Dockerfile
       └── pull_bucket.py

Reuses the same template shape as the existing jupyter-agent-eval-v1 suite —
only the per-task fields (id, question, gold answer, kaggle, files) differ.

Intentionally minimal: no fancy multi-mode grader yet, no rubric.yaml. The
existing eval-v1 grader does exact → numeric → llm-judge fallback which
matches what we'd want for the default `flexible` reward mode anyway.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Mapping


# Layout constants (relative to rl/)
RL_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = RL_ROOT / "harbor" / "tasks"
DEFAULT_SUITE = "data-agent-eval-v1"

# Reuse the reference template's static files (Dockerfile, pull_bucket.py,
# grader.py). The eval-v1 task suite lives on disk locally — we copy from
# any of its tasks.
REFERENCE_TASK_DIR = TASKS_DIR / "jupyter-agent-eval-v1" / "0000_419_419825_qa_1"

# The HF bucket containing every Kaggle dataset we've mirrored.
DEFAULT_DATA_BUCKET_ID = "AdithyaSK/jupyter-agent-kaggle-all"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def id_safe(task_id: str) -> str:
    """`0074/276/74276642.ipynb_qa_3` → `0074_276_74276642_qa_3`."""
    return task_id.replace("/", "_").replace(".ipynb", "")


def kaggle_to_bucket_prefix(kaggle_name: str) -> str:
    """`pavansubhasht/ibm-hr-analytics-attrition-dataset`
       → `pavansubhasht__ibm-hr-analytics-attrition-dataset`."""
    return kaggle_name.replace("/", "__")


def _file_list_for_instruction(files_used) -> str:
    """Render the file list as instruction.md does: 'database.sqlite\n- foo.csv'."""
    names = []
    for f in files_used or []:
        s = str(f).strip()
        # paths in the dataset look like `../input/foo.csv` — keep just the basename
        names.append(Path(s).name)
    return "\n".join(f"- {n}" for n in names) if names else "- (see /home/user/input)"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

TASK_TOML_TEMPLATE = '''schema_version = "1.2"
artifacts = []

[task]
name = "{suite}/{id_safe}"
description = "{question_esc}"
authors = []
keywords = ["data-agent", "data-analysis", "kaggle"]

[metadata]
source_dataset = "jupyter-agent/jupyter-agent-dataset"
source_row_id = "{task_id}"
kaggle_dataset_name = "{kaggle}"
gold_answer = "{answer_esc}"
reward_mode_initial = "{reward_mode}"
package_tier = {package_tier}

[environment]
build_timeout_sec = 600.0
os = "linux"
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
allow_internet = true
mcp_servers = []

# Pre-agent hook: Harbor runs the command AFTER container start and BEFORE the
# agent setup begins. We use it to pull this task's bucket prefix into
# /home/user/input/. See environment/pull_bucket.py.
[environment.healthcheck]
command = "python3 /opt/pull_bucket.py && [ -n \\"$(ls /home/user/input)\\" ]"
interval_sec = 2.0
timeout_sec = 180.0
start_period_sec = 5.0
start_interval_sec = 2.0
retries = 30

[environment.env]
HF_BUCKET = "{data_bucket_id}"
BUCKET_PREFIX = "{bucket_prefix}"
HF_TOKEN = "${{HF_TOKEN}}"
KAGGLE_DATASET_NAME = "{kaggle}"

[verifier]
timeout_sec = 120.0

[verifier.env]
EXPECTED_ANSWER = "{answer_esc}"
QUESTION = "{question_esc}"
REWARD_MODE = "{reward_mode}"
ATOL = "{atol}"
RTOL = "{rtol}"
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"

[agent]
# Capped at 600s (10 min) to kill the long-tail stuck-agent cases without
# cutting off legitimate complex trials. Median Phase B trial is 60-120s;
# legitimate L4/L5 tasks can hit 200-300s; anything past 600s is almost
# certainly a stuck agent loop.
timeout_sec = 600.0

[solution.env]
'''


INSTRUCTION_TEMPLATE = '''You are an intelligent data science assistant with access to a stateful jupyter notebook environment you can interact with it using tool calling. For example, you have access to the add_and_execute_jupyter_code_cell tool.

You have access to the following files:
{files_list}
All of the files are located only in the '/home/user/input' folder without any folders inside 'input'. Do not use '/kaggle/input/' folder as it does not exist.

The following packages are already installed:
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly.

You are also allowed to install additional packages if needed via `pip install ...`.

Answer the following question based on the provided files:
{question}

Those are the guidelines for how to format your answer:
Answer must be short and concise. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.

To provide your final answer, you should call the final_answer tool using your tool calling capabilities. Do not do everything at once - break down your solution into smaller steps and code cell chunks, like data exploration, planning, data preprocessing required to answer the question and execution. Do not plot figures as they would not be visible. Look into previous conversation history and try not to get stuck on generating repetitive code.

---
**Work it out step by step.** Inspect the data first (head, shape, dtypes), write down what you observe, plan the computation, then execute it. If your agent has a notes/scratchpad tool, USE IT — jot down intermediate results, the columns you found, and the exact formula you're applying before the final calc. This is more reliable than reasoning silently across many tool calls.

**Submission protocol (READ CAREFULLY):**
1. Compute the answer in your sandbox.
2. Write **only the answer value** (no labels, no units, no trailing newline noise) to the absolute path `/workdir/answer.txt`. Examples:
   - Shell: `echo -n "<value>" > /workdir/answer.txt`
   - Python: `open("/workdir/answer.txt","w").write(str(<value>))`
3. **Do NOT use patch-style tools** (`apply_patch`, `edit`, diff patches) to write `answer.txt` — they resolve paths relative to a workspace root which may not include `/workdir/`. Always use a direct-write tool (shell redirect, file write) with the **absolute** path `/workdir/answer.txt`.
4. After the file is written, stop calling tools.

The grader does exact match → numeric tolerance → LLM judge against the gold answer. Keep the answer short and concise.'''


def _toml_escape(s: str) -> str:
    """Minimal TOML string escaping (inside double quotes)."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _infer_tolerances(answer: str) -> tuple[float, float]:
    """Pick per-task numeric (atol, rtol) from the gold answer's magnitude.

    Heuristic, not formal:
      - non-numeric or integer gold     → (0,        0)        exact int match
      - 1-2 decimals (e.g. "21.3")     → (0.05,     0.01)
      - 3+ decimals (e.g. "0.544341") → (1e-3,     0.005)
      - very large numbers (>1e6)      → (0,        0.005)
    """
    s = str(answer).strip()
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return (0.0, 0.0)
    # integer
    if "." not in s and "e" not in s.lower():
        return (0.0, 0.0)
    # count decimals
    decimals = len(s.split(".")[1]) if "." in s else 0
    if abs(v) > 1e6:
        return (0.0, 0.005)
    if decimals >= 3:
        return (1e-3, 0.005)
    return (0.05, 0.01)


def build_spec(row: Mapping, *, suite: str = DEFAULT_SUITE,
               data_bucket_id: str = DEFAULT_DATA_BUCKET_ID,
               out_root: Path | None = None,
               overwrite: bool = False) -> Path:
    """Generate a Harbor task folder for one manifest row.

    New specs go to `<suite>/pending/<id>/`. If the task already exists in
    *any* bucket (verified/dropped/phase_b_failed/pending), it's returned
    in-place — unless `overwrite=True`, in which case the existing dir is
    deleted and a fresh one created in pending/.

    Returns the path to the (re)created task dir.
    """
    from .buckets import find_task_dir, ensure_bucket

    out_root = out_root or TASKS_DIR
    suite_dir = out_root / suite
    task_id = str(row["id"])
    safe = id_safe(task_id)

    existing = find_task_dir(suite_dir, safe)
    if existing is not None and not overwrite:
        return existing
    if existing is not None and overwrite:
        shutil.rmtree(existing)

    out_dir = ensure_bucket(suite_dir, "pending") / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(exist_ok=True)
    (out_dir / "environment").mkdir(exist_ok=True)

    kaggle = str(row["kaggle_dataset_name"])
    reward_mode = str(row.get("reward_mode_initial") or "flexible")
    pt = row.get("package_tier")
    package_tier = int(pt) if pt is not None else 0
    fu = row.get("files_used")
    files_used = list(fu) if fu is not None else []

    # --- per-task numeric tolerance (only used by numeric / flexible modes)
    atol, rtol = _infer_tolerances(str(row["answer"]))

    # --- task.toml
    task_toml = TASK_TOML_TEMPLATE.format(
        suite=suite,
        id_safe=safe,
        task_id=task_id,
        question_esc=_toml_escape(str(row["question"])),
        answer_esc=_toml_escape(str(row["answer"])),
        kaggle=kaggle,
        reward_mode=reward_mode,
        package_tier=package_tier,
        data_bucket_id=data_bucket_id,
        bucket_prefix=kaggle_to_bucket_prefix(kaggle),
        atol=atol,
        rtol=rtol,
    )
    (out_dir / "task.toml").write_text(task_toml)

    # --- instruction.md
    instruction = INSTRUCTION_TEMPLATE.format(
        files_list=_file_list_for_instruction(files_used),
        question=str(row["question"]),
    )
    (out_dir / "instruction.md").write_text(instruction)

    # --- static files copied from the reference task
    ref = REFERENCE_TASK_DIR
    if not ref.exists():
        raise FileNotFoundError(
            f"reference template not found at {ref}. "
            "Need an existing eval-v1 task to copy Dockerfile + pull_bucket.py + grader.py from."
        )
    shutil.copy(ref / "tests" / "test.sh", out_dir / "tests" / "test.sh")
    shutil.copy(ref / "tests" / "grader.py", out_dir / "tests" / "grader.py")
    shutil.copy(ref / "environment" / "Dockerfile", out_dir / "environment" / "Dockerfile")
    shutil.copy(ref / "environment" / "pull_bucket.py", out_dir / "environment" / "pull_bucket.py")

    return out_dir
