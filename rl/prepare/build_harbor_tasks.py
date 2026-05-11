"""Build a Harbor task suite from the cached source dataset.

`--name <slug>` controls all derived names:
  - local task dir:      rl/harbor/tasks/jupyter-agent-<slug>/
  - local data mirror:   rl/harbor/tasks/jupyter-agent-<slug>/data/<prefix>/   (gitignored)
  - HF Bucket:           hf://buckets/<user>/jupyter-agent-<slug>-data
  - HF Dataset repo:     hf://datasets/<user>/jupyter-agent-<slug>-harbor

Pipeline order (now does Kaggle download up front, persistently):

  Phase 1 — pick `N * --candidate-multiplier` candidate rows; download their
            UNIQUE Kaggle datasets in parallel via kagglehub. Rows whose
            dataset fails (auth-gated, removed, quota) are filtered out.
  Phase 2 — pick the first N rows whose data is present. Hardlink files into
            `data/<prefix>/` (persistent local mirror).
  Phase 3 — emit one Harbor task folder per row + manifest.jsonl + README.md.

Local layout after a run:
    harbor/tasks/jupyter-agent-<slug>/
    ├── README.md
    ├── manifest.jsonl
    ├── dropped.jsonl                 (only if some kaggle datasets failed)
    ├── data/                         ← persistent local data mirror (gitignored)
    │   └── <bucket_prefix>/...
    └── <safe_id>/
        ├── instruction.md
        ├── task.toml
        ├── tests/{test.sh, grader.py}
        └── environment/{Dockerfile, entrypoint.sh}

Usage:
  uv run python -m prepare.build_harbor_tasks --name test --n-tasks 5
  uv run python -m prepare.build_harbor_tasks --name test --n-tasks 5 --candidate-multiplier 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
RAW_DIR = RL_ROOT / "cache" / "raw"
HARBOR_ROOT = RL_ROOT / "harbor" / "tasks"
GRADER_SRC = RL_ROOT / "grader.py"
SOURCE_PROMPT = REPO_ROOT / "data" / "pipelines" / "prompts" / "agent_prompt_e2b.md"

DEFAULT_USER = "AdithyaSK"
BASE_PREFIX = "jupyter-agent"

KEEP_COLS = [
    "id", "question", "answer", "kaggle_dataset_name",
    "executor_type", "files_used", "packages_used", "edu_score",
]


# ---------------------------------------------------------------------------
# Naming (single source of truth for derived names)
# ---------------------------------------------------------------------------


def derive_names(slug: str, user: str = DEFAULT_USER) -> dict[str, str]:
    """One `slug` -> everything else.

    --name test  →  jupyter-agent-test  →
        bucket   = <user>/jupyter-agent-test-data
        repo     = <user>/jupyter-agent-test-harbor
        local    = rl/harbor/tasks/jupyter-agent-test/
        data     = rl/harbor/tasks/jupyter-agent-test/data/
    """
    base = f"{BASE_PREFIX}-{slug}"
    local_dir = HARBOR_ROOT / base
    return {
        "slug": slug,
        "base": base,
        "bucket_id": f"{user}/{base}-data",
        "repo_id": f"{user}/{base}-harbor",
        "local_dir": str(local_dir),
        "local_data_dir": str(local_dir / "data"),
    }


# ---------------------------------------------------------------------------
# Kaggle auth (accepts new KGAT_ token or legacy username/key)
# ---------------------------------------------------------------------------


def setup_kaggle_auth() -> str:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy KAGGLE_USERNAME/KAGGLE_KEY"
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return f"kaggle.json at {cfg}"
    token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY_TOKEN")
    if token and token.startswith("KGAT_"):
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({"username": "anonymous", "key": token}))
        cfg.chmod(0o600)
        os.environ["KAGGLE_USERNAME"] = "anonymous"
        os.environ["KAGGLE_KEY"] = token
        return f"wrote {cfg} from KAGGLE_API_TOKEN"
    raise RuntimeError(
        "No Kaggle credentials. Set KAGGLE_USERNAME+KAGGLE_KEY, place "
        "~/.kaggle/kaggle.json, or export KAGGLE_API_TOKEN=KGAT_..."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_id(raw_id: str) -> str:
    cleaned = raw_id.replace(".ipynb", "")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", cleaned).strip("_")


def _bucket_prefix(kaggle_dataset_name: str) -> str:
    return kaggle_dataset_name.replace("/", "__")


def _basename(path: str) -> str:
    return path.split("/")[-1] if "/" in path else path


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


# ---------------------------------------------------------------------------
# Phase 1: parallel Kaggle download
# ---------------------------------------------------------------------------


def download_kaggle(name: str) -> tuple[str, Path | None, str | None]:
    """Returns (kaggle_name, local_path or None, error or None)."""
    import kagglehub
    try:
        path = Path(kagglehub.dataset_download(name))
        if not path.exists():
            return name, None, f"missing local path: {path}"
        return name, path, None
    except Exception as exc:  # noqa: BLE001
        return name, None, f"{type(exc).__name__}: {exc}"


def parallel_download(unique_names: Iterable[str], max_workers: int) -> tuple[dict[str, Path], dict[str, str]]:
    successes: dict[str, Path] = {}
    failures: dict[str, str] = {}
    names = list(unique_names)
    if not names:
        return successes, failures
    print(f"[kaggle] downloading {len(names)} unique dataset(s) ({max_workers} workers)…")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_kaggle, n): n for n in names}
        for fut in as_completed(futures):
            name, path, err = fut.result()
            if err is None:
                files = sum(1 for _ in path.rglob("*") if _.is_file())  # type: ignore[union-attr]
                size_mb = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)  # type: ignore[union-attr]
                print(f"  [ok]   {name}: {files} file(s), {size_mb:.1f} MB")
                successes[name] = path  # type: ignore[assignment]
            else:
                print(f"  [drop] {name}: {err}")
                failures[name] = err
    return successes, failures


# ---------------------------------------------------------------------------
# Phase 2: hardlink/copy into the local data mirror
# ---------------------------------------------------------------------------


def mirror_into_data_dir(local_path: Path, dest_dir: Path) -> int:
    """Flatten files from kagglehub's nested path into dest_dir/<basename>."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in local_path.rglob("*"):
        if not src.is_file():
            continue
        dst = dest_dir / src.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_instruction(question: str, files_used: list[str], packages_used: list[str]) -> str:
    template = SOURCE_PROMPT.read_text(encoding="utf-8")
    files_str = "\n".join(f"- {_basename(f)}" for f in files_used) or "(none staged)"
    packages_str = ", ".join(packages_used) if packages_used else "(none)"
    body = template.format(files=files_str, packages=packages_str, question=question)
    footer = (
        "\n\n---\nWhen you have the final answer, call the `final_answer` tool "
        "with a short concise value, OR (Harbor CLI path) write it to "
        "`/workdir/answer.txt`. The grader compares (case-insensitive, "
        "numeric tolerance, LLM judge) to the gold answer."
    )
    return body + footer


def render_task_toml(
    *, base: str, task_id_safe: str, raw_id: str, kaggle_dataset_name: str,
    bucket_id: str, bucket_prefix: str, expected_answer: str, question: str,
) -> str:
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    desc = question if len(question) <= 200 else question[:197] + "..."
    return f"""schema_version = "1.2"
artifacts = []

[task]
name = "{base}/{task_id_safe}"
description = {q(desc)}
authors = []
keywords = ["jupyter-agent", "data-analysis", "kaggle"]

[metadata]
source_dataset = "jupyter-agent/jupyter-agent-dataset"
source_row_id = {q(raw_id)}
kaggle_dataset_name = {q(kaggle_dataset_name)}
gold_answer = {q(expected_answer)}

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
timeout_sec = 120.0
start_period_sec = 5.0
start_interval_sec = 2.0
retries = 30

[environment.env]
HF_BUCKET = "{bucket_id}"
BUCKET_PREFIX = "{bucket_prefix}"
HF_TOKEN = "${{HF_TOKEN}}"
KAGGLE_DATASET_NAME = {q(kaggle_dataset_name)}

[verifier]
timeout_sec = 120.0

[verifier.env]
EXPECTED_ANSWER = {q(expected_answer)}
QUESTION = {q(question)}
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"

[agent]
timeout_sec = 900.0

[solution.env]
"""


TEST_SH = r'''#!/usr/bin/env bash
set -u
mkdir -p /logs/verifier

answer_path="/workdir/answer.txt"
if [ ! -s "$answer_path" ]; then
  echo "0.0" > /logs/verifier/reward.txt
  echo "[grader] no answer at $answer_path" >&2
  exit 0
fi

pip install --quiet openai >/dev/null 2>&1 || true
python3 /tests/grader.py < "$answer_path" > /logs/verifier/reward.txt
'''


# Bucket-only design — no `COPY data`. The HF Bucket is the canonical data
# store. Harbor's `[environment.healthcheck]` (declared in task.toml) runs
# /opt/pull_bucket.py BEFORE the agent setup, fetching this task's per-prefix
# files into /home/user/input/. Works with stock CLI agents (opencode/codex
# bypass our ENTRYPOINT but always run the healthcheck) and custom agents.
DOCKERFILE = r"""FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "huggingface_hub>=1.12" \
        "openai>=1.40" \
        pandas numpy matplotlib seaborn scipy scikit-learn statsmodels tabulate

ENV HF_XET_HIGH_PERFORMANCE=1

RUN mkdir -p /home/user/input /workdir

# Bucket-pull script invoked by [environment.healthcheck] in task.toml.
COPY pull_bucket.py /opt/pull_bucket.py

WORKDIR /workdir
"""


# environment/pull_bucket.py — staged into every task's build context. Pulls
# the per-task HF_BUCKET/BUCKET_PREFIX into /home/user/input/. Idempotent.
PULL_BUCKET_PY = r'''"""Pull this task's bucket prefix into /home/user/input/.

Invoked by Harbor's [environment.healthcheck] command (declared in task.toml)
— runs after container start, before the agent. Idempotent: skips work if
files are already present from a prior pull.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import download_bucket_files, list_bucket_tree


def main() -> int:
    bucket = os.environ["HF_BUCKET"]
    prefix = os.environ["BUCKET_PREFIX"].rstrip("/") + "/"
    dest = Path("/home/user/input")
    dest.mkdir(parents=True, exist_ok=True)

    existing = [p for p in dest.iterdir() if p.is_file()]
    if existing:
        print(f"[pull_bucket] {dest}/ already has {len(existing)} file(s); skipping", flush=True)
        return 0

    targets = [
        (it.path, str(dest / Path(it.path).name))
        for it in list_bucket_tree(bucket, prefix=prefix, recursive=True)
        if getattr(it, "type", None) == "file"
    ]
    if not targets:
        print(f"[pull_bucket] FATAL: no files at hf://buckets/{bucket}/{prefix}", flush=True)
        return 2

    download_bucket_files(bucket, files=targets)
    print(f"[pull_bucket] staged {len(targets)} file(s) from {bucket}/{prefix}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="test")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument(
        "--candidate-multiplier", type=int, default=3,
        help="Over-pick N*M candidates so we can survive failed Kaggle downloads.",
    )
    parser.add_argument(
        "--source-split", default="non_thinking",
        choices=["thinking", "non_thinking"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    names = derive_names(args.name, user=args.user)
    out_dir = Path(names["local_dir"])
    data_dir = Path(names["local_data_dir"])
    print(f"[plan] --name={names['slug']}")
    print(f"  base name:    {names['base']}")
    print(f"  local dir:    {names['local_dir']}")
    print(f"  data dir:     {names['local_data_dir']}")
    print(f"  bucket:       hf://buckets/{names['bucket_id']}")
    print(f"  hub repo:     hf://datasets/{names['repo_id']}")

    if not GRADER_SRC.exists():
        print(f"[err] {GRADER_SRC} missing — write rl/grader.py first.")
        return 1

    in_path = RAW_DIR / f"{args.source_split}.parquet"
    if not in_path.exists():
        print(f"[miss] {in_path} — run prepare.cache_dataset first.")
        return 1

    print(f"\n[load] {in_path}")
    from datasets import Dataset
    ds = Dataset.from_parquet(str(in_path))

    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    if drop:
        ds = ds.remove_columns(drop)
    ds = ds.filter(
        lambda r: bool(r.get("kaggle_dataset_name"))
        and r.get("executor_type") == "e2b"
        and bool(r.get("answer")),
        desc="filter e2b + kaggle + answer",
    )

    # Dedup by (kaggle_dataset_name, question); keep highest edu_score.
    seen: dict[tuple[str, str], dict] = {}
    for row in ds:
        key = (row["kaggle_dataset_name"], (row.get("question") or "").strip())
        prev = seen.get(key)
        if prev is None or (row.get("edu_score") or 0) > (prev.get("edu_score") or 0):
            seen[key] = row
    rows = list(seen.values())

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    candidates = rows[: args.n_tasks * args.candidate_multiplier]
    print(f"[candidates] {len(candidates)} (target N={args.n_tasks}, multiplier={args.candidate_multiplier})")

    # ── Phase 1: parallel Kaggle download ───────────────────────────────────
    setup_kaggle_auth()
    unique_kaggles = []
    seen_set: set[str] = set()
    for r in candidates:
        n = r["kaggle_dataset_name"]
        if n not in seen_set:
            seen_set.add(n)
            unique_kaggles.append(n)
    successes, failures = parallel_download(unique_kaggles, max_workers=args.max_workers)

    # ── Phase 2: pick first N rows whose kaggle is downloaded ───────────────
    accepted: list[tuple[dict, Path]] = []
    for r in candidates:
        kg = r["kaggle_dataset_name"]
        if kg in successes:
            accepted.append((r, successes[kg]))
            if len(accepted) >= args.n_tasks:
                break

    if len(accepted) < args.n_tasks:
        print(
            f"[warn] only {len(accepted)} task(s) survived kaggle download; "
            f"raise --candidate-multiplier (now {args.candidate_multiplier})."
        )

    if not accepted:
        print("[err] no successful tasks. exiting.", file=sys.stderr)
        return 1

    # ── Phase 3: write Harbor folders + local data mirror ───────────────────
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    bucket_datasets: set[str] = set()
    mirrored_prefixes: set[str] = set()

    for row, kaggle_local in accepted:
        safe_id = _safe_id(row["id"])
        kaggle = row["kaggle_dataset_name"]
        prefix = _bucket_prefix(kaggle)
        files = _coerce_list(row.get("files_used"))
        packages = _coerce_list(row.get("packages_used"))
        answer = (row.get("answer") or "").strip()
        question = (row.get("question") or "").strip()

        # Mirror data once per unique dataset into the shared data/<prefix>/.
        dataset_dest = data_dir / prefix
        if prefix not in mirrored_prefixes:
            n_mirrored = mirror_into_data_dir(kaggle_local, dataset_dest)
            mirrored_prefixes.add(prefix)
            print(f"  [mirror] {prefix}: {n_mirrored} file(s) → {dataset_dest}")

        task_dir = out_dir / safe_id
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)

        # NOTE: no per-task environment/data/ hardlink anymore. The container's
        # [environment.healthcheck] pulls this task's prefix from the bucket at
        # runtime. Local data/<prefix>/ is kept only as an offline-dev cache.

        (task_dir / "instruction.md").write_text(
            render_instruction(question, files, packages), encoding="utf-8"
        )
        (task_dir / "task.toml").write_text(
            render_task_toml(
                base=names["base"], task_id_safe=safe_id, raw_id=row["id"],
                kaggle_dataset_name=kaggle, bucket_id=names["bucket_id"],
                bucket_prefix=prefix, expected_answer=answer, question=question,
            ),
            encoding="utf-8",
        )
        (task_dir / "tests" / "test.sh").write_text(TEST_SH, encoding="utf-8")
        (task_dir / "tests" / "test.sh").chmod(0o755)
        shutil.copy2(GRADER_SRC, task_dir / "tests" / "grader.py")
        (task_dir / "environment" / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
        (task_dir / "environment" / "pull_bucket.py").write_text(PULL_BUCKET_PY, encoding="utf-8")

        bucket_datasets.add(kaggle)
        manifest_lines.append(
            json.dumps({
                "safe_id": safe_id,
                "raw_id": row["id"],
                "kaggle_dataset_name": kaggle,
                "bucket_id": names["bucket_id"],
                "bucket_prefix": prefix,
                "local_data_dir": str(dataset_dest.relative_to(out_dir)),
                "files_used": files,
                "edu_score": int(row.get("edu_score") or 0),
            })
        )
        print(f"  [+] {safe_id}  kaggle={kaggle}  gold={answer!r}")

    (out_dir / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    if failures:
        dropped_path = out_dir / "dropped.jsonl"
        dropped_path.write_text(
            "\n".join(json.dumps({"kaggle_dataset_name": n, "reason": r}) for n, r in failures.items())
            + "\n",
            encoding="utf-8",
        )
        print(f"[dropped] {len(failures)} kaggle dataset(s) recorded at {dropped_path}")

    bucket_url = f"https://huggingface.co/buckets/{names['bucket_id']}"
    (out_dir / "README.md").write_text(
        f"""---
license: mit
tags:
  - harbor
  - jupyter-agent
  - data-analysis
  - kaggle
---

# {names['base']} — Harbor task suite

{len(accepted)} Harbor task(s) for the Jupyter data-analysis agent. Each task
is one (question, gold answer) pair against a real Kaggle dataset.

> 📦 **Data for these tasks lives in the HF Bucket** → [`{names['bucket_id']}`]({bucket_url}) — `hf://buckets/{names['bucket_id']}`
>
> The Harbor task definitions in this repo are tiny (instruction + grader +
> Dockerfile). The actual CSVs are pulled from the bucket at container start
> via `[environment.healthcheck]` → `pull_bucket.py`. **No data is committed
> to this repo**; the bucket is the canonical store.

## What's in this repo

| Path | What it is |
|---|---|
| `<task_id>/instruction.md` | Prompt the agent sees |
| `<task_id>/task.toml` | Harbor schema 1.2 — env vars (`HF_BUCKET`, `BUCKET_PREFIX`), healthcheck, verifier config, oracle |
| `<task_id>/environment/Dockerfile` | Minimal Python image — no data baked in |
| `<task_id>/environment/pull_bucket.py` | Pulls the per-task bucket prefix into `/home/user/input/` at container start |
| `<task_id>/tests/test.sh` | Calls `grader.py` against `/workdir/answer.txt` |
| `<task_id>/tests/grader.py` | 3-tier grader: exact / numeric tolerance / LLM-judge (gpt-4o-mini, simple-evals prompt) |
| `manifest.jsonl` | Maps each task → kaggle dataset → bucket prefix |

## Quick run (3rd-person flow)

```bash
# 1. Clone the suite
huggingface-cli download {names['repo_id']} --repo-type dataset --local-dir ./{names['base']}

# 2. Set env vars (HF_TOKEN reads the bucket, OPENAI_API_KEY for the agent + grader)
export HF_TOKEN=hf_…           # READ access to the bucket
export OPENAI_API_KEY=sk-…

# 3. Run Harbor with any built-in agent
harbor run -p ./{names['base']} \\
    --agent opencode --env docker --model openai/gpt-5 \\
    --ae OPENAI_API_KEY="$OPENAI_API_KEY"
```

Per-task flow when Harbor runs:

1. Build the per-task image from `<task_id>/environment/Dockerfile` (no data baked).
2. Container starts. `[environment.healthcheck]` invokes `python3 /opt/pull_bucket.py`
   which reads `HF_BUCKET` + `BUCKET_PREFIX` from the task's `[environment.env]`
   and pulls just that prefix from [`{names['bucket_id']}`]({bucket_url}) into
   `/home/user/input/`.
3. Healthcheck passes; Harbor brings up the agent (opencode, codex, etc., or
   our custom `JupyterToolAgent`) and runs the task.
4. Agent writes `/workdir/answer.txt`.
5. `tests/test.sh` runs `grader.py` against the gold answer; reward to
   `/logs/verifier/reward.txt`.

## Provenance

- **Source dataset:** [`jupyter-agent/jupyter-agent-dataset`](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset) — 51,389 rows × 2 splits
- **Pipeline:** [github.com/your-org/jupyter-agent](https://github.com/) (rl/ subfolder)
- **Sampling:** `slug={names['slug']}`, `seed={args.seed}`, top {len(accepted)} of filtered candidates
- **Unique Kaggle datasets:** {len(bucket_datasets)}
- **Generated:** {Path(__file__).name}

## See also

- 📦 **Data bucket:** [`{names['bucket_id']}`]({bucket_url}) — `hf://buckets/{names['bucket_id']}`
- 📁 **Spec repo (this one):** [`{names['repo_id']}`](https://huggingface.co/datasets/{names['repo_id']})
""",
        encoding="utf-8",
    )

    print(f"\n[done] {len(accepted)} task(s) at {out_dir}; "
          f"{len(bucket_datasets)} unique Kaggle dataset(s) mirrored to {data_dir}.")
    print(f"[manifest] {out_dir / 'manifest.jsonl'}")
    print(f"\nNext: upload data to bucket, then push spec to Hub:")
    print(f"  uv run python -m prepare.stage_data --name {args.name}")
    print(f"  uv run python -m prepare.push_harbor --name {args.name}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
