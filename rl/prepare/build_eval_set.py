"""Build a 150-task eval set stratified by difficulty (easy / medium / hard).

The eval set is a small frozen benchmark we run every checkpoint against. The
training set is built later by `build_harbor_tasks.py --exclude-ids` so eval
rows are never seen during training.

Pipeline:
  1. Load cached source parquet (`cache/raw/non_thinking.parquet`).
  2. Filter to `executor_type == "e2b"`, non-null kaggle + answer, and drop
     rows whose Kaggle dataset is NOT in the all-bucket (license-walled etc).
  3. Dedup by (kaggle_dataset_name, question), keep highest edu_score.
  4. Compute heuristic features (package tier, answer type, q word count).
  5. Heuristic-prefilter to `--candidate-pool` rows, biased toward diversity
     across package tier × heuristic-difficulty so the LLM sees a varied pool.
  6. LLM-score difficulty 1-5 in parallel. Candidates are split 50/50 between
     OpenAI (gpt-4o-mini) and Anthropic (claude-haiku-4-5) for throughput.
  7. Bucket: score 1-2=easy, 3=medium, 4-5=hard.
  8. Stratified final sample: 50 each, max 2 per kaggle dataset, balance
     package tiers and answer types within each bucket.
  9. Persist (`cache/eval/`):
       candidates.parquet     — every LLM-scored candidate (for audit/reuse)
       eval_manifest.parquet  — the chosen 150 with all features
       eval_ids.txt           — one source-row id per line (suite generator input)

Usage:
  uv run python -m prepare.build_eval_set --n-per-bucket 50
  uv run python -m prepare.build_eval_set --candidate-pool 1500 --llm-workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "rl"
RAW_DIR = RL_ROOT / "cache" / "raw"
EVAL_DIR = RL_ROOT / "cache" / "eval"

BUCKET_ID = "AdithyaSK/jupyter-agent-kaggle-all"
META_PREFIX = "_meta/"

KEEP_COLS = [
    "id", "question", "answer", "kaggle_dataset_name",
    "executor_type", "files_used", "packages_used", "edu_score",
]

# ---------------------------------------------------------------------------
# Heuristic features
# ---------------------------------------------------------------------------

PKG_BASIC = {"pandas", "numpy", "os", "sys", "json", "re", "datetime",
             "collections", "math", "itertools", "csv", "io", "pathlib",
             "warnings", "subprocess", "tqdm", "sqlite3"}
PKG_VIZ = {"matplotlib", "seaborn", "plotly", "plotnine", "altair", "bokeh",
           "wordcloud", "geopandas", "folium", "missingno", "cufflinks",
           "graphviz", "yellowbrick"}
# Names normalized with `.lower().replace("-", "_")`.
PKG_ML = {"sklearn", "scikit_learn", "scipy", "statsmodels", "xgboost",
          "lightgbm", "catboost", "keras", "tensorflow", "torch", "pytorch",
          "transformers", "nltk", "spacy", "gensim", "lightning",
          "pytorch_lightning", "fastai", "tslearn", "lifelines", "shap",
          "tensorflow_addons", "imblearn", "imbalanced_learn",
          "category_encoders", "mlxtend", "eli5", "pandas_profiling"}


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    # numpy arrays + tuples + other iterables coming back from pyarrow.to_pandas
    try:
        if hasattr(value, "tolist"):
            return [str(x) for x in value.tolist()]
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value]
    try:
        return [str(x) for x in value]
    except TypeError:
        return [str(value)]


def _pkg_set(packages: list[str]) -> set[str]:
    out: set[str] = set()
    for p in packages or []:
        out.add(str(p).lower().split(".")[0].replace("-", "_"))
    return out


def package_tier(packages: list[str]) -> int:
    pkgs = _pkg_set(packages)
    if pkgs & PKG_ML:
        return 2  # ml/stats
    if pkgs & PKG_VIZ:
        return 1  # viz
    return 0      # basic


_NUMERIC_RE = re.compile(r"^-?[\d.,eE+\-]+%?$")


def answer_type(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "empty"
    # Strip common suffixes
    a_norm = a.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        float(a_norm)
        return "numeric"
    except ValueError:
        pass
    if _NUMERIC_RE.match(a):
        return "numeric"
    return "string"


def compute_features(row: dict) -> dict:
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    files = _coerce_list(row.get("files_used"))
    packages = _coerce_list(row.get("packages_used"))
    q_words = len(q.split())
    return {
        "q_word_count": q_words,
        "n_files": len(files),
        "n_packages": len(packages),
        "package_tier": package_tier(packages),
        "answer_type": answer_type(a),
        "answer_len": len(a),
    }


def heuristic_bucket(features: dict) -> str:
    """Coarse pre-LLM bucket used only to balance the candidate pool.

    The LLM is the actual difficulty signal; this just ensures we send a
    varied mix into it instead of 95% pandas one-liners.
    """
    tier = features["package_tier"]
    wc = features["q_word_count"]
    if tier == 2:
        return "hard"          # uses sklearn / scipy / statsmodels / ...
    if tier == 1 or wc >= 22 or features["answer_len"] > 20:
        return "medium"
    return "easy"


# ---------------------------------------------------------------------------
# Bucket coverage
# ---------------------------------------------------------------------------


def fetch_in_bucket_datasets(refresh: bool = False) -> set[str]:
    """Return set of kaggle_dataset_name's known uploaded to the all-bucket."""
    cache = EVAL_DIR / "_uploaded_kaggle_names.txt"
    if cache.exists() and not refresh:
        return set(cache.read_text().splitlines())
    from huggingface_hub import download_bucket_files

    tmp = EVAL_DIR / "_meta_pull"
    tmp.mkdir(parents=True, exist_ok=True)
    remote = f"{META_PREFIX}uploaded.jsonl"
    local = tmp / "uploaded.jsonl"
    download_bucket_files(BUCKET_ID, files=[(remote, str(local))])
    path = local
    names: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                names.add(json.loads(line)["kaggle_dataset_name"])
            except (json.JSONDecodeError, KeyError):
                pass
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(names)) + "\n")
    return names


# ---------------------------------------------------------------------------
# Loading + dedup
# ---------------------------------------------------------------------------


def load_source_rows(parquet_path: Path) -> list[dict]:
    print(f"[load] {parquet_path}")
    t0 = time.time()
    table = pq.read_table(str(parquet_path), columns=KEEP_COLS)
    df = table.to_pandas()
    rows = df.to_dict(orient="records")
    elapsed = time.time() - t0
    print(f"  read {len(rows):,} rows in {elapsed:.1f}s")
    return rows


def filter_rows(rows: list[dict], in_bucket: set[str]) -> list[dict]:
    out: list[dict] = []
    miss_kaggle = miss_answer = miss_exec = miss_bucket = 0
    for r in rows:
        kg = r.get("kaggle_dataset_name")
        if not kg:
            miss_kaggle += 1
            continue
        if (r.get("executor_type") or "") != "e2b":
            miss_exec += 1
            continue
        if not (r.get("answer") or "").strip():
            miss_answer += 1
            continue
        if kg not in in_bucket:
            miss_bucket += 1
            continue
        out.append(r)
    print(f"[filter] kept={len(out):,}  "
          f"-kaggle={miss_kaggle:,}  -nonE2B={miss_exec:,}  "
          f"-noAnswer={miss_answer:,}  -notInBucket={miss_bucket:,}")
    return out


def dedup_rows(rows: list[dict]) -> list[dict]:
    seen: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["kaggle_dataset_name"], (r.get("question") or "").strip())
        prev = seen.get(key)
        if prev is None or (r.get("edu_score") or 0) > (prev.get("edu_score") or 0):
            seen[key] = r
    out = list(seen.values())
    print(f"[dedup] {len(rows):,} → {len(out):,} unique (kaggle, question)")
    return out


# ---------------------------------------------------------------------------
# Pre-LLM heuristic candidate selection (diversity-aware)
# ---------------------------------------------------------------------------


def prefilter_candidates(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample `n` candidates roughly balanced across heuristic buckets.

    Hard rows are rare in this data (only ones using sklearn / scipy etc.),
    so we take them first and let the remaining slots fall to medium/easy.
    Caps per-kaggle at 5 to avoid one popular dataset eating the pool.
    """
    rng = random.Random(seed)
    rng.shuffle(rows)

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["_h_bucket"]].append(r)

    # LLM-hard rows are rarer than heuristic-hard rows (the heuristic only
    # checks whether ML packages exist anywhere in the notebook, not whether
    # the question itself requires them). So oversample heuristic-hard to
    # land enough genuine hard cases after LLM rating.
    targets = {
        "hard":   int(n * 0.55),
        "medium": int(n * 0.30),
        "easy":   n - int(n * 0.55) - int(n * 0.30),  # remainder
    }
    per_kaggle: Counter[str] = Counter()
    picked: list[dict] = []

    for bucket_name in ("hard", "medium", "easy"):  # hard first (rare)
        pool = by_bucket.get(bucket_name, [])
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            if taken >= targets[bucket_name]:
                break
            if per_kaggle[r["kaggle_dataset_name"]] >= 5:
                continue
            picked.append(r)
            per_kaggle[r["kaggle_dataset_name"]] += 1
            taken += 1

    # Top up to `n` from leftover (any bucket) keeping the per-kaggle cap.
    if len(picked) < n:
        picked_ids = {id(r) for r in picked}
        leftover = [r for r in rows if id(r) not in picked_ids]
        rng.shuffle(leftover)
        for r in leftover:
            if len(picked) >= n:
                break
            if per_kaggle[r["kaggle_dataset_name"]] >= 5:
                continue
            picked.append(r)
            per_kaggle[r["kaggle_dataset_name"]] += 1

    rng.shuffle(picked)
    print(f"[prefilter] {len(picked):,} candidates  "
          f"(easy={sum(1 for r in picked if r['_h_bucket']=='easy')}, "
          f"medium={sum(1 for r in picked if r['_h_bucket']=='medium')}, "
          f"hard={sum(1 for r in picked if r['_h_bucket']=='hard')}, "
          f"unique_kaggle={len(set(r['kaggle_dataset_name'] for r in picked))})")
    return picked


# ---------------------------------------------------------------------------
# LLM scoring (parallel; split between OpenAI + Anthropic)
# ---------------------------------------------------------------------------

RUBRIC = """\
Rate the difficulty of this DATA-ANALYSIS task on a 1-5 scale. The agent has
access to a Jupyter sandbox with the listed file(s) loaded and the listed
Python packages preinstalled.

Scale:
 1 = Trivial: single-stat lookup (max/mean/count/unique of one column).
 2 = Easy: one filter or one groupby + one aggregation.
 3 = Medium: 2-3 step pandas pipeline (filter -> groupby -> sort/top-k), or a
     visualization-derived value.
 4 = Hard: 4+ pipeline steps OR requires a scipy/statsmodels test or a
     classical-ML fit + metric report.
 5 = Very Hard: complex multi-step ML/DL pipeline, careful preprocessing,
     custom modeling, or expert-domain reasoning.

Task:
Question: {question}
Files available: {files}
Packages available: {packages}
Gold answer (for difficulty calibration only): {answer}

Respond JSON: {{"score": <1|2|3|4|5>, "reason": "<one short sentence>"}}
"""


def _format_prompt(row: dict) -> str:
    files = ", ".join(_coerce_list(row.get("files_used"))[:5]) or "(none)"
    packages = ", ".join(_coerce_list(row.get("packages_used"))[:10]) or "(none)"
    return RUBRIC.format(
        question=(row.get("question") or "").strip()[:600],
        files=files,
        packages=packages,
        answer=(row.get("answer") or "").strip()[:200],
    )


def _score_with_openai(prompt: str, client) -> tuple[int, str] | None:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=120,
        )
        raw = r.choices[0].message.content or "{}"
        obj = json.loads(raw)
        score = int(obj["score"])
        if score not in (1, 2, 3, 4, 5):
            return None
        return score, str(obj.get("reason", ""))[:200]
    except Exception:  # noqa: BLE001
        return None


def _score_with_anthropic(prompt: str, client) -> tuple[int, str] | None:
    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.0,
            tools=[{
                "name": "rate",
                "description": "Rate task difficulty 1-5.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                },
            }],
            tool_choice={"type": "tool", "name": "rate"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in r.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "rate":
                score = int(block.input.get("score", 0))
                if score not in (1, 2, 3, 4, 5):
                    return None
                return score, str(block.input.get("reason", ""))[:200]
        return None
    except Exception:  # noqa: BLE001
        return None


def llm_score_difficulty(
    candidates: list[dict], max_workers: int, openai_share: float = 0.5,
) -> list[dict]:
    from openai import OpenAI
    from anthropic import Anthropic

    oai = OpenAI()
    anth = Anthropic()

    rng = random.Random(7)
    assignments = []
    for i, r in enumerate(candidates):
        provider = "openai" if rng.random() < openai_share else "anthropic"
        assignments.append((i, provider, r))

    n_oai = sum(1 for _, p, _ in assignments if p == "openai")
    n_anth = len(assignments) - n_oai
    print(f"[llm] scoring {len(assignments):,} candidates "
          f"({n_oai} OpenAI, {n_anth} Anthropic, {max_workers} workers)…")

    def _worker(args):
        i, provider, row = args
        prompt = _format_prompt(row)
        if provider == "openai":
            res = _score_with_openai(prompt, oai)
            if res is None:  # fallback
                res = _score_with_anthropic(prompt, anth)
        else:
            res = _score_with_anthropic(prompt, anth)
            if res is None:
                res = _score_with_openai(prompt, oai)
        return i, provider, res

    t0 = time.time()
    done = 0
    last_print = t0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, a) for a in assignments]
        for fut in as_completed(futures):
            i, provider, res = fut.result()
            r = candidates[i]
            if res is None:
                r["llm_score"] = None
                r["llm_reason"] = "rate_failed"
                r["llm_provider"] = provider
            else:
                r["llm_score"] = res[0]
                r["llm_reason"] = res[1]
                r["llm_provider"] = provider
            done += 1
            now = time.time()
            if now - last_print >= 5 or done == len(assignments):
                rate = done / max(1.0, now - t0)
                print(f"  [{done:>4d}/{len(assignments):>4d}] {rate:.1f} req/s")
                last_print = now

    ok = sum(1 for r in candidates if r.get("llm_score") is not None)
    print(f"[llm] scored {ok}/{len(candidates)} (failures dropped)")
    return [r for r in candidates if r.get("llm_score") is not None]


def llm_to_bucket(score: int) -> str:
    if score <= 2:
        return "easy"
    if score == 3:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Stratified final sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    candidates: list[dict], n_per_bucket: int, max_per_kaggle: int, seed: int,
) -> list[dict]:
    """Pick `n_per_bucket` each from easy/medium/hard with diversity caps."""
    by_bucket: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for r in candidates:
        by_bucket[llm_to_bucket(r["llm_score"])].append(r)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for bucket_name in ("easy", "medium", "hard"):
        pool = by_bucket[bucket_name]
        rng.shuffle(pool)

        # Round-robin by (package_tier, answer_type) to balance within bucket
        cells: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for r in pool:
            cells[(r["_features"]["package_tier"], r["_features"]["answer_type"])].append(r)

        per_kaggle: Counter[str] = Counter()
        picked: list[dict] = []
        cell_keys = list(cells.keys())
        rng.shuffle(cell_keys)
        # Iterate cells round-robin until we hit n_per_bucket
        while len(picked) < n_per_bucket and any(cells[k] for k in cell_keys):
            for k in cell_keys:
                if len(picked) >= n_per_bucket:
                    break
                while cells[k]:
                    cand = cells[k].pop(0)
                    if per_kaggle[cand["kaggle_dataset_name"]] >= max_per_kaggle:
                        continue
                    picked.append(cand)
                    per_kaggle[cand["kaggle_dataset_name"]] += 1
                    break

        # If under-quota, fill from leftover ignoring cell balance but keeping
        # the kaggle cap.
        if len(picked) < n_per_bucket:
            leftover = [r for r in pool if r not in picked]
            for cand in leftover:
                if len(picked) >= n_per_bucket:
                    break
                if per_kaggle[cand["kaggle_dataset_name"]] >= max_per_kaggle:
                    continue
                picked.append(cand)
                per_kaggle[cand["kaggle_dataset_name"]] += 1

        if len(picked) < n_per_bucket:
            print(f"[warn] only {len(picked)}/{n_per_bucket} for bucket={bucket_name} "
                  f"(pool={len(pool)}); relaxing kaggle cap as last resort")
            leftover = [r for r in pool if r not in picked]
            for cand in leftover:
                if len(picked) >= n_per_bucket:
                    break
                picked.append(cand)

        print(f"[bucket={bucket_name}] picked={len(picked)} "
              f"unique_kaggle={len(set(r['kaggle_dataset_name'] for r in picked))} "
              f"tiers={Counter(r['_features']['package_tier'] for r in picked)} "
              f"answers={Counter(r['_features']['answer_type'] for r in picked)}")
        for r in picked:
            r["difficulty"] = bucket_name
        chosen.extend(picked)

    return chosen


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        flat_row = {k: v for k, v in r.items() if k != "_features"}
        for fk, fv in (r.get("_features") or {}).items():
            flat_row[f"feat_{fk}"] = fv
        flat.append(flat_row)
    df = pd.DataFrame(flat)
    df.to_parquet(path, index=False)
    print(f"[save] {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-split", default="non_thinking",
                   choices=["thinking", "non_thinking"])
    p.add_argument("--n-per-bucket", type=int, default=50,
                   help="Final picks per difficulty bucket (default 50 → 150 total).")
    p.add_argument("--max-per-kaggle", type=int, default=2)
    p.add_argument("--candidate-pool", type=int, default=1500,
                   help="Heuristic candidate pool size sent to the LLM.")
    p.add_argument("--llm-workers", type=int, default=24)
    p.add_argument("--openai-share", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh-bucket-list", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Stop before the LLM step; print pool stats only.")
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("[err] OPENAI_API_KEY missing", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[err] ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 1

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = RAW_DIR / f"{args.source_split}.parquet"
    if not parquet_path.exists():
        print(f"[err] {parquet_path} missing; run prepare.cache_dataset first")
        return 1

    in_bucket = fetch_in_bucket_datasets(refresh=args.refresh_bucket_list)
    print(f"[bucket] {len(in_bucket)} kaggle datasets uploaded to {BUCKET_ID}")

    rows = load_source_rows(parquet_path)
    rows = filter_rows(rows, in_bucket)
    rows = dedup_rows(rows)

    for r in rows:
        r["_features"] = compute_features(r)
        r["_h_bucket"] = heuristic_bucket(r["_features"])

    print(f"[heuristic] easy={sum(1 for r in rows if r['_h_bucket']=='easy'):,}  "
          f"medium={sum(1 for r in rows if r['_h_bucket']=='medium'):,}  "
          f"hard={sum(1 for r in rows if r['_h_bucket']=='hard'):,}")

    candidates = prefilter_candidates(rows, n=args.candidate_pool, seed=args.seed)

    if args.dry_run:
        print("[dry-run] stopping before LLM step")
        return 0

    candidates = llm_score_difficulty(
        candidates,
        max_workers=args.llm_workers,
        openai_share=args.openai_share,
    )
    save_parquet(candidates, EVAL_DIR / "candidates.parquet")

    score_dist = Counter(r["llm_score"] for r in candidates)
    print(f"[scores] {dict(sorted(score_dist.items()))}")

    chosen = stratified_sample(
        candidates,
        n_per_bucket=args.n_per_bucket,
        max_per_kaggle=args.max_per_kaggle,
        seed=args.seed,
    )
    save_parquet(chosen, EVAL_DIR / "eval_manifest.parquet")

    ids_path = EVAL_DIR / "eval_ids.txt"
    ids_path.write_text("\n".join(r["id"] for r in chosen) + "\n")
    print(f"[save] {ids_path}  ({len(chosen)} ids)")

    # Summary
    print()
    print("─" * 60)
    print(f"Total: {len(chosen)} tasks")
    for bucket in ("easy", "medium", "hard"):
        bucket_rows = [r for r in chosen if r["difficulty"] == bucket]
        print(f"  {bucket:6s}: {len(bucket_rows)} "
              f"unique_kaggle={len({r['kaggle_dataset_name'] for r in bucket_rows})}")
    print(f"\nNext: build the Harbor suite from these ids:")
    print(f"  uv run python -m prepare.build_harbor_tasks \\")
    print(f"      --name eval-v1 --ids-from cache/eval/eval_ids.txt \\")
    print(f"      --data-bucket-id {BUCKET_ID} --skip-data-download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
