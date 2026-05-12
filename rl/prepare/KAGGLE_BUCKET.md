# Kaggle datasets → HF Bucket

All Kaggle datasets referenced by `jupyter-agent/jupyter-agent-dataset` (`non_thinking` split, `executor_type=e2b`) are mirrored to a single HF Bucket so they're reachable from anywhere without re-hitting the Kaggle API.

## Where things live

| What | Where |
|---|---|
| **Datasets (786)** | `hf://buckets/AdithyaSK/jupyter-agent-kaggle-all/<owner>__<slug>/...` |
| **Per-dataset layout** | files preserved as-is from the Kaggle archive; e.g. `uciml__iris/Iris.csv` |
| **Metadata / manifest** | `hf://buckets/AdithyaSK/jupyter-agent-kaggle-all/_meta/` |
| **Source HF dataset** | `jupyter-agent/jupyter-agent-dataset` (split `non_thinking`) |

Bucket prefix convention: `kaggle_dataset_name.replace("/", "__")` — e.g. `4quant/soft-tissue-sarcoma` → `4quant__soft-tissue-sarcoma`.

## What's in `_meta/`

| File | Contents |
|---|---|
| `manifest.json` | Bucket summary: totals, source, notes on failures. Start here. |
| `uploaded.jsonl` | 786 successful uploads with files / bytes / size_mb / source row ids (cap 50 per dataset) |
| `failed.jsonl` | 4 datasets that returned `403` (Kaggle license clickthrough required) |
| `skipped.jsonl` | 1 dataset manually skipped (`4quant/eye-gaze` — 11 GB for 1 source row) |
| `fetch_all_kaggle.jsonl` | Raw append-only run report (resumable index) |
| `source_index.parquet` | `id, kaggle_dataset_name, executor_type` from the source split — for reverse lookup |
| `eval_v1/` | 100-task eval split (see [Eval split](#eval-split-eval_v1) below) |

## Eval split (`_meta/eval_v1/`)

The 100-task public eval suite at [`AdithyaSK/jupyter-agent-eval-v1-harbor`](https://huggingface.co/datasets/AdithyaSK/jupyter-agent-eval-v1-harbor) is built from a fixed list of source-row ids.

**Exclusion is row-level, not dataset-level.** Each row in the source dataset is one `(kaggle_dataset, question)` pair — e.g. `0001_573_1573460_qa_5`. Only those exact rows are held out of training. The underlying Kaggle dataset (`mmoreaux/audio-cats-and-dogs` in that example) is free to appear in many other train rows with different questions. That's intentional — Kaggle datasets are scarce (786 unique total), questions are plentiful (~29k), and the agent needs to generalise across questions on the same data.

The exclusion list lives in the bucket so every machine that builds a train suite can pull it without going through the source repo.

| File | Contents | Size |
|---|---|---|
| `eval_v1/eval_ids.txt` | 100 source-row ids, one per line. The canonical exclusion list. | 2.9 KB |
| `eval_v1/eval_manifest.parquet` | Full per-row metadata: `id, question, answer, kaggle_dataset_name, difficulty, llm_score, feat_*`. Use for audit / inspection. | 35 KB |
| `eval_v1/candidates.parquet` | All 2,000 LLM-scored candidates the eval was sampled from. Useful if you want to expand the eval or use a different stratification. | 345 KB |

### Composition (as of 2026-05-13)

| Bucket | Count | Unique Kaggle datasets | LLM score range |
|---|---|---|---|
| easy   | 50 | 47 | 1–2 |
| medium | 25 | 24 | 3 |
| hard   | 25 | 23 | 4–5 |
| **Total** | **100** | **94** | 1–5 |

Within each bucket: tiers ~⅓ each (basic pandas / +viz / +ML), answer types ~½ numeric / ½ string, max 2 tasks per Kaggle dataset.

### Use it: drop eval ids from a train build

```bash
# 1. Pull the exclusion list from the bucket (one-liner; no HF dataset clone needed).
python - <<'PY'
from huggingface_hub import HfApi
HfApi().download_bucket_files(
    "AdithyaSK/jupyter-agent-kaggle-all",
    files=[("_meta/eval_v1/eval_ids.txt", "cache/eval/eval_ids.txt")],
)
PY

# 2. Build train suite with the exclusion in effect.
uv run python -m prepare.build_harbor_tasks \
    --name train-v1 --n-tasks 1500 \
    --exclude-ids cache/eval/eval_ids.txt \
    --data-bucket-id AdithyaSK/jupyter-agent-kaggle-all \
    --skip-data-download
```

`build_harbor_tasks.py`'s `--exclude-ids` filters the dedup pool by `id` (row-level), so the exact eval `(kaggle, question)` rows are never seen during training. Other rows from the same Kaggle datasets are kept — the agent can train on `mmoreaux/audio-cats-and-dogs` with all questions *except* `…_qa_5` (which is in eval).

### Quick check: how often does the same Kaggle dataset show up in both?

```python
import pyarrow.parquet as pq
from huggingface_hub import HfApi
HfApi().download_bucket_files(
    "AdithyaSK/jupyter-agent-kaggle-all",
    files=[("_meta/eval_v1/eval_manifest.parquet", "/tmp/eval.parquet")],
)
eval_df = pq.read_table("/tmp/eval.parquet").to_pandas()
print(f"{len(eval_df)} eval rows across {eval_df.kaggle_dataset_name.nunique()} unique Kaggle datasets")
# 100 eval rows across 94 unique Kaggle datasets → ~6 datasets contribute 2 eval rows,
# the rest contribute 1. ~700 Kaggle datasets remain fully available for training.
```

### Inspect the eval

```python
import pyarrow.parquet as pq
df = pq.read_table(
    "hf://buckets/AdithyaSK/jupyter-agent-kaggle-all/_meta/eval_v1/eval_manifest.parquet"
).to_pandas()
print(df.groupby('difficulty').size())
df[df.difficulty == 'hard'][['id','question','answer','llm_score']].head()
```

## Numbers (as of 2026-05-12)

| | Count | Bytes |
|---|---|---|
| Datasets uploaded | **786** | **126.68 GB** |
| Datasets failed (license-walled) | 4 | – |
| Datasets skipped (manual) | 1 | – |
| **Expected total** | **791** | – |
| Source rows referenced (e2b) | 29,561 | – |

## Failed datasets (require manual license accept)

Visit each URL in a browser signed in to Kaggle, accept the rules, then run the retry command below.

- https://www.kaggle.com/datasets/NUFORC/ufo-sightings — 34 source rows
- https://www.kaggle.com/datasets/sazid28/advertising.csv — 10 source rows
- https://www.kaggle.com/datasets/shahir/protein-data-set — 13 source rows
- https://www.kaggle.com/datasets/akshay4/road-accidents-incidence — 2 source rows

Coverage gap: 59 rows / 29,561 total (0.2%).

## How to access from anywhere

### Read the manifest

```python
from huggingface_hub import download_bucket_files

download_bucket_files(
    "AdithyaSK/jupyter-agent-kaggle-all",
    paths=["_meta/manifest.json"],
    local_dir="./meta",
)
```

### List one dataset's files

```python
from huggingface_hub import list_bucket_tree

for entry in list_bucket_tree("AdithyaSK/jupyter-agent-kaggle-all", path="uciml__iris"):
    print(entry.path, entry.size)
```

### Lazy-mount the whole bucket (read-only, FUSE)

```bash
hf-mount AdithyaSK/jupyter-agent-kaggle-all ~/mnt/kaggle-all
ls ~/mnt/kaggle-all/uciml__iris/
```

Use the mount for agents that touch a small slice of many datasets without pulling everything.

### Download a whole dataset

```python
from huggingface_hub import download_bucket_files

download_bucket_files(
    "AdithyaSK/jupyter-agent-kaggle-all",
    paths=["uciml__iris/"],   # trailing slash = prefix
    local_dir="./data",
)
```

### Find which datasets back a set of source rows

```python
import pyarrow.parquet as pq
idx = pq.read_table("hf://buckets/AdithyaSK/jupyter-agent-kaggle-all/_meta/source_index.parquet").to_pandas()
needed = set(idx.loc[idx.id.isin(my_row_ids), "kaggle_dataset_name"].unique())
# bucket prefixes:
prefixes = [n.replace("/", "__") for n in needed]
```

## How the upload was done

Source script: [`fetch_all_kaggle.py`](./fetch_all_kaggle.py)

1. Scan the source parquet (`cache/raw/non_thinking.parquet`), extract unique `kaggle_dataset_name` × source-row backlinks.
2. For each unique dataset, in a 16-worker thread pool:
   1. `kagglehub.dataset_download(name)` — fetches the archive into `~/.cache/kagglehub`.
   2. `huggingface_hub.sync_bucket(local_path, "hf://buckets/<bucket>/<prefix>")` — xet-chunked parallel upload, dedup-aware.
   3. If `--purge-local-after-upload`, delete the local copy.
   3. Append a JSON line to `cache/fetch_all_kaggle.jsonl` (resume index).

Resumability: re-running the same command skips any entry already marked `uploaded` / `downloaded` / `skipped-already`. Failures are retried only with `--retry-failed`.

### Re-run / extend

```bash
cd /fsx/$USER/projects/jupyter-agent/rl
source .venv/bin/activate

# Retry the 4 license-walled failures (after accepting rules in browser):
python -m prepare.fetch_all_kaggle \
    --retry-failed \
    --upload-bucket AdithyaSK/jupyter-agent-kaggle-all \
    --report cache/fetch_all_kaggle.jsonl \
    --max-workers 4

# Cold rebuild from scratch (don't do this unless you really mean it):
rm cache/fetch_all_kaggle.jsonl
python -m prepare.fetch_all_kaggle \
    --upload-bucket AdithyaSK/jupyter-agent-kaggle-all \
    --max-workers 16 \
    --purge-local-after-upload \
    --report cache/fetch_all_kaggle.jsonl
```

### Refresh `_meta/` after a retry

After any successful retry, regenerate the split files and re-upload:

```python
# Run the same logic that produced the current _meta/ — see git history of this
# README for the inline script, or re-derive from cache/fetch_all_kaggle.jsonl.
```

## Source parquet (how the index is built)

The source `jupyter-agent/jupyter-agent-dataset` is published as 103 sharded parquets. We don't materialize the full 67 GB; we read only the columns we need (`id`, `kaggle_dataset_name`, `executor_type`) from the cached shards into a tiny `cache/raw/non_thinking.parquet` (~0.6 MB). See `cache_dataset.py` for the full-materialization path if you need it.

## Bucket-prefix convention rationale

Underscored prefixes (`<owner>__<slug>`) instead of nested paths (`<owner>/<slug>`) because:
- Buckets are flat; nested paths add no structure server-side.
- The double-underscore is unambiguous — Kaggle slugs can contain hyphens and dots but never `__`.
- Reversible: `prefix.replace("__", "/", 1)` recovers `<owner>/<slug>`.

## Known gotchas

- `kagglehub` ignores `KAGGLEHUB_CACHE_DIR`; it uses `KAGGLEHUB_CACHE_FOLDER`. The previous run wrote 80+ GB through `/admin/home/` (slow weka) — set `KAGGLEHUB_CACHE_FOLDER=/fsx/$USER/.cache/kagglehub` next time.
- `tqdm.write` races across worker threads when `sync_bucket` and `kagglehub` both create tqdm instances. The script wraps it in `try/except RuntimeError` (see `fetch_all_kaggle.py`); without that fix you'll see `RuntimeError: Set changed size during iteration` and lose the run.
- `as_completed` only writes a JSONL entry when a worker fully returns. If you see the JSONL frozen while the bucket grows, that's expected — workers are tied up on multi-GB datasets and the bucket is the more accurate view.
- xet dedup is aggressive: a worker can "upload" a previously-seen dataset's files in well under a second by xet-hash match. Don't read short upload times as a bug.

## Cost of large datasets

These 5 datasets account for 72 GB / 127 GB (57% of the bucket) but only 22 of 29,561 source rows (0.07%). Worth pruning if storage matters more than coverage:

| Bucket prefix | Size | Source rows |
|---|---|---|
| `nih-chest-xrays__data` | 45.1 GB | 9 |
| `kmader__rsna-bone-age` | 10.0 GB | 5 |
| `crawford__emnist` | 6.1 GB | 3 |
| `kmader__food41` | 5.8 GB | 4 |
| (`4quant__eye-gaze`) | (5.1 GB) | (1) — already pruned |

Prune by setting `--max-size-mb 5000` on a future cold run, or delete after the fact with `batch_bucket_files(..., delete=[...])`.
