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
