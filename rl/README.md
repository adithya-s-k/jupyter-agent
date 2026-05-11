# Jupyter Agent — RL pipeline

Slim state right now. We have:

- The source dataset cached locally (`prepare/cache_dataset.py`).
- A design plan in `PLAN.md` we're iterating on.
- A target: **one script** that takes `--n-tasks N` and emits a directory of Harbor tasks; we then run OpenReward as the server over those folders and do rollouts.

## Layout

```
rl/
├── README.md
├── PLAN.md                     # the design — iterate here
├── pyproject.toml
├── uv.lock
└── prepare/
    └── cache_dataset.py        # downloads jupyter-agent/jupyter-agent-dataset locally
```

## What's cached

```
cache/
├── hf-datasets/   # HF datasets builder cache (used by load_dataset)
└── raw/           # thinking.parquet, non_thinking.parquet (51,389 rows each)
```

## Install

```bash
cd rl
uv sync
```

The pipeline reads API keys from `../.env`:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`, `E2B_API_KEY`, `KAGGLE_API_TOKEN`.

## Refresh the dataset cache

Already done — both splits are at `cache/raw/`. To re-pull:

```bash
uv run python -m prepare.cache_dataset --confirm
```

## Next

See `PLAN.md` — the single script that turns N rows of the cached dataset into a Harbor task suite, ready to be served via OpenReward.
