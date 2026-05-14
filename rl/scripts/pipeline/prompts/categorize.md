You are categorizing a data-analysis task's difficulty on a 1-5 scale.

You will receive:
- The original question (`QUESTION`)
- The gold answer (`GOLD`)
- An excerpt of the **passing trajectory** that solved this task (which tools the agent called, what code it ran)

Output a JSON object with these fields:
```
{"level": <1|2|3|4|5>, "reasoning": "<one sentence>", "confidence": <0.0-1.0>, "signal": "<what tipped you off>"}
```

# Rubric

## 1 — Trivial
- Single column lookup, simple aggregation (`min`, `max`, `mean`, `count`).
- ≤2 meaningful cells of code.
- No groupby, no joins, no transformation beyond `.max()` / `.count()`.
- *Examples:* "What is the highest votes?", "How many rows are NaN?"

## 2 — Simple
- Single-file dataframe operation with a filter and an aggregation.
- Boolean / percentage computation with a single condition.
- ≤4 cells.
- No multi-step transformation.
- *Examples:* "What % of users are aged > 30?", "Most common category?"

## 3 — Moderate
- `groupby` / `pivot` / sort + aggregation.
- Two-step transformation (filter → group → top-k).
- Up to 1 join across at most 2 files.
- Simple statistics (correlation, basic descriptives).
- ≤8 cells.
- *Examples:* "Which year had the highest mean revenue per group?"

## 4 — Complex
- Multi-file join with cleaning.
- Non-trivial feature engineering (encoding, scaling, binning).
- Basic ML (one model train with default hyperparams).
- Statistical inference (CI, p-value, hypothesis test).
- Time-series ops (resample, rolling).
- 8-15 cells.
- *Examples:* "Train logistic regression and report test accuracy."

## 5 — Hard
- Multi-step ML pipeline (feature eng + multiple model train + comparison).
- Deep-learning training or fine-tuning.
- Multi-file research-grade analysis with non-trivial cleaning.
- Time-series forecasting.
- >15 cells, or requires non-trivial library usage (transformers, torch, etc.).
- *Examples:* "Train a CNN and report top-5 accuracy on the test set."

# Tiebreakers

- Count meaningful cells *in the passing trajectory*, not what was theoretically possible.
- "Yes/No" questions about ML clusters or visualizations (e.g., "is there distinct clustering?") are typically **3** unless they require training a model first (then **4**).
- Statistical hypothesis tests = **4**.
- Plotting-only with the answer derivable from the plot = **easy** (level 1-2).

Be decisive. Pick exactly one level. Don't dither between two.
