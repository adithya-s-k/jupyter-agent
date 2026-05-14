"""Token-usage accumulator + price lookup for our custom agents.

The lookup table lives in `rl/cache/sweep/cost_table.yaml`. We strip the
provider prefix and `:provider` suffix from a model name before the lookup,
so `hf/Qwen/Qwen3-8B:nscale` and `Qwen/Qwen3-8B:fastest` both map to the
canonical key `Qwen/Qwen3-8B`.

Usage inside an agent's run() loop:

    from rl.harbor_agents._shared.cost import UsageTracker
    tracker = UsageTracker(model_name=self.model_name)

    for turn in range(max_turns):
        resp = client.chat.completions.create(...)
        tracker.add_response(resp)
        ...

    # at end of run():
    tracker.populate(context)      # sets context.cost_usd / n_*_tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TABLE_PATH = Path(__file__).resolve().parent / "cost_table.yaml"
_TABLE_CACHE: dict[str, dict[str, float]] | None = None


def _load_table() -> dict[str, dict[str, float]]:
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        if not _TABLE_PATH.exists():
            _TABLE_CACHE = {}
        else:
            data = yaml.safe_load(_TABLE_PATH.read_text())
            _TABLE_CACHE = data.get("prices", {}) if isinstance(data, dict) else {}
    return _TABLE_CACHE


def canonical_model_key(model_name: str | None) -> str:
    """`hf/Qwen/Qwen3-8B:nscale` → `Qwen/Qwen3-8B`. Idempotent on clean ids."""
    if not model_name:
        return ""
    s = model_name
    # Strip provider prefix
    for pfx in ("hf/", "openai/", "anthropic/", "huggingface/"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    # Strip provider suffix `:provider` (`Qwen/Qwen3-8B:nscale`,
    # `gpt-5.5:fastest`, etc.) — but only at the END.
    # For OpenAI/Anthropic native ids (no `:`), this is a no-op.
    if ":" in s:
        # Be careful: only strip the LAST colon segment if it looks like a
        # provider name (lowercase, alnum + hyphen). Otherwise keep.
        head, _, tail = s.rpartition(":")
        if tail and all(c.isalnum() or c == "-" for c in tail) and not tail.startswith("v"):
            s = head
    return s


def compute_cost(
    model_name: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """USD cost for one LLM call. Returns 0.0 if model isn't in the table."""
    table = _load_table()
    key = canonical_model_key(model_name)
    entry = table.get(key)
    if entry is None:
        return 0.0
    in_per_tok = entry["input"] / 1_000_000
    out_per_tok = entry["output"] / 1_000_000
    cached_per_tok = entry.get("cached", entry["input"] / 10) / 1_000_000
    non_cached_in = max(0, prompt_tokens - cached_tokens)
    return non_cached_in * in_per_tok + cached_tokens * cached_per_tok + completion_tokens * out_per_tok


@dataclass
class UsageTracker:
    """Accumulates token usage + cost over the lifetime of an agent's run()."""

    model_name: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    n_calls: int = 0

    def add_response(self, resp: Any) -> None:
        """Take a single OpenAI ChatCompletion response and add its usage."""
        u = getattr(resp, "usage", None)
        if u is None:
            return
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.cached_tokens += cached
        self.cost_usd += compute_cost(self.model_name, pt, ct, cached)
        self.n_calls += 1

    def populate(self, context: Any) -> None:
        """Write the accumulated usage to Harbor's AgentContext.

        Harbor's job-level result.json surfaces these fields as top-level
        `cost_usd`, `n_input_tokens`, etc., on the trial — so the sweep
        orchestrator just reads result.json for cost aggregation.
        """
        try:
            context.cost_usd = round(self.cost_usd, 6)
            context.n_input_tokens = self.prompt_tokens
            context.n_output_tokens = self.completion_tokens
            context.n_cache_tokens = self.cached_tokens
        except Exception:  # noqa: BLE001
            pass

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "canonical_model": canonical_model_key(self.model_name),
        }
