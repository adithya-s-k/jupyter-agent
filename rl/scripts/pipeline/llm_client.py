"""Minimal LLM client used by Phase C (doctor) and Phase D (categorize).

Talks to Anthropic + OpenAI via the OpenAI-compatible REST shim (same
pattern as the existing seta agent). Returns an assistant message with
its tool_calls and usage. Tool-calling format follows OpenAI's spec.

A future migration to LiteLLM would replace this — for now we mirror what
already works in `harbor_agents/_shared/providers.py`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Reuse the same provider routing the seta agent uses.
_HARBOR_AGENTS = Path(__file__).resolve().parents[2] / "harbor_agents"
sys.path.insert(0, str(_HARBOR_AGENTS.parent))  # so we can `from harbor_agents._shared...`
from harbor_agents._shared.providers import parse_model, provider_credentials  # noqa: E402
from harbor_agents._shared.cost import compute_cost, canonical_model_key  # noqa: E402


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]              # [{id, name, arguments(str)}, ...]
    finish_reason: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def _get_client(model: str):
    from openai import OpenAI
    provider, _ = parse_model(model)
    api_key, base_url = provider_credentials(provider)
    if not api_key:
        raise RuntimeError(f"missing API key for provider={provider} (model={model})")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def call(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMResponse:
    """One chat-completion call. Synchronous."""
    _, model_id = parse_model(model)
    client = _get_client(model)
    kwargs: dict = dict(model=model_id, messages=list(messages), temperature=temperature)
    if tools:
        kwargs["tools"] = list(tools)
        kwargs["tool_choice"] = "auto"
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    tool_calls: list[dict] = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments,  # JSON string per OpenAI spec
        })

    usage = resp.usage
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    # cached tokens — provider-specific shape; try a few keys
    cached = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = int(getattr(ptd, "cached_tokens", 0) or 0)

    cost = compute_cost(model_name=model, prompt_tokens=pt,
                        completion_tokens=ct, cached_tokens=cached)

    return LLMResponse(
        content=msg.content or "",
        tool_calls=tool_calls,
        finish_reason=resp.choices[0].finish_reason or "",
        cost_usd=cost,
        prompt_tokens=pt,
        completion_tokens=ct,
        cached_tokens=cached,
    )
