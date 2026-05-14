"""LLM client used by Phase C (doctor) and Phase D (categorize).

Uses the **native Anthropic SDK** for `anthropic/*` models (so we can use
prompt caching, which the anthropic OpenAI-compat shim does not honor) and
the **OpenAI SDK** for `openai/*` models.

Provides:
  - Sync `call(...)` with tool-calling, per-call cost & token reporting, and
    Anthropic prompt caching via system `cache_control`.
  - Per-call cost.jsonl logging when a StateStore is provided.

Anthropic prompt caching:
  Sets `cache_control: {"type": "ephemeral"}` on the system prompt block.
  Anthropic surfaces cache stats as `cache_creation_input_tokens` and
  `cache_read_input_tokens` in `response.usage` — we add the latter to our
  uniform `cached_tokens` field.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_HARBOR_AGENTS = Path(__file__).resolve().parents[2] / "harbor_agents"
sys.path.insert(0, str(_HARBOR_AGENTS.parent))  # for `from harbor_agents._shared...`
from harbor_agents._shared.providers import parse_model  # noqa: E402
from harbor_agents._shared.cost import compute_cost  # noqa: E402


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]              # [{id, name, arguments(str)}, ...]
    finish_reason: str
    cost_usd: float
    prompt_tokens: int                  # input tokens NOT served from cache
    completion_tokens: int
    cached_tokens: int                  # tokens served from prompt cache
    cache_creation_tokens: int = 0      # tokens written to cache on this call
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Anthropic native path
# ---------------------------------------------------------------------------

def _call_anthropic(*, model_id: str, messages: Sequence[Mapping[str, Any]],
                    tools, temperature: float, max_tokens: int | None) -> LLMResponse:
    import anthropic
    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    # Anthropic expects system as a top-level argument (string OR list of blocks
    # with optional cache_control). User/assistant/tool messages are in `messages`.
    system_blocks: list[dict] | str = ""
    rest: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            content = m.get("content") or ""
            if isinstance(content, str):
                system_blocks = [{
                    "type": "text", "text": content,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                system_blocks = list(content)
        else:
            rest.append(_normalize_for_anthropic(m))

    # Translate OpenAI-style tool schema → Anthropic-style.
    anth_tools = None
    if tools:
        anth_tools = []
        for t in tools:
            fn = t.get("function") or t
            anth_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })

    kwargs = dict(
        model=model_id,
        system=system_blocks,
        messages=rest,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
    )
    if anth_tools:
        kwargs["tools"] = anth_tools

    resp = client.messages.create(**kwargs)

    # Extract text + tool_use blocks
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "arguments": json.dumps(block.input or {}),
            })

    usage = resp.usage
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

    cost = compute_cost(
        model_name=f"anthropic/{model_id}",
        prompt_tokens=prompt + cache_create,  # cache_creation IS billable input
        completion_tokens=completion,
        cached_tokens=cache_read,
    )

    return LLMResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        finish_reason=resp.stop_reason or "",
        cost_usd=cost,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cache_read,
        cache_creation_tokens=cache_create,
    )


def _normalize_for_anthropic(m: Mapping[str, Any]) -> dict:
    """Translate a message from OpenAI shape into Anthropic shape."""
    role = m["role"]
    content = m.get("content")

    if role == "tool":
        # OpenAI: {role: "tool", tool_call_id, content}
        # Anthropic: {role: "user", content: [{type:"tool_result", tool_use_id, content}]}
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": content if isinstance(content, str) else json.dumps(content),
            }],
        }

    if role == "assistant":
        # An OpenAI assistant message can carry text content AND tool_calls.
        # Anthropic wants one list of content blocks: text blocks + tool_use blocks.
        blocks: list[dict] = []
        if content:
            blocks.append({"type": "text", "text": content if isinstance(content, str) else json.dumps(content)})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                inp = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                inp = {}
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": fn.get("name"),
                "input": inp,
            })
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        return {"role": "assistant", "content": blocks}

    # user / others — pass through (Anthropic accepts str or list-of-blocks)
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# OpenAI native path (also handles anything else not Anthropic)
# ---------------------------------------------------------------------------

def _call_openai(*, provider: str, model_id: str, messages, tools, temperature,
                 max_tokens, response_format):
    """OpenAI-compatible call. `provider` selects the API endpoint:
       - "openai" → api.openai.com with OPENAI_API_KEY
       - "hf"     → router.huggingface.co/v1 with HF_TOKEN
    """
    from openai import OpenAI
    if provider == "hf":
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise RuntimeError("HF_TOKEN missing — required for hf/* models")
        client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
        is_reasoning = False  # HF inference doesn't use OpenAI's reasoning param shape
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        client = OpenAI(api_key=api_key)
        # GPT-5.x reasoning models reject `max_tokens` and `temperature`.
        is_reasoning = model_id.startswith(("gpt-5", "o1", "o3", "o4"))

    kwargs: dict = dict(model=model_id, messages=list(messages))
    if not is_reasoning:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = list(tools)
        kwargs["tool_choice"] = "auto"
    if max_tokens is not None:
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
    if response_format is not None and provider == "openai":
        kwargs["response_format"] = dict(response_format)

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message

    tool_calls: list[dict] = []
    for tc in (msg.tool_calls or []):
        tool_calls.append({
            "id": tc.id, "name": tc.function.name,
            "arguments": tc.function.arguments,
        })

    usage = resp.usage
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = int(getattr(ptd, "cached_tokens", 0) or 0)

    cost = compute_cost(
        model_name=f"openai/{model_id}",
        prompt_tokens=pt, completion_tokens=ct, cached_tokens=cached,
    )

    return LLMResponse(
        content=msg.content or "",
        tool_calls=tool_calls,
        finish_reason=resp.choices[0].finish_reason or "",
        cost_usd=cost,
        prompt_tokens=pt,
        completion_tokens=ct,
        cached_tokens=cached,
    )


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def call(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: Mapping[str, Any] | None = None,
    state=None,
    task_id: str | None = None,
    phase: str | None = None,
    call_kind: str | None = None,
) -> LLMResponse:
    provider, model_id = parse_model(model)
    t0 = time.time()

    if provider == "anthropic":
        resp = _call_anthropic(model_id=model_id, messages=messages, tools=tools,
                               temperature=temperature, max_tokens=max_tokens)
    else:
        # openai OR hf — both use OpenAI-compatible HTTP API
        resp = _call_openai(provider=provider, model_id=model_id,
                            messages=messages, tools=tools,
                            temperature=temperature, max_tokens=max_tokens,
                            response_format=response_format)
    resp.elapsed_sec = time.time() - t0

    if state is not None:
        try:
            state.append_cost_event(
                task_id=task_id, phase=phase, call_kind=call_kind,
                model=model, cost_usd=round(resp.cost_usd, 6),
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cached_tokens=resp.cached_tokens,
                cache_creation_tokens=resp.cache_creation_tokens,
                elapsed_sec=round(resp.elapsed_sec, 3),
                finish_reason=resp.finish_reason,
            )
        except Exception:  # noqa: BLE001
            pass

    return resp
