"""Provider routing shared by every Harbor agent in `rl/harbor_agents/`.

All three providers we target — OpenAI, Anthropic, HF Inference — speak the
OpenAI chat-completions wire format, so each agent can use ONE OpenAI client
and just swap `base_url` + `api_key` per provider. This module owns that
mapping so individual agents don't need to repeat it.

The `--model` value Harbor passes through to the agent uses a prefix to pick
the provider:

  openai/gpt-5                                     → OpenAI native
  anthropic/claude-sonnet-4-6                      → Anthropic via api.anthropic.com/v1/
  hf/Qwen/Qwen3-235B-A22B-Instruct-2507:nscale     → HF Inference router

No prefix → assume OpenAI (for backward compatibility with bare `gpt-5`).
"""

from __future__ import annotations

import os

_PREFIXES = (
    ("openai/", "openai"),
    ("anthropic/", "anthropic"),
    ("hf/", "hf"),
)


def parse_model(name: str | None) -> tuple[str, str]:
    """Return (provider, bare_model_id).

    Examples:
        parse_model("openai/gpt-5") → ("openai", "gpt-5")
        parse_model("anthropic/claude-sonnet-4-6") → ("anthropic", "claude-sonnet-4-6")
        parse_model("hf/Qwen/Qwen3-235B:nscale") → ("hf", "Qwen/Qwen3-235B:nscale")
        parse_model("gpt-5") → ("openai", "gpt-5")  # implicit OpenAI
        parse_model(None) → ("openai", "gpt-4o-mini")  # safe default
    """
    if not name:
        return "openai", "gpt-4o-mini"
    for prefix, provider in _PREFIXES:
        if name.startswith(prefix):
            return provider, name[len(prefix):]
    return "openai", name


def provider_credentials(provider: str) -> tuple[str | None, str | None]:
    """Return (api_key, base_url) for `provider`. Raises if unknown.

    `api_key` may be None if the env var isn't set — callers should check
    and raise their own error message.
    """
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY"), None
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY"), "https://api.anthropic.com/v1/"
    if provider == "hf":
        return os.environ.get("HF_TOKEN"), "https://router.huggingface.co/v1"
    raise RuntimeError(f"unknown provider: {provider!r}")
