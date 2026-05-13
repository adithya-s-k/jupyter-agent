"""Minimal 1-tool bash agent — multi-provider, file-based submission."""
from .agent import BashOnlyAgent, TOOLS, SYSTEM_PROMPT

__all__ = ["BashOnlyAgent", "TOOLS", "SYSTEM_PROMPT"]
