"""Native OpenAI-compatible agent backend (hand-coded tool-calling loop)."""

from .backend import NativeOpenAIAgent, tool_to_openai

__all__ = ["NativeOpenAIAgent", "tool_to_openai"]
