"""Adaptadores LLM: Protocol + parsers + engines concretos."""

from .base import LLMEngine, LLMUnavailableError
from .llama_cpp_engine import LlamaCppEngine
from .openai_compat_engine import OpenAICompatEngine
from .tool_call_parser import parse_tool_calls

__all__ = [
    "LLMEngine",
    "LLMUnavailableError",
    "LlamaCppEngine",
    "OpenAICompatEngine",
    "parse_tool_calls",
]
