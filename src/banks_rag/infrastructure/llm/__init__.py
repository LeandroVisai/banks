"""Adaptadores LLM: Protocol + parsers + engines concretos."""

from .base import LLMEngine
from .llama_cpp_engine import LlamaCppEngine
from .tool_call_parser import parse_tool_calls

__all__ = ["LLMEngine", "LlamaCppEngine", "parse_tool_calls"]
