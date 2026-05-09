"""Adaptadores LLM: Protocol + parsers + (futuro) llamacpp/vllm engines."""

from .base import LLMEngine
from .tool_call_parser import parse_tool_calls

__all__ = ["LLMEngine", "parse_tool_calls"]
