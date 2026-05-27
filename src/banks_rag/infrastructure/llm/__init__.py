"""Adaptadores LLM: Protocol + parsers + engines concretos."""

from .base import LLMEngine
from .llama_cpp_engine import LlamaCppEngine
from .tool_call_parser import parse_tool_calls
from .vllm_engine import VLLMEngine

__all__ = ["LLMEngine", "LlamaCppEngine", "VLLMEngine", "parse_tool_calls"]
