"""Casos de uso del agente: loop conversacional + prompts + verificador de citas."""

from .citation_verifier import CITATION_RE, verify_citations
from .conversation_loop import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_RESULT_TOKENS,
    run_agent,
)
from .prompts import (
    MAX_ITERATIONS_FALLBACK_MESSAGE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)

__all__ = [
    "CITATION_RE",
    "verify_citations",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_RESULT_TOKENS",
    "run_agent",
    "MAX_ITERATIONS_FALLBACK_MESSAGE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
]
