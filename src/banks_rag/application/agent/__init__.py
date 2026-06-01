"""Casos de uso del agente: router determinista + sub-agentes + síntesis."""

from .citation_verifier import CITATION_RE, verify_citations
from .conversation_loop import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_RESULT_TOKENS,
    DEFAULT_SUBAGENT_MAX_ITERATIONS,
    SubAgentResult,
    run_agent,
    run_subagent,
)
from .prompts import (
    MAX_ITERATIONS_FALLBACK_MESSAGE,
    PROMPT_VERSION,
    SYNTHESIS_PROMPT,
)
from .router import select_specialists
from .subagents import SUBAGENTS, SubAgentSpec

__all__ = [
    "CITATION_RE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_RESULT_TOKENS",
    "DEFAULT_SUBAGENT_MAX_ITERATIONS",
    "MAX_ITERATIONS_FALLBACK_MESSAGE",
    "PROMPT_VERSION",
    "SUBAGENTS",
    "SYNTHESIS_PROMPT",
    "SubAgentResult",
    "SubAgentSpec",
    "run_agent",
    "run_subagent",
    "select_specialists",
    "verify_citations",
]
