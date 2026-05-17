"""Casos de uso del agente: orquestador multi-agente + sub-agentes + prompts."""

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
    ORCHESTRATOR_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)
from .subagents import SUBAGENTS, SubAgentSpec

__all__ = [
    "CITATION_RE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_RESULT_TOKENS",
    "DEFAULT_SUBAGENT_MAX_ITERATIONS",
    "MAX_ITERATIONS_FALLBACK_MESSAGE",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "SUBAGENTS",
    "SYSTEM_PROMPT",
    "SubAgentResult",
    "SubAgentSpec",
    "run_agent",
    "run_subagent",
    "verify_citations",
]
