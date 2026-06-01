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
    REPORT_PROMPT,
    SYNTHESIS_PROMPT,
)
from .report import run_report
from .router import select_specialists
from .subagents import SUBAGENTS, SubAgentSpec

__all__ = [
    "CITATION_RE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_RESULT_TOKENS",
    "DEFAULT_SUBAGENT_MAX_ITERATIONS",
    "MAX_ITERATIONS_FALLBACK_MESSAGE",
    "PROMPT_VERSION",
    "REPORT_PROMPT",
    "SUBAGENTS",
    "SYNTHESIS_PROMPT",
    "SubAgentResult",
    "SubAgentSpec",
    "run_agent",
    "run_report",
    "run_subagent",
    "select_specialists",
    "verify_citations",
]
