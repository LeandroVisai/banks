"""Resultados de una conversación con el agente."""

from __future__ import annotations

from dataclasses import dataclass, field

from .tool_call import ToolCall


@dataclass
class GenerationResult:
    """Resultado de una llamada al LLM en modo agentic.

    Campos:
        text: respuesta del LLM **sin** los markers ``<tool_call>``.
        tool_calls: tool calls parseados del output (puede ser lista vacía).
        raw_text: output completo tal como salió del modelo (debugging).
        n_tokens: tokens generados.
        finish_reason: ``"stop"``, ``"length"``, ``"tool_calls"``, etc.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str = ""
    n_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class AgentResult:
    """Resultado final de un turno del agente (run completo)."""

    response: str
    iterations: int
    tool_trace: list[dict]
    chunks_seen: list[dict]
    series_used: list[dict]
    cited_refs: list[int]
    finish_reason: str
    total_tokens: int
    latency_ms: int
    # Refs [N] que el LLM citó fuera del rango de chunks vistos: señal directa
    # de alucinación. Se eliminan del texto pero se reportan aquí en vez de
    # ocultarse, para que la evaluación de faithfulness pueda medirlas.
    invalid_refs: list[int] = field(default_factory=list)
    # Cifras financieras de la respuesta que no provienen de ninguna
    # herramienta de este turno (grounding numérico). Señal de alucinación de
    # valores; se reportan para evaluación y para que el frontend las marque.
    ungrounded_numbers: list[float] = field(default_factory=list)
    # Gráficos (specs Vega-Lite) generados por plot_series; el frontend los
    # renderiza junto a la respuesta.
    charts: list[dict] = field(default_factory=list)
