"""Definición de los sub-agentes especializados (arquitectura router-v1).

Cada especialista tiene:

  - un **system prompt** propio (cómo razona ese tipo de analista) — en ``prompts.py``;
  - un **subconjunto de las tools** registradas (qué dominio puede tocar).

El ruteo a los especialistas es DETERMINISTA (``router.select_specialists``): no
hay un LLM orquestador que decida a quién delegar — eso, con Qwen, inventaba
delegados, re-delegaba y nunca sintetizaba. ``run_agent`` corre los
especialistas seleccionados y luego sintetiza UNA sola vez. Un especialista no
invoca a otro: no hay recursión ni tools de delegación.
"""

from __future__ import annotations

from dataclasses import dataclass

# Importar el paquete de tools puebla TOOL_REGISTRY / TOOL_SCHEMAS vía los
# decoradores @register de cada submódulo. Explícito para no depender del
# orden de imports de otros módulos.
import banks_rag.application.agent.tools  # noqa: F401
from banks_rag.application.agent.tools.registry import TOOL_SCHEMAS

from .prompts import (
    AFP_ANALYST_PROMPT,
    COYUNTURA_ANALYST_PROMPT,
    DOCUMENT_ANALYST_PROMPT,
    FFMM_ANALYST_PROMPT,
    FX_ANALYST_PROMPT,
    LIQUIDEZ_ANALYST_PROMPT,
    NR_ANALYST_PROMPT,
    POLICY_ANALYST_PROMPT,
    RENTA_FIJA_ANALYST_PROMPT,
)


@dataclass(frozen=True)
class SubAgentSpec:
    """Especificación de un sub-agente especialista.

    Campos:
        key: identificador corto, usado como etiqueta en la traza. Debe coincidir
            con el valor que ``domain_knowledge.financial_aliases`` retorna en
            ``.specialist`` para enrutar por vocabulario.
        display_name: nombre legible para la respuesta y los logs.
        system_prompt: prompt de sistema del sub-agente.
        tool_names: tools de dominio que el sub-agente puede usar (en el orden
            en que se le presentan).
        default_segments: segmentos del parquet_catalog que cubre el especialista
            (documentación + scoping de discover_query). Vacío si es transversal
            (NR) o documental.
        multi_step: True si su flujo de tools es multi-paso (discover → execute →
            compute), donde el razonamiento más rinde. En ``thinking_mode=adaptive``
            estos especialistas usan thinking; los demás (documentales) no. Ver
            ``conversation_loop._should_think`` y el diagnóstico de literatura.
    """

    key: str
    display_name: str
    system_prompt: str
    tool_names: tuple[str, ...]
    default_segments: tuple[str, ...] = ()
    multi_step: bool = False


# Las series de tiempo se consultan exclusivamente vía el catálogo de parquets
# (discover_query / execute_query / analytics). La extracción SQL en vivo del DW
# fue eliminada del proyecto.
#
# Roster híbrido: especialistas POR MERCADO sobre el catálogo de parquets + dos
# especialistas del corpus documental. Las `key` coinciden con los valores de
# financial_aliases (`.specialist`) para enrutar por vocabulario.
_QUANT_TOOLS = (
    "discover_query",
    "execute_query",
    "compute_variation",
    "compute_spread",
    "compute_composition",
    "compute_aggregate",
    "get_series_stats",
    "detect_anomaly",
    "plot_series",
)

SUBAGENTS: dict[str, SubAgentSpec] = {
    # ── Especialistas de mercado (catálogo de parquets) ──────────────────────
    "fx": SubAgentSpec(
        key="fx",
        display_name="Analista de Mercado Cambiario (FX)",
        system_prompt=FX_ANALYST_PROMPT,
        tool_names=(*_QUANT_TOOLS, "get_market_snapshot"),
        default_segments=("mercado_cambiario", "posiciones_cambiarias"),
        multi_step=True,
    ),
    "no_residentes": SubAgentSpec(
        key="no_residentes",
        display_name="Analista de No Residentes",
        system_prompt=NR_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        multi_step=True,
    ),
    "afp": SubAgentSpec(
        key="afp",
        display_name="Analista de Fondos de Pensiones (AFP)",
        system_prompt=AFP_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("fondos_pension",),
        multi_step=True,
    ),
    "fondos_mutuos": SubAgentSpec(
        key="fondos_mutuos",
        display_name="Analista de Fondos Mutuos (FFMM)",
        system_prompt=FFMM_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("fondos_pension",),
        multi_step=True,
    ),
    "renta_fija": SubAgentSpec(
        key="renta_fija",
        display_name="Analista de Renta Fija",
        system_prompt=RENTA_FIJA_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("renta_fija_chile", "instrumentos_bcch", "spreads_credito"),
        multi_step=True,
    ),
    "liquidez": SubAgentSpec(
        key="liquidez",
        display_name="Analista de Liquidez y Balance",
        system_prompt=LIQUIDEZ_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("liquidez_bancaria", "balance_bancario"),
        multi_step=True,
    ),
    # ── Especialistas del corpus documental ──────────────────────────────────
    "document": SubAgentSpec(
        key="document",
        display_name="Analista de Documentos",
        system_prompt=DOCUMENT_ANALYST_PROMPT,
        tool_names=(
            "search_documents",
            "search_visuals",
            "list_documents",
            "get_document_chunks",
            "compare_meetings",
            # Contexto noticioso para corroborar/explicar la coyuntura (base aislada).
            "search_current_context",
        ),
    ),
    "policy": SubAgentSpec(
        key="policy",
        display_name="Analista de Política Monetaria",
        system_prompt=POLICY_ANALYST_PROMPT,
        tool_names=(
            "search_documents",
            "get_document_chunks",
            "compare_meetings",
            "get_recent_policy_decisions",
            "discover_query",
            "execute_query",
            # Contexto noticioso para explicar el porqué del momento (base aislada).
            "search_current_context",
        ),
    ),
    # ── Especialista de coyuntura (noticias / contexto actual) ────────────────
    "coyuntura": SubAgentSpec(
        key="coyuntura",
        display_name="Analista de Coyuntura",
        system_prompt=COYUNTURA_ANALYST_PROMPT,
        tool_names=(
            "search_current_context",
        ),
    ),
}


def tool_schemas_for(spec: SubAgentSpec) -> list[dict]:
    """Subconjunto de ``TOOL_SCHEMAS`` que ve el sub-agente, en su orden declarado.

    Levanta ``ValueError`` si el spec referencia una tool no registrada — señal
    de que falta importar el módulo de la tool en ``tools/__init__.py``.
    """
    by_name = {schema["function"]["name"]: schema for schema in TOOL_SCHEMAS}
    missing = [name for name in spec.tool_names if name not in by_name]
    if missing:
        raise ValueError(
            f"Sub-agente {spec.key!r}: tools no registradas {missing}. "
            "¿Falta importar el módulo de la tool en tools/__init__.py?"
        )
    return [by_name[name] for name in spec.tool_names]


# Validación temprana: al importar este módulo, cada sub-agente debe referir
# solo tools efectivamente registradas. Falla rápido si el catálogo de tools
# y los specs se desincronizan.
for _spec in SUBAGENTS.values():
    tool_schemas_for(_spec)
del _spec
