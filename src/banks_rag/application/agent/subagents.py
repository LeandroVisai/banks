"""Definición de los sub-agentes especializados y el wiring de delegación.

La Fase B convierte el agente único en un orquestador que delega en cuatro
especialistas. Cada uno tiene:

  - un **system prompt** propio (cómo razona ese tipo de analista) — en ``prompts.py``;
  - un **subconjunto de las tools** registradas (qué dominio puede tocar);
  - una **tool de delegación** ``delegate_to_*`` que el orquestador usa para invocarlo.

El orquestador NO ve las tools de dominio: solo ve las cuatro ``delegate_to_*``.
Esa es la única interfaz entre el orquestador y los especialistas, lo que
mantiene a cada capa con un contexto acotado y evita recursión (un especialista
no puede delegar).
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
            con el valor que ``domain_knowledge.financial_aliases.specialist_for``
            retorna para enrutar por vocabulario.
        display_name: nombre legible para la respuesta y los logs.
        delegate_tool: nombre de la tool ``delegate_to_*`` que lo invoca.
        delegate_description: descripción que ve el orquestador para decidir
            cuándo delegar a este especialista.
        system_prompt: prompt de sistema del sub-agente.
        tool_names: tools de dominio que el sub-agente puede usar (en el orden
            en que se le presentan).
        default_segments: segmentos del parquet_catalog que cubre el especialista
            (documentación + scoping de discover_query). Vacío si es transversal
            (NR) o documental.
    """

    key: str
    display_name: str
    delegate_tool: str
    delegate_description: str
    system_prompt: str
    tool_names: tuple[str, ...]
    default_segments: tuple[str, ...] = ()


# Las series de tiempo se consultan exclusivamente vía el catálogo de parquets
# (discover_query / execute_query / analytics). La extracción SQL en vivo del DW
# fue eliminada del proyecto.
#
# Roster híbrido (Fase 1): especialistas POR MERCADO sobre el catálogo de
# parquets + dos especialistas del corpus documental. Las `key` coinciden con
# los valores de financial_aliases.specialist_for para enrutar por vocabulario.
_QUANT_TOOLS = (
    "discover_query",
    "execute_query",
    "compute_variation",
    "compute_spread",
    "compute_composition",
    "compute_aggregate",
    "get_series_stats",
    "detect_anomaly",
)

SUBAGENTS: dict[str, SubAgentSpec] = {
    # ── Especialistas de mercado (catálogo de parquets) ──────────────────────
    "fx": SubAgentSpec(
        key="fx",
        display_name="Analista de Mercado Cambiario (FX)",
        delegate_tool="delegate_to_fx_analyst",
        delegate_description=(
            "Delega al Analista de Mercado Cambiario (FX): tipo de cambio "
            "spot/forward, flujos cambiarios por sector, puntos forward, "
            "posiciones en derivados y commodities (cobre, petróleo, DXY). "
            "Úsalo para el dólar, el mercado cambiario y los commodities."
        ),
        system_prompt=FX_ANALYST_PROMPT,
        tool_names=(*_QUANT_TOOLS, "get_market_snapshot"),
        default_segments=("mercado_cambiario", "posiciones_cambiarias"),
    ),
    "no_residentes": SubAgentSpec(
        key="no_residentes",
        display_name="Analista de No Residentes",
        delegate_tool="delegate_to_nr_analyst",
        delegate_description=(
            "Delega al Analista de No Residentes (NR): posición y flujos de "
            "inversionistas no residentes en instrumentos chilenos (renta fija "
            "local, spot, forward, derivados). Úsalo para preguntas sobre NR."
        ),
        system_prompt=NR_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
    ),
    "afp": SubAgentSpec(
        key="afp",
        display_name="Analista de Fondos de Pensiones (AFP)",
        delegate_tool="delegate_to_afp_analyst",
        delegate_description=(
            "Delega al Analista de Fondos de Pensiones (AFP): cartera/allocation "
            "nacional vs. internacional, stock por fondo, DV01, MTM, atribución "
            "y posición cambiaria de las AFP. Úsalo para fondos de pensiones."
        ),
        system_prompt=AFP_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("fondos_pension",),
    ),
    "fondos_mutuos": SubAgentSpec(
        key="fondos_mutuos",
        display_name="Analista de Fondos Mutuos (FFMM)",
        delegate_tool="delegate_to_ffmm_analyst",
        delegate_description=(
            "Delega al Analista de Fondos Mutuos (FFMM): flujos, stock, "
            "duración, DV01 y composición de los fondos mutuos. Úsalo para "
            "preguntas sobre fondos mutuos."
        ),
        system_prompt=FFMM_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("fondos_pension",),
    ),
    "renta_fija": SubAgentSpec(
        key="renta_fija",
        display_name="Analista de Renta Fija",
        delegate_tool="delegate_to_renta_fija_analyst",
        delegate_description=(
            "Delega al Analista de Renta Fija: curvas soberanas (BTP/BTU/SPC/"
            "OIS), break-evens, spreads, montos y volatilidad, PDBC e "
            "instrumentos BCCh, spreads de crédito. Úsalo para bonos y tasas."
        ),
        system_prompt=RENTA_FIJA_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("renta_fija_chile", "instrumentos_bcch", "spreads_credito"),
    ),
    "liquidez": SubAgentSpec(
        key="liquidez",
        display_name="Analista de Liquidez y Balance",
        delegate_tool="delegate_to_liquidez_analyst",
        delegate_description=(
            "Delega al Analista de Liquidez y Balance: LCR, NSFR, caja, reserva "
            "técnica, operaciones de liquidez, TIB y balance bancario "
            "(activos/pasivos por banco). Úsalo para liquidez y balance bancario."
        ),
        system_prompt=LIQUIDEZ_ANALYST_PROMPT,
        tool_names=_QUANT_TOOLS,
        default_segments=("liquidez_bancaria", "balance_bancario"),
    ),
    # ── Especialistas del corpus documental ──────────────────────────────────
    "document": SubAgentSpec(
        key="document",
        display_name="Analista de Documentos",
        delegate_tool="delegate_to_document_analyst",
        delegate_description=(
            "Delega al Analista de Documentos: busca y sintetiza evidencia del "
            "corpus del BCCh (Comunicados, Minutas, IPoM, IEF, Fed Statements, "
            "research JPMorgan, Monitor PM). Úsalo para preguntas sobre qué "
            "dicen los documentos."
        ),
        system_prompt=DOCUMENT_ANALYST_PROMPT,
        tool_names=(
            "search_documents",
            "search_visuals",
            "list_documents",
            "get_document_chunks",
            "compare_meetings",
        ),
    ),
    "policy": SubAgentSpec(
        key="policy",
        display_name="Analista de Política Monetaria",
        delegate_tool="delegate_to_policy_analyst",
        delegate_description=(
            "Delega al Analista de Política Monetaria: decisiones de TPM, "
            "razonamiento y votaciones del Consejo, trayectoria de la política "
            "y expectativas de mercado. Úsalo para preguntas sobre política "
            "monetaria del BCCh."
        ),
        system_prompt=POLICY_ANALYST_PROMPT,
        tool_names=(
            "search_documents",
            "get_document_chunks",
            "compare_meetings",
            "get_recent_policy_decisions",
            "discover_query",
            "execute_query",
        ),
    ),
}

_DELEGATE_TO_KEY: dict[str, str] = {
    spec.delegate_tool: key for key, spec in SUBAGENTS.items()
}


def subagent_for_delegate(delegate_tool: str) -> SubAgentSpec | None:
    """Resuelve la tool ``delegate_to_*`` a su ``SubAgentSpec`` (o ``None``)."""
    key = _DELEGATE_TO_KEY.get(delegate_tool)
    return SUBAGENTS[key] if key is not None else None


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


def _delegate_schema(spec: SubAgentSpec) -> dict:
    """Schema OpenAI-compatible de la tool ``delegate_to_*`` de un especialista."""
    return {
        "type": "function",
        "function": {
            "name": spec.delegate_tool,
            "description": spec.delegate_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Instrucción concreta y autocontenida para el "
                            "especialista. Incluye variables, fechas y el "
                            "contexto necesario: el especialista no ve la "
                            "conversación, solo este texto."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    }


# Las cuatro tools de delegación que ve el orquestador.
DELEGATE_SCHEMAS: list[dict] = [_delegate_schema(spec) for spec in SUBAGENTS.values()]

# Validación temprana: al importar este módulo, cada sub-agente debe referir
# solo tools efectivamente registradas. Falla rápido si el catálogo de tools
# y los specs se desincronizan.
for _spec in SUBAGENTS.values():
    tool_schemas_for(_spec)
del _spec
