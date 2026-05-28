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
    DOCUMENT_ANALYST_PROMPT,
    MARKET_ANALYST_PROMPT,
    POLICY_ANALYST_PROMPT,
    QUANT_ANALYST_PROMPT,
)


@dataclass(frozen=True)
class SubAgentSpec:
    """Especificación de un sub-agente especialista.

    Campos:
        key: identificador corto, usado como etiqueta en la traza.
        display_name: nombre legible para la respuesta y los logs.
        delegate_tool: nombre de la tool ``delegate_to_*`` que lo invoca.
        delegate_description: descripción que ve el orquestador para decidir
            cuándo delegar a este especialista.
        system_prompt: prompt de sistema del sub-agente.
        tool_names: tools de dominio que el sub-agente puede usar (en el orden
            en que se le presentan).
    """

    key: str
    display_name: str
    delegate_tool: str
    delegate_description: str
    system_prompt: str
    tool_names: tuple[str, ...]


# Las series de tiempo se consultan exclusivamente vía el catálogo de parquets
# (discover_query / execute_query / analytics). La extracción SQL en vivo del DW
# fue eliminada del proyecto.
SUBAGENTS: dict[str, SubAgentSpec] = {
    "document": SubAgentSpec(
        key="document",
        display_name="Analista de Documentos",
        delegate_tool="delegate_to_document_analyst",
        delegate_description=(
            "Delega al Analista de Documentos: busca y sintetiza evidencia del "
            "corpus del BCCh (Comunicados, Minutas, Fed Statements, research "
            "JPMorgan, Monitor PM). Úsalo para preguntas sobre qué dicen los "
            "documentos."
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
    "quant": SubAgentSpec(
        key="quant",
        display_name="Analista Cuantitativo",
        delegate_tool="delegate_to_quant_analyst",
        delegate_description=(
            "Delega al Analista Cuantitativo: consulta series del catálogo SQL "
            "(tipo de cambio, tasas, bonos, liquidez, commodities) y las "
            "interpreta — variaciones, spreads, estadística descriptiva. Úsalo "
            "para preguntas sobre cifras y su lectura."
        ),
        system_prompt=QUANT_ANALYST_PROMPT,
        tool_names=(
            "discover_query",
            "execute_query",
            "compute_variation",
            "compute_spread",
            "get_series_stats",
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
    "market": SubAgentSpec(
        key="market",
        display_name="Analista de Mercados",
        delegate_tool="delegate_to_market_analyst",
        delegate_description=(
            "Delega al Analista de Mercados: mercado cambiario, renta fija, "
            "liquidez bancaria, commodities y mercados internacionales; detecta "
            "anomalías y arma el panorama de mercado. Úsalo para preguntas "
            "sobre el estado y los movimientos del mercado."
        ),
        system_prompt=MARKET_ANALYST_PROMPT,
        tool_names=(
            "get_market_snapshot",
            "discover_query",
            "execute_query",
            "compute_variation",
            "compute_spread",
            "get_series_stats",
            "detect_anomaly",
            "search_documents",
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
