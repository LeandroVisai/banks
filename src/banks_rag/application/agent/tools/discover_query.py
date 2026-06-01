"""Tool ``discover_query`` — busca datasets del parquet_catalog para responder
una pregunta sobre series financieras o macroeconómicas.

El agente la usa para encontrar qué *dataset* del catálogo tiene los datos
que necesita (USD/CLP, curva BTP, LCR, precio del cobre, etc.) antes de
llamar a ``execute_query`` o a alguna de las analytics tools.

Estrategia de scoring: BM25-like delegado en
``parquet_catalog_loader.search_datasets`` (solapamiento de tokens entre la
consulta del usuario y los campos ``id+name+description+segment+unit`` de
cada dataset)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from banks_rag.domain_knowledge.financial_aliases import resolve_segment
from banks_rag.infrastructure.sql.catalog_index import catalog_semantic_scores
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    load_parquet_catalog,
    search_datasets,
)

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState


_SCHEMA = {
    "type": "function",
    "function": {
        "name": "discover_query",
        "description": (
            "Descubre qué datasets del catálogo de parquets responden a una "
            "pregunta sobre datos financieros o macroeconómicos (tipo de "
            "cambio, tasas, bonos, liquidez bancaria, commodities, balance "
            "del sistema, etc.). Retorna los top-k datasets con su id, "
            "esquema de columnas, unidad y rango de fechas disponibles. "
            "Úsala ANTES de execute_query (para traer las filas) o de las "
            "analytics tools (compute_variation, compute_spread, "
            "get_series_stats, detect_anomaly), todas las cuales reciben el "
            "`id` del dataset que esta tool devuelve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Pregunta o descripción en lenguaje natural de los "
                        "datos que necesitas (ej. 'curva de bonos BTP en "
                        "pesos', 'tipo de cambio dólar último mes', "
                        "'activos del banco BCI en pesos'). OPCIONAL: si lo "
                        "omites, lista los datasets disponibles (filtrados por "
                        "`segment` si lo entregas) para que elijas uno."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Número de datasets a retornar (default 5, máx 20).",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
                "segment": {
                    "type": "string",
                    "description": (
                        "Filtrar por segmento del catálogo (ej. "
                        "'mercado_cambiario', 'balance_bancario', "
                        "'renta_fija_chile', 'commodities', "
                        "'liquidez_bancaria', 'politica_monetaria')."
                    ),
                },
            },
            "required": [],
        },
    },
}


@register("discover_query", _SCHEMA)
async def discover_query(
    state: AgentState,
    query: str | None = None,
    top_k: int = 5,
    segment: str | None = None,
) -> dict[str, Any]:
    entries = load_parquet_catalog()
    available_segments = {e.segment for e in entries}

    segment_note: str | None = None
    if segment:
        # Resolver alias del LLM ("AFP" → "fondos_pension"); si no mapea, NO
        # fallar: buscar en todo el catálogo y avisar (evita el error duro de
        # los logs que dejaba al modelo sin datos → alucinación).
        resolved = resolve_segment(segment, available_segments)
        if resolved is not None:
            entries = [e for e in entries if e.segment == resolved]
            if resolved != segment:
                segment_note = f"Segmento {segment!r} interpretado como {resolved!r}."
        else:
            segment_note = (
                f"Segmento {segment!r} no reconocido; se buscó en todo el "
                "catálogo. Segmentos válidos: "
                f"{', '.join(sorted(available_segments))}."
            )

    top_k = max(1, min(int(top_k), 20))
    query = (query or "").strip()
    if not query:
        # Modo browse: sin texto de búsqueda, lista los datasets disponibles
        # (ya filtrados por segment si se entregó). Robusto ante modelos que
        # llaman discover_query solo con `segment` esperando un listado.
        top = entries[:top_k]
    else:
        # Scoring semántico best-effort (embeddings del catálogo); {} si la
        # feature está off o el modelo no está → ranking léxico+alias.
        semantic = catalog_semantic_scores(query, entries)
        top = search_datasets(entries, query, top_k=top_k, extra_scores=semantic)

    result: dict[str, Any] = {
        "results": [e.to_dict() for e in top],
        "n_results": len(top),
        "hint": (
            "Cada resultado tiene `id` (úsalo como `dataset_id`), `columns` "
            "(con tipos y valores de enum cuando aplica) y `date_range` "
            "(primera y última fecha del parquet). Para traer filas crudas "
            "usa execute_query(dataset_id, columns?, fecha_inicio?, "
            "fecha_fin?, filters?, limit?). Para análisis usa "
            "compute_variation / compute_spread / get_series_stats / "
            "detect_anomaly con (dataset_id, column)."
        ),
    }
    if segment_note:
        result["segment_note"] = segment_note
    if not top:
        result["message"] = (
            "Ningún dataset coincide. Reformula con términos del dominio "
            "(p.ej. 'AFP', 'no residentes', 'curva BTP') o quita el filtro de "
            "segmento. Si el dato no existe en el catálogo, NO lo inventes: "
            "indica que no está disponible."
        )
    return result
