"""Orquestador unificado de búsqueda híbrida (legacy 04_search + 04_advanced_search).

Pipeline:

    parse_query → vector_recall + lexical_recall → RRF → MMR → importance_boost
    → [CrossEncoderReranker opcional] → top-k

Si el primer recall viene vacío con filtros estrictos (variables/sections),
relaja esos dos filtros y reintenta. Si sigue vacío, cae a date_importance_fallback.

Acepta filtros adicionales explícitos vía ``extra_filters`` (no detectables del NL).
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from banks_rag.domain.retrieval import ParsedQuery, SearchFilters, SearchResult
from banks_rag.infrastructure.persistence.postgres_repo import PostgresRepo
from banks_rag.infrastructure.sql.recall_queries import (
    date_importance_fallback,
    get_db_embedding_dim,
    lexical_recall,
    vector_recall,
)

from .filters import build_filters_sql
from .fusion import (
    DEFAULT_IMPORTANCE_BOOST,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_RECENCY_WEIGHT,
    DEFAULT_RRF_K,
    importance_boost,
    mark_low_confidence,
    mmr_select,
    rrf_fuse,
)
from .query_parser import parse_query, row_matches_month

if TYPE_CHECKING:
    pass

DEFAULT_RECALL_N = 50

# Instrucción para Qwen3-Embedding en modo retrieval financiero (queries).
_QWEN_QUERY_INSTRUCTION = (
    "Instruct: Dado el texto de una consulta financiera en español, "
    "recupera los fragmentos de documentos del Banco Central de Chile "
    "más relevantes.\nQuery: "
)


def embed_query_text(query_text: str, model_name: str) -> str:
    """Construye el texto de query con el prefijo correcto según familia del modelo.

    Refleja la lógica usada por el legacy en ``04_search.embed_query``.
    """
    name_lower = model_name.lower()
    if "e5" in name_lower:
        return "query: " + query_text
    if "qwen" in name_lower and "embedding" in name_lower:
        return _QWEN_QUERY_INSTRUCTION + query_text
    return query_text


def _merge_filters(
    parsed: SearchFilters,
    extra: SearchFilters | None,
) -> SearchFilters:
    """Combina filtros parseados con extras explícitos. Los extras tienen prioridad
    para campos auto-detectables si vienen no-default."""
    if extra is None:
        return parsed

    merged = SearchFilters(
        # Auto-detectados: ``extra`` overridea solo si vienen no-default.
        exact_date=extra.exact_date or parsed.exact_date,
        day=extra.day if extra.day is not None else parsed.day,
        year_from=extra.year_from if extra.year_from is not None else parsed.year_from,
        year_to=extra.year_to if extra.year_to is not None else parsed.year_to,
        month=extra.month if extra.month is not None else parsed.month,
        doc_types=extra.doc_types or parsed.doc_types,
        variables=extra.variables or parsed.variables,
        sections=extra.sections or parsed.sections,
        # Avanzados (siempre vienen del extra).
        years=extra.years,
        months=extra.months,
        institutions=extra.institutions,
        entities=extra.entities,
        tags=extra.tags,
        min_importance=extra.min_importance,
        max_importance=extra.max_importance,
        min_section_confidence=extra.min_section_confidence,
        exclude_boilerplate=extra.exclude_boilerplate,
        exclude_doc_types=extra.exclude_doc_types,
        exclude_institutions=extra.exclude_institutions,
    )
    return merged


def _adjust_query_vec_to_db(
    query_vec: np.ndarray,
    db_dim: int | None,
) -> np.ndarray:
    """Ajusta la dimensión del query vector al de la BD si difieren.

    Trunca si es más grande, padea con ceros si es más chico. Tras truncar
    se **re-normaliza L2**: truncar un vector normalizado reduce su norma y
    distorsiona la distancia coseno contra los embeddings de la BD.
    """
    if db_dim is None or query_vec.shape[0] == db_dim:
        return query_vec
    if query_vec.shape[0] > db_dim:
        truncated = query_vec[:db_dim]
        norm = np.linalg.norm(truncated)
        return truncated / norm if norm > 0 else truncated
    return np.pad(query_vec, (0, db_dim - query_vec.shape[0]))


def _recall_and_fuse(
    conn,
    *,
    query_vec: np.ndarray,
    query_text: str,
    filters: SearchFilters,
    docs_table: str,
    chunks_table: str,
    n: int = DEFAULT_RECALL_N,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    where_sql, where_params = build_filters_sql(
        filters, docs_table=docs_table, chunks_table=chunks_table,
        chunks_alias="c", docs_alias="d",
    )

    vec_hits = vector_recall(
        conn,
        query_embedding=query_vec.tolist(),
        where_sql=where_sql,
        where_params=where_params,
        n=n,
        docs_table=docs_table,
        chunks_table=chunks_table,
    )
    lex_hits = lexical_recall(
        conn,
        query_text=query_text,
        where_sql=where_sql,
        where_params=where_params,
        n=n,
        docs_table=docs_table,
        chunks_table=chunks_table,
    )
    return rrf_fuse(vec_hits, lex_hits, k=rrf_k)


def hybrid_search(
    query: str,
    *,
    query_embedder,
    repo: PostgresRepo | None = None,
    extra_filters: SearchFilters | None = None,
    k: int = 5,
    use_mmr: bool = True,
    recall_n: int = DEFAULT_RECALL_N,
    rrf_k: int = DEFAULT_RRF_K,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    importance_weight: float = DEFAULT_IMPORTANCE_BOOST,
    recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    reranker: Any | None = None,
) -> SearchResult:
    """Búsqueda híbrida unificada con filtros automáticos + explícitos.

    Args:
        query: NL del usuario.
        query_embedder: objeto con ``encode_text(list[str], batch_size) → np.ndarray``
            y atributo ``name`` (típicamente ``SentenceTransformersEmbedder``).
        repo: ``PostgresRepo``; si ``None``, crea uno con env vars.
        extra_filters: filtros explícitos (institutions, entities, tags, importance...)
            que sobrescriben/complementan los auto-detectados.
        k: número de resultados finales.
        use_mmr: aplicar diversificación MMR antes del importance_boost.
        recall_n: top-N por rama antes de fusionar.
        rrf_k, mmr_lambda, importance_weight, recency_weight: hiperparámetros.
        reranker: objeto opcional con ``rerank(query, chunks, top_k) → list[dict]``
            (e.g. ``CrossEncoderReranker``). Si se provee, re-ordena el pool
            post-MMR+importance antes de cortar a ``k``.

    Returns:
        ``SearchResult`` con ``hits`` ordenados (top-K), ``parsed_filters``
        snapshot para serialización.
    """
    if repo is None:
        repo = PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))

    parsed: ParsedQuery = parse_query(query)
    filters = _merge_filters(parsed.filters, extra_filters)

    # Embed de query
    embed_text = embed_query_text(parsed.clean_query, query_embedder.name)
    query_vec = query_embedder.encode_text([embed_text], batch_size=1)[0]

    docs_table, chunks_table = repo.docs_table, repo.chunks_table

    with repo.connect() as conn:
        db_dim = get_db_embedding_dim(conn, chunks_table)
        query_vec = _adjust_query_vec_to_db(query_vec, db_dim)

        fused = _recall_and_fuse(
            conn,
            query_vec=query_vec, query_text=parsed.clean_query,
            filters=filters,
            docs_table=docs_table, chunks_table=chunks_table,
            n=recall_n, rrf_k=rrf_k,
        )

    relaxed_filters: list[str] = []
    fallback_used = False

    # Si hay filtros estrictos y no hubo resultados, relajar y reintentar.
    if not fused and filters.has_strict_filters():
        relaxed_filters = [
            name for name in ("variables", "sections") if getattr(filters, name)
        ]
        relaxed = SearchFilters(**{**asdict(filters), "variables": [], "sections": []})
        with repo.connect() as conn:
            fused = _recall_and_fuse(
                conn,
                query_vec=query_vec, query_text=parsed.clean_query,
                filters=relaxed,
                docs_table=docs_table, chunks_table=chunks_table,
                n=recall_n, rrf_k=rrf_k,
            )

    # Filtro post-SQL por mes (los PDFs sin chunk_date no se filtran en SQL).
    if filters.month is not None:
        fused = [hit for hit in fused if row_matches_month(hit, filters.month)]
    if filters.months:
        fused = [
            hit for hit in fused
            if any(row_matches_month(hit, m) for m in filters.months)
        ]

    # Fallback final: importance + página
    if not fused:
        fallback_used = True
        with repo.connect() as conn:
            where_sql, where_params = build_filters_sql(
                filters, docs_table=docs_table, chunks_table=chunks_table,
                chunks_alias="c", docs_alias="d",
            )
            fused = date_importance_fallback(
                conn,
                where_sql=where_sql, where_params=where_params,
                n=max(k * 4, 20),
                docs_table=docs_table, chunks_table=chunks_table,
            )

    if use_mmr and fused:
        fused = mmr_select(fused, query_vec, k=min(k * 2, len(fused)), lambda_param=mmr_lambda)

    ranked = importance_boost(fused, weight=importance_weight, recency_weight=recency_weight)

    # Reranker de segunda pasada: re-score con cross-encoder antes de cortar a k.
    if reranker is not None and ranked:
        top = reranker.rerank(parsed.clean_query, ranked, top_k=k)
    else:
        top = ranked[:k]

    mark_low_confidence(top)

    return SearchResult(
        query=parsed.raw_query,
        clean_query=parsed.clean_query,
        hits=top,
        parsed_filters=asdict(filters),
        relaxed_filters=relaxed_filters,
        fallback_used=fallback_used,
    )
