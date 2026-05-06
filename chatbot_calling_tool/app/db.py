"""
Pool async + helpers para el chatbot agentic.

Diferencias con chatbot/db.py:
  - Tablas distintas: `agent_sessions`, `agent_messages` (no `chat_*`).
  - `save_message` persiste `tool_trace`, `cited_chunks`, `iterations`.
  - Funciones de search expuestas con FILTROS (doc_type, year, date_range)
    porque las tools necesitan ese poder; el RAG clásico hace una sola pasada.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, AsyncIterator, Optional

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .settings import SCHEMA_DIR, settings

log = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_ident(name: str) -> str:
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Identificador SQL inválido: {name!r}")
    return name


async def init_pool() -> None:
    global _pool
    _pool = AsyncConnectionPool(
        conninfo=settings.dsn,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
        kwargs={"row_factory": dict_row, "autocommit": False},
        open=False,
    )
    await _pool.open(wait=True)
    log.info(
        "PostgreSQL pool listo db=%s host=%s min=%d max=%d",
        settings.pgdatabase, settings.pghost, settings.pg_pool_min, settings.pg_pool_max,
    )


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn() -> AsyncIterator[AsyncConnection]:
    if _pool is None:
        raise RuntimeError("DB pool no inicializado")
    async with _pool.connection() as conn:
        yield conn


async def healthcheck() -> bool:
    try:
        async with get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        return True
    except Exception as e:
        log.warning("DB healthcheck failed: %s", e)
        return False


async def apply_schema_files() -> None:
    paths = sorted(SCHEMA_DIR.glob("*.sql"))
    if not paths:
        return
    async with get_conn() as conn:
        for path in paths:
            sql = path.read_text(encoding="utf-8")
            async with conn.cursor() as cur:
                await cur.execute(sql)
            log.info("Schema aplicado: %s", path.name)
        await conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Sesiones e historial (tablas agent_*)
# ─────────────────────────────────────────────────────────────────────────────

async def create_session(metadata: Optional[dict] = None) -> str:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO agent_sessions (metadata) VALUES (%s) RETURNING session_id",
                [json.dumps(metadata or {})],
            )
            row = await cur.fetchone()
        await conn.commit()
    return str(row["session_id"])


async def session_exists(session_id: str) -> bool:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM agent_sessions WHERE session_id = %s",
                [session_id],
            )
            return (await cur.fetchone()) is not None


async def save_message(
    session_id: str,
    role: str,
    content: str,
    *,
    tool_trace: Optional[list[dict]] = None,
    cited_chunks: Optional[list[dict]] = None,
    historical_series: Optional[list[dict]] = None,
    iterations: Optional[int] = None,
    token_count: Optional[int] = None,
    latency_ms: Optional[int] = None,
) -> str:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_messages
                    (session_id, role, content, tool_trace, cited_chunks,
                     historical_series, iterations, token_count, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING message_id
                """,
                [
                    session_id, role, content,
                    json.dumps(tool_trace) if tool_trace is not None else None,
                    json.dumps(cited_chunks) if cited_chunks is not None else None,
                    json.dumps(historical_series) if historical_series is not None else None,
                    iterations, token_count, latency_ms,
                ],
            )
            row = await cur.fetchone()
        await conn.commit()
    return str(row["message_id"])


async def get_recent_history(session_id: str, max_turns: int) -> list[dict]:
    if max_turns <= 0:
        return []
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at
                    FROM   agent_messages
                    WHERE  session_id = %s AND role IN ('user', 'assistant')
                    ORDER  BY created_at DESC
                    LIMIT  %s
                ) sub
                ORDER BY created_at ASC
                """,
                [session_id, max_turns * 2],
            )
            return [{"role": r["role"], "content": r["content"]} for r in await cur.fetchall()]


async def get_full_history(session_id: str) -> list[dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT role, content, tool_trace, cited_chunks, historical_series,
                       iterations, token_count, latency_ms, created_at
                FROM   agent_messages
                WHERE  session_id = %s
                ORDER  BY created_at ASC
                """,
                [session_id],
            )
            return [dict(r) for r in await cur.fetchall()]


async def list_sessions(limit: int = 20) -> list[dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT s.session_id, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS num_messages
                FROM   agent_sessions s
                LEFT JOIN agent_messages m USING (session_id)
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT  %s
                """,
                [limit],
            )
            return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# RAG con FILTROS (esto es lo que hace potente a las tools)
# ─────────────────────────────────────────────────────────────────────────────

def _build_where(
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list[Any] = []
    if doc_type:
        clauses.append("d.doc_type_category = %s")
        params.append(doc_type)
    if year is not None:
        clauses.append("d.document_year = %s")
        params.append(year)
    if date_from:
        clauses.append("COALESCE(c.chunk_date, d.document_date::date) >= %s")
        params.append(date_from)
    if date_to:
        clauses.append("COALESCE(c.chunk_date, d.document_date::date) <= %s")
        params.append(date_to)
    return (" AND ".join(clauses) if clauses else ""), params


async def vector_recall(
    query_vec: list[float],
    n: int,
    *,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    docs_t = safe_ident(settings.docs_table)
    chunks_t = safe_ident(settings.chunks_table)
    emb_str = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"
    where, where_params = _build_where(doc_type, year, date_from, date_to)
    where_sql = f"WHERE {where}" if where else ""

    sql = f"""
        SELECT c.chunk_id, c.document_id, c.text,
               c.page_start, c.page_end, c.section_type,
               c.importance_score, c.economic_variables,
               c.chunk_date,
               d.filename, d.doc_type_category, d.document_date,
               1 - (c.embedding <=> %s::vector) AS vector_score,
               c.embedding
        FROM   {chunks_t} c
        JOIN   {docs_t}   d USING (document_id)
        {where_sql}
        ORDER  BY c.embedding <=> %s::vector
        LIMIT  %s
    """
    params = [emb_str, *where_params, emb_str, n]

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return [dict(r) for r in await cur.fetchall()]


async def lexical_recall(
    query: str,
    n: int,
    *,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    docs_t = safe_ident(settings.docs_table)
    chunks_t = safe_ident(settings.chunks_table)
    where, where_params = _build_where(doc_type, year, date_from, date_to)
    extra = f" AND {where}" if where else ""

    sql = f"""
        SELECT c.chunk_id, c.document_id, c.text,
               c.page_start, c.page_end, c.section_type,
               c.importance_score, c.economic_variables,
               c.chunk_date,
               d.filename, d.doc_type_category, d.document_date,
               ts_rank_cd(c.text_tsv, plainto_tsquery('simple', %s)) AS lexical_score,
               c.embedding
        FROM   {chunks_t} c
        JOIN   {docs_t}   d USING (document_id)
        WHERE  c.text_tsv @@ plainto_tsquery('simple', %s){extra}
        ORDER  BY lexical_score DESC
        LIMIT  %s
    """
    params = [query, query, *where_params, n]

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return [dict(r) for r in await cur.fetchall()]


async def get_document_by_filename(filename: str) -> Optional[dict]:
    docs_t = safe_ident(settings.docs_table)
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT * FROM {docs_t} WHERE filename = %s LIMIT 1",
                [filename],
            )
            row = await cur.fetchone()
            return dict(row) if row else None


async def get_chunks_by_document_id(document_id: str, limit: int = 50) -> list[dict]:
    docs_t = safe_ident(settings.docs_table)
    chunks_t = safe_ident(settings.chunks_table)
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT c.chunk_id, c.text, c.page_start, c.page_end,
                       c.section_type, c.importance_score, c.chunk_date,
                       d.filename, d.doc_type_category, d.document_date
                FROM   {chunks_t} c
                JOIN   {docs_t}   d USING (document_id)
                WHERE  c.document_id = %s
                ORDER  BY c.page_start ASC, c.chunk_id ASC
                LIMIT  %s
                """,
                [document_id, limit],
            )
            return [dict(r) for r in await cur.fetchall()]


async def list_documents(
    *,
    doc_type: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 50,
) -> list[dict]:
    docs_t = safe_ident(settings.docs_table)
    where: list[str] = []
    params: list[Any] = []
    if doc_type:
        where.append("doc_type_category = %s")
        params.append(doc_type)
    if year is not None:
        where.append("document_year = %s")
        params.append(year)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT document_id, filename, doc_type_category,
                       document_date, document_year
                FROM   {docs_t}
                {where_sql}
                ORDER  BY document_date DESC
                LIMIT  %s
                """,
                params,
            )
            return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Series históricas
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_series_meta(series_ids: list[str]) -> dict[str, dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT series_id, series_name, unit, frequency, source,
                       economic_variable, description
                FROM   historical_series
                WHERE  series_id = ANY(%s)
                """,
                [series_ids],
            )
            return {r["series_id"]: dict(r) for r in await cur.fetchall()}


async def fetch_series_rows(
    series_id: str,
    limit: int,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> list[dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            if date_from and date_to:
                await cur.execute(
                    """
                    SELECT date, value, notes
                    FROM   historical_data
                    WHERE  series_id = %s AND date BETWEEN %s AND %s
                    ORDER  BY date ASC
                    """,
                    [series_id, date_from, date_to],
                )
            else:
                await cur.execute(
                    """
                    SELECT date, value, notes FROM (
                        SELECT date, value, notes
                        FROM   historical_data
                        WHERE  series_id = %s AND value IS NOT NULL
                        ORDER  BY date DESC
                        LIMIT  %s
                    ) sub
                    ORDER BY date ASC
                    """,
                    [series_id, limit],
                )
            return [dict(r) for r in await cur.fetchall()]


async def list_series_catalog(variable: Optional[str] = None) -> list[dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            if variable:
                await cur.execute(
                    """
                    SELECT  s.series_id, s.series_name, s.unit, s.frequency,
                            s.source, s.economic_variable, s.description,
                            COUNT(d.date) AS num_observations,
                            MIN(d.date)   AS first_date,
                            MAX(d.date)   AS last_date
                    FROM    historical_series s
                    LEFT JOIN historical_data d USING (series_id)
                    WHERE   s.economic_variable = %s
                    GROUP BY s.series_id
                    ORDER BY s.series_id
                    """,
                    [variable],
                )
            else:
                await cur.execute(
                    """
                    SELECT  s.series_id, s.series_name, s.unit, s.frequency,
                            s.source, s.economic_variable, s.description,
                            COUNT(d.date) AS num_observations,
                            MIN(d.date)   AS first_date,
                            MAX(d.date)   AS last_date
                    FROM    historical_series s
                    LEFT JOIN historical_data d USING (series_id)
                    GROUP BY s.series_id
                    ORDER BY s.series_id
                    """
                )
            return [dict(r) for r in await cur.fetchall()]
