"""
Pool async de PostgreSQL + helpers de acceso para el chatbot.

Centraliza:
  - El pool (`AsyncConnectionPool` de psycopg3).
  - Aplicación de los `schema/*.sql` al startup.
  - CRUD de sesiones e historial de chat.
  - Lectura de la tabla `historical_data`.

El pool se abre en `lifespan` (api.py) y se cierra ahí mismo. Los handlers
HTTP toman conexiones con `async with get_conn() as conn:` — el pool se
encarga del retorno.
"""
from __future__ import annotations

import sys
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from .settings import SCHEMA_DIR, settings

log = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None


def _skip_db() -> bool:
    return settings.chatbot_skip_db


def is_enabled() -> bool:
    return not _skip_db()


# ── Validación de identificadores SQL (defensa contra SQL injection en interpolación) ─
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_ident(name: str) -> str:
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Identificador SQL inválido: {name!r}")
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Ciclo de vida del pool
# ─────────────────────────────────────────────────────────────────────────────

async def init_pool() -> None:
    global _pool
    if _skip_db():
        _pool = None
        log.warning("CHATBOT_SKIP_DB=1: PostgreSQL omitido (modo testing)")
        return
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
        log.info("PostgreSQL pool cerrado")


@asynccontextmanager
async def get_conn() -> AsyncIterator[AsyncConnection]:
    if _pool is None:
        raise RuntimeError("DB pool no inicializado. Llama init_pool() primero.")
    async with _pool.connection() as conn:
        yield conn


async def healthcheck() -> bool:
    if _skip_db():
        return False
    try:
        async with get_conn() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        return True
    except Exception as e:
        log.warning("DB healthcheck failed: %s", e)
        return False


async def apply_schema_files() -> None:
    """Aplica todos los `schema/*.sql` ordenados — idempotentes (CREATE IF NOT EXISTS)."""
    if _skip_db():
        log.warning("CHATBOT_SKIP_DB=1: schema omitido")
        return
    paths = sorted(SCHEMA_DIR.glob("*.sql"))
    if not paths:
        log.warning("No se encontraron schemas en %s", SCHEMA_DIR)
        return
    async with get_conn() as conn:
        for path in paths:
            sql = path.read_text(encoding="utf-8")
            async with conn.cursor() as cur:
                await cur.execute(sql)
            log.info("Schema aplicado: %s", path.name)
        await conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Sesiones e historial
# ─────────────────────────────────────────────────────────────────────────────

async def create_session(metadata: Optional[dict] = None) -> str:
    if _skip_db():
        return str(uuid.uuid4())
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO chat_sessions (metadata) VALUES (%s) RETURNING session_id",
                [json.dumps(metadata or {})],
            )
            row = await cur.fetchone()
        await conn.commit()
    return str(row["session_id"])


async def session_exists(session_id: str) -> bool:
    if _skip_db():
        return False
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = %s",
                [session_id],
            )
            return (await cur.fetchone()) is not None


async def save_message(
    session_id: str,
    role: str,
    content: str,
    rag_sources: Optional[list[dict]] = None,
    historical_series: Optional[list[dict]] = None,
    token_count: Optional[int] = None,
    latency_ms: Optional[int] = None,
) -> str:
    if _skip_db():
        return str(uuid.uuid4())
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO chat_messages
                    (session_id, role, content, rag_sources, historical_series,
                     token_count, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING message_id
                """,
                [
                    session_id, role, content,
                    json.dumps(rag_sources) if rag_sources is not None else None,
                    json.dumps(historical_series) if historical_series is not None else None,
                    token_count, latency_ms,
                ],
            )
            row = await cur.fetchone()
        await conn.commit()
    return str(row["message_id"])


async def get_recent_history(session_id: str, max_turns: int) -> list[dict]:
    if max_turns <= 0 or _skip_db():
        return []
    limit = max_turns * 2
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at
                    FROM   chat_messages
                    WHERE  session_id = %s
                    ORDER  BY created_at DESC
                    LIMIT  %s
                ) sub
                ORDER BY created_at ASC
                """,
                [session_id, limit],
            )
            return [{"role": r["role"], "content": r["content"]} for r in await cur.fetchall()]


async def get_full_history(session_id: str) -> list[dict]:
    if _skip_db():
        return []
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT role, content, rag_sources, historical_series,
                       token_count, latency_ms, created_at
                FROM   chat_messages
                WHERE  session_id = %s
                ORDER  BY created_at ASC
                """,
                [session_id],
            )
            return [dict(r) for r in await cur.fetchall()]


async def list_sessions(limit: int = 20) -> list[dict]:
    if _skip_db():
        return []
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT s.session_id, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS num_messages
                FROM   chat_sessions s
                LEFT JOIN chat_messages m USING (session_id)
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT  %s
                """,
                [limit],
            )
            return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# RAG: vector + lexical
# ─────────────────────────────────────────────────────────────────────────────

async def vector_recall(query_vec: list[float], n: int) -> list[dict]:
    if _skip_db():
        return []
    docs_t = safe_ident(settings.docs_table)
    chunks_t = safe_ident(settings.chunks_table)
    emb_str = "[" + ",".join(f"{x:.7f}" for x in query_vec) + "]"
    sql = f"""
        SELECT c.chunk_id, c.document_id, c.text,
               c.page_start, c.page_end, c.section_type,
               c.importance_score, c.economic_variables,
               c.numeric_values, c.tags, c.chunk_date,
               d.filename, d.doc_type_category, d.document_date,
               1 - (c.embedding <=> %s::vector) AS vector_score,
               c.embedding
        FROM   {chunks_t} c
        JOIN   {docs_t}   d USING (document_id)
        ORDER  BY c.embedding <=> %s::vector
        LIMIT  %s
    """
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, [emb_str, emb_str, n])
            return [dict(r) for r in await cur.fetchall()]


async def lexical_recall(query: str, n: int) -> list[dict]:
    if _skip_db():
        return []
    docs_t = safe_ident(settings.docs_table)
    chunks_t = safe_ident(settings.chunks_table)
    sql = f"""
        SELECT c.chunk_id, c.document_id, c.text,
               c.page_start, c.page_end, c.section_type,
               c.importance_score, c.economic_variables,
               c.numeric_values, c.tags, c.chunk_date,
               d.filename, d.doc_type_category, d.document_date,
               ts_rank_cd(c.text_tsv, plainto_tsquery('simple', %s)) AS lexical_score,
               c.embedding
        FROM   {chunks_t} c
        JOIN   {docs_t}   d USING (document_id)
        WHERE  c.text_tsv @@ plainto_tsquery('simple', %s)
        ORDER  BY lexical_score DESC
        LIMIT  %s
    """
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, [query, query, n])
            return [dict(r) for r in await cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Series históricas
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_series_meta(series_ids: list[str]) -> dict[str, dict]:
    if _skip_db():
        return {}
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
    if _skip_db():
        return []
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
                # N más recientes, devuelta en orden cronológico ascendente
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


async def list_series_catalog() -> list[dict]:
    if _skip_db():
        return []
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT  s.series_id, s.series_name, s.unit, s.frequency,
                        s.source, s.economic_variable,
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
