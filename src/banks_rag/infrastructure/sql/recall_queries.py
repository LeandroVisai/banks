"""Queries SQL de recall: vector (HNSW) + lexical (tsvector) + fallback.

Las funciones reciben una conexión psycopg2 y la cláusula WHERE pre-construida
(con sus params). Retornan ``list[dict]`` (vía RealDictCursor) compatible con
el formato legacy.

Nombres de tabla deben venir validados (``safe_ident``) por el caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2.extensions


def format_pgvector(embedding: list[float]) -> str:
    """Literal pgvector ``[0.1234567,...]`` (7 decimales)."""
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


_BASE_COLUMNS = """
    c.chunk_id,
    c.document_id,
    c.text,
    c.page_start, c.page_end,
    c.section_type,
    c.importance_score,
    c.economic_variables,
    c.numeric_values,
    c.tags,
    c.chunk_date,
    c.image_path,
    d.filename, d.doc_type_category, d.document_date
"""


def vector_recall(
    conn,
    *,
    query_embedding: list[float],
    where_sql: str,
    where_params: list,
    n: int,
    docs_table: str,
    chunks_table: str,
) -> list[dict]:
    """Recall vector con HNSW (cosine). Retorna top-N con ``vector_score``."""
    from psycopg2.extras import RealDictCursor

    emb_str = format_pgvector(query_embedding)
    sql = f"""
    SELECT
        {_BASE_COLUMNS},
        1 - (c.embedding <=> %s::vector) AS vector_score,
        c.embedding
    FROM {chunks_table} c
    JOIN {docs_table} d USING (document_id)
    WHERE {where_sql}
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, [emb_str] + where_params + [emb_str, n])
        return cur.fetchall()


def lexical_recall(
    conn,
    *,
    query_text: str,
    where_sql: str,
    where_params: list,
    n: int,
    docs_table: str,
    chunks_table: str,
) -> list[dict]:
    """Recall lexical full-text con ``ts_rank_cd`` (cover density).

    Usa config ``'simple'`` para el tsquery — agnóstico de idioma, robusto
    para corpus mixto ES/EN. ``plainto_tsquery`` es permisivo.
    """
    from psycopg2.extras import RealDictCursor

    sql = f"""
    SELECT
        {_BASE_COLUMNS},
        ts_rank_cd(c.text_tsv, plainto_tsquery('simple', %s)) AS lexical_score,
        c.embedding
    FROM {chunks_table} c
    JOIN {docs_table} d USING (document_id)
    WHERE ({where_sql})
      AND c.text_tsv @@ plainto_tsquery('simple', %s)
    ORDER BY lexical_score DESC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, [query_text] + where_params + [query_text, n])
        return cur.fetchall()


def date_importance_fallback(
    conn,
    *,
    where_sql: str,
    where_params: list,
    n: int,
    docs_table: str,
    chunks_table: str,
) -> list[dict]:
    """Fallback cuando recalls primarios no devuelven nada: orden por importancia."""
    from psycopg2.extras import RealDictCursor

    sql = f"""
    SELECT
        {_BASE_COLUMNS},
        c.embedding
    FROM {chunks_table} c
    JOIN {docs_table} d USING (document_id)
    WHERE {where_sql}
    ORDER BY c.importance_score DESC, c.page_start ASC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, where_params + [n])
        return cur.fetchall()


def get_db_embedding_dim(conn, chunks_table: str) -> int | None:
    """Inspecciona la dimensión real del vector en la tabla. ``None`` si no existe."""
    import re

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (chunks_table,),
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    match = re.search(r"vector\((\d+)\)", row[0])
    return int(match.group(1)) if match else None
