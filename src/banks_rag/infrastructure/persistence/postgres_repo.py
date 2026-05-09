"""Adapter PostgreSQL + pgvector con psycopg2 (sync) para el pipeline de ingesta.

Uso típico:

    repo = PostgresRepo(prefix="qwen_")
    repo.ensure_database()
    repo.setup_schema(dim=4096, drop_first=False)
    repo.upsert_documents(docs)
    repo.delete_chunks_for_docs([d["document_id"] for d in docs])
    repo.insert_chunks(chunks)

La API async (con psycopg3 + pool) se expondrá en una clase paralela
``AsyncPostgresRepo`` cuando se construya el FastAPI service (Sub-fase G).
Ambas comparten el mismo schema + queries de ``sql_templates.py``.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from . import sql_templates as sql

if TYPE_CHECKING:
    import psycopg2.extensions

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def safe_ident(name: str) -> str:
    """Valida un identificador SQL para evitar inyección por interpolación.

    Levanta ``ValueError`` si contiene caracteres fuera de ``[A-Za-z_][A-Za-z0-9_]*``.
    Usado para validar nombres de tablas con prefijo dinámico.
    """
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Identificador SQL inválido: {name!r}")
    return name


def format_vector(embedding: list[float]) -> str:
    """Literal pgvector: ``[0.1234567,0.7654321,...]`` (7 decimales)."""
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


def document_year_from_date(date: str | None) -> int | None:
    """Extrae el año de un ISO date (``YYYY-MM-DD`` o solo ``YYYY``)."""
    if not date:
        return None
    try:
        return int(date[:4])
    except (ValueError, TypeError):
        return None


@contextmanager
def _autocommit_conn(connect_fn):
    conn = connect_fn()
    conn.autocommit = True
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _tx_conn(connect_fn):
    conn = connect_fn()
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class PostgresRepo:
    """Adapter sincrónico (psycopg2) para el corpus en PostgreSQL.

    Args:
        prefix: prefijo de tablas (``"qwen_"``, ``"gemma_"`` o ``""``).
            Validado contra ``[A-Za-z_][A-Za-z0-9_]*``.
        database: override del nombre de BD (default: env ``PGDATABASE`` o ``rag_banco``).
        connect_params: dict adicional de parámetros para ``psycopg2.connect``.
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        database: str | None = None,
        connect_params: dict[str, Any] | None = None,
    ) -> None:
        # El prefijo puede ser vacío; valida solo el nombre completo de tabla.
        self.prefix = prefix
        self.docs_table = safe_ident(f"{prefix}documents")
        self.chunks_table = safe_ident(f"{prefix}chunks")
        self.database = database or os.getenv("PGDATABASE", "rag_banco")
        self._connect_params = connect_params or {}

    # ── Conexión ──────────────────────────────────────────────────────────────

    def _conn_kwargs(self, *, database: str | None = None) -> dict[str, Any]:
        return {
            "host": os.getenv("PGHOST", "localhost"),
            "port": int(os.getenv("PGPORT", "5432")),
            "user": os.getenv("PGUSER", os.getenv("USER", "postgres")),
            "password": os.getenv("PGPASSWORD", "postgres"),
            "database": database or self.database,
            **self._connect_params,
        }

    def connect(self, *, database: str | None = None) -> "psycopg2.extensions.connection":
        import psycopg2

        return psycopg2.connect(**self._conn_kwargs(database=database))

    # ── Setup ─────────────────────────────────────────────────────────────────

    def ensure_database(self) -> bool:
        """Crea la BD si no existe. Conecta a ``postgres`` para hacer ``CREATE DATABASE``.

        Returns:
            ``True`` si la BD ya existía, ``False`` si la creó.
        """

        def _connect_postgres():
            return self.connect(database="postgres")

        with _autocommit_conn(_connect_postgres) as conn, conn.cursor() as cur:
            cur.execute(sql.DATABASE_EXISTS_SQL, (self.database,))
            if cur.fetchone() is not None:
                return True
            # CREATE DATABASE no soporta parámetros; el nombre ya viene de env
            # validado, pero envolvemos en comillas dobles.
            cur.execute(f'CREATE DATABASE "{self.database}"')
            return False

    def pgvector_available(self) -> bool:
        try:
            with self.connect() as conn, conn.cursor() as cur:
                cur.execute(sql.PGVECTOR_AVAILABLE_SQL)
                return cur.fetchone() is not None
        except Exception:  # noqa: BLE001
            return False

    def setup_schema(self, *, dim: int, drop_first: bool = False) -> None:
        """Crea schema + 18 índices. Idempotente salvo ``drop_first=True``."""
        with _tx_conn(self.connect) as conn, conn.cursor() as cur:
            if drop_first:
                drop_chunks, drop_docs = sql.drop_sql(self.docs_table, self.chunks_table)
                cur.execute(drop_chunks)
                cur.execute(drop_docs)

            cur.execute(sql.schema_sql(dim, self.docs_table, self.chunks_table))
            for migration in sql.post_schema_migrations(self.chunks_table):
                cur.execute(migration)
            for stmt in sql.index_statements(self.docs_table, self.chunks_table):
                cur.execute(stmt)

    # ── Inspección ────────────────────────────────────────────────────────────

    def existing_embedding_dim(self) -> int | None:
        """Si la tabla chunks ya existe, devuelve la dimensión del vector."""
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.table_exists_sql(), (self.chunks_table,))
            if cur.fetchone()[0] is None:
                return None
            cur.execute(sql.EMBEDDING_COLUMN_TYPE_SQL, (self.chunks_table,))
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            m = re.search(r"vector\((\d+)\)", row[0])
            return int(m.group(1)) if m else None

    # ── DML ──────────────────────────────────────────────────────────────────

    def upsert_documents(self, documents: list[dict]) -> None:
        from psycopg2.extras import Json, execute_values

        rows = [self._doc_to_row(d) for d in documents]
        with _tx_conn(self.connect) as conn, conn.cursor() as cur:
            execute_values(cur, sql.upsert_documents_sql(self.docs_table), rows)

    def delete_chunks_for_docs(self, document_ids: list[str]) -> int:
        if not document_ids:
            return 0
        with _tx_conn(self.connect) as conn, conn.cursor() as cur:
            cur.execute(sql.delete_chunks_for_docs_sql(self.chunks_table), (document_ids,))
            return cur.rowcount

    def insert_chunks(self, chunks: list[dict], *, expected_dim: int) -> int:
        """Inserta chunks. Filtra chunks con dimensión incorrecta (loggea warning).

        Returns:
            Número de chunks efectivamente insertados.
        """
        from psycopg2.extras import execute_values

        rows = self._chunks_to_rows(chunks, expected_dim=expected_dim)
        if not rows:
            return 0

        with _tx_conn(self.connect) as conn, conn.cursor() as cur:
            execute_values(
                cur,
                sql.insert_chunks_sql(self.chunks_table),
                rows,
                template=sql.INSERT_CHUNKS_TEMPLATE,
            )
            cur.execute(sql.update_text_tsv_sql(self.chunks_table))
        return len(rows)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        queries = sql.stats_queries(self.docs_table, self.chunks_table)
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(queries["n_docs"])
            n_docs = cur.fetchone()[0]
            cur.execute(queries["n_chunks"])
            n_chunks = cur.fetchone()[0]
            cur.execute(queries["by_doc_type"])
            by_type = cur.fetchall()
            cur.execute(queries["by_section"])
            by_section = cur.fetchall()
            cur.execute(queries["importance"])
            avg_imp, high_imp = cur.fetchone()
            cur.execute(queries["n_policy"])
            n_policy = cur.fetchone()[0]
        return {
            "database": self.database,
            "documents": n_docs,
            "chunks": n_chunks,
            "importance_avg": float(avg_imp) if avg_imp is not None else None,
            "high_importance_count": high_imp,
            "policy_decision_count": n_policy,
            "by_doc_type": list(by_type),
            "by_section": list(by_section),
        }

    # ── Builders de filas (privados) ──────────────────────────────────────────

    @staticmethod
    def _doc_to_row(doc: dict) -> tuple:
        from psycopg2.extras import Json

        return (
            doc["document_id"],
            doc["filename"],
            doc["filepath"],
            doc["doc_type_category"],
            doc["institution"],
            doc.get("document_date"),
            document_year_from_date(doc.get("document_date")),
            doc.get("total_pages"),
            doc.get("total_chunks"),
            doc.get("char_count"),
            Json(doc.get("extraction_warnings", [])),
        )

    @staticmethod
    def _chunks_to_rows(chunks: list[dict], *, expected_dim: int) -> list[tuple]:
        from psycopg2.extras import Json

        rows: list[tuple] = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            if not embedding or len(embedding) != expected_dim:
                # Skip silenciosamente; el orquestador imprime el resumen.
                continue
            rows.append((
                chunk["chunk_id"],
                chunk["document_id"],
                chunk["text"],
                chunk.get("char_count"),
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk.get("position_in_doc"),
                chunk.get("section_type", "CONTENIDO"),
                float(chunk.get("section_confidence", 0)),
                Json(chunk.get("economic_variables", {})),
                Json(chunk.get("numeric_values", [])),
                Json(chunk.get("entities", {})),
                Json(chunk.get("temporal_refs", {})),
                chunk.get("tags", []),
                float(chunk.get("importance_score", 0)),
                bool(chunk.get("is_policy_decision", False)),
                bool(chunk.get("is_forward_looking", False)),
                chunk.get("chunk_date"),
                chunk.get("image_path"),
                format_vector(embedding),
                chunk.get("embedding_model"),
            ))
        return rows
