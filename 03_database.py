#!/usr/bin/env python3
"""
PASO 3 — Base de datos PostgreSQL con pgvector.

Schema (normalizado):
  documents                  → 1 fila por PDF (metadata del documento)
  chunks                     → N filas por documento, con embedding vector(dim)
                              y text_tsv (tsvector) para BM25 híbrido.
                              La dimensión se auto-detecta desde chunks_vectorized.json
                              o mediante la variable de entorno RAG_EMBEDDING_DIM.

Índices:
  chunks.embedding           → HNSW con vector_cosine_ops (embeddings normalizados
                              L2 → cosine similarity)
  chunks.text_tsv            → GIN (full-text, config 'simple' para multilingüe)
  chunks.tags                → GIN sobre TEXT[]
  chunks.economic_variables  → GIN sobre JSONB
  chunks.entities            → GIN sobre JSONB
  chunks.importance_score    → BTREE (orden DESC)
  chunks.section_type        → BTREE
  chunks.document_id         → BTREE (FK)
  documents.doc_type_category, institution, document_year → BTREE

Uso:
  python3 03_database.py setup    # crea extension, schema, índices
  python3 03_database.py load     # inserta documents + chunks desde logs/
  python3 03_database.py reset    # DROP y recrea desde cero
  python3 03_database.py stats    # muestra métricas de la BD actual

Variables de entorno (opcional):
  PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE (default: rag_banco)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values, Json

DB_NAME = os.getenv("PGDATABASE", "rag_banco")

INPUT_DOCUMENTS = Path("logs/documents.json")
INPUT_CHUNKS = Path("logs/chunks_vectorized.json")

# ---------------------------------------------------------------------------
# Dimensión del embedding — dinámica según el modelo usado
# ---------------------------------------------------------------------------

def detect_embedding_dim() -> int:
    """Detecta la dimensión del embedding en orden de prioridad:
    1. Variable de entorno RAG_EMBEDDING_DIM
    2. Campo embedding_dim en el primer chunk de chunks_vectorized.json
    3. Longitud real del primer embedding en chunks_vectorized.json
    4. Fallback 384 (multilingual-e5-small, compatibilidad con main branch)
    """
    env_dim = os.getenv("RAG_EMBEDDING_DIM")
    if env_dim:
        return int(env_dim)
    if INPUT_CHUNKS.exists():
        with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if chunks:
            dim = chunks[0].get("embedding_dim") or len(chunks[0].get("embedding") or [])
            if dim:
                return int(dim)
    return 384


# ---------------------------------------------------------------------------
# SQL (schema)
# ---------------------------------------------------------------------------

def build_schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id         TEXT PRIMARY KEY,
    filename            TEXT NOT NULL,
    filepath            TEXT NOT NULL,
    doc_type_category   TEXT NOT NULL,
    institution         TEXT NOT NULL,
    document_date       TEXT,
    document_year       INT,
    total_pages         INT,
    total_chunks        INT,
    char_count          INT,
    extraction_warnings JSONB DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id           TEXT PRIMARY KEY,
    document_id        TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    text               TEXT NOT NULL,
    text_tsv           TSVECTOR,
    char_count         INT,
    page_start         INT,
    page_end           INT,
    position_in_doc    INT,
    section_type       TEXT NOT NULL DEFAULT 'CONTENIDO',
    section_confidence REAL DEFAULT 0,
    economic_variables JSONB DEFAULT '{{}}'::jsonb,
    numeric_values     JSONB DEFAULT '[]'::jsonb,
    entities           JSONB DEFAULT '{{}}'::jsonb,
    temporal_refs      JSONB DEFAULT '{{}}'::jsonb,
    tags               TEXT[] DEFAULT '{{}}',
    importance_score   REAL NOT NULL DEFAULT 0,
    is_policy_decision BOOLEAN DEFAULT FALSE,
    is_forward_looking BOOLEAN DEFAULT FALSE,
    chunk_date         DATE,
    embedding          vector({dim}) NOT NULL,
    embedding_model    TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

INDICES_SQL = [
    # documents
    "CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(doc_type_category)",
    "CREATE INDEX IF NOT EXISTS idx_docs_institution ON documents(institution)",
    "CREATE INDEX IF NOT EXISTS idx_docs_year ON documents(document_year)",
    # chunks — filtros
    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_type)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_importance ON chunks(importance_score DESC)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_policy ON chunks(is_policy_decision) "
    "WHERE is_policy_decision = TRUE",
    "CREATE INDEX IF NOT EXISTS idx_chunks_fwd ON chunks(is_forward_looking) "
    "WHERE is_forward_looking = TRUE",
    # chunks — GIN
    "CREATE INDEX IF NOT EXISTS idx_chunks_tags ON chunks USING GIN(tags)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_variables ON chunks USING GIN(economic_variables jsonb_path_ops)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_entities ON chunks USING GIN(entities jsonb_path_ops)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN(text_tsv)",
    # chunks — vector (HNSW con cosine distance). Los embeddings están
    # normalizados L2, así que cosine es numéricamente equivalente a inner
    # product; usamos vector_cosine_ops porque el operador <=> es el más
    # legible y convencional en queries.
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_date ON chunks(chunk_date) WHERE chunk_date IS NOT NULL",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn_params(database: Optional[str] = None) -> dict:
    return {
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", 5432)),
        "user": os.getenv("PGUSER", os.getenv("USER", "postgres")),
        "password": os.getenv("PGPASSWORD", "postgres"),
        "database": database or DB_NAME,
    }


def connect(database: Optional[str] = None):
    return psycopg2.connect(**get_conn_params(database))


def ensure_database_exists() -> None:
    """Crea la base de datos si no existe (conectándose a 'postgres')."""
    conn = connect(database="postgres")
    conn.autocommit = True  # must be set before any cursor use; CREATE DATABASE needs no transaction
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{DB_NAME}"')
                print(f"[03] ✓ base de datos '{DB_NAME}' creada")
            else:
                print(f"[03] ✓ base de datos '{DB_NAME}' ya existe")
    finally:
        conn.close()


def check_pgvector_available() -> bool:
    """Verifica que la extensión vector esté instalada en el servidor."""
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
            )
            return cur.fetchone() is not None
    except Exception as e:
        print(f"[03] ⚠ no pude verificar pgvector: {e}")
        return False


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_setup(force_drop: bool = False) -> int:
    dim = detect_embedding_dim()
    print(f"[03] SETUP — base de datos '{DB_NAME}', embedding dim={dim}")
    ensure_database_exists()

    if not check_pgvector_available():
        print("[03] ❌ la extensión 'vector' no está disponible en el servidor.\n"
              "     Instálala desde el source de pgvector:\n"
              "       cd pgvector && make && sudo make install\n"
              "     (Windows: abrir 'x64 Native Tools Command Prompt' y ejecutar "
              "nmake /F Makefile.win install)\n")
        return 2

    with connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            if force_drop:
                print("[03] DROP tablas (reset)")
                cur.execute("DROP TABLE IF EXISTS chunks CASCADE")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE")

            cur.execute(build_schema_sql(dim))
            # Migración: agrega chunk_date si la tabla ya existía sin ella
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_date DATE"
            )
            print("[03] ✓ schema (documents, chunks) creado")

            for sql in INDICES_SQL:
                cur.execute(sql)
            print(f"[03] ✓ {len(INDICES_SQL)} índices creados")
        conn.commit()

    return 0


def cmd_reset() -> int:
    return cmd_setup(force_drop=True)


def _format_vector(embedding: list[float]) -> str:
    """pgvector acepta el formato '[0.1,0.2,...]' como literal."""
    return "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"


def _document_year(doc: dict) -> Optional[int]:
    date = doc.get("document_date")
    if not date:
        return None
    try:
        return int(date[:4])
    except (ValueError, TypeError):
        return None


def _build_doc_rows(documents: list[dict]) -> list[tuple]:
    return [
        (
            doc["document_id"],
            doc["filename"],
            doc["filepath"],
            doc["doc_type_category"],
            doc["institution"],
            doc.get("document_date"),
            _document_year(doc),
            doc.get("total_pages"),
            doc.get("total_chunks"),
            doc.get("char_count"),
            Json(doc.get("extraction_warnings", [])),
        )
        for doc in documents
    ]


def _build_chunk_rows(chunks: list[dict], expected_dim: int) -> list[tuple]:
    rows = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding or len(embedding) != expected_dim:
            print(f"[03] ⚠ chunk {chunk.get('chunk_id')} embedding inválido "
                  f"(esperado dim={expected_dim}, got {len(embedding) if embedding else 0}) — skip")
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
            _format_vector(embedding),
            chunk.get("embedding_model"),
        ))
    return rows


def _upsert_documents(cur, doc_rows: list[tuple]) -> None:
    execute_values(
        cur,
        """
        INSERT INTO documents (
            document_id, filename, filepath, doc_type_category, institution,
            document_date, document_year, total_pages, total_chunks, char_count,
            extraction_warnings
        ) VALUES %s
        ON CONFLICT (document_id) DO UPDATE SET
            filename = EXCLUDED.filename,
            filepath = EXCLUDED.filepath,
            doc_type_category = EXCLUDED.doc_type_category,
            institution = EXCLUDED.institution,
            document_date = EXCLUDED.document_date,
            document_year = EXCLUDED.document_year,
            total_pages = EXCLUDED.total_pages,
            total_chunks = EXCLUDED.total_chunks,
            char_count = EXCLUDED.char_count,
            extraction_warnings = EXCLUDED.extraction_warnings
        """,
        doc_rows,
    )
    print(f"[03] ✓ upsert {len(doc_rows)} documents")


def _insert_chunks(cur, chunk_rows: list[tuple]) -> None:
    # text_tsv se rellena con UPDATE post-insert porque plainto_tsquery
    # no puede usarse directamente en el template de execute_values.
    execute_values(
        cur,
        """
        INSERT INTO chunks (
            chunk_id, document_id, text, char_count, page_start, page_end,
            position_in_doc, section_type, section_confidence,
            economic_variables, numeric_values, entities, temporal_refs,
            tags, importance_score, is_policy_decision, is_forward_looking,
            chunk_date, embedding, embedding_model
        ) VALUES %s
        """,
        chunk_rows,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                 "%s, %s, %s, %s, %s::vector, %s)",
    )
    cur.execute(
        "UPDATE chunks SET text_tsv = to_tsvector('simple', text) WHERE text_tsv IS NULL"
    )


def cmd_load() -> int:
    if not INPUT_DOCUMENTS.exists() or not INPUT_CHUNKS.exists():
        print("❌ Faltan archivos. Corre primero los pasos 0-2.", file=sys.stderr)
        return 1

    with open(INPUT_DOCUMENTS, "r", encoding="utf-8") as f:
        documents = json.load(f)
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    dim = detect_embedding_dim()
    print(f"[03] LOAD — {len(documents)} documents, {len(chunks)} chunks, embedding dim={dim}")

    doc_rows = _build_doc_rows(documents)
    chunk_rows = _build_chunk_rows(chunks, expected_dim=dim)

    with connect() as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            _upsert_documents(cur, doc_rows)

            # Delete existing chunks for re-loaded docs to ensure consistency.
            # ON DELETE CASCADE would only fire on document deletion, not re-load.
            doc_ids = tuple(doc["document_id"] for doc in documents)
            if doc_ids:
                cur.execute("DELETE FROM chunks WHERE document_id IN %s", (doc_ids,))
                print("[03] ✓ limpieza de chunks previos")

            _insert_chunks(cur, chunk_rows)
        conn.commit()

    print("[03] ✓ text_tsv poblado")
    print(f"[03] ✓ LOAD completo: {len(chunk_rows)} chunks insertados")
    return cmd_stats()


def cmd_stats() -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute(
            "SELECT doc_type_category, COUNT(*) FROM chunks c "
            "JOIN documents d USING(document_id) GROUP BY 1 ORDER BY 2 DESC"
        )
        by_type = cur.fetchall()
        cur.execute(
            "SELECT section_type, COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC"
        )
        by_section = cur.fetchall()
        cur.execute(
            "SELECT AVG(importance_score)::numeric(4,3), "
            "COUNT(*) FILTER (WHERE importance_score >= 0.6) FROM chunks"
        )
        avg_imp, high_imp = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM chunks WHERE is_policy_decision")
        n_policy = cur.fetchone()[0]

    print(f"\n[03] STATS para '{DB_NAME}':")
    print(f"       documents: {n_docs}")
    print(f"       chunks:    {n_chunks}")
    print(f"       importance: media={avg_imp}, con score≥0.6: {high_imp}")
    print(f"       policy_decision: {n_policy}")
    print(f"\n       Por tipo de documento:")
    for t, n in by_type:
        print(f"         {t:<20} {n}")
    print(f"\n       Por sección:")
    for s, n in by_section:
        print(f"         {s:<20} {n}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMANDS = {
    "setup": cmd_setup,
    "reset": cmd_reset,
    "load": cmd_load,
    "stats": cmd_stats,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"\nComandos válidos: {', '.join(COMMANDS)}")
        return 1
    return COMMANDS[sys.argv[1]]() or 0


if __name__ == "__main__":
    sys.exit(main())
