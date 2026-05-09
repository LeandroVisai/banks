"""Templates SQL para schema, índices y queries del corpus en PostgreSQL.

Las plantillas se parametrizan en Python (no en archivos .sql) porque dependen
de:
  - ``dim``: dimensión del embedding (varía según modelo: 384 para E5-small,
    4096 para Qwen3-Embedding-8B).
  - ``docs_table``, ``chunks_table``: nombres con prefijo según
    ``RAG_TABLE_PREFIX`` (e.g. ``qwen_``, ``gemma_``, vacío).

Los nombres de tabla se validan con regex al ser construidos
(ver ``postgres_repo.safe_ident``) — nunca se interpolan strings sin validar.
"""

from __future__ import annotations

# ── DDL: schema base ──────────────────────────────────────────────────────────


def schema_sql(dim: int, docs_table: str, chunks_table: str) -> str:
    """DDL para crear la extensión vector + tablas documents y chunks.

    Idempotente vía ``IF NOT EXISTS``. Para forzar recreación, hacer DROP
    explícito antes (ver ``DROP_SQL``).
    """
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {docs_table} (
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

CREATE TABLE IF NOT EXISTS {chunks_table} (
    chunk_id           TEXT PRIMARY KEY,
    document_id        TEXT NOT NULL REFERENCES {docs_table}(document_id) ON DELETE CASCADE,
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
    image_path         TEXT,
    kind               TEXT NOT NULL DEFAULT 'TEXT'
                       CHECK (kind IN ('TEXT', 'VISUAL', 'TABLE')),
    visual_caption     TEXT,
    embedding          vector({dim}) NOT NULL,
    embedding_model    TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
"""


def drop_sql(docs_table: str, chunks_table: str) -> tuple[str, str]:
    """Pareja de DROPs en orden: chunks primero (FK), luego documents."""
    return (
        f"DROP TABLE IF EXISTS {chunks_table} CASCADE",
        f"DROP TABLE IF EXISTS {docs_table} CASCADE",
    )


def post_schema_migrations(chunks_table: str) -> list[str]:
    """ALTER TABLE para columnas añadidas a tablas existentes (idempotente).

    Para tablas creadas antes de Fase 2 que aún no tienen ``kind`` y
    ``visual_caption``: las añadimos con default ``'TEXT'`` (todos los chunks
    legacy son texto) y un CHECK constraint. La constraint se crea solo si no
    existe (vía ``DO $$`` block).
    """
    return [
        f"ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS chunk_date DATE",
        f"ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS image_path TEXT",
        f"ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'TEXT'",
        f"ALTER TABLE {chunks_table} ADD COLUMN IF NOT EXISTS visual_caption TEXT",
        # CHECK constraint idempotente. Si ya existe se ignora.
        f"""DO $$ BEGIN
            ALTER TABLE {chunks_table} ADD CONSTRAINT {chunks_table}_kind_check
                CHECK (kind IN ('TEXT', 'VISUAL', 'TABLE'));
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$""",
    ]


def index_statements(docs_table: str, chunks_table: str) -> list[str]:
    """18 índices: BTREE para filtros, GIN para JSONB/array/tsvector, HNSW para vector.

    HNSW usa ``vector_cosine_ops`` con ``m=16, ef_construction=64`` — ajustado
    para corpus de ~10K-100K chunks. Para corpus mayores subir ``ef_construction``
    a 128 mejora recall a costa de tiempo de build.
    """
    return [
        # documents
        f"CREATE INDEX IF NOT EXISTS idx_docs_type ON {docs_table}(doc_type_category)",
        f"CREATE INDEX IF NOT EXISTS idx_docs_institution ON {docs_table}(institution)",
        f"CREATE INDEX IF NOT EXISTS idx_docs_year ON {docs_table}(document_year)",
        # chunks — filtros
        f"CREATE INDEX IF NOT EXISTS idx_chunks_document ON {chunks_table}(document_id)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_section ON {chunks_table}(section_type)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_importance ON {chunks_table}(importance_score DESC)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_policy ON {chunks_table}(is_policy_decision) "
        f"WHERE is_policy_decision = TRUE",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_fwd ON {chunks_table}(is_forward_looking) "
        f"WHERE is_forward_looking = TRUE",
        # chunks — GIN
        f"CREATE INDEX IF NOT EXISTS idx_chunks_tags ON {chunks_table} USING GIN(tags)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_variables ON {chunks_table} "
        f"USING GIN(economic_variables jsonb_path_ops)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_entities ON {chunks_table} "
        f"USING GIN(entities jsonb_path_ops)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON {chunks_table} USING GIN(text_tsv)",
        # chunks — HNSW (cosine; embeddings deben venir L2-normalizados de Fase 1C)
        f"CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON {chunks_table} "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        # chunks — fechas y compuestos para filtros frecuentes
        f"CREATE INDEX IF NOT EXISTS idx_chunks_chunk_date ON {chunks_table}(chunk_date) "
        f"WHERE chunk_date IS NOT NULL",
        # chunks — kind para filtrar TEXT vs VISUAL en search_visuals.
        f"CREATE INDEX IF NOT EXISTS idx_chunks_kind ON {chunks_table}(kind)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_doc_section ON {chunks_table}(document_id, section_type)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_year_section ON {chunks_table}(document_id, section_type) "
        f"INCLUDE (importance_score)",
        f"CREATE INDEX IF NOT EXISTS idx_chunks_date_imp ON {chunks_table}(chunk_date, importance_score DESC) "
        f"WHERE chunk_date IS NOT NULL",
    ]


# ── DML: upsert / insert / update tsv ─────────────────────────────────────────


def upsert_documents_sql(docs_table: str) -> str:
    return f"""
INSERT INTO {docs_table} (
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
"""


def insert_chunks_sql(chunks_table: str) -> str:
    return f"""
INSERT INTO {chunks_table} (
    chunk_id, document_id, text, char_count, page_start, page_end,
    position_in_doc, section_type, section_confidence,
    economic_variables, numeric_values, entities, temporal_refs,
    tags, importance_score, is_policy_decision, is_forward_looking,
    chunk_date, image_path, kind, visual_caption, embedding, embedding_model
) VALUES %s
"""


# Template para execute_values: el embedding va como string casteado a vector,
# el resto como parámetros normales. 23 placeholders.
INSERT_CHUNKS_TEMPLATE = (
    "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)"
)


def update_text_tsv_sql(chunks_table: str) -> str:
    """Popula text_tsv con to_tsvector('simple', text) para chunks que aún no lo tienen.

    Config 'simple' es agnóstica de idioma — buena para corpus multilingüe
    (ES + EN + jerga financiera). Si en el futuro se quiere stemming en español,
    usar 'spanish' (PostgreSQL incluye el dictionary).
    """
    return (
        f"UPDATE {chunks_table} SET text_tsv = to_tsvector('simple', text) "
        f"WHERE text_tsv IS NULL"
    )


def delete_chunks_for_docs_sql(chunks_table: str) -> str:
    """Borra chunks por document_id para re-load idempotente."""
    return f"DELETE FROM {chunks_table} WHERE document_id = ANY(%s)"


# ── Queries de inspección y existencia ───────────────────────────────────────


def table_exists_sql() -> str:
    return "SELECT to_regclass(%s)"


EMBEDDING_COLUMN_TYPE_SQL = (
    "SELECT format_type(a.atttypid, a.atttypmod) "
    "FROM pg_attribute a JOIN pg_class c ON a.attrelid = c.oid "
    "WHERE c.relname = %s AND a.attname = 'embedding'"
)

DATABASE_EXISTS_SQL = "SELECT 1 FROM pg_database WHERE datname = %s"
PGVECTOR_AVAILABLE_SQL = "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"


def list_documents_sql(docs_table: str) -> str:
    """Lista documentos con filtros opcionales por tipo y/o año.

    El caller pasa los filtros como params; la cláusula WHERE se construye
    con ``COALESCE`` para que cada filtro sea opcional sin tocar SQL.
    """
    return f"""
SELECT document_id, filename, doc_type_category, document_date, document_year
FROM {docs_table}
WHERE (%s::text IS NULL OR doc_type_category = %s)
  AND (%s::int IS NULL OR document_year = %s)
ORDER BY COALESCE(document_year, 0) DESC, document_date DESC NULLS LAST, filename
LIMIT %s
"""


def get_document_by_filename_sql(docs_table: str) -> str:
    return f"""
SELECT document_id, filename, doc_type_category, institution,
       document_date, document_year, total_pages, total_chunks
FROM {docs_table}
WHERE filename = %s
LIMIT 1
"""


def get_chunk_image_sql(chunks_table: str) -> str:
    """Resuelve image_path + kind para un chunk_id (endpoint /v1/images)."""
    return (
        f"SELECT chunk_id, image_path, kind, visual_caption "
        f"FROM {chunks_table} WHERE chunk_id = %s"
    )


def get_chunks_by_document_id_sql(chunks_table: str) -> str:
    """Trae chunks de un documento ordenados por página."""
    return f"""
SELECT chunk_id, document_id, text, char_count, page_start, page_end,
       position_in_doc, section_type, section_confidence,
       economic_variables, numeric_values, entities, tags,
       importance_score, is_policy_decision, is_forward_looking,
       chunk_date, image_path
FROM {chunks_table}
WHERE document_id = %s
ORDER BY position_in_doc
LIMIT %s
"""


def stats_queries(docs_table: str, chunks_table: str) -> dict[str, str]:
    """Set de queries para reporting; el caller los ejecuta y agrega."""
    return {
        "n_docs": f"SELECT COUNT(*) FROM {docs_table}",
        "n_chunks": f"SELECT COUNT(*) FROM {chunks_table}",
        "by_doc_type": (
            f"SELECT doc_type_category, COUNT(*) FROM {chunks_table} c "
            f"JOIN {docs_table} d USING(document_id) GROUP BY 1 ORDER BY 2 DESC"
        ),
        "by_section": (
            f"SELECT section_type, COUNT(*) FROM {chunks_table} GROUP BY 1 ORDER BY 2 DESC"
        ),
        "importance": (
            f"SELECT AVG(importance_score)::numeric(4,3), "
            f"COUNT(*) FILTER (WHERE importance_score >= 0.6) FROM {chunks_table}"
        ),
        "n_policy": f"SELECT COUNT(*) FROM {chunks_table} WHERE is_policy_decision",
    }
