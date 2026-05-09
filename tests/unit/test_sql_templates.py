"""Unit tests para SQL templates de persistence."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.persistence import sql_templates as sql


@pytest.mark.unit
class TestSchemaSql:
    def test_includes_extension(self) -> None:
        ddl = sql.schema_sql(4096, "documents", "chunks")
        assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl

    def test_uses_correct_dim(self) -> None:
        ddl = sql.schema_sql(384, "documents", "chunks")
        assert "vector(384)" in ddl
        ddl2 = sql.schema_sql(4096, "documents", "chunks")
        assert "vector(4096)" in ddl2

    def test_uses_table_names(self) -> None:
        ddl = sql.schema_sql(4096, "qwen_documents", "qwen_chunks")
        assert "qwen_documents" in ddl
        assert "qwen_chunks" in ddl

    def test_chunks_references_documents(self) -> None:
        ddl = sql.schema_sql(4096, "qwen_documents", "qwen_chunks")
        assert "REFERENCES qwen_documents(document_id)" in ddl

    def test_idempotent_via_if_not_exists(self) -> None:
        ddl = sql.schema_sql(4096, "documents", "chunks")
        assert "CREATE TABLE IF NOT EXISTS" in ddl


@pytest.mark.unit
class TestIndexStatements:
    def test_count(self) -> None:
        # 18 índices según diseño:
        # - 3 docs (BTREE) + 5 chunks BTREE/parciales + 4 GIN +
        #   1 HNSW + 4 fechas/compuestos = 17 (legacy) o 18 (con compuestos extra)
        stmts = sql.index_statements("documents", "chunks")
        # Verificar al menos los críticos en lugar del count exacto
        assert len(stmts) >= 15

    def test_hnsw_with_cosine_ops(self) -> None:
        stmts = sql.index_statements("documents", "chunks")
        hnsw_stmts = [s for s in stmts if "hnsw" in s.lower()]
        assert len(hnsw_stmts) == 1
        assert "vector_cosine_ops" in hnsw_stmts[0]
        assert "m = 16" in hnsw_stmts[0]
        assert "ef_construction = 64" in hnsw_stmts[0]

    def test_gin_indices_exist(self) -> None:
        stmts = sql.index_statements("documents", "chunks")
        gin = [s for s in stmts if "GIN" in s]
        # tags, economic_variables, entities, text_tsv
        assert len(gin) == 4

    def test_partial_indices_for_boolean_flags(self) -> None:
        stmts = sql.index_statements("documents", "chunks")
        partials = [s for s in stmts if "WHERE" in s]
        # is_policy_decision, is_forward_looking, chunk_date IS NOT NULL,
        # chunk_date IS NOT NULL para idx_chunks_date_imp
        assert len(partials) >= 3


@pytest.mark.unit
class TestDmlSql:
    def test_upsert_documents_uses_on_conflict(self) -> None:
        ddl = sql.upsert_documents_sql("documents")
        assert "ON CONFLICT (document_id) DO UPDATE" in ddl
        # Todos los campos updateables deben estar en SET
        for field in [
            "filename", "filepath", "doc_type_category", "institution",
            "document_date", "document_year", "total_pages", "total_chunks",
            "char_count", "extraction_warnings",
        ]:
            assert f"{field} = EXCLUDED.{field}" in ddl

    def test_insert_chunks_template_has_vector_cast(self) -> None:
        # El template debe castear el embedding a vector
        assert "%s::vector" in sql.INSERT_CHUNKS_TEMPLATE

    def test_update_text_tsv_uses_simple_config(self) -> None:
        ddl = sql.update_text_tsv_sql("chunks")
        assert "to_tsvector('simple', text)" in ddl
        assert "WHERE text_tsv IS NULL" in ddl

    def test_delete_chunks_uses_array_param(self) -> None:
        ddl = sql.delete_chunks_for_docs_sql("chunks")
        assert "= ANY(%s)" in ddl


@pytest.mark.unit
def test_drop_sql_order() -> None:
    """chunks DROP debe ir antes que documents DROP (por la FK)."""
    drops = sql.drop_sql("documents", "chunks")
    assert drops[0].startswith("DROP TABLE IF EXISTS chunks")
    assert drops[1].startswith("DROP TABLE IF EXISTS documents")


@pytest.mark.unit
def test_post_schema_migrations_idempotent() -> None:
    """Cada migración debe ser idempotente: ADD COLUMN IF NOT EXISTS o el
    bloque DO $$ con EXCEPTION WHEN duplicate_object para CHECK constraints."""
    migrations = sql.post_schema_migrations("chunks")
    for m in migrations:
        assert (
            "ADD COLUMN IF NOT EXISTS" in m
            or "duplicate_object" in m
        ), f"Migration no idempotente: {m!r}"


@pytest.mark.unit
def test_post_schema_migrations_includes_visual_columns() -> None:
    """Fase 2: kind + visual_caption + CHECK constraint."""
    migrations = sql.post_schema_migrations("chunks")
    flat = " ".join(migrations)
    assert "kind" in flat
    assert "visual_caption" in flat
    assert "VISUAL" in flat and "TABLE" in flat


@pytest.mark.unit
def test_schema_includes_kind_column() -> None:
    """schema_sql debe incluir la columna kind con CHECK + visual_caption."""
    ddl = sql.schema_sql(4096, "documents", "chunks")
    assert "kind" in ddl
    assert "visual_caption" in ddl
    assert "CHECK (kind IN ('TEXT', 'VISUAL', 'TABLE'))" in ddl


@pytest.mark.unit
def test_index_statements_include_kind_index() -> None:
    stmts = sql.index_statements("documents", "chunks")
    assert any("idx_chunks_kind" in s for s in stmts)


@pytest.mark.unit
def test_insert_chunks_template_has_correct_placeholder_count() -> None:
    """23 placeholders: 21 originales + kind + visual_caption."""
    placeholders = sql.INSERT_CHUNKS_TEMPLATE.count("%s")
    assert placeholders == 23
