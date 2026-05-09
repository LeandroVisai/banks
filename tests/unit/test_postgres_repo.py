"""Unit tests para PostgresRepo helpers (sin DB)."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.persistence.postgres_repo import (
    PostgresRepo,
    document_year_from_date,
    format_vector,
    safe_ident,
)


@pytest.mark.unit
class TestSafeIdent:
    @pytest.mark.parametrize("name", ["documents", "qwen_chunks", "_private", "tbl1"])
    def test_valid_identifiers(self, name: str) -> None:
        assert safe_ident(name) == name

    @pytest.mark.parametrize(
        "bad", ["1tbl", "tbl-name", "tbl name", "tbl;DROP", "", "tbl/x"]
    )
    def test_rejects_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError, match="inválido"):
            safe_ident(bad)


@pytest.mark.unit
class TestFormatVector:
    def test_basic_format(self) -> None:
        result = format_vector([0.1, 0.2, 0.3])
        assert result == "[0.1000000,0.2000000,0.3000000]"

    def test_seven_decimals(self) -> None:
        result = format_vector([0.123456789])
        # Trunca a 7 decimales
        assert "0.1234568" in result

    def test_negative_values(self) -> None:
        result = format_vector([-0.5, 0.5])
        assert result == "[-0.5000000,0.5000000]"

    def test_empty_vector(self) -> None:
        assert format_vector([]) == "[]"


@pytest.mark.unit
class TestDocumentYearFromDate:
    @pytest.mark.parametrize(
        "date,expected",
        [
            ("2024-03-15", 2024),
            ("1999-12-31", 1999),
            ("2024", 2024),  # solo año
            (None, None),
            ("", None),
            ("invalid", None),
        ],
    )
    def test_extraction(self, date, expected) -> None:
        assert document_year_from_date(date) == expected


@pytest.mark.unit
class TestPostgresRepoInit:
    def test_default_prefix_empty(self) -> None:
        repo = PostgresRepo()
        assert repo.docs_table == "documents"
        assert repo.chunks_table == "chunks"

    def test_qwen_prefix(self) -> None:
        repo = PostgresRepo(prefix="qwen_")
        assert repo.docs_table == "qwen_documents"
        assert repo.chunks_table == "qwen_chunks"

    def test_invalid_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="inválido"):
            PostgresRepo(prefix="qwen-")

    def test_database_override(self) -> None:
        repo = PostgresRepo(database="rag_test")
        assert repo.database == "rag_test"


@pytest.mark.unit
class TestChunksToRowsValidation:
    def test_skips_chunks_with_wrong_dim(self) -> None:
        chunks = [
            {
                "chunk_id": "ok",
                "document_id": "doc1",
                "text": "hola",
                "embedding": [0.1] * 4,
            },
            {
                "chunk_id": "wrong",
                "document_id": "doc1",
                "text": "wrong dim",
                "embedding": [0.1] * 8,  # diferente dim
            },
            {
                "chunk_id": "missing",
                "document_id": "doc1",
                "text": "no embedding",
            },
        ]
        rows = PostgresRepo._chunks_to_rows(chunks, expected_dim=4)
        assert len(rows) == 1
        assert rows[0][0] == "ok"

    def test_uses_default_section_type(self) -> None:
        chunks = [{
            "chunk_id": "x",
            "document_id": "doc1",
            "text": "hola",
            "embedding": [0.1] * 4,
        }]
        rows = PostgresRepo._chunks_to_rows(chunks, expected_dim=4)
        # section_type es el campo 7 (0-indexed)
        assert rows[0][7] == "CONTENIDO"
