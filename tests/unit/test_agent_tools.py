"""Tests para las tools concretas del agente, con infra mockeada."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from banks_rag.application.agent.tools import dispatch
from banks_rag.domain.agent import AgentState
from banks_rag.domain.retrieval import SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# search_documents
# ─────────────────────────────────────────────────────────────────────────────


def _fake_search_result(n: int = 2) -> SearchResult:
    hits = [
        {
            "chunk_id": f"x{i}",
            "text": f"Texto del chunk {i}.",
            "filename": f"doc{i}.pdf",
            "doc_type_category": "COMUNICADO",
            "section_type": "DECISION",
            "page_start": 1,
            "page_end": 1,
            "chunk_date": None,
            "document_date": "2024-03-15",
            "importance_score": 0.85,
        }
        for i in range(n)
    ]
    return SearchResult(query="q", clean_query="q", hits=hits, parsed_filters={})


@pytest.mark.unit
class TestSearchDocumentsTool:
    @pytest.mark.asyncio
    async def test_returns_results_with_refs(self) -> None:
        state = AgentState()

        with (
            patch(
                "banks_rag.application.agent.tools.search_documents.hybrid_search",
                return_value=_fake_search_result(2),
            ),
            patch(
                "banks_rag.application.agent.tools.search_documents.build_default_embedder",
                return_value=type("E", (), {"name": "fake", "encode_text": lambda *a, **k: None})(),
            ),
            patch(
                "banks_rag.application.agent.tools.search_documents.PostgresRepo",
            ),
        ):
            result, _ = await dispatch(state, "search_documents", {"query": "TPM 2024"})

        assert result["n_results"] == 2
        assert result["results"][0]["ref"] == 1
        assert result["results"][1]["ref"] == 2
        assert result["filters_applied"]["doc_type"] is None
        assert len(state.chunks_seen) == 2

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        state = AgentState()
        empty = SearchResult(query="q", clean_query="q", hits=[], parsed_filters={})

        with (
            patch(
                "banks_rag.application.agent.tools.search_documents.hybrid_search",
                return_value=empty,
            ),
            patch(
                "banks_rag.application.agent.tools.search_documents.build_default_embedder",
                return_value=type("E", (), {"name": "fake"})(),
            ),
            patch("banks_rag.application.agent.tools.search_documents.PostgresRepo"),
        ):
            result, _ = await dispatch(state, "search_documents", {"query": "x"})

        assert result["n_results"] == 0
        assert "results" in result
        assert "message" in result

    @pytest.mark.asyncio
    async def test_truncates_long_text(self) -> None:
        state = AgentState()
        long_text = "x" * 1000
        sr = SearchResult(
            query="q", clean_query="q",
            hits=[{
                "chunk_id": "x", "text": long_text,
                "filename": "f.pdf", "doc_type_category": "X", "section_type": "Y",
                "page_start": 1, "page_end": 1, "chunk_date": None,
                "document_date": "2024", "importance_score": 0.5,
            }],
            parsed_filters={},
        )
        with (
            patch(
                "banks_rag.application.agent.tools.search_documents.hybrid_search",
                return_value=sr,
            ),
            patch(
                "banks_rag.application.agent.tools.search_documents.build_default_embedder",
                return_value=type("E", (), {"name": "fake"})(),
            ),
            patch("banks_rag.application.agent.tools.search_documents.PostgresRepo"),
        ):
            result, _ = await dispatch(state, "search_documents", {"query": "x"})

        assert len(result["results"][0]["text"]) <= 600
        assert result["results"][0]["text"].endswith("...")


@pytest.mark.unit
class TestSearchDocumentsBuildFilters:
    """build_filters convierte args a SearchFilters correctamente."""

    def test_doc_type_only(self) -> None:
        from banks_rag.application.agent.tools.search_documents import _build_filters
        f = _build_filters("COMUNICADO", None, None, None)
        assert f.doc_types == ["COMUNICADO"]
        assert f.year_from is None

    def test_year_only(self) -> None:
        from banks_rag.application.agent.tools.search_documents import _build_filters
        f = _build_filters(None, 2024, None, None)
        assert f.year_from == 2024 and f.year_to == 2024

    def test_exact_date_when_from_equals_to(self) -> None:
        from banks_rag.application.agent.tools.search_documents import _build_filters
        f = _build_filters(None, None, "2024-03-15", "2024-03-15")
        assert f.exact_date == "2024-03-15"
        assert f.year_from == 2024

    def test_date_range(self) -> None:
        from banks_rag.application.agent.tools.search_documents import _build_filters
        f = _build_filters(None, None, "2023-01-01", "2024-12-31")
        assert f.year_from == 2023 and f.year_to == 2024
        assert f.exact_date is None


# ─────────────────────────────────────────────────────────────────────────────
# document_lookup
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestListDocumentsTool:
    @pytest.mark.asyncio
    async def test_returns_documents(self) -> None:
        state = AgentState()
        fake_rows = [
            {
                "document_id": "doc1",
                "filename": "comunicado1.pdf",
                "doc_type_category": "COMUNICADO",
                "document_date": "2024-03-15",
                "document_year": 2024,
            },
            {
                "document_id": "doc2",
                "filename": "minuta1.pdf",
                "doc_type_category": "MINUTA",
                "document_date": "2024-04-01",
                "document_year": 2024,
            },
        ]

        class FakeRepo:
            def __init__(self, *args, **kwargs):
                pass

            def list_documents(self, **kwargs):
                return fake_rows

        with patch(
            "banks_rag.application.agent.tools.document_lookup.PostgresRepo",
            FakeRepo,
        ):
            result, _ = await dispatch(state, "list_documents", {"year": 2024})

        assert result["n"] == 2
        assert result["documents"][0]["filename"] == "comunicado1.pdf"
        assert result["documents"][1]["doc_type"] == "MINUTA"


@pytest.mark.unit
class TestGetDocumentChunksTool:
    @pytest.mark.asyncio
    async def test_returns_chunks_with_refs(self) -> None:
        state = AgentState()

        class FakeRepo:
            def __init__(self, *args, **kwargs):
                pass

            def get_document_by_filename(self, fn):
                return {
                    "document_id": "doc1",
                    "filename": fn,
                    "doc_type_category": "COMUNICADO",
                    "document_date": "2024-03-15",
                }

            def get_chunks_by_document_id(self, doc_id, **kwargs):
                return [
                    {
                        "chunk_id": f"{doc_id}_{i}",
                        "text": f"chunk {i}",
                        "page_start": i + 1, "page_end": i + 1,
                        "section_type": "X", "importance_score": 0.5,
                    }
                    for i in range(3)
                ]

        with patch(
            "banks_rag.application.agent.tools.document_lookup.PostgresRepo",
            FakeRepo,
        ):
            result, _ = await dispatch(
                state, "get_document_chunks", {"filename": "x.pdf"},
            )

        assert result["n_chunks"] == 3
        assert result["filename"] == "x.pdf"
        # Refs asignadas en orden
        assert [c["ref"] for c in result["chunks"]] == [1, 2, 3]
        assert len(state.chunks_seen) == 3

    @pytest.mark.asyncio
    async def test_returns_error_when_not_found(self) -> None:
        state = AgentState()

        class FakeRepo:
            def __init__(self, *args, **kwargs):
                pass

            def get_document_by_filename(self, fn):
                return None

        with patch(
            "banks_rag.application.agent.tools.document_lookup.PostgresRepo",
            FakeRepo,
        ):
            result, _ = await dispatch(
                state, "get_document_chunks", {"filename": "missing.pdf"},
            )

        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# historical_series — graceful fallback cuando dw_store no está
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSearchVisualsTool:
    @pytest.mark.asyncio
    async def test_filters_by_visual_kind(self) -> None:
        state = AgentState()
        captured_filters = {}

        def fake_search(query, *, query_embedder, repo, extra_filters, k, use_mmr):
            captured_filters["kinds"] = list(extra_filters.kinds)
            return SearchResult(
                query=query, clean_query=query,
                hits=[{
                    "chunk_id": "v1", "filename": "doc.pdf",
                    "doc_type_category": "COMUNICADO",
                    "page_start": 5, "page_end": 5,
                    "kind": "VISUAL",
                    "visual_caption": "Gráfico 3: TPM",
                    "chunk_date": None, "document_date": "2024-03-15",
                    "importance_score": 0.7,
                }],
                parsed_filters={},
            )

        with (
            patch(
                "banks_rag.application.agent.tools.search_visuals.hybrid_search",
                fake_search,
            ),
            patch(
                "banks_rag.application.agent.tools.search_visuals.build_default_embedder",
                return_value=type("E", (), {"name": "fake"})(),
            ),
            patch("banks_rag.application.agent.tools.search_visuals.PostgresRepo"),
        ):
            result, _ = await dispatch(
                state, "search_visuals", {"query": "curva swap"},
            )

        assert captured_filters["kinds"] == ["VISUAL"]
        assert result["n_results"] == 1
        assert result["results"][0]["caption"] == "Gráfico 3: TPM"
        assert result["results"][0]["image_url"] == "/v1/images/v1"
        assert result["filters_applied"]["kinds"] == ["VISUAL"]

    @pytest.mark.asyncio
    async def test_visual_kind_table(self) -> None:
        state = AgentState()
        captured = {}

        def fake_search(*a, **kw):
            captured["kinds"] = list(kw["extra_filters"].kinds)
            return SearchResult(query="x", clean_query="x", hits=[], parsed_filters={})

        with (
            patch(
                "banks_rag.application.agent.tools.search_visuals.hybrid_search",
                fake_search,
            ),
            patch(
                "banks_rag.application.agent.tools.search_visuals.build_default_embedder",
                return_value=type("E", (), {"name": "fake"})(),
            ),
            patch("banks_rag.application.agent.tools.search_visuals.PostgresRepo"),
        ):
            await dispatch(
                state, "search_visuals",
                {"query": "tabla", "visual_kind": "TABLE"},
            )

        assert captured["kinds"] == ["TABLE"]


@pytest.mark.unit
class TestHistoricalSeriesTool:
    @pytest.mark.asyncio
    async def test_list_returns_error_without_dw_store(self) -> None:
        state = AgentState()
        with patch(
            "banks_rag.application.agent.tools.historical_series._import_dw_store",
            return_value=None,
        ):
            result, _ = await dispatch(state, "list_historical_series", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_returns_error_without_get_data_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = AgentState()
        monkeypatch.delenv("GET_DATA_PATH", raising=False)

        # dw_store disponible pero sin GET_DATA_PATH → error temprano
        class FakeDw:
            @staticmethod
            def get_series_meta(series_id, catalog_path=None):
                return {"name": "Mock", "unit": "%"}

        with patch(
            "banks_rag.application.agent.tools.historical_series._import_dw_store",
            return_value=FakeDw,
        ):
            result, _ = await dispatch(
                state, "get_historical_series", {"series_id": "tpm"},
            )
        assert "error" in result
        assert "GET_DATA_PATH" in result["error"]
