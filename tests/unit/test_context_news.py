"""Tests del corpus de CONTEXTO ACTUAL (noticias): extract, enrich, router, tool.

Todo con datos sintéticos y mocks — sin BD, sin modelos, sin red.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from banks_rag.application.agent.router import select_specialists
from banks_rag.application.agent.tools import dispatch
from banks_rag.application.ingestion.enrich_news import (
    detect_sentiment,
    enrich_news,
    enrich_news_chunk,
)
from banks_rag.application.ingestion.extract_news import (
    chunk_article_text,
    extract_article,
    extract_news,
    fix_mojibake,
    news_date_from_filename,
    parse_source_institution,
)
from banks_rag.domain.agent import AgentState
from banks_rag.domain.retrieval import SearchResult

# ─────────────────────────────────────────────────────────────────────────────
# extract_news
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestExtractNews:
    def test_date_from_filename(self) -> None:
        assert news_date_from_filename("noticias_2026_01_20.json") == "2026-01-20"
        assert news_date_from_filename("noticias-2026-01-20.json") == "2026-01-20"
        assert news_date_from_filename("2026_05_01_news.json") == "2026-05-01"

    def test_date_from_filename_invalid(self) -> None:
        assert news_date_from_filename("sin_fecha.json") is None
        assert news_date_from_filename("noticias_2026_13_40.json") is None  # mes/día fuera de rango

    def test_parse_source_institution(self) -> None:
        src = "LA TERCERA (PULSO) - CHILE - ACTUALIDAD - 29/03/2026 0:00:00"
        assert parse_source_institution(src) == "La Tercera (Pulso)"
        assert parse_source_institution("DF - ECONOMIA") == "DF"  # sigla real preservada
        assert parse_source_institution("") == "Prensa"

    def test_chunk_short_text_not_split(self) -> None:
        assert chunk_article_text("Una nota breve.") == ["Una nota breve."]
        assert chunk_article_text("") == []

    def test_chunk_long_text_splits(self) -> None:
        para = "Párrafo con bastante contenido económico. " * 30  # ~1290 chars
        text = "\n\n".join([para, para, para])  # ~3900 chars > MIN_SPLIT
        chunks = chunk_article_text(text)
        assert len(chunks) >= 2
        assert all(c.strip() for c in chunks)

    def test_extract_article_builds_doc_and_chunks(self) -> None:
        art = {
            "title": "Dólar sube",
            "source": "LA TERCERA - CHILE",
            "topic": "Economía",
            "text": "El dólar subió fuerte por la incertidumbre global.",
        }
        out = extract_article(
            art, file_date="2026-01-20", filename="noticias_2026_01_20.json",
            filepath="noticias_2026_01_20.json", position=0,
        )
        assert out is not None
        doc, chunks = out
        assert doc.doc_type_category == "NOTICIA"
        assert doc.institution == "La Tercera"
        assert doc.document_date == "2026-01-20"
        assert doc.document_id.startswith("news_2026-01-20_")
        assert chunks and chunks[0].chunk_date == "2026-01-20"
        assert chunks[0].section_title_raw == "Economía"
        # El titular entra como lead del cuerpo (para el embedding).
        assert "Dólar sube" in chunks[0].text

    def test_extract_article_empty_returns_none(self) -> None:
        assert extract_article(
            {"source": "X"}, file_date="2026-01-20", filename="f.json",
            filepath="f.json", position=0,
        ) is None

    def test_article_id_is_stable(self) -> None:
        """Mismo artículo → mismo document_id (ingesta idempotente)."""
        art = {"title": "T", "source": "S", "text": "cuerpo"}
        kw = dict(file_date="2026-01-20", filename="f.json", filepath="f.json", position=0)
        d1, _ = extract_article(art, **kw)
        d2, _ = extract_article(art, **kw)
        assert d1.document_id == d2.document_id

    def test_extract_news_directory(self, tmp_path) -> None:
        articles = [
            {"title": "Inflación al alza", "source": "El Mercurio - CHILE",
             "topic": "Macro", "text": "El IPC sorprendió al alza este mes."},
            {"title": "Dólar a la baja", "source": "Pulso - CHILE",
             "topic": "Mercados", "text": "El tipo de cambio cayó tras el dato."},
        ]
        f = tmp_path / "noticias_2026_05_01.json"
        f.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
        result = extract_news(tmp_path)
        assert result.report.files_processed == 1
        assert result.report.documents_count == 2
        assert all(d.document_date == "2026-05-01" for d in result.documents)

    def test_extract_news_skips_undated_file(self, tmp_path) -> None:
        (tmp_path / "sin_fecha.json").write_text(json.dumps([{"title": "x", "text": "y"}]))
        result = extract_news(tmp_path)
        assert result.report.documents_count == 0
        assert "sin_fecha.json" in result.report.files_skipped

    @staticmethod
    def _mangle(s: str) -> str:
        """Simula el mojibake del scraper: UTF-8 guardado como si fuera cp1252."""
        return s.encode("utf-8").decode("cp1252")

    def test_fix_mojibake(self) -> None:
        for clean in ("inflación", "país", "¿Qué?", "CHAÑARCILLO"):
            assert fix_mojibake(self._mangle(clean)) == clean
        # Texto ya limpio no se toca.
        assert fix_mojibake("La inflación subió") == "La inflación subió"
        assert fix_mojibake("") == ""

    def test_extract_article_repairs_mojibake(self) -> None:
        art = {
            "emailTitle": self._mangle("Sube la inflación"),
            "source": "DIARIO FINANCIERO - CHILE - ECONOMIA",
            "topic": "Banco Central",
            "text": self._mangle("El IPC sorprendió al alza en el país."),
        }
        _, chunks = extract_article(
            art, file_date="2026-06-04", filename="f.json", filepath="f.json", position=0,
        )
        assert "inflación" in chunks[0].text
        assert "país" in chunks[0].text

    def test_internal_date_overrides_file_date(self) -> None:
        """Un informe diario incluye notas de días previos: usa la fecha propia."""
        art = {"title": "T", "source": "S", "text": "cuerpo", "date": "02/06/2026"}
        doc, chunks = extract_article(
            art, file_date="2026-06-04", filename="f.json", filepath="f.json", position=0,
        )
        assert doc.document_date == "2026-06-02"
        assert chunks[0].chunk_date == "2026-06-02"

    def test_file_date_when_internal_null(self) -> None:
        art = {"title": "T", "source": "S", "text": "cuerpo", "date": None}
        doc, _ = extract_article(
            art, file_date="2026-06-04", filename="f.json", filepath="f.json", position=0,
        )
        assert doc.document_date == "2026-06-04"


# ─────────────────────────────────────────────────────────────────────────────
# enrich_news
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestEnrichNews:
    def _doc_chunk(self, text: str):
        art = {"title": "T", "source": "La Tercera - CHILE", "topic": "Macro", "text": text}
        doc, chunks = extract_article(
            art, file_date="2026-01-20", filename="f.json", filepath="f.json", position=0,
        )
        return doc, chunks[0]

    def test_news_taxonomy_distinct_from_bank(self) -> None:
        doc, chunk = self._doc_chunk("La TPM se mantuvo en 4,5% pese a la incertidumbre.")
        ec = enrich_news_chunk(chunk, doc)
        assert ec.doc_type_category == "NOTICIA"
        assert ec.section_type == "NOTICIA"
        assert ec.is_policy_decision is False  # una noticia NUNCA es la decisión
        assert ec.schema_version == "news-1.0"
        assert "NOTICIA" in ec.tags
        # Reusa el detector de variables del banco (útil para cruzar coyuntura).
        assert ec.economic_variables  # detecta TASA_INTERES / TIPO_CAMBIO

    def test_sentiment(self) -> None:
        assert detect_sentiment("crecimiento alza recuperacion repunte") == "POSITIVO"
        assert detect_sentiment("crisis caida desplome riesgo") == "NEGATIVO"
        assert detect_sentiment("la reunion fue el martes") == "NEUTRO"

    def test_sentiment_tag_present(self) -> None:
        doc, chunk = self._doc_chunk("Hubo una fuerte caída y crisis en los mercados.")
        ec = enrich_news_chunk(chunk, doc)
        assert "SENTIMIENTO_NEGATIVO" in ec.tags

    def test_importance_rises_with_econ_signal(self) -> None:
        doc_a, ca = self._doc_chunk("Hoy es un día soleado en la capital.")
        doc_b, cb = self._doc_chunk(
            "La TPM bajó a 4,5%, el IPC subió 0,3% y el dólar cerró en alza."
        )
        low = enrich_news_chunk(ca, doc_a).importance_score
        high = enrich_news_chunk(cb, doc_b).importance_score
        assert high > low

    def test_enrich_news_report(self) -> None:
        doc, chunk = self._doc_chunk("El dólar subió por la incertidumbre.")
        res = enrich_news([doc], [chunk])
        assert res.report is not None
        assert res.report.chunks_total == 1
        assert res.report.institution_histogram.get("La Tercera") == 1


# ─────────────────────────────────────────────────────────────────────────────
# router → especialista coyuntura
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRouterCoyuntura:
    def test_news_question_routes_to_coyuntura(self) -> None:
        keys = [s.key for s in select_specialists("qué noticias hay sobre inflación")]
        assert "coyuntura" in keys

    def test_why_question_adds_coyuntura(self) -> None:
        keys = [s.key for s in select_specialists("por qué sube el dólar?")]
        assert "coyuntura" in keys and "fx" in keys

    def test_plain_data_question_no_coyuntura(self) -> None:
        keys = [s.key for s in select_specialists("cuál es la TPM vigente")]
        assert "coyuntura" not in keys

    def test_greeting_no_specialists(self) -> None:
        assert select_specialists("hola") == []


# ─────────────────────────────────────────────────────────────────────────────
# tool search_current_context
# ─────────────────────────────────────────────────────────────────────────────


def _news_hits(items: list[dict]) -> SearchResult:
    hits = []
    for i, it in enumerate(items):
        hits.append({
            "chunk_id": it.get("chunk_id", f"n{i}"),
            "text": it.get("text", f"Noticia {i}."),
            "filename": it.get("title", f"Titular {i}"),
            "institution": it.get("source", "La Tercera"),
            "doc_type_category": "NOTICIA",
            "section_type": it.get("topic", "Macro"),
            "chunk_date": it.get("date"),
            "document_date": it.get("date"),
            "tags": it.get("tags", ["NOTICIA", "SENTIMIENTO_NEGATIVO"]),
            "importance_score": 0.5,
        })
    return SearchResult(query="q", clean_query="q", hits=hits, parsed_filters={})


def _patch_tool(search_result: SearchResult):
    base = "banks_rag.application.agent.tools.search_current_context"
    return (
        patch(f"{base}.hybrid_search", return_value=search_result),
        patch(f"{base}.build_default_embedder",
              return_value=type("E", (), {"name": "fake", "encode_text": lambda *a, **k: None})()),
        patch(f"{base}.build_default_reranker", return_value=None),
        patch(f"{base}.PostgresRepo"),
    )


@pytest.mark.unit
class TestSearchCurrentContextTool:
    @pytest.mark.asyncio
    async def test_returns_results_with_source_and_note(self) -> None:
        state = AgentState()
        sr = _news_hits([
            {"date": "2026-06-01", "source": "El Mercurio", "title": "Dólar al alza"},
            {"date": "2026-05-30", "source": "Pulso", "title": "IPC sorprende"},
        ])
        p1, p2, p3, p4 = _patch_tool(sr)
        with p1, p2, p3, p4:
            result, _ = await dispatch(state, "search_current_context", {"query": "dólar"})
        assert result["n_results"] == 2
        assert result["source_kind"] == "prensa_externa"
        assert "note" in result  # sanitización: marca prensa externa
        assert result["results"][0]["ref"] == 1
        assert result["results"][0]["source"] in {"El Mercurio", "Pulso"}
        assert result["results"][0]["sentiment"] == "negativo"
        assert len(state.chunks_seen) == 2

    @pytest.mark.asyncio
    async def test_recency_lifts_fresh_over_stale(self) -> None:
        """Una noticia fresca de menor relevancia sube sobre una stale de mayor."""
        state = AgentState()
        sr = _news_hits([
            {"chunk_id": "top", "date": "2026-06-03"},    # rank0, fresca → se queda arriba
            {"chunk_id": "stale", "date": "2026-01-01"},  # rank1, vieja
            {"chunk_id": "fresh", "date": "2026-06-03"},  # rank2, fresca → supera a 'stale'
        ])
        p1, p2, p3, p4 = _patch_tool(sr)
        with patch("banks_rag.application.agent.tools.search_current_context.date") as mdate:
            import datetime as _dt
            mdate.today.return_value = _dt.date(2026, 6, 3)
            mdate.fromisoformat = _dt.date.fromisoformat
            with p1, p2, p3, p4:
                await dispatch(state, "search_current_context", {"query": "x", "k": 3})
        ids = [c["chunk_id"] for c in state.chunks_seen]
        # 'fresh' (rank2 fresca) debe quedar por delante de 'stale' (rank1 vieja).
        assert ids.index("fresh") < ids.index("stale")

    @pytest.mark.asyncio
    async def test_date_filter(self) -> None:
        state = AgentState()
        sr = _news_hits([
            {"chunk_id": "old", "date": "2026-01-01"},
            {"chunk_id": "new", "date": "2026-06-01"},
        ])
        p1, p2, p3, p4 = _patch_tool(sr)
        with p1, p2, p3, p4:
            result, _ = await dispatch(
                state, "search_current_context",
                {"query": "x", "date_from": "2026-05-01"},
            )
        assert result["n_results"] == 1
        assert state.chunks_seen[0]["chunk_id"] == "new"

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        state = AgentState()
        p1, p2, p3, p4 = _patch_tool(_news_hits([]))
        with p1, p2, p3, p4:
            result, _ = await dispatch(state, "search_current_context", {"query": "x"})
        assert result["n_results"] == 0
        assert "message" in result
