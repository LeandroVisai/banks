"""Tests del modo análisis de documento adjunto (map-reduce).

``build_batches`` y ``select_visuals`` son puras (contador de tokens fake).
``run_document_analysis`` y la integración con ``run_agent`` usan un LLM mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banks_rag.application.agent import run_agent
from banks_rag.application.agent.document_analysis import (
    build_batches,
    run_document_analysis,
    select_visuals,
)
from banks_rag.domain.agent import GenerationResult


def _tok(text: str) -> int:
    """Contador de tokens fake: ~1 token cada 4 chars."""
    return max(1, len(text) // 4)


@dataclass
class _MockLLM:
    """LLM mock que enruta MAP vs REDUCE por el contenido del prompt.

    El MAP pregunta por un "Fragmento del documento"; el REDUCE entrega "Tus
    notas del documento". Así el guion es determinista sin depender del nº de
    lotes."""

    map_text: str = "nota"
    reduce_text: str = "ANÁLISIS FINAL"
    calls: list[dict] = field(default_factory=list)

    async def generate(self, messages, *, tools=None, **kwargs):
        user = messages[-1]["content"]
        is_map = "Fragmento del documento" in user
        self.calls.append({"system": messages[0]["content"], "user": user, "tools": tools, "is_map": is_map})
        if is_map:
            return GenerationResult(text=self.map_text, n_tokens=3)
        return GenerationResult(text=self.reduce_text, n_tokens=20)

    def count_text_tokens(self, text: str) -> int:
        return _tok(text)


# ─────────────────────────────────────────────────────────────────────────────
# build_batches (puro)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildBatches:
    def test_groups_pages_within_budget(self) -> None:
        # Cada página ~25 tokens (100 chars). Budget 60 → ~2 páginas por lote.
        pages = [{"page": i, "text": "A" * 100} for i in range(1, 7)]
        batches = build_batches(pages, batch_tokens=60, count_tokens=_tok, max_batches=99)
        assert len(batches) >= 3                 # 6 páginas no caben en 1-2 lotes
        # Cada lote respeta el budget (con el marcador [pág. N] incluido).
        for b in batches:
            assert _tok(b) <= 60 or "[pág." in b  # páginas grandes pueden ir solas

    def test_splits_a_single_huge_page(self) -> None:
        pages = [{"page": 1, "text": "Z" * 4000}]   # ~1000 tokens, budget 100
        batches = build_batches(pages, batch_tokens=100, count_tokens=_tok, max_batches=99)
        assert len(batches) > 1                  # se partió la página gigante

    def test_respects_max_batches(self) -> None:
        pages = [{"page": i, "text": "B" * 400} for i in range(1, 50)]
        batches = build_batches(pages, batch_tokens=80, count_tokens=_tok, max_batches=5)
        assert len(batches) == 5

    def test_skips_empty_pages(self) -> None:
        pages = [{"page": 1, "text": "   "}, {"page": 2, "text": "contenido"}]
        batches = build_batches(pages, batch_tokens=100, count_tokens=_tok, max_batches=99)
        assert len(batches) == 1
        assert "contenido" in batches[0]


# ─────────────────────────────────────────────────────────────────────────────
# select_visuals (puro)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSelectVisuals:
    def test_builds_image_urls_and_caps(self) -> None:
        records = [{
            "upload_id": "up_abc",
            "visuals": [
                {"asset_id": "a1", "page": 2, "caption": "Gráfico 1: TPM", "kind": "CHART", "image_file": "p2.png"},
                {"asset_id": "a2", "page": 5, "caption": None, "kind": "TABLE", "image_file": "p5.png"},
            ],
        }]
        out = select_visuals(records, response_text="", max_visuals=6)
        assert out[0]["image_url"] == "/v1/uploads/up_abc/images/p2.png"
        # El que tiene caption va primero.
        assert out[0]["caption"] == "Gráfico 1: TPM"
        assert len(out) == 2

    def test_mentioned_visual_ranks_first(self) -> None:
        records = [{
            "upload_id": "up_x",
            "visuals": [
                {"page": 9, "caption": "Otra cosa", "image_file": "p9.png"},
                {"page": 3, "caption": "Inflación proyectada", "image_file": "p3.png"},
            ],
        }]
        resp = "El documento proyecta la inflación proyectada en (pág. 3)."
        out = select_visuals(records, response_text=resp, max_visuals=6)
        assert out[0]["page"] == 3

    def test_skips_visuals_without_file(self) -> None:
        records = [{"upload_id": "up_x", "visuals": [{"page": 1, "caption": "x"}]}]
        assert select_visuals(records, "", max_visuals=6) == []


# ─────────────────────────────────────────────────────────────────────────────
# run_document_analysis (LLM mock)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunDocumentAnalysis:
    @pytest.mark.asyncio
    async def test_map_per_batch_then_one_reduce(self) -> None:
        records = [{
            "upload_id": "up_1",
            "name": "informe.pdf",
            "pages": [{"page": i, "text": "X" * 200} for i in range(1, 5)],
            "visuals": [{"page": 2, "caption": "Gráfico 2", "image_file": "p2.png"}],
        }]
        # Con budget chico habrá varios lotes; el mock enruta MAP vs REDUCE.
        llm = _MockLLM(map_text="nota", reduce_text="ANÁLISIS FINAL")
        res = await run_document_analysis(
            records, "Resume el documento", llm=llm,
            batch_tokens=60, map_max_tokens=128, reduce_max_tokens=512, max_batches=99, max_visuals=6,
        )
        # n_batches llamadas de MAP + 1 de REDUCE.
        assert sum(c["is_map"] for c in llm.calls) == res.n_batches
        assert len(llm.calls) == res.n_batches + 1
        assert res.response == "ANÁLISIS FINAL"
        assert res.visuals and res.visuals[0]["image_url"] == "/v1/uploads/up_1/images/p2.png"

    @pytest.mark.asyncio
    async def test_empty_document_returns_message(self) -> None:
        records = [{"upload_id": "up_2", "name": "vacio.pdf", "pages": []}]
        llm = _MockLLM()
        res = await run_document_analysis(records, "Resume", llm=llm)
        assert res.finish_reason == "empty"
        assert llm.calls == []                   # no se llama al LLM sin contenido

    @pytest.mark.asyncio
    async def test_irrelevant_notes_are_dropped(self) -> None:
        records = [{"upload_id": "up_3", "name": "d.pdf",
                    "pages": [{"page": 1, "text": "hola"}, {"page": 2, "text": "mundo"}]}]
        # El MAP devuelve "sin contenido relevante" → notas vacías, pero igual REDUCE.
        llm = _MockLLM(map_text="(sin contenido relevante)", reduce_text="RESPUESTA")
        res = await run_document_analysis(
            records, "X", llm=llm, batch_tokens=10_000, max_batches=99,
        )
        assert res.partial_notes == []
        assert res.response == "RESPUESTA"


# ─────────────────────────────────────────────────────────────────────────────
# run_agent — desvío a modo documento cuando hay adjuntos
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunAgentDocumentMode:
    @pytest.mark.asyncio
    async def test_attachments_bypass_specialists(self) -> None:
        records = [{
            "upload_id": "up_9",
            "name": "doc.pdf",
            "n_pages": 1,
            "pages": [{"page": 1, "text": "El PIB creció 2,5% en 2025."}],
            "visuals": [{"page": 1, "caption": "PIB", "image_file": "p1.png"}],
        }]
        llm = _MockLLM(map_text="nota PIB", reduce_text="Análisis del documento.")
        result = await run_agent(
            "¿Qué dice el documento?", history=[], llm=llm,
            attachment_records=records,
        )
        assert result.response == "Análisis del documento."
        assert result.tool_trace == []           # NO se ruteó a especialistas
        assert result.attachment_visuals
        assert result.attachment_visuals[0]["image_url"] == "/v1/uploads/up_9/images/p1.png"
        assert len(llm.calls) == 2               # 1 MAP + 1 REDUCE (sin tools)
        assert all(c["tools"] is None for c in llm.calls)
