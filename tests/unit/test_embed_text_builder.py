"""Unit tests para builder de texto-de-embedding y detectores de familia."""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.embeddings import (
    E5_PASSAGE_PREFIX,
    build_embed_text,
    is_e5_model,
    is_qwen_embedding,
    is_vl_model,
)


def _chunk(**overrides) -> dict:
    base = {
        "text": "El Consejo decidió mantener la TPM en 8,25%.",
        "doc_type_category": "COMUNICADO",
        "section_type": "DECISION",
        "economic_variables": {
            "TASA_INTERES": {
                "importance": "CRITICAL",
                "indicator_type": "CONTEMPORANEOUS",
                "mentions": 2,
                "confidence": 0.6,
            },
        },
        "document_date": "2024-03-15",
        "chunk_date": None,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestFamilyDetectors:
    def test_is_e5_model(self) -> None:
        assert is_e5_model("intfloat/multilingual-e5-small")
        assert is_e5_model("intfloat/e5-large-v2")
        assert not is_e5_model("Qwen/Qwen3-Embedding-8B")

    def test_is_qwen_embedding(self) -> None:
        assert is_qwen_embedding("Qwen/Qwen3-Embedding-8B")
        assert is_qwen_embedding("Qwen/Qwen3-VL-Embedding-8B")
        assert not is_qwen_embedding("Qwen/Qwen3.6-27B")  # LLM, no embedding
        assert not is_qwen_embedding("intfloat/multilingual-e5-small")

    def test_is_vl_model(self) -> None:
        assert is_vl_model("Qwen/Qwen3-VL-Embedding-8B")
        assert is_vl_model("CLIP-vision-embed")
        assert not is_vl_model("Qwen/Qwen3-Embedding-8B")  # texto puro
        assert not is_vl_model("intfloat/multilingual-e5-small")


@pytest.mark.unit
class TestBuildEmbedText:
    def test_pure_text_mode(self) -> None:
        """RAG_PURE_TEXT=1 → solo el texto, sin contexto."""
        text = build_embed_text(_chunk(), "Qwen/Qwen3-Embedding-8B", with_metadata=False)
        assert text == "El Consejo decidió mantener la TPM en 8,25%."

    def test_metadata_context_qwen(self) -> None:
        text = build_embed_text(_chunk(), "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        # Prefijo temporal mensual (PERIOD_MONTHLY) + categoría + sección + variable
        assert "[PERIOD_MONTHLY 2024-03]" in text
        assert "COMUNICADO" in text
        assert "DECISION" in text
        assert "TASA_INTERES" in text
        # Sin prefijo "passage: " (es Qwen, no E5)
        assert not text.startswith(E5_PASSAGE_PREFIX)

    def test_metadata_context_e5_adds_passage_prefix(self) -> None:
        text = build_embed_text(_chunk(), "intfloat/multilingual-e5-small", with_metadata=True)
        assert text.startswith(E5_PASSAGE_PREFIX)
        assert "[PERIOD_MONTHLY 2024-03]" in text

    def test_monitor_pm_uses_daily_prefix(self) -> None:
        chunk = _chunk(
            doc_type_category="MONITOR_PM",
            section_type="MERCADO_CAMBIARIO",
            chunk_date="2024-03-15",
            document_date=None,
        )
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        assert "[DAILY 2024-03-15]" in text

    def test_top_three_variables_only(self) -> None:
        many_vars = {
            f"VAR_{i}": {"importance": "HIGH", "indicator_type": "C", "mentions": 1, "confidence": 0.5}
            for i in range(10)
        }
        chunk = _chunk(economic_variables=many_vars, text="cuerpo")
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        # El cuerpo "cuerpo" separa el prefijo del texto. Solo las top-3 vars
        # deberían estar en el prefijo.
        prefix = text[: text.index("cuerpo")]
        var_mentions = sum(1 for i in range(10) if f"VAR_{i}" in prefix)
        assert var_mentions == 3

    def test_critical_variables_first(self) -> None:
        chunk = _chunk(economic_variables={
            "MEDIUM_VAR": {"importance": "MEDIUM", "indicator_type": "C", "mentions": 1, "confidence": 0.5},
            "CRITICAL_VAR": {"importance": "CRITICAL", "indicator_type": "LEADING", "mentions": 1, "confidence": 0.5},
            "HIGH_VAR": {"importance": "HIGH", "indicator_type": "C", "mentions": 1, "confidence": 0.5},
        })
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        # CRITICAL debe aparecer antes que HIGH y MEDIUM
        assert text.index("CRITICAL_VAR") < text.index("HIGH_VAR") < text.index("MEDIUM_VAR")

    def test_indicator_suffix_leading(self) -> None:
        chunk = _chunk(economic_variables={
            "V": {"importance": "CRITICAL", "indicator_type": "LEADING", "mentions": 1, "confidence": 0.5},
        })
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        assert "→LEADING" in text

    def test_indicator_suffix_lagging(self) -> None:
        chunk = _chunk(economic_variables={
            "V": {"importance": "CRITICAL", "indicator_type": "LAGGING", "mentions": 1, "confidence": 0.5},
        })
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        assert "→LAGGING" in text

    def test_no_variables_omits_var_section(self) -> None:
        chunk = _chunk(economic_variables={})
        text = build_embed_text(chunk, "Qwen/Qwen3-Embedding-8B", with_metadata=True)
        # No debe haber sufijo →LEADING/→LAGGING ni listado de vars
        assert "→LEADING" not in text
        assert "→LAGGING" not in text
