"""Unit tests para VisualAsset domain + chart_detector helpers."""

from __future__ import annotations

import pytest

from banks_rag.domain.documents import (
    BoundingBox,
    VisualAsset,
)
from banks_rag.infrastructure.extractors.chart_detector import (
    detect_caption,
    extract_surrounding_text,
)


@pytest.mark.unit
class TestBoundingBox:
    def test_dimensions(self) -> None:
        b = BoundingBox(0, 0, 100, 50)
        assert b.width == 100
        assert b.height == 50
        assert b.area == 5000

    def test_union(self) -> None:
        b1 = BoundingBox(0, 0, 100, 50)
        b2 = BoundingBox(50, 25, 200, 100)
        u = b1.union(b2)
        assert u.to_tuple() == (0, 0, 200, 100)

    def test_to_tuple(self) -> None:
        b = BoundingBox(1.5, 2.5, 3.5, 4.5)
        assert b.to_tuple() == (1.5, 2.5, 3.5, 4.5)


@pytest.mark.unit
class TestVisualAsset:
    def test_to_chunk_dict_includes_caption_and_text(self) -> None:
        asset = VisualAsset(
            asset_id="doc1_v0001",
            document_id="doc1",
            page=5,
            kind="CHART",
            image_path="/tmp/x.png",
            caption="Gráfico 3: TPM",
            surrounding_text="Texto previo y posterior.",
        )
        d = asset.to_chunk_dict(chunk_id="doc1_img_0001", position_in_doc=42)
        assert d["chunk_id"] == "doc1_img_0001"
        assert d["kind"] == "VISUAL"
        assert d["image_path"] == "/tmp/x.png"
        assert d["page_start"] == 5 == d["page_end"]
        assert d["position_in_doc"] == 42
        assert "CHART" in d["text"]
        assert "TPM" in d["text"]
        assert d["visual_caption"] == "Gráfico 3: TPM"

    def test_to_chunk_dict_without_caption(self) -> None:
        asset = VisualAsset(
            asset_id="x", document_id="d", page=1,
            kind="TABLE", image_path="/tmp/t.png",
        )
        d = asset.to_chunk_dict(chunk_id="x", position_in_doc=0)
        assert "TABLE" in d["text"]
        assert d["visual_caption"] is None


@pytest.mark.unit
class TestDetectCaption:
    def test_detects_grafico_with_title(self) -> None:
        assert (
            detect_caption("Gráfico 3: Evolución de la TPM")
            == "Gráfico 3: Evolución de la TPM"
        )

    def test_detects_figura(self) -> None:
        assert detect_caption("Figura 12: Curva swap CLP") == "Figura 12: Curva swap CLP"

    def test_detects_cuadro_no_title(self) -> None:
        assert detect_caption("Cuadro 5") == "Cuadro 5"

    def test_detects_in_middle_of_text(self) -> None:
        text = "Texto antes. Gráfico 7: Inflación anual. Texto después."
        result = detect_caption(text)
        assert result is not None
        assert "Gráfico 7" in result
        assert "Inflación anual" in result

    def test_strips_trailing_punctuation(self) -> None:
        assert detect_caption("Cuadro 1: Tabla;") == "Cuadro 1: Tabla"

    def test_returns_none_when_no_match(self) -> None:
        assert detect_caption("Texto sin captions") is None

    def test_returns_none_for_empty(self) -> None:
        assert detect_caption("") is None

    def test_english_table(self) -> None:
        assert detect_caption("Table 5: Annual GDP growth") == "Table 5: Annual GDP growth"

    def test_case_insensitive(self) -> None:
        assert detect_caption("GRÁFICO 8: x") == "GRÁFICO 8: x"


@pytest.mark.unit
class TestExtractSurroundingText:
    def test_returns_first_chars(self) -> None:
        text = "x" * 1000
        result = extract_surrounding_text(text)
        assert len(result) == 400
        assert result == "x" * 400

    def test_collapses_whitespace(self) -> None:
        text = "uno   dos    tres\n\n\ncuatro"
        result = extract_surrounding_text(text)
        assert result == "uno dos tres cuatro"

    def test_short_text_returned_full(self) -> None:
        assert extract_surrounding_text("hola mundo") == "hola mundo"

    def test_empty(self) -> None:
        assert extract_surrounding_text("") == ""

    def test_respects_max_chars_param(self) -> None:
        text = "abc " * 500  # 2000 chars
        result = extract_surrounding_text(text, max_chars=50)
        assert len(result) <= 50
