"""Tests del gating de extracción visual a IPoM only.

El usuario pidió extraer gráficos/figuras SOLO de IPoM (no de Comunicados ni
Minutas RPM). Verifica que ``_process_pdf`` solo invoca ``extract_visual_assets``
para ``doc_type == 'IPOM'``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import importlib

# El __init__ del paquete rebindea el nombre `extract_corpus` a la función
# homónima, sombreando el submódulo; import_module devuelve el módulo real.
ec = importlib.import_module("banks_rag.application.ingestion.extract_corpus")


def _run_process_pdf(doc_type: str, visual_mock: MagicMock):
    raw_chunk = {
        "text": "contenido",
        "page_start": 1,
        "page_end": 1,
        "section_title_raw": None,
    }
    with (
        patch.object(ec, "extract_pages", return_value=(["pág 1"], [])),
        patch.object(ec, "normalize_page_text", side_effect=lambda p: p),
        patch.object(ec, "chunk_pages", return_value=[raw_chunk]),
        patch.object(ec, "detect_doc_type", return_value=doc_type),
        patch.object(ec, "detect_institution", return_value="x"),
        patch.object(ec, "detect_date", return_value=None),
        patch.object(ec, "slugify_document_id", return_value="docid"),
        patch.object(ec, "extract_visual_assets", visual_mock),
    ):
        return ec._process_pdf(
            Path("/data/X/doc.pdf"),
            Path("/data"),
            Path("/imgs"),
            MagicMock(),
        )


@pytest.mark.unit
class TestVisualGating:
    def test_constant_is_ipom_only(self) -> None:
        assert set(ec.VISUAL_DOC_TYPES) == {"IPOM"}

    def test_ipom_extracts_visuals(self) -> None:
        visual_mock = MagicMock(return_value=[])
        _run_process_pdf("IPOM", visual_mock)
        visual_mock.assert_called_once()

    @pytest.mark.parametrize("doc_type", ["COMUNICADO_RPM", "MINUTA_RPM", "MINUTA_IPOM", "IEF"])
    def test_non_ipom_skips_visuals(self, doc_type: str) -> None:
        visual_mock = MagicMock(return_value=[])
        _, chunks = _run_process_pdf(doc_type, visual_mock)
        visual_mock.assert_not_called()
        # Sin visuales: todos los chunks son TEXT.
        assert all(c.kind == ec.ChunkKind.TEXT for c in chunks)
