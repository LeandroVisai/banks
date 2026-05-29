"""Unit tests para la normalización de doc_type compartida por las tools."""

from __future__ import annotations

import pytest

from banks_rag.application.agent.tools._doc_types import (
    DOC_TYPE_VALUES,
    normalize_doc_types,
)


@pytest.mark.unit
class TestNormalizeDocTypes:
    def test_none_and_empty(self) -> None:
        assert normalize_doc_types(None) == []
        assert normalize_doc_types("") == []
        assert normalize_doc_types([]) == []

    def test_canonical_passthrough(self) -> None:
        assert normalize_doc_types("IPOM") == ["IPOM"]
        assert normalize_doc_types(["COMUNICADO_RPM"]) == ["COMUNICADO_RPM"]

    def test_llm_aliases(self) -> None:
        # El vocabulario que el LLM usó en los logs.
        assert normalize_doc_types("COMUNICADO") == ["COMUNICADO_RPM"]
        assert normalize_doc_types("RESEARCH") == ["REPORTE_RESEARCH"]
        assert normalize_doc_types("FED STATEMENT") == ["FED_STATEMENT"]
        assert normalize_doc_types("MONITOR PM") == ["MONITOR_PM"]

    def test_minuta_expands_to_both(self) -> None:
        assert normalize_doc_types("MINUTA") == ["MINUTA_RPM", "MINUTA_IPOM"]

    def test_array_dedup_and_order(self) -> None:
        out = normalize_doc_types(["COMUNICADO", "COMUNICADO_RPM", "IPOM"])
        assert out == ["COMUNICADO_RPM", "IPOM"]

    def test_unknown_dropped(self) -> None:
        assert normalize_doc_types("BANANA") == []
        assert normalize_doc_types(["BANANA", "IPOM"]) == ["IPOM"]

    def test_array_from_logs_does_not_crash(self) -> None:
        # El caso exacto que reventaba list_documents en los logs.
        out = normalize_doc_types(
            ["COMUNICADO", "MINUTA", "RESEARCH", "MONITOR PM", "FED STATEMENT"]
        )
        for v in out:
            assert v in DOC_TYPE_VALUES
        assert "COMUNICADO_RPM" in out and "REPORTE_RESEARCH" in out
