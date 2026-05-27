"""Unit tests para detectores de enriquecimiento."""

from __future__ import annotations

import pytest

from banks_rag.application.ingestion.enrich_corpus import enrich_chunk
from banks_rag.domain.documents.chunk import Chunk, ChunkKind
from banks_rag.domain.documents.document import Document
from banks_rag.domain_knowledge.enrichment import (
    DEVIATION_PATTERN,
    compute_signal_strength,
    compute_trend_direction,
    detect_section,
    detect_variables,
    extract_forward_guidance,
    extract_numerics,
    extract_temporal,
)
from banks_rag.domain_knowledge.taxonomy import (
    DECISION_SENTENCE_PATTERN,
    normalize_text,
)


@pytest.mark.unit
class TestDetectVariables:
    def test_detects_tasa_interes(self) -> None:
        text = normalize_text("La TPM se ubicó en 8,25% durante 2024")
        result = detect_variables(text)
        # TASA_INTERES debería estar entre las detectadas
        assert any("TASA" in k for k in result)

    def test_returns_empty_for_irrelevant_text(self) -> None:
        text = normalize_text("hoy hace mucho calor en santiago")
        result = detect_variables(text)
        assert result == {}

    def test_confidence_grows_with_mentions(self) -> None:
        single = detect_variables(normalize_text("la inflación subió"))
        multiple = detect_variables(
            normalize_text("la inflación, la inflación, la inflación crece")
        )
        if "INFLACION" in single and "INFLACION" in multiple:
            assert multiple["INFLACION"]["confidence"] >= single["INFLACION"]["confidence"]


@pytest.mark.unit
class TestDetectSection:
    def test_hint_takes_precedence(self) -> None:
        section, conf = detect_section(
            text_norm="cualquier texto",
            position_in_doc=5,
            total_chunks_in_doc=10,
            section_hint="DECISION",
        )
        assert section == "DECISION"
        assert conf >= 0.9

    def test_first_chunk_bias_to_encabezado(self) -> None:
        section, conf = detect_section(
            text_norm="texto sin keywords de seccion",
            position_in_doc=0,
            total_chunks_in_doc=20,
            section_hint=None,
        )
        # Primer 5% del doc → ENCABEZADO con baja confianza
        assert section == "ENCABEZADO"
        assert conf < 0.5

    def test_last_chunk_bias_to_riesgos(self) -> None:
        section, _ = detect_section(
            text_norm="texto neutro sin keywords",
            position_in_doc=18,
            total_chunks_in_doc=20,
            section_hint=None,
        )
        # rel_pos = 18/19 = 0.947 > 0.85 → RIESGOS
        assert section == "RIESGOS"

    def test_default_when_no_signal(self) -> None:
        section, _ = detect_section(
            text_norm="texto medio sin keywords",
            position_in_doc=10,
            total_chunks_in_doc=20,
            section_hint=None,
        )
        # Posición media (~52%) sin matches → CONTENIDO
        assert section == "CONTENIDO"

    def test_decision_sentence_overrides_frequency(self) -> None:
        # Párrafo de apertura real (abril 2026): la decisión + contexto de
        # riesgos. Antes ganaba RIESGOS (riesgos+incertidumbre=2 vs acordo=1);
        # ahora la frase canónica de decisión fuerza DECISION.
        text = normalize_text(
            "En su Reunión de Política Monetaria, el Consejo del Banco Central "
            "de Chile acordó mantener la tasa de interés de política monetaria "
            "en 4,5%. La evolución del panorama internacional continúa marcada "
            "por la incertidumbre en torno a la guerra; han aumentado los riesgos."
        )
        section, conf = detect_section(text, 0, 12, None)
        assert section == "DECISION"
        assert conf >= 0.9

    def test_riesgos_sin_frase_de_decision(self) -> None:
        # Sin la frase de decisión, un párrafo de riesgos sigue siendo RIESGOS.
        text = normalize_text(
            "El escenario macroeconómico sigue sujeto a incertidumbre. Los "
            "principales riesgos están al alza por la guerra en Medio Oriente."
        )
        section, _ = detect_section(text, 5, 12, None)
        assert section == "RIESGOS"


@pytest.mark.unit
class TestDecisionSentencePattern:
    def test_matches_canonical(self) -> None:
        assert DECISION_SENTENCE_PATTERN.search(normalize_text(
            "el Consejo acordó mantener la tasa de interés de política "
            "monetaria en 4,5%"
        ))

    def test_matches_tpm_sigla(self) -> None:
        assert DECISION_SENTENCE_PATTERN.search(normalize_text(
            "el Consejo decidió reducir la TPM en 25 puntos base"
        ))

    def test_no_matches_truncated(self) -> None:
        # El chunk-imagen truncado corta antes de "monetaria" → sin match.
        assert not DECISION_SENTENCE_PATTERN.search(normalize_text(
            "el Consejo acordó mantener la tasa de interés de política monetari"
        ))

    def test_no_matches_forward_looking(self) -> None:
        # "la evolución futura de la TPM irá evaluándose" no es una decisión.
        assert not DECISION_SENTENCE_PATTERN.search(normalize_text(
            "la evolución futura de la TPM irá evaluándose Reunión a Reunión"
        ))


@pytest.mark.unit
class TestVisualChunkNotPolicyDecision:
    """Un chunk visual (preview truncado de la página) NO debe ser la decisión
    autoritativa, aunque su snippet contenga 'acordó mantener la tasa...'."""

    def _doc(self) -> Document:
        return Document(
            document_id="d1", filename="comunicado abril 2026.pdf", filepath="x",
            doc_type_category="COMUNICADO_RPM", institution="banco_central_chile",
            document_date="2026-04-28", total_pages=1, total_chunks=2, char_count=100,
        )

    def _chunk(self, *, visual: bool) -> Chunk:
        text = ("En su Reunión de Política Monetaria, el Consejo del Banco "
                "Central de Chile acordó mantener la tasa de interés de "
                "política monetaria en 4,5%.")
        return Chunk(
            chunk_id="d1c0", document_id="d1", text=text, char_count=len(text),
            token_estimate=len(text) // 4, page_start=1, page_end=1,
            position_in_doc=0, section_title_raw=None,
            image_path="/img/p1.png" if visual else None,
            kind=ChunkKind.VISUAL if visual else ChunkKind.TEXT,
        )

    def test_text_decision_is_policy_decision(self) -> None:
        enriched = enrich_chunk(self._chunk(visual=False), self._doc(), 2)
        assert enriched.section_type == "DECISION"
        assert enriched.is_policy_decision is True

    def test_visual_decision_is_not_policy_decision(self) -> None:
        enriched = enrich_chunk(self._chunk(visual=True), self._doc(), 2)
        assert enriched.is_policy_decision is False


@pytest.mark.unit
class TestExtractNumerics:
    def test_extracts_percentage(self) -> None:
        result = extract_numerics("La TPM bajó a 8,25%")
        assert any("8,25" in r["value"] or "8" in r["value"] for r in result)

    def test_caps_at_max(self) -> None:
        # Texto con muchos %
        text = " ".join(f"{i}%" for i in range(20))
        result = extract_numerics(text, max_results=5)
        assert len(result) <= 5

    def test_basis_points(self) -> None:
        result = extract_numerics("Recortó en 75 puntos base la tasa")
        assert any(r["unit"] == "basis_points" for r in result)


@pytest.mark.unit
class TestExtractTemporal:
    def test_finds_year(self) -> None:
        result = extract_temporal(normalize_text("la inflación de 2022 fue alta"))
        assert "years" in result
        assert "2022" in result["years"]

    def test_finds_quarter(self) -> None:
        result = extract_temporal(normalize_text("en el q3 del año"))
        assert "quarters" in result

    def test_omits_empty_keys(self) -> None:
        result = extract_temporal(normalize_text("texto sin fechas"))
        assert "years" not in result
        assert "quarters" not in result


@pytest.mark.unit
class TestComputeSignalStrength:
    def test_up_direction(self) -> None:
        text = normalize_text("la TPM subió 25 puntos base")
        result = compute_signal_strength(text, {"TASA_INTERES": {}}, [{"value": "25"}])
        assert result["TASA_INTERES"]["direction"] == "UP"
        assert result["TASA_INTERES"]["magnitude"] == 25.0

    def test_down_direction(self) -> None:
        text = normalize_text("recorte de 50 pb en la tasa")
        result = compute_signal_strength(text, {"V": {}}, [])
        assert result["V"]["direction"] == "DOWN"

    def test_stable_when_neither(self) -> None:
        text = normalize_text("la tasa permaneció en niveles")
        result = compute_signal_strength(text, {"V": {}}, [])
        assert result["V"]["direction"] == "STABLE"

    def test_stable_when_both(self) -> None:
        # Si ambas señales aparecen, default a STABLE
        text = normalize_text("subio y bajo en distintos momentos")
        result = compute_signal_strength(text, {"V": {}}, [])
        assert result["V"]["direction"] == "STABLE"


@pytest.mark.unit
class TestComputeTrendDirection:
    def test_rising(self) -> None:
        text = normalize_text("aumento sostenido en los ultimos meses")
        result = compute_trend_direction(text, {"V": {}})
        assert result["V"] == "RISING"

    def test_falling(self) -> None:
        text = normalize_text("recorte significativo y desaceleracion")
        result = compute_trend_direction(text, {"V": {}})
        assert result["V"] == "FALLING"

    def test_mixed(self) -> None:
        text = normalize_text("aumento inicial seguido de recorte")
        result = compute_trend_direction(text, {"V": {}})
        assert result["V"] == "MIXED"

    def test_empty_when_no_signal(self) -> None:
        text = normalize_text("texto neutro sin direccion")
        result = compute_trend_direction(text, {"V": {}})
        assert result == {}


@pytest.mark.unit
class TestExtractForwardGuidance:
    def test_returns_none_without_forward_signal(self) -> None:
        text = "La inflación subió"
        result = extract_forward_guidance(text, normalize_text(text), {"INFLACION": {"importance": "CRITICAL", "indicator_type": "C"}})
        assert result is None


@pytest.mark.unit
def test_deviation_pattern_matches() -> None:
    assert DEVIATION_PATTERN.search(normalize_text("contrasta con lo esperado"))
    assert DEVIATION_PATTERN.search(normalize_text("sorpresa para los analistas"))
    assert not DEVIATION_PATTERN.search(normalize_text("comportamiento normal"))
