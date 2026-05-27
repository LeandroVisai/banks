"""Orquestador del enriquecimiento semántico (legacy paso 01).

Toma documentos + chunks crudos y produce ``EnrichedChunk`` con:

  - section_type (+ confianza)
  - economic_variables {VAR: {importance, mentions, confidence, indicator_type}}
  - numeric_values [{value, unit, raw}]
  - entities {BANCO_X: mentions, PAIS_X: mentions}
  - temporal_refs {years, quarters}
  - tags (DECISION_POLITICA, FORWARD_LOOKING, DATOS_NUMERICOS, VARIABLE_CRITICA)
  - importance_score 0.0–1.0
  - is_policy_decision, is_forward_looking
  - v1.1: indicator_types, signal_strength, deviation_flag, trend_direction,
    forward_guidance, schema_version

Política: ``doc_type_category`` e ``institution`` se HEREDAN del documento,
no se re-detectan desde el contenido del chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from banks_rag.domain.documents import Chunk, ChunkKind, Document, EnrichedChunk
from banks_rag.domain_knowledge.enrichment import (
    DEVIATION_PATTERN,
    compute_signal_strength,
    compute_trend_direction,
    detect_entities,
    detect_section,
    detect_variables,
    extract_forward_guidance,
    extract_numerics,
    extract_temporal,
)
from banks_rag.domain_knowledge.importance_rules import (
    VISUAL_CHUNK_FLOOR,
    calculate_importance,
)
from banks_rag.domain_knowledge.taxonomy import (
    FORWARD_LOOKING_PATTERN,
    MONITOR_PM_SECTION_MAP,
    derive_tags,
    normalize_text,
)


@dataclass
class EnrichmentReport:
    """Estadísticas de la corrida de enriquecimiento."""

    documents_count: int
    chunks_total: int
    section_histogram: dict[str, int]
    variable_histogram: dict[str, int]
    avg_importance: float
    high_importance_count: int  # chunks con score >= 0.6

    def to_dict(self) -> dict:
        return {
            "documents_count": self.documents_count,
            "chunks_total": self.chunks_total,
            "section_histogram": self.section_histogram,
            "variable_histogram": self.variable_histogram,
            "avg_importance": self.avg_importance,
            "high_importance_count": self.high_importance_count,
        }


@dataclass
class EnrichmentResult:
    enriched_chunks: list[EnrichedChunk] = field(default_factory=list)
    report: EnrichmentReport | None = None


def enrich_chunk(chunk: Chunk, doc: Document, total_chunks_in_doc: int) -> EnrichedChunk:
    """Enriquece un chunk individual con todas las señales semánticas."""
    text_norm = normalize_text(chunk.text)
    doc_type = doc.doc_type_category

    variables = detect_variables(text_norm)
    numerics = extract_numerics(chunk.text)
    entities = detect_entities(text_norm)
    temporal = extract_temporal(text_norm)

    # Monitor PM: la sección viene directamente del nombre de columna del Excel.
    section_hint = chunk.section_title_raw or ""
    if (
        doc_type == "MONITOR_PM"
        and section_hint
        and section_hint.strip() in MONITOR_PM_SECTION_MAP
    ):
        section_type = MONITOR_PM_SECTION_MAP[section_hint.strip()]
        section_conf = 1.0
    else:
        section_type, section_conf = detect_section(
            text_norm,
            chunk.position_in_doc,
            total_chunks_in_doc,
            section_hint,
        )

    is_fwd = bool(FORWARD_LOOKING_PATTERN.search(text_norm))
    tags = derive_tags(text_norm, variables, numerics, section_type)

    # v1.1 — señales diferenciadas por tipo de fuente
    if doc_type == "MONITOR_PM":
        deviation_flag = bool(DEVIATION_PATTERN.search(text_norm))
        signal_strength: dict | None = compute_signal_strength(text_norm, variables, numerics)
        trend_direction: dict | None = None
        forward_guidance: str | None = None
    else:
        deviation_flag = False
        signal_strength = None
        trend_direction = compute_trend_direction(text_norm, variables) if variables else None
        forward_guidance = (
            extract_forward_guidance(chunk.text, text_norm, variables) if is_fwd else None
        )

    importance = calculate_importance(
        variables, numerics, section_type, entities, is_fwd, text_norm, deviation_flag
    )
    # Piso para chunks visuales: poco texto pero alta señal contextual.
    if chunk.image_path:
        importance = max(importance, VISUAL_CHUNK_FLOOR)

    indicator_types = {k: v["indicator_type"] for k, v in variables.items()}

    return EnrichedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        char_count=chunk.char_count,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        position_in_doc=chunk.position_in_doc,
        doc_type_category=doc_type or "REPORTE_RESEARCH",
        institution=doc.institution or "unknown",
        document_date=doc.document_date,
        section_type=section_type,
        section_confidence=section_conf,
        economic_variables=variables,
        numeric_values=numerics,
        entities=entities,
        temporal_refs=temporal,
        tags=tags,
        importance_score=importance,
        # Un chunk visual es un preview truncado de la página (imagen): no puede
        # ser la decisión autoritativa, aunque su snippet contenga "acordó...".
        # La decisión vive en el chunk de texto correspondiente.
        is_policy_decision=("DECISION_POLITICA" in tags) and not bool(chunk.image_path),
        is_forward_looking=is_fwd,
        chunk_date=chunk.chunk_date,
        image_path=chunk.image_path,
        visual_caption=chunk.visual_caption,
        kind=chunk.kind if isinstance(chunk.kind, ChunkKind) else ChunkKind.TEXT,
        indicator_types=indicator_types,
        signal_strength=signal_strength,
        deviation_flag=deviation_flag,
        trend_direction=trend_direction,
        forward_guidance=forward_guidance,
        schema_version="1.1",
    )


def _count_chunks_per_doc(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    return counts


def enrich_corpus(documents: list[Document], chunks: list[Chunk]) -> EnrichmentResult:
    """Enriquece un corpus completo (documentos + chunks crudos).

    Retorna ``EnrichmentResult`` con la lista de ``EnrichedChunk`` y un reporte
    agregado (histogramas de secciones y variables, distribución de importance).
    """
    docs_by_id = {d.document_id: d for d in documents}
    chunks_per_doc = _count_chunks_per_doc(chunks)

    enriched: list[EnrichedChunk] = []
    section_hist: dict[str, int] = {}
    var_hist: dict[str, int] = {}

    for chunk in chunks:
        doc = docs_by_id.get(chunk.document_id)
        if doc is None:
            # Chunk huérfano — usamos un Document mínimo para preservar comportamiento legacy.
            doc = Document(
                document_id=chunk.document_id,
                filename="",
                filepath="",
                doc_type_category="REPORTE_RESEARCH",
                institution="unknown",
                document_date=None,
                total_pages=0,
                total_chunks=0,
                char_count=0,
            )
        enriched_chunk = enrich_chunk(chunk, doc, chunks_per_doc[chunk.document_id])
        enriched.append(enriched_chunk)

        section_hist[enriched_chunk.section_type] = (
            section_hist.get(enriched_chunk.section_type, 0) + 1
        )
        for var_name in enriched_chunk.economic_variables:
            var_hist[var_name] = var_hist.get(var_name, 0) + 1

    avg_importance = (
        sum(e.importance_score for e in enriched) / len(enriched) if enriched else 0.0
    )
    high_importance_count = sum(1 for e in enriched if e.importance_score >= 0.6)

    report = EnrichmentReport(
        documents_count=len(documents),
        chunks_total=len(enriched),
        section_histogram=section_hist,
        variable_histogram=var_hist,
        avg_importance=round(avg_importance, 4),
        high_importance_count=high_importance_count,
    )

    return EnrichmentResult(enriched_chunks=enriched, report=report)
