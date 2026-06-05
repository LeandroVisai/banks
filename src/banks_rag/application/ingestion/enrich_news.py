"""Enriquecimiento del corpus de **contexto actual** (noticias).

Paralelo a ``enrich_corpus`` (banco), pero con una taxonomía DISTINTA, como
corresponde a una fuente distinta: una noticia de prensa NO tiene secciones de
política ("decisión", "votación"), no es una decisión del Consejo y su valor
está en la **coyuntura** (qué está pasando y con qué tono), no en la citación
normativa.

Qué REUSA del banco (porque sigue siendo útil cruzar la coyuntura con el
vocabulario económico): los detectores ``detect_variables`` / ``detect_entities``
/ ``extract_numerics`` / ``extract_temporal``. Una noticia que menciona "TPM",
"IPC" o "Rosanna Costa" es justo lo que el agente querrá traer como contexto.

Qué es PROPIO de noticias (no del banco):
  - ``section_type = "NOTICIA"`` (sin detección de secciones del corpus).
  - ``is_policy_decision`` SIEMPRE False (una noticia jamás es la decisión).
  - **sentimiento** por léxico (POSITIVO/NEGATIVO/NEUTRO) como tag.
  - ``importance_score`` por **relevancia económica** (cuántas variables/entidades/
    cifras trae), no por reglas de sección de política.
  - ``schema_version = "news-1.0"`` para distinguir el origen en la BD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from banks_rag.domain.documents import Chunk, ChunkKind, Document, EnrichedChunk
from banks_rag.domain_knowledge.enrichment import (
    detect_entities,
    detect_variables,
    extract_numerics,
    extract_temporal,
)
from banks_rag.domain_knowledge.taxonomy import FORWARD_LOOKING_PATTERN, normalize_text

SECTION_NEWS = "NOTICIA"
SCHEMA_VERSION_NEWS = "news-1.0"

# Léxico de sentimiento económico (sobre texto normalizado, sin acentos). No
# pretende ser un clasificador fino: marca el tono dominante de la noticia para
# que el agente module la lectura de coyuntura.
_POSITIVE_WORDS = {
    "subio", "sube", "suben", "alza", "crecimiento", "crece", "recuperacion",
    "repunte", "expansion", "fortaleza", "avance", "avanza", "ganancia",
    "ganancias", "superavit", "optimismo", "mejora", "mejoro", "impulso",
    "solido", "favorable", "positivo", "auge", "dinamismo",
}
_NEGATIVE_WORDS = {
    "cae", "caen", "caida", "baja", "bajan", "desplome", "desploma", "recesion",
    "crisis", "incertidumbre", "riesgo", "riesgos", "pesimismo", "deterioro",
    "contraccion", "debil", "debilidad", "perdida", "perdidas", "deficit",
    "conflicto", "guerra", "temor", "temores", "preocupacion", "tension",
    "tensiones", "desaceleracion", "shock", "golpe", "presion", "presiones",
    "negativo", "desfavorable",
}
_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


@dataclass
class NewsEnrichmentReport:
    documents_count: int = 0
    chunks_total: int = 0
    sentiment_histogram: dict[str, int] = field(default_factory=dict)
    variable_histogram: dict[str, int] = field(default_factory=dict)
    institution_histogram: dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "documents_count": self.documents_count,
            "chunks_total": self.chunks_total,
            "sentiment_histogram": self.sentiment_histogram,
            "variable_histogram": self.variable_histogram,
            "institution_histogram": self.institution_histogram,
            "avg_importance": self.avg_importance,
        }


@dataclass
class NewsEnrichmentResult:
    enriched_chunks: list[EnrichedChunk] = field(default_factory=list)
    report: NewsEnrichmentReport | None = None


def detect_sentiment(text_norm: str) -> str:
    """Tono económico dominante del texto: ``POSITIVO`` | ``NEGATIVO`` | ``NEUTRO``.

    Conteo de léxico con umbral: empate o pocas señales → NEUTRO."""
    tokens = _WORD_RE.findall(text_norm)
    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if pos == neg:
        return "NEUTRO"
    return "POSITIVO" if pos > neg else "NEGATIVO"


def _news_importance(
    variables: dict, numerics: list, entities: dict, is_fwd: bool,
) -> float:
    """Importancia por **relevancia económica** de la noticia (no por sección).

    Una noticia con varias variables económicas, entidades y cifras es más útil
    como contexto que una nota genérica. Calibrado a [0, 1]."""
    score = 0.15  # piso: toda noticia indexada aporta algo de contexto
    score += min(len(variables), 4) * 0.12
    score += min(len(numerics), 4) * 0.05
    score += min(len(entities), 3) * 0.06
    if is_fwd:
        score += 0.10
    return round(min(score, 1.0), 4)


def _news_tags(
    variables: dict, numerics: list, is_fwd: bool, sentiment: str,
) -> list[str]:
    """Tags propios de noticia (NO los del banco). Incluye el sentimiento."""
    tags = ["NOTICIA", f"SENTIMIENTO_{sentiment}"]
    if variables:
        tags.append("VARIABLE_CRITICA")
    if numerics:
        tags.append("DATOS_NUMERICOS")
    if is_fwd:
        tags.append("FORWARD_LOOKING")
    return tags


def enrich_news_chunk(chunk: Chunk, doc: Document) -> EnrichedChunk:
    """Enriquece un chunk de noticia con la taxonomía noticiera."""
    text_norm = normalize_text(chunk.text)

    variables = detect_variables(text_norm)
    numerics = extract_numerics(chunk.text)
    entities = detect_entities(text_norm)
    temporal = extract_temporal(text_norm)
    is_fwd = bool(FORWARD_LOOKING_PATTERN.search(text_norm))
    sentiment = detect_sentiment(text_norm)

    importance = _news_importance(variables, numerics, entities, is_fwd)
    tags = _news_tags(variables, numerics, is_fwd, sentiment)
    indicator_types = {k: v.get("indicator_type") for k, v in variables.items()}

    return EnrichedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        char_count=chunk.char_count,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        position_in_doc=chunk.position_in_doc,
        doc_type_category=doc.doc_type_category or "NOTICIA",
        institution=doc.institution or "Prensa",
        document_date=doc.document_date,
        section_type=SECTION_NEWS,
        section_confidence=1.0,
        economic_variables=variables,
        numeric_values=numerics,
        entities=entities,
        temporal_refs=temporal,
        tags=tags,
        importance_score=importance,
        is_policy_decision=False,           # una noticia nunca es la decisión
        is_forward_looking=is_fwd,
        chunk_date=chunk.chunk_date,
        image_path=None,
        visual_caption=None,
        kind=chunk.kind if isinstance(chunk.kind, ChunkKind) else ChunkKind.TEXT,
        indicator_types=indicator_types,
        signal_strength=None,
        deviation_flag=False,
        trend_direction=None,
        forward_guidance=None,
        schema_version=SCHEMA_VERSION_NEWS,
    )


def enrich_news(
    documents: list[Document], chunks: list[Chunk],
) -> NewsEnrichmentResult:
    """Enriquece un corpus de noticias completo (documentos + chunks crudos)."""
    docs_by_id = {d.document_id: d for d in documents}
    fallback_doc = Document(
        document_id="", filename="", filepath="", doc_type_category="NOTICIA",
        institution="Prensa", document_date=None, total_pages=0,
        total_chunks=0, char_count=0,
    )

    enriched: list[EnrichedChunk] = []
    sent_hist: dict[str, int] = {}
    var_hist: dict[str, int] = {}
    inst_hist: dict[str, int] = {}

    for chunk in chunks:
        doc = docs_by_id.get(chunk.document_id, fallback_doc)
        ec = enrich_news_chunk(chunk, doc)
        enriched.append(ec)

        for tag in ec.tags:
            if tag.startswith("SENTIMIENTO_"):
                key = tag.removeprefix("SENTIMIENTO_")
                sent_hist[key] = sent_hist.get(key, 0) + 1
        for var_name in ec.economic_variables:
            var_hist[var_name] = var_hist.get(var_name, 0) + 1
        inst_hist[ec.institution] = inst_hist.get(ec.institution, 0) + 1

    avg_importance = (
        round(sum(e.importance_score for e in enriched) / len(enriched), 4)
        if enriched else 0.0
    )

    report = NewsEnrichmentReport(
        documents_count=len(documents),
        chunks_total=len(enriched),
        sentiment_histogram=sent_hist,
        variable_histogram=var_hist,
        institution_histogram=inst_hist,
        avg_importance=avg_importance,
    )
    return NewsEnrichmentResult(enriched_chunks=enriched, report=report)
