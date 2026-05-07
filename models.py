"""
Modelos de datos centralizados del pipeline RAG.

Estos dataclasses son la única fuente de verdad para las estructuras de datos
que se pasan entre pasos. Importar desde aquí, no duplicar en cada script.

Jerarquía:
  Document  — metadata de un PDF (1 por archivo)
  Chunk     — fragmento de texto crudo con posición (N por documento)
  EnrichedChunk — Chunk + señales semánticas del enriquecimiento (paso 1)

El embedding se almacena como list[float] en los JSON intermedios; la BD usa
vector(384) de pgvector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Paso 0 — Extracción y chunking
# ---------------------------------------------------------------------------

@dataclass
class Document:
    document_id: str
    filename: str
    filepath: str
    doc_type_category: str        # COMUNICADO | MINUTA | FED_STATEMENT | REPORTE_RESEARCH
    institution: str              # banco_central_chile | federal_reserve | jpmorgan | unknown
    document_date: Optional[str]  # ISO YYYY-MM-DD o solo año
    total_pages: int
    total_chunks: int
    char_count: int
    extraction_warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str                      # "<doc_id>_<NNNN>"
    document_id: str
    text: str
    char_count: int
    token_estimate: int                # ~char_count/4 (heurística rápida)
    page_start: int
    page_end: int
    position_in_doc: int               # 0-based
    section_title_raw: Optional[str]   # título crudo detectado durante chunking; 01 lo normaliza
    chunk_date: Optional[str] = None   # ISO YYYY-MM-DD; solo Excel (Monitor PM), None para PDFs
    image_path: Optional[str] = None   # ruta al PNG extraído; solo páginas visuales, None para texto


# ---------------------------------------------------------------------------
# Paso 1 — Enriquecimiento semántico
# ---------------------------------------------------------------------------

@dataclass
class EnrichedChunk:
    # Identidad
    chunk_id: str
    document_id: str
    text: str
    char_count: int
    page_start: int
    page_end: int
    position_in_doc: int
    # Heredados del documento (no re-detectados por chunk)
    doc_type_category: str
    institution: str
    document_date: Optional[str]
    # Señales semánticas
    section_type: str
    section_confidence: float
    economic_variables: dict   # {VAR: {importance, mentions, confidence}}
    numeric_values: list       # [{value, unit, raw}]
    entities: dict             # {ENTIDAD: mentions}
    temporal_refs: dict        # {years: [...], quarters: [...]}
    tags: list                 # DECISION_POLITICA, FORWARD_LOOKING, ...
    importance_score: float    # 0.0–1.0, calibrado
    is_policy_decision: bool
    is_forward_looking: bool
    chunk_date: Optional[str] = None   # ISO YYYY-MM-DD; solo Excel (Monitor PM), None para PDFs
    image_path: Optional[str] = None   # ruta al PNG extraído; solo páginas visuales, None para texto
    # --- v1.1 fields ---
    indicator_types: dict = field(default_factory=dict)  # {var_name: LEADING|CONTEMPORANEOUS|LAGGING}
    signal_strength: Optional[dict] = None               # solo MONITOR_PM
    deviation_flag: bool = False                          # solo MONITOR_PM
    trend_direction: Optional[dict] = None               # solo PDFs
    forward_guidance: Optional[str] = None               # solo PDFs, is_forward_looking=True
    schema_version: str = "1.1"
