"""EnrichedChunk — Chunk + señales semánticas (sección, variables, importancia)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunk import ChunkKind


@dataclass
class EnrichedChunk:
    """Chunk enriquecido con análisis semántico (paso 1 del pipeline).

    Incluye señales de sección canónica, variables económicas detectadas,
    valores numéricos, entidades, referencias temporales, tags categóricos,
    e ``importance_score`` calibrado [0.0, 1.0].

    Los campos v1.1 (``signal_strength``, ``deviation_flag``, ``trend_direction``,
    ``forward_guidance``) son parcialmente nulables según el tipo de documento:

    - ``signal_strength`` y ``deviation_flag`` solo se calculan para MONITOR_PM.
    - ``trend_direction`` solo para PDFs.
    - ``forward_guidance`` solo para PDFs con ``is_forward_looking=True``.
    """

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
    document_date: str | None

    # Señales semánticas
    section_type: str
    section_confidence: float
    economic_variables: dict  # {VAR: {importance, mentions, confidence, indicator_type}}
    numeric_values: list  # [{value, unit, raw}]
    entities: dict  # {ENTIDAD: mentions}
    temporal_refs: dict  # {years: [...], quarters: [...]}
    tags: list  # DECISION_POLITICA | FORWARD_LOOKING | DATOS_NUMERICOS | VARIABLE_CRITICA
    importance_score: float
    is_policy_decision: bool
    is_forward_looking: bool

    chunk_date: str | None = None
    image_path: str | None = None
    visual_caption: str | None = None
    kind: ChunkKind = ChunkKind.TEXT

    # v1.1
    indicator_types: dict = field(default_factory=dict)
    signal_strength: dict | None = None
    deviation_flag: bool = False
    trend_direction: dict | None = None
    forward_guidance: str | None = None
    schema_version: str = "1.1"
