"""``VisualAsset`` — gráfico, tabla o figura extraída de un PDF.

Hasta Fase 1 los PDFs con charts se renderizan a PNG y se asocian al chunk
mediante ``Chunk.image_path``, pero el embedding sigue siendo del texto
``"[Imagen p.N]"``. Fase 2 cierra el ciclo:

  1. Detectar la región del chart en la página (bbox).
  2. Extraer la caption ("Gráfico 3", "Figura 12", "Cuadro 5") del texto
     cercano vía regex.
  3. Capturar las primeras N chars del texto que la rodean (contexto).
  4. Embeber con ``Qwen3-VL-Embedding-8B`` (modelo multimodal en el mismo
     espacio 4096-dim que el texto).
  5. Almacenar como ``Chunk`` con ``kind=VISUAL`` y ``visual_caption`` en
     el schema PostgreSQL.

``VisualAsset`` es el value object que viaja entre el extractor y el
chunker antes de convertirse en un ``Chunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VisualKind = Literal["CHART", "TABLE", "IMAGE"]


@dataclass(frozen=True)
class BoundingBox:
    """Rectángulo en coordenadas PDF (origen en esquina inferior-izquierda).

    Las coords son floats en pt (1/72 inch). PyMuPDF usa este sistema.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class VisualAsset:
    """Un activo visual extraído de un documento.

    Attributes:
        asset_id: único, formato ``{document_id}_v{NNNN}``.
        document_id: documento de origen.
        page: número de página (1-based).
        kind: ``CHART``, ``TABLE`` o ``IMAGE``.
        bbox: rectángulo en la página (None si se renderizó la página completa).
        image_path: ruta absoluta o relativa al PNG renderizado.
        caption: caption detectada (e.g. ``"Gráfico 3: Evolución de la TPM"``).
        surrounding_text: texto de la página cercano a la región (para fallback
            de embedding cuando el VL falla, y para enriquecer el contexto).
        embedding: vector L2-normalizado (poblado en Fase 2.b por ``vectorize_corpus``).
    """

    asset_id: str
    document_id: str
    page: int
    kind: VisualKind = "CHART"
    bbox: BoundingBox | None = None
    image_path: str = ""
    caption: str | None = None
    surrounding_text: str = ""
    embedding: list[float] | None = None
    embedding_model: str | None = None

    def to_chunk_dict(
        self,
        *,
        chunk_id: str,
        position_in_doc: int,
    ) -> dict:
        """Convierte el asset a un dict compatible con el flujo de chunks legacy.

        ``Chunk.text`` se construye a partir de la caption + surrounding text,
        de modo que el TF-IDF / BM25 todavía pueda encontrar el chunk visual
        por texto descriptivo aunque el embedding venga de la imagen.
        """
        text_parts: list[str] = [f"[{self.kind} p.{self.page}]"]
        if self.caption:
            text_parts.append(self.caption)
        if self.surrounding_text:
            text_parts.append(self.surrounding_text)
        text = "\n".join(text_parts)

        return {
            "chunk_id": chunk_id,
            "document_id": self.document_id,
            "text": text,
            "char_count": len(text),
            "token_estimate": max(1, len(text) // 4),
            "page_start": self.page,
            "page_end": self.page,
            "position_in_doc": position_in_doc,
            "section_title_raw": self.caption,
            "image_path": self.image_path,
            "visual_caption": self.caption,
            "kind": "VISUAL",  # se mapea a ChunkKind.VISUAL
        }


@dataclass
class VisualExtractionStats:
    """Estadísticas de la extracción visual sobre un documento."""

    pages_with_visuals: int = 0
    assets_extracted: int = 0
    captions_detected: int = 0
    pages_skipped_no_visuals: int = 0
    bbox_extractions: int = 0
    full_page_fallbacks: int = 0
    warnings: list[str] = field(default_factory=list)
