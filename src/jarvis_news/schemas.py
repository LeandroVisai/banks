"""Schemas request/response de jarvis_news (aislados del core)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NewsReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Nombre del JSON en Noticias_scrapping/; None = el más reciente.
    json_path: str | None = Field(default=None, max_length=255)
    top_n: int = Field(default=25, ge=1, le=60)


class NewsReportResponse(BaseModel):
    report: str
    source_file: str
    n_total: int        # noticias en el archivo
    n_used: int         # noticias usadas (top-N)


class TtsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=20_000)
    # La voz JARVIS es inglés (en_GB); por defecto se traduce el texto con el LLM
    # antes de sintetizar. Pon false si el texto ya está en inglés.
    translate_to_en: bool = True
