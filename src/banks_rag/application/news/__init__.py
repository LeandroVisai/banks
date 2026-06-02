"""Analizador de noticias: lee un JSON de noticias scrapeadas (informe diario) y
genera un reporte estructurado con las más importantes, vía map-reduce sobre el
LLM (los ~100 artículos no caben en una sola ventana de contexto)."""

from .report import (
    generate_news_report,
    list_news_files,
    load_news,
    prioritize,
)

__all__ = [
    "generate_news_report",
    "list_news_files",
    "load_news",
    "prioritize",
]
