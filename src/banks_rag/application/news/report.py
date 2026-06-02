"""Analizador de noticias — reporte diario por map-reduce.

Un informe scrapeado trae ~100 noticias (cientos de miles de tokens) que NO
caben en una sola ventana de contexto. Pipeline:

  1. load_news      → carga el JSON (el más reciente de Noticias_scrapping/).
  2. prioritize     → ordena por relevancia (tema del dominio × alcance) y corta
                      a top-N (las más importantes del día).
  3. map            → resume cada lote de noticias (varias llamadas al LLM).
  4. reduce         → con los resúmenes, redacta el REPORTE estructurado.

Determinista en la priorización (testeable sin LLM); el LLM solo resume y
redacta. Pensado para un endpoint dedicado (no el chat interactivo): puede
tardar minutos porque el LLM es un recurso serializado.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from banks_rag.config.paths import NEWS_SCRAPING_DIR

log = logging.getLogger(__name__)

DEFAULT_TOP_N = 25
DEFAULT_BATCH_SIZE = 8
# El texto de cada noticia se trunca en el prompt de map (el resumen no necesita
# el cuerpo completo; evita inflar el contexto).
_MAP_TEXT_CHARS = 2500

# Peso por tema: las del dominio financiero/económico pesan más para un analista
# del BCCh. El resto entra por alcance, pero con menor prioridad.
_TOPIC_WEIGHTS = {
    "banco central": 3.0,
    "banca y finanzas": 2.5,
    "economía internacional": 2.5,
    "economia internacional": 2.5,
    "economía": 2.0,
    "economia": 2.0,
    "mercados": 2.0,
}
_DEFAULT_TOPIC_WEIGHT = 1.0

_METRIC_RE = re.compile(r"([\d.,]+)\s*([KkMm])?")


def _parse_metric(raw: Any) -> float:
    """Convierte 'audience'/'vpe' a número. Tolera ``$``/``€``, coma decimal y
    sufijos K/M. ``'271,02K'`` → 271020.0; ``'$ 18,90M'`` → 18900000.0; None → 0."""
    if raw is None:
        return 0.0
    s = str(raw).strip()
    m = _METRIC_RE.search(s.replace(" ", ""))
    if not m:
        return 0.0
    num, suffix = m.group(1), (m.group(2) or "").upper()
    # Formato es-CL: '.' es separador de miles, ',' es decimal.
    num = num.replace(".", "").replace(",", ".")
    try:
        value = float(num)
    except ValueError:
        return 0.0
    if suffix == "K":
        value *= 1_000
    elif suffix == "M":
        value *= 1_000_000
    return value


def _topic_weight(topic: Any) -> float:
    return _TOPIC_WEIGHTS.get(str(topic or "").strip().lower(), _DEFAULT_TOPIC_WEIGHT)


def _score(noticia: dict) -> float:
    """Score de relevancia: peso del tema × alcance (log para no dejar que una
    audiencia gigante aplaste todo). Usa vpe como proxy si falta la audiencia."""
    import math

    reach = _parse_metric(noticia.get("audience"))
    if reach <= 0:
        reach = _parse_metric(noticia.get("vpe")) / 1_000.0  # proxy grosero
    return math.log1p(max(reach, 0.0)) * _topic_weight(noticia.get("topic"))


def list_news_files() -> list[Path]:
    """JSONs de Noticias_scrapping/ ordenados por mtime (más reciente primero)."""
    if not NEWS_SCRAPING_DIR.exists():
        return []
    return sorted(NEWS_SCRAPING_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_news(json_path: str | Path | None = None) -> tuple[Path, list[dict]]:
    """Carga las noticias del JSON indicado o del más reciente. Lanza
    ``FileNotFoundError`` si no hay archivos, ``ValueError`` si el JSON no sirve."""
    if json_path is not None:
        path = Path(json_path)
        if not path.is_absolute():
            path = NEWS_SCRAPING_DIR / path
    else:
        files = list_news_files()
        if not files:
            raise FileNotFoundError(f"No hay JSONs de noticias en {NEWS_SCRAPING_DIR}")
        path = files[0]
    if not path.exists():
        raise FileNotFoundError(f"No existe el JSON de noticias: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else (
        next((v for v in data.values() if isinstance(v, list)), None)
        if isinstance(data, dict) else None
    )
    if not items:
        raise ValueError(f"El JSON {path.name} no contiene una lista de noticias.")
    return path, [n for n in items if isinstance(n, dict)]


def prioritize(news: list[dict], top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """Top-N noticias por score de relevancia (tema × alcance), desc."""
    return sorted(news, key=_score, reverse=True)[: max(1, top_n)]


def _noticia_brief(n: dict) -> str:
    """Encabezado compacto de una noticia para el prompt de map."""
    text = (n.get("text") or "").strip()
    if len(text) > _MAP_TEXT_CHARS:
        text = text[:_MAP_TEXT_CHARS] + "…"
    return (
        f"Título: {n.get('title') or n.get('emailTitle') or '(s/t)'}\n"
        f"Medio: {n.get('source') or '?'} | Tema: {n.get('topic') or '?'} | "
        f"Alcance: {n.get('audience') or '?'} | VPE: {n.get('vpe') or '?'}\n"
        f"Texto: {text}"
    )


_MAP_SYSTEM = (
    "Eres analista de prensa de la División de Mercados Financieros del Banco "
    "Central de Chile. Resume cada noticia para un brief diario."
)
_REDUCE_SYSTEM = (
    "Eres analista senior del Banco Central de Chile. Redactas el reporte de "
    "prensa del día para el Consejo, claro y jerarquizado."
)


def _map_prompt(batch: list[dict]) -> str:
    bloques = "\n\n---\n\n".join(_noticia_brief(n) for n in batch)
    return (
        "Resume CADA una de las siguientes noticias en 2-3 frases, destacando lo "
        "relevante para política monetaria, mercados o economía chilena. Para cada "
        "una indica: **Titular** (medio · tema) y el resumen. NO inventes datos; "
        "usa solo lo que dice la noticia.\n\n"
        f"{bloques}"
    )


def _reduce_prompt(summaries: list[str], n_total: int, n_used: int) -> str:
    cuerpo = "\n\n".join(summaries)
    return (
        f"A partir de estos resúmenes de las {n_used} noticias más relevantes del "
        f"día (de {n_total} del informe), redacta el REPORTE DE PRENSA del día:\n"
        "- Empieza con un RESUMEN EJECUTIVO (3-5 viñetas con lo más importante).\n"
        "- Agrupa por temas (Banco Central / política monetaria, economía "
        "internacional, banca y finanzas, otros).\n"
        "- Dentro de cada tema, destaca las noticias clave e interpreta su "
        "relevancia para el BCCh y los mercados.\n"
        "- Sé concreto y no inventes: usa solo lo de los resúmenes.\n\n"
        f"RESÚMENES:\n{cuerpo}"
    )


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def generate_news_report(
    llm,
    *,
    json_path: str | Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    batch_size: int = DEFAULT_BATCH_SIZE,
    map_max_tokens: int = 1500,
    report_max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Genera el reporte de prensa del día (map-reduce). Devuelve dict con
    ``report``, ``source_file``, ``n_total`` y ``n_used``."""
    path, news = load_news(json_path)
    top = prioritize(news, top_n)
    log.info("news-report: %s — %d noticias, top %d", path.name, len(news), len(top))

    # MAP: resumir por lotes (cada lote, una llamada al LLM).
    summaries: list[str] = []
    for batch in _chunks(top, batch_size):
        res = await llm.generate(
            [{"role": "system", "content": _MAP_SYSTEM},
             {"role": "user", "content": _map_prompt(batch)}],
            tools=None, temperature=temperature, max_tokens=map_max_tokens,
        )
        if res.text.strip():
            summaries.append(res.text.strip())

    if not summaries:
        return {"report": "No se pudo resumir ninguna noticia.", "source_file": path.name,
                "n_total": len(news), "n_used": 0}

    # REDUCE: redactar el reporte a partir de los resúmenes.
    red = await llm.generate(
        [{"role": "system", "content": _REDUCE_SYSTEM},
         {"role": "user", "content": _reduce_prompt(summaries, len(news), len(top))}],
        tools=None, temperature=temperature, max_tokens=report_max_tokens,
    )
    return {
        "report": red.text.strip(),
        "source_file": path.name,
        "n_total": len(news),
        "n_used": len(top),
    }
