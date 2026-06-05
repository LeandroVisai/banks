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

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import NEWS_SCRAPING_DIR

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
    """Encabezado compacto de una noticia para el prompt de map.

    Incluye ``date`` además de medio/título: el reduce arma la sección de
    Referencias en APA y necesita fuente + título + fecha de cada noticia."""
    text = (n.get("text") or "").strip()
    if len(text) > _MAP_TEXT_CHARS:
        text = text[:_MAP_TEXT_CHARS] + "…"
    return (
        f"Título: {n.get('title') or n.get('emailTitle') or '(s/t)'}\n"
        f"Medio: {n.get('source') or '?'} | Fecha: {n.get('date') or '?'} | "
        f"Tema: {n.get('topic') or '?'} | "
        f"Alcance: {n.get('audience') or '?'} | VPE: {n.get('vpe') or '?'}\n"
        f"Texto: {text}"
    )


_MAP_SYSTEM = (
    "Eres analista experto en economía, finanzas y políticas públicas. Resumes "
    "noticias de prensa con rigor técnico, sin inventar datos."
)
_REDUCE_SYSTEM = (
    "Eres un analista experto en economía, finanzas y políticas públicas que "
    "redacta resúmenes de prensa técnicos, profundos y analíticos para analistas "
    "senior. Escribes en Markdown limpio y bien estructurado."
)
# Persona del relato hablado: el mismo informe contado como una narración.
_NARRATE_SYSTEM = (
    "Eres un analista económico que NARRA en voz alta el informe del día para un "
    "panel de analistas senior. Hablas en español, con un relato fluido, claro y "
    "didáctico, sin leer marcas de formato."
)


def _map_prompt(batch: list[dict]) -> str:
    bloques = "\n\n---\n\n".join(_noticia_brief(n) for n in batch)
    return (
        "Resume CADA una de las siguientes noticias en 2-4 frases, basándote SOLO "
        "en el campo Texto (no agregues datos externos). Para cada noticia escribe "
        "exactamente una línea de cabecera con la cita en este formato:\n"
        "  [CITA] Medio | Título | Fecha\n"
        "y debajo el resumen técnico, destacando cifras, diagnósticos y efectos "
        "económicos o financieros mencionados explícitamente.\n\n"
        f"{bloques}"
    )


# Texto fijo del recuadro bajo el título (lo pide el usuario; también lo usa el HTML).
DISCLAIMER = (
    "Este informe fue generado por IA a partir de fuentes periodísticas incluidas "
    "en el Informe Diario de Prensa"
)
REPORT_TITLE = "Informe Analítico de Coyuntura Económica y Financiera"


def _reduce_prompt(summaries: list[str], n_total: int, n_used: int) -> str:
    hoy = datetime.date.today().strftime("%d de %B de %Y")
    cuerpo = "\n\n".join(summaries)
    return (
        "# ROL\n"
        "Eres un analista experto en economía, finanzas y políticas públicas.\n\n"
        "# OBJETIVO\n"
        "Elabora un resumen de prensa técnico, profundo y analítico basado "
        "EXCLUSIVAMENTE en los resúmenes de noticias que se entregan abajo, con una "
        "extensión máxima equivalente a 5 páginas. No agregues datos externos, "
        "proyecciones ni interpretación fuera de lo explícitamente mencionado.\n\n"
        "# FORMATO DE SALIDA (Markdown estricto)\n"
        "Responde SOLO con el Markdown del informe, sin texto antes ni después, "
        "siguiendo EXACTAMENTE esta estructura para que se pueda convertir a HTML:\n"
        f"- Una única línea de título de nivel 1:  `# {REPORT_TITLE}`\n"
        "- Luego uno o más bloques temáticos, cada uno con un encabezado de nivel 2 "
        "numerado:  `## 1. <Tema>`,  `## 2. <Tema>`, … Agrupa noticias afines "
        "(p. ej. política fiscal, política monetaria, mercados y riesgos "
        "geopolíticos, mercado laboral, sistema previsional, sectores productivos, "
        "comercio exterior, energía, etc.).\n"
        "- Dentro de cada bloque, en este orden, usa encabezados de nivel 3:\n"
        "    `### Síntesis técnica`  → uno o dos párrafos.\n"
        "    `### Implicancias económicas`  → viñetas con `- ` (efectos económicos "
        "o financieros identificados directamente en los textos).\n"
        "    `### Relación entre noticias`  → un párrafo, SOLO si el tema reúne "
        "varias noticias relacionadas.\n"
        "- Separa cada bloque temático del siguiente con una línea `---`.\n"
        "- Cierra con un bloque  `## Referencias`  y una lista de viñetas `- ` con "
        "TODAS las noticias citadas en formato APA, usando únicamente la información "
        "de las cabeceras [CITA] (medio, título, fecha): "
        "`- Medio. (Fecha). Título.`\n\n"
        "# REGLAS\n"
        "- Tono riguroso, objetivo y técnico, para analistas senior.\n"
        "- Identifica relaciones, diagnósticos y efectos económicos presentes en "
        "los textos; NO inventes cifras ni fuentes.\n"
        "- Usa solo viñetas `- ` (no numeradas) dentro de los bloques.\n\n"
        f"# DATOS (resúmenes de las {n_used} noticias más relevantes de {n_total} "
        f"del informe del {hoy}; cada noticia trae su línea [CITA])\n{cuerpo}"
    )


def _narrate_prompt(report_md: str) -> str:
    return (
        "Convierte el siguiente INFORME en un RELATO HABLADO en español, pensado "
        "para leerse en voz alta ante analistas. Reglas:\n"
        "- Es un relato continuo y fluido: nada de títulos, viñetas, numeración, "
        "asteriscos, enlaces ni la sección de Referencias.\n"
        "- Usa frases completas y transiciones naturales entre temas ('en materia "
        "fiscal…', 'por el lado de los mercados…', 'en el plano laboral…').\n"
        "- Explica con claridad las cifras y sus efectos, pero NO agregues datos "
        "que no estén en el informe.\n"
        "- Empieza presentando que es el informe analítico de coyuntura económica y "
        "financiera del día, y cierra con una frase de síntesis.\n\n"
        f"INFORME:\n{report_md}"
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

    # REDUCE: redactar el informe analítico estructurado a partir de los resúmenes.
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


async def narrate_report(
    llm,
    report_md: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.4,
) -> str:
    """Convierte el informe estructurado (Markdown) en un RELATO hablado en español.

    El audio se sintetiza a partir de este relato (no del Markdown crudo): prosa
    fluida, sin títulos/viñetas/links/referencias, para que suene a una narración y
    no a un documento leído. Se basa SOLO en el informe (no agrega datos)."""
    if not (report_md or "").strip():
        return ""
    res = await llm.generate(
        [{"role": "system", "content": _NARRATE_SYSTEM},
         {"role": "user", "content": _narrate_prompt(report_md)}],
        tools=None, temperature=temperature, max_tokens=max_tokens,
    )
    return (res.text or "").strip() or report_md
