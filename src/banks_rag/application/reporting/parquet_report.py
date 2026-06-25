"""Informe descriptivo de datasets parquet (map-reduce SIN tools).

Análogo al informe de noticias de ``jarvis_news`` pero sobre los datos del
catálogo de parquets: el usuario pide por segmento ("ffmm" / "fondos mutuos"),
por lista de ids/archivos, o por texto libre, y el proceso genera un Markdown
con una síntesis global y UN párrafo descriptivo por dataset que describe SOLO
el comportamiento relevante de los datos (nivel, variación última semana/mes,
máximos/mínimos, composición, anomalías) — NO qué mide la variable.

Pipeline:
    select_datasets        → resuelve la selección (segmento | ids | query)
    MAP  _describe_dataset → Python lee el parquet REAL y calcula los hechos
                             (``parquet_facts.compute_facts``, reutilizando
                             ``series_analytics``); luego UNA llamada al LLM SIN
                             tools redacta el párrafo a partir de esos números.
    REDUCE _synthesize_overview → una llamada al LLM SIN tools que sintetiza
    ParquetReport.to_markdown   → ensamblado determinista (sin LLM)

El cómputo en Python (no tool-calling) elimina el loop multi-paso contra el
llama-server, que se caía sistemáticamente, y ancla las ventanas a la última
fecha REAL de cada parquet (no a "hoy" ni al date_range del catálogo).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from banks_rag.application.agent.prompts import NO_THINK_DIRECTIVE, PARQUET_REPORTER_PROMPT
from banks_rag.application.reporting.parquet_facts import compute_facts, facts_to_text
from banks_rag.domain_knowledge.financial_aliases import expand_query, resolve_segment
from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    ParquetDataset,
    get_dataset,
    get_parquet_dir,
    load_parquet_catalog,
    search_datasets,
)

log = logging.getLogger(__name__)

# Sampling del MAP (redacción del párrafo desde hechos ya calculados):
# perfil no-thinking de Qwen3 (MODE_PROFILES["off"]).
_MAP_TEMPERATURE = 0.4
_MAP_TOP_P = 0.80
# Sampling del REDUCE: igual que la síntesis del agente (fiel/determinista).
_SYNTH_TEMPERATURE = 0.3
_SYNTH_TOP_P = 0.8

_NO_DATA_PARAGRAPH = (
    "Los datos de este dataset aún no están disponibles localmente (el archivo "
    "parquet no se ha copiado); la sección se completará en la próxima "
    "actualización de datos."
)
_ERROR_PARAGRAPH = (
    "La consulta de este dataset falló durante la generación del informe; "
    "revisar los logs para el detalle."
)


def _error_paragraph(stage: str, exc: Exception) -> str:
    """Mensaje de error que NOMBRA la etapa que reventó y el error real.

    ``stage='datos'`` → falló ``compute_facts`` (DuckDB/parquet en esta máquina).
    ``stage='LLM'``   → falló ``llm.generate`` (servidor llama-server / backend).
    Antes ambas etapas devolvían el mismo texto genérico y el informe no permitía
    distinguir un problema de datos de uno del modelo.
    """
    detail = f"{type(exc).__name__}: {exc}".strip()
    if len(detail) > 300:
        detail = detail[:297] + "…"
    if stage == "LLM":
        return (
            "Los datos de este dataset SÍ se calcularon, pero el modelo no pudo "
            f"redactar el párrafo (error del servidor LLM): {detail}."
        )
    return (
        "No se pudieron calcular los hechos de este dataset a partir del parquet "
        f"(error de datos, no del modelo): {detail}."
    )
_EMPTY_OVERVIEW = (
    "No fue posible obtener datos de ningún dataset de la selección; el informe "
    "no incluye síntesis."
)

_REPORT_SYNTHESIS_SYSTEM = """\
Eres un analista senior de la División de Mercados Financieros del BCCh. \
Recibes los párrafos descriptivos de un informe SEMANAL (uno por dataset del \
catálogo) y redactas la SÍNTESIS EJECUTIVA que lo encabeza. El foco es la SEMANA: \
lo prioritario son los movimientos de la última semana; lo mensual es contexto \
secundario.

Estructura EXACTA (en este orden):
1. UNA frase de apertura con el estado general del segmento en la semana.
2. Una línea ``**Principales movimientos de la semana:**`` seguida de 3 a 4 \
viñetas ``- `` con lo más relevante de la última semana (cada viñeta: qué se \
movió, cuánto y su cifra/fecha/unidad exacta; marca si confirma o contrasta con \
el mes).
3. Una línea ``**Contexto del mes:**`` seguida de 1 a 2 viñetas ``- `` BREVES con \
el telón de fondo mensual, solo si aporta.

Reglas:
- Usa SOLO las cifras que aparecen en los párrafos recibidos: NO inventes ni \
recalcules números, y conserva su unidad y fecha al citarlos.
- INDICA SIEMPRE la VENTANA de datos de cada cifra (el rango de fechas hasta el que \
llega), porque la fecha de corte puede diferir entre series.
- Prioriza los movimientos SEMANALES más significativos; no listes todos los datasets.
- Cada viñeta es una sola línea, concisa.
- Si te indican que algunos datasets quedaron sin datos, menciónalo en una \
frase al final.
- Usa solo ``-`` para viñetas y ``**…**`` para los dos rótulos; nada de tablas \
ni encabezados ``#``."""


# ─────────────────────────────────────────────────────────────────────────────
# Selección de datasets
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSelection:
    """Resultado de resolver la selección del usuario contra el catálogo."""

    datasets: tuple[ParquetDataset, ...]
    selector_label: str  # slug corto para el nombre de archivo ("ffmm", "ids", …)
    selector_desc: str   # texto legible para el título del informe
    missing_ids: tuple[str, ...] = ()


def _slug(text: str) -> str:
    out = re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_")
    return out[:40] or "informe"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-záéíóúüñ0-9]+", text.lower()) if len(t) >= 3}


def select_datasets(
    entries: list[ParquetDataset],
    *,
    segment: str | None = None,
    dataset_ids: Sequence[str] | None = None,
    query: str | None = None,
    top_k: int = 12,
) -> DatasetSelection:
    """Resuelve la selección de datasets por exactamente UNA de las tres vías.

    - ``segment``: nombre canónico o alias del dominio ("ffmm", "fondos mutuos").
    - ``dataset_ids``: ids del catálogo o nombres de archivo (``flujos_ffmm`` /
      ``flujos_ffmm.parquet``). Los no hallados van a ``missing_ids``.
    - ``query``: texto libre, resuelto con el descubrimiento del catálogo.
    """
    provided = [name for name, value in (
        ("segment", segment), ("dataset_ids", dataset_ids), ("query", query),
    ) if value]
    if len(provided) != 1:
        raise ValueError(
            "Indica exactamente UNA vía de selección: segment, dataset_ids o "
            f"query (recibí: {provided or 'ninguna'})."
        )

    if segment:
        valid = {e.segment for e in entries if e.segment}
        canonical = resolve_segment(segment, valid_segments=valid)
        if canonical is None:
            raise ValueError(
                f"Segmento {segment!r} no reconocido. Segmentos del catálogo: "
                f"{', '.join(sorted(valid))}."
            )
        datasets = tuple(e for e in entries if e.segment == canonical)
        desc = f"segmento {canonical}"
        if segment.strip().lower() != canonical:
            desc += f" ({segment.strip()})"
        return DatasetSelection(datasets=datasets, selector_label=_slug(canonical), selector_desc=desc)

    if dataset_ids:
        found: list[ParquetDataset] = []
        seen: set[str] = set()
        missing: list[str] = []
        for item in dataset_ids:
            ds = _match_id_or_file(entries, item.strip())
            if ds is None:
                missing.append(item.strip())
            elif ds.id not in seen:
                seen.add(ds.id)
                found.append(ds)
        if not found:
            raise ValueError(
                f"Ningún dataset del catálogo coincide con: {', '.join(missing)}."
            )
        if missing:
            log.warning("Datasets pedidos no hallados en el catálogo: %s", missing)
        return DatasetSelection(
            datasets=tuple(found),
            selector_label="ids",
            selector_desc="datasets: " + ", ".join(d.id for d in found),
            missing_ids=tuple(missing),
        )

    assert query is not None
    candidates = search_datasets(entries, query, top_k=top_k)
    # search_datasets siempre devuelve top_k aunque el score sea 0: filtramos
    # los que no comparten ningún token con la query expandida por el dominio
    # (así "fondos mutuos" alcanza ids con "ffmm" pero no arrastra ruido).
    query_tokens = _tokens(expand_query(query))
    relevant = tuple(
        e for e in candidates
        if _tokens(f"{e.id} {e.name} {e.description} {e.segment} {e.unit}") & query_tokens
    )
    if not relevant:
        raise ValueError(f"Ningún dataset del catálogo coincide con la consulta {query!r}.")
    return DatasetSelection(
        datasets=relevant,
        selector_label=_slug(query),
        selector_desc=f'consulta "{query.strip()}"',
    )


def _match_id_or_file(entries: list[ParquetDataset], item: str) -> ParquetDataset | None:
    """Matchea un item del usuario por id exacto o por nombre de archivo (con
    o sin ruta, con o sin extensión)."""
    if not item:
        return None
    ds = get_dataset(entries, item)
    if ds is not None:
        return ds
    base = item.replace("\\", "/").rsplit("/", 1)[-1]
    stem = base[:-len(".parquet")] if base.endswith(".parquet") else base
    for e in entries:
        e_base = e.file.replace("\\", "/").rsplit("/", 1)[-1]
        if e.file == item or e_base == base or e_base == f"{stem}.parquet":
            return e
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Ventanas de análisis
# ─────────────────────────────────────────────────────────────────────────────

_WINDOW_RE = re.compile(r"^(\d+)\s*([dwmy])$", re.IGNORECASE)
_WINDOW_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def _parse_window(token: str) -> tuple[int, str]:
    """``"7d"/"30d"/"2w"/"3m"`` → (días, etiqueta legible en español)."""
    m = _WINDOW_RE.match(token.strip())
    if not m:
        raise ValueError(f"Ventana {token!r} inválida (usa Nd, Nw, Nm o Ny — ej. 7d, 30d).")
    n = int(m.group(1))
    days = n * _WINDOW_DAYS[m.group(2).lower()]
    if days == 7:
        return days, "última semana"
    if days == 30:
        return days, "último mes"
    if days == 365:
        return days, "último año"
    return days, f"últimos {days} días"


def _window_specs(windows: Sequence[str]) -> list[tuple[str, int]]:
    """``["7d","30d"]`` → ``[("última semana", 7), ("último mes", 30)]``."""
    out = []
    for token in windows:
        days, label = _parse_window(token)
        out.append((label, days))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAP — un párrafo por dataset: Python calcula los hechos, el LLM solo redacta
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetSection:
    """Una sección del informe: el párrafo descriptivo de un dataset."""

    dataset_id: str
    name: str
    chart_type: str
    unit: str
    segment: str
    last_date: str | None
    paragraph: str
    status: str  # "ok" | "no_data" | "error"
    total_tokens: int = 0
    # Hechos calculados (compute_facts) que vio el LLM: el verificador los usa como
    # fuente de verdad para contrastar la prosa. Se conservan, no se descartan.
    facts: dict | None = None
    # Texto tal como lo generó el LLM, ANTES del verificador (para diff antes/después).
    raw_paragraph: str = ""
    # ¿El párrafo lleva sub-párrafo MENSUAL? (False = solo semanal; mensual una vez
    # por sección). Lo respeta la regeneración del verificador.
    include_monthly: bool = True


def _build_user_prompt(
    dataset: ParquetDataset, facts: dict, *, think: bool = True, include_monthly: bool = True,
) -> str:
    """Mensaje de usuario para el LLM: identidad del dataset + hechos calculados.

    ``include_monthly`` controla cuántos párrafos se piden: con ``True`` el redactor
    escribe DOS (semanal primero, mensual después); con ``False`` UNO solo (semanal)
    — la sección ya tiene su lectura mensual en otro bloque (mensual una vez por
    sección). El informe ffmm es SEMANAL: el énfasis va en la última semana.

    Con ``think=False`` antepone ``/no_think`` al turno de usuario — es el
    único lugar donde el template Jinja de Qwen3 lo lee para suprimir la
    inyección de ``<think>``. En el system message el template lo ignora y el
    modelo piensa de todas formas, consumiendo los tokens y dejando ``content``
    vacío en la respuesta.
    """
    if include_monthly:
        ask = (
            "Redacta DOS párrafos: el PRIMERO sobre la variación SEMANAL (última "
            "semana) y el SEGUNDO sobre la MENSUAL (último mes). Describe SOLO el "
            "comportamiento relevante; no expliques qué mide la variable ni para qué "
            "sirve la serie."
        )
    else:
        ask = (
            "Redacta UN SOLO párrafo sobre la variación SEMANAL (última semana). NO "
            "escribas párrafo mensual: esta sección ya tiene su lectura del mes en "
            "otro bloque. Describe SOLO el comportamiento relevante; no expliques qué "
            "mide la variable ni para qué sirve la serie."
        )
    head = [
        f"Dataset: {dataset.name} (`{dataset.id}`)",
        f"Unidad: {dataset.unit}" if dataset.unit else "",
        f"Segmento: {dataset.segment}" if dataset.segment else "",
        "",
        "DATOS YA CALCULADOS (úsalos tal cual, no recalcules):",
        facts_to_text(facts),
        "",
        ask,
    ]
    text = "\n".join(line for line in head if line != "")
    if not think:
        text = f"{NO_THINK_DIRECTIVE}\n\n{text}"
    return text


# Etiquetas INTERNAS de los facts (scaffolding para el LLM): no deben aparecer en
# el informe. El modelo chico tiende a copiarlas literal ("[ENTRADA: flujo
# positivo]") pese al prompt; se eliminan de forma determinista.
_FACT_TAG_RE = re.compile(
    r"[ \t]*\[\s*(?:ENTRADA|SALIDA|MENOR\s+ENTRADA)\b[^\]]*\]", re.IGNORECASE
)


def _strip_fact_tags(text: str) -> str:
    """Quita las etiquetas internas ``[ENTRADA…]``/``[SALIDA…]``/``[MENOR ENTRADA…]``
    que el redactor pudo copiar de los datos al texto final. Colapsa solo espacios/
    tabs (NO newlines: conserva la separación en párrafos)."""
    out = _FACT_TAG_RE.sub("", text or "")
    return re.sub(r"[ \t]{2,}", " ", out)


def _clean_paragraph(text: str) -> str:
    """Limpia la salida del LLM CONSERVANDO la separación en párrafos.

    El redactor produce uno o dos párrafos (semanal y, si se pidió, mensual)
    separados por una línea en blanco. Se quitan encabezados/viñetas que el modelo
    haya metido pese al formato pedido, las etiquetas internas de los facts, se
    colapsan los saltos de línea DENTRO de cada párrafo, y se devuelven los párrafos
    unidos por ``\\n\\n`` (separador estable que el ensamblado a markdown/HTML
    interpreta como párrafos distintos)."""
    blocks: list[str] = []
    current: list[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s or re.fullmatch(r"[-*_]{3,}", s):
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        s = re.sub(r"^#{1,6}\s+", "", s)
        s = re.sub(r"^[-*+]\s+", "", s)
        current.append(s)
    if current:
        blocks.append(" ".join(current))
    return _strip_fact_tags("\n\n".join(b for b in blocks if b))


async def _describe_dataset(
    dataset: ParquetDataset,
    *,
    llm,
    window_specs: Sequence[tuple[str, int]],
    parquet_dir: Path,
    map_max_tokens: int = 32768,
    think: bool = False,
    weekly_asof: str | None = None,
    include_monthly: bool = True,
) -> DatasetSection:
    """Genera la sección de un dataset: facts en Python + UNA llamada al LLM.

    ``weekly_asof`` = corte / T común del informe (ancla TODAS las ventanas de los
    facts: semanal y mensual). ``include_monthly`` decide si el párrafo lleva también
    la lectura mensual (solo el primer dataset de cada sección) o solo la semanal."""
    section = DatasetSection(
        dataset_id=dataset.id,
        name=dataset.name,
        chart_type=dataset.chart_type,
        unit=dataset.unit,
        segment=dataset.segment,
        last_date=None,
        paragraph="",
        status="ok",
        include_monthly=include_monthly,
    )
    try:
        facts = compute_facts(dataset, parquet_dir, list(window_specs), weekly_asof=weekly_asof)
    except Exception as exc:
        log.exception("[%s] el cálculo de hechos falló", dataset.id)
        section.status = "error"
        section.paragraph = _error_paragraph("datos", exc)
        return section

    # Sin parquet (None) o sin datos útiles → sección "sin datos".
    if facts is None or facts.get("shape") in (None, "empty", "unknown") or _facts_empty(facts):
        section.status = "no_data"
        section.paragraph = _NO_DATA_PARAGRAPH
        return section

    section.facts = facts  # fuente de verdad para el verificador
    section.last_date = facts.get("last_date")

    try:
        result = await llm.generate(
            [
                {"role": "system", "content": PARQUET_REPORTER_PROMPT},
                {"role": "user", "content": _build_user_prompt(
                    dataset, facts, think=think, include_monthly=include_monthly)},
            ],
            tools=None,
            temperature=_MAP_TEMPERATURE,
            top_p=_MAP_TOP_P,
            max_tokens=map_max_tokens,
        )
    except Exception as exc:
        log.exception("[%s] la redacción del párrafo falló", dataset.id)
        section.status = "error"
        section.paragraph = _error_paragraph("LLM", exc)
        return section

    section.total_tokens = getattr(result, "n_tokens", 0) or 0
    section.paragraph = _clean_paragraph(result.text)
    # Si completion_tokens >> len(párrafo)/4, el extra son tokens de thinking.
    # Un párrafo de 600 chars ≈ 150 tokens; si completion_tokens es 5000+, pensó.
    _think_indicator = " (thinking)" if section.total_tokens > len(section.paragraph) // 4 + 400 else ""
    log.info("[%s] %d completion_tokens → %d chars párrafo%s",
             dataset.id, section.total_tokens, len(section.paragraph), _think_indicator)
    if not section.paragraph:
        section.status = "error"
        log.error("[%s] el LLM respondió con texto vacío (raw=%r)", dataset.id, result.text)
        section.paragraph = _error_paragraph("LLM", RuntimeError("el modelo respondió con texto vacío"))
    return section


def _facts_empty(facts: dict) -> bool:
    """True si los hechos no traen ningún número útil (parquet vacío/sin señal)."""
    shape = facts.get("shape")
    if shape == "snapshot":
        return not facts.get("composition")
    if shape == "timeseries_categorical":
        return not facts.get("por_categoria") and not facts.get("composicion_corte")
    if shape == "timeseries_wide":
        return not facts.get("por_columna")
    if shape == "timeseries_single":
        return not facts.get("estadisticas")
    if shape == "return_index":
        return not facts.get("por_fondo")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE — síntesis global sin tools
# ─────────────────────────────────────────────────────────────────────────────


async def _synthesize_overview(
    sections: Sequence[DatasetSection],
    *,
    llm,
    selector_desc: str,
    max_tokens: int = 2048,
    redundant_ids: set[str] | None = None,
) -> str:
    # Excluye de la síntesis las vistas redundantes (bloques no_text del spec):
    # sus gráficos se mantienen, pero su párrafo NO entra al reduce, así la síntesis
    # no mezcla dos comentarios del mismo concepto con direcciones distintas.
    redundant_ids = redundant_ids or set()
    ok = [s for s in sections if s.status == "ok" and s.dataset_id not in redundant_ids]
    if not ok:
        return _EMPTY_OVERVIEW
    parts = [f"Informe descriptivo — {selector_desc}.", "", "Párrafos por dataset:"]
    for s in ok:
        parts.append(f"\n[{s.dataset_id}] {s.name}\n{s.paragraph}")
    # "Sin datos" cuenta SOLO los datasets sin estado ok (no_data/error), NO los
    # excluidos por redundantes (esos tienen gráfico y párrafo propio, solo no van
    # a la síntesis).
    skipped = sum(1 for s in sections if s.status != "ok")
    if skipped:
        parts.append(f"\n(Nota: {skipped} dataset(s) de la selección quedaron sin datos.)")
    parts.append(
        "\nRedacta ahora la síntesis ejecutiva con la estructura pedida: una frase "
        "de apertura, luego los principales movimientos de la SEMANA en viñetas y, al "
        "final, un breve contexto del mes."
    )
    result = await llm.generate(
        [
            {"role": "system", "content": _REPORT_SYNTHESIS_SYSTEM},
            {"role": "user", "content": f"{NO_THINK_DIRECTIVE}\n\n" + "\n".join(parts)},
        ],
        tools=None,
        temperature=_SYNTH_TEMPERATURE,
        top_p=_SYNTH_TOP_P,
        max_tokens=max_tokens,
    )
    return _strip_fact_tags((result.text or "").strip()) or _EMPTY_OVERVIEW


# ─────────────────────────────────────────────────────────────────────────────
# Verificación: regeneración de párrafo + utilidades
# ─────────────────────────────────────────────────────────────────────────────


async def _regenerate_paragraph(
    section: DatasetSection, issues: list, *, llm, map_max_tokens: int, think: bool = False,
) -> str:
    """Reescribe el párrafo de una sección con una NOTA de las fallas detectadas por
    el verificador (cifras sin sustento / contradicción). Reusa los mismos facts y el
    prompt del redactor; SOLO redacta sobre los datos calculados."""
    note = "; ".join(getattr(i, "detail", str(i)) for i in issues[:4])
    que = (
        "los DOS párrafos (semanal y mensual)" if section.include_monthly
        else "el párrafo (solo semanal)"
    )
    head = [
        f"Dataset: {section.name} (`{section.dataset_id}`)",
        f"Unidad: {section.unit}" if section.unit else "",
        f"Segmento: {section.segment}" if section.segment else "",
        "",
        "DATOS YA CALCULADOS (úsalos tal cual, no recalcules):",
        facts_to_text(section.facts or {}),
        "",
        "Tu borrador anterior tenía problemas que detectó el control de calidad: "
        f"{note}. Reescribe {que} corrigiéndolos, usando EXCLUSIVAMENTE "
        "las cifras y direcciones de los datos de arriba; no inventes números.",
    ]
    text = "\n".join(line for line in head if line != "")
    if not think:
        text = f"{NO_THINK_DIRECTIVE}\n\n{text}"
    result = await llm.generate(
        [
            {"role": "system", "content": PARQUET_REPORTER_PROMPT},
            {"role": "user", "content": text},
        ],
        tools=None, temperature=_MAP_TEMPERATURE, top_p=_MAP_TOP_P, max_tokens=map_max_tokens,
    )
    return _clean_paragraph(result.text or "")


def _redundant_dataset_ids(selected_ids: set[str]) -> set[str]:
    """``source_id`` de bloques ``no_text`` (vistas redundantes) presentes en la
    selección, leídos de los specs curados. La síntesis los excluye."""
    from banks_rag.application.reporting.specs import available_families, get_spec

    out: set[str] = set()
    for fam in available_families():
        spec = get_spec(fam)
        if spec is not None:
            out |= {sid for sid in spec.no_text_source_ids() if sid in selected_ids}
    return out


def _facts_digest(sections: Sequence[DatasetSection]) -> str:
    """Digest compacto de los facts (texto) de las secciones ok, para el crítico LLM."""
    parts = []
    for s in sections:
        if s.status == "ok" and s.facts:
            parts.append(f"[{s.dataset_id}] {s.name}\n{facts_to_text(s.facts)}")
    return "\n\n".join(parts)[:8000]


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador + ensamblado Markdown
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ParquetReport:
    """Informe completo: síntesis + secciones por dataset + metadata."""

    title: str
    selector_label: str
    selector_desc: str
    generated_at: str
    windows: tuple[str, ...]
    overview_md: str
    sections: list[DatasetSection] = field(default_factory=list)
    missing_ids: tuple[str, ...] = ()
    # Resultado del verificador post-síntesis (None si no se corrió). Lleva los
    # hallazgos, las autocorrecciones y las marcas residuales.
    verification: Any = None
    # Síntesis tal como la generó el LLM, ANTES del verificador (diff antes/después).
    raw_overview_md: str = ""

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sections:
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts

    def _residual_flags(self, where: str) -> list[str]:
        """Marcas de verificación NO resueltas para ``where`` (dataset o síntesis)."""
        vr = self.verification
        if vr is None:
            return []
        try:
            return [i.detail for i in vr.residual_for(where)]
        except Exception:  # verification con shape inesperado: no romper el render
            return []

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", "## Síntesis", "", self.overview_md.strip(), ""]
        from banks_rag.application.reporting.verify import SYNTHESIS_KEY
        for flag in self._residual_flags(SYNTHESIS_KEY):
            lines.append(f"> ⚠ Verificación: {flag}")
        if self._residual_flags(SYNTHESIS_KEY):
            lines.append("")
        for i, s in enumerate(self.sections, 1):
            meta = [f"Dataset `{s.dataset_id}`"]
            if s.unit:
                meta.append(s.unit)
            if s.segment:
                meta.append(s.segment)
            if s.last_date:
                meta.append(f"datos hasta {s.last_date}")
            lines.append(f"## {i}. {s.name}")
            lines.append("")
            lines.append(f"*{' · '.join(meta)}*")
            lines.append("")
            lines.append(s.paragraph)
            for flag in self._residual_flags(s.dataset_id):
                lines.append("")
                lines.append(f"> ⚠ Verificación: {flag}")
            lines.append("")
        counts = self.status_counts()
        lines.append("## Referencias y metodología")
        lines.append("")
        lines.append(
            f"- Datasets analizados: {len(self.sections)} "
            f"(ok: {counts.get('ok', 0)}, sin datos: {counts.get('no_data', 0)}, "
            f"con error: {counts.get('error', 0)})"
        )
        if self.missing_ids:
            lines.append(f"- No hallados en el catálogo: {', '.join(self.missing_ids)}")
        lines.append(
            f"- Ventanas de análisis: {', '.join(self.windows)} "
            "(ancladas a la última fecha disponible de cada dataset)"
        )
        if self.verification is not None:
            lines.append(f"- Verificación de afirmaciones: {self.verification.summary()}")
        lines.append(f"- Generado: {self.generated_at}")
        lines.append("- Fuente: sql_catalog/parquet_catalog.yaml")
        lines.append("")
        return "\n".join(lines)


async def generate_parquet_report(
    llm,
    *,
    segment: str | None = None,
    dataset_ids: Sequence[str] | None = None,
    query: str | None = None,
    windows: Sequence[str] = ("7d", "30d"),
    top_k: int = 12,
    concurrency: int = 1,
    map_max_tokens: int = 32768,
    synthesis_max_tokens: int = 8192,
    think: bool = False,
    verify: bool = True,
    catalog_path: Path | str | None = None,
    entries: list[ParquetDataset] | None = None,
    parquet_dir: Path | None = None,
) -> ParquetReport:
    """Genera el informe descriptivo completo (map-reduce SIN tools + verificación).

    ``entries``/``catalog_path``/``parquet_dir`` son inyectables para tests; por
    defecto se carga ``sql_catalog/parquet_catalog.yaml`` y su ``parquet_dir``.
    ``verify`` corre el verificador post-síntesis (determinista + crítico LLM):
    autocorrige direcciones, regenera párrafos con cifras sin sustento y marca lo
    residual. Best-effort: nunca tumba la generación.
    """
    window_specs = _window_specs(windows)  # valida temprano, antes de cargar nada

    if entries is None:
        entries = load_parquet_catalog(catalog_path)
    selection = select_datasets(
        entries, segment=segment, dataset_ids=dataset_ids, query=query, top_k=top_k,
    )
    if parquet_dir is None:
        parquet_dir = get_parquet_dir(catalog_path)

    # Corte semanal común + qué datasets reciben el sub-párrafo MENSUAL (una vez por
    # sección). Ambos derivan del spec curado de la familia (segment), así el corte
    # coincide EXACTAMENTE con el del proceso de gráficos (build_curated_report).
    weekly_asof: str | None = None
    monthly_ids: set[str] = set()
    spec_ids: set[str] = set()
    anchor_ids: set[str] = set()
    try:
        from banks_rag.application.reporting.curated_report import (
            _spec_source_ids,
            compute_weekly_cutoff,
            monthly_section_source_ids,
            weekly_anchor_source_ids,
        )
        from banks_rag.application.reporting.specs import get_spec

        spec = get_spec(segment) if segment else None
        if spec is not None:
            weekly_asof = compute_weekly_cutoff(spec, entries, parquet_dir)
            monthly_ids = monthly_section_source_ids(spec)
            spec_ids = set(_spec_source_ids(spec))
            anchor_ids = weekly_anchor_source_ids(spec)
        else:
            from banks_rag.application.reporting.parquet_facts import weekly_cutoff

            weekly_asof = weekly_cutoff(list(selection.datasets), parquet_dir)
            anchor_ids = {d.id for d in selection.datasets}
    except Exception:
        log.exception("corte semanal/secciones: cálculo falló; se sigue sin corte")
    if weekly_asof:
        log.info("Informe parquet: corte semanal común = %s", weekly_asof)

    def _include_monthly(ds_id: str) -> bool:
        # Con spec: el sub-párrafo mensual solo en el primer dataset de cada sección;
        # los datasets fuera del spec (standalone) lo conservan; sin spec, todos
        # (comportamiento previo).
        if not spec_ids:
            return True
        return ds_id not in spec_ids or ds_id in monthly_ids

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(ds: ParquetDataset) -> DatasetSection:
        async with semaphore:
            return await _describe_dataset(
                ds,
                llm=llm,
                window_specs=window_specs,
                parquet_dir=parquet_dir,
                map_max_tokens=map_max_tokens,
                think=think,
                # El corte solo ancla la variación semanal de las secciones de anclaje
                # (Flujos+DCV en ffmm); el resto usa el máximo de su propio parquet.
                weekly_asof=weekly_asof if ds.id in anchor_ids else None,
                include_monthly=_include_monthly(ds.id),
            )

    log.info(
        "Informe parquet (%s): %d datasets, ventanas %s, concurrencia %d",
        selection.selector_desc, len(selection.datasets), list(windows), concurrency,
    )
    sections = list(await asyncio.gather(*(_bounded(ds) for ds in selection.datasets)))

    redundant_ids = _redundant_dataset_ids({d.id for d in selection.datasets})
    overview = await _synthesize_overview(
        sections, llm=llm, selector_desc=selection.selector_desc,
        max_tokens=synthesis_max_tokens, redundant_ids=redundant_ids,
    )

    report = ParquetReport(
        title=f"Informe descriptivo — {selection.selector_desc}",
        selector_label=selection.selector_label,
        selector_desc=selection.selector_desc,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        windows=tuple(windows),
        overview_md=overview,
        sections=sections,
        missing_ids=selection.missing_ids,
    )

    if verify:
        from banks_rag.application.reporting.verify import verify_report

        # Snapshot del texto ANTES de verificar (para comparar antes/después).
        report.raw_overview_md = report.overview_md
        for s in report.sections:
            s.raw_paragraph = s.paragraph

        async def _regen(section: DatasetSection, issues: list) -> str:
            return await _regenerate_paragraph(
                section, issues, llm=llm, map_max_tokens=map_max_tokens, think=think,
            )

        report.verification = await verify_report(
            report, llm=llm, regenerate=_regen, facts_digest=_facts_digest(sections),
            redundant_ids=redundant_ids,
        )

    return report
