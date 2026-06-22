"""Loader del catálogo de parquets analíticos (sql_catalog/parquet_catalog.yaml).

Carga la metadata de los ~144 datasets en data_pipeline/parquet/ y expone
funciones de búsqueda por keyword para que el agente pueda descubrir qué
dataset usar (``discover_query``) antes de leerlo con ``execute_query`` —
la SQL la arma siempre la tool (``_parquet_query``), nunca el LLM.

Funciones públicas:
    load_parquet_catalog(path)        → list[ParquetDataset]
    search_datasets(entries, query)   → list[ParquetDataset]
    get_dataset(entries, id)          → ParquetDataset | None
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from banks_rag.config.paths import ROOT

_STOP_WORDS = frozenset({
    "de", "del", "la", "el", "los", "las", "un", "una", "en", "con",
    "por", "para", "que", "y", "o", "a", "al", "se", "es", "son",
    "fue", "ser", "como", "más", "pero", "si", "the", "of", "in",
    "for", "and", "or", "to", "is",
})


@dataclass
class ColumnSpec:
    name: str
    type: str
    values: list[str] = field(default_factory=list)

    def hint(self) -> str:
        base = f"{self.name} ({self.type})"
        if self.values:
            sample = ", ".join(str(v) for v in self.values[:8])
            suffix = "…" if len(self.values) > 8 else ""
            return f"{base} — values: [{sample}{suffix}]"
        return base


@dataclass
class ParquetDataset:
    id: str
    file: str
    name: str
    description: str
    segment: str
    unit: str
    date_range: list[str] | None
    columns: list[ColumnSpec]
    # Tipo de gráfico canónico del tablero para este dataset (diccionario de
    # parquets): "line", "stacked_area", "grouped_bar", "market_monitor_table",
    # etc. Las tools lo usan como default al graficar; el frontend lo reduce a
    # una familia renderizable vía domain/agent/chart_types.chart_family.
    chart_type: str = "line"
    # Factor de corrección de escala del dato crudo del parquet a la unidad
    # declarada en ``unit`` (1.0 = el parquet ya está en esa unidad). Algunos
    # parquets vienen en otra escala que su unidad nominal (p.ej. ``Monto_USD``
    # en MILES de USD declarado como "Mill US$" → 0.001; un retorno en FRACCIÓN
    # declarado como "%" → 100). Lo aplican el informe (``compute_facts`` para el
    # texto y el builder curado para el gráfico), así prosa y gráfico coinciden.
    value_scale: float = 1.0
    # Naturaleza del valor de la serie, para que el informe describa la dirección
    # correctamente: "flow" (entrada/salida — el SIGNO del flujo del período manda;
    # la variación por ventana se SUMA, no se resta punto a punto), "stock"/"level"
    # (un nivel — la dirección es el signo de la variación), "return" (un índice de
    # retorno), "rate" (una tasa/%). Vacío = desconocido → se cae a la heurística
    # de palabras clave (``_FLOW_KEYWORDS``). Robustece la detección de flujos en
    # datasets cuyo nombre no contiene una palabra-clave (afp/nr).
    value_kind: str = ""

    def parquet_path(self, parquet_dir: Path) -> Path:
        return parquet_dir / self.file

    def schema_hint(self) -> str:
        """Representación compacta de schema para prompt del LLM."""
        cols = " | ".join(c.hint() for c in self.columns)
        dr = f" | dates: {self.date_range[0]} → {self.date_range[1]}" if self.date_range else ""
        return f"[{self.id}] {self.name} — {cols}{dr}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "name": self.name,
            "description": self.description.strip(),
            "segment": self.segment,
            "unit": self.unit,
            "date_range": self.date_range,
            "chart_type": self.chart_type,
            **({"value_scale": self.value_scale} if self.value_scale != 1.0 else {}),
            **({"value_kind": self.value_kind} if self.value_kind else {}),
            "columns": [
                {"name": c.name, "type": c.type, **({"values": c.values} if c.values else {})}
                for c in self.columns
            ],
        }


def load_parquet_catalog(path: Path | str | None = None) -> list[ParquetDataset]:
    """Carga el catálogo desde YAML. Path por defecto: sql_catalog/parquet_catalog.yaml."""
    if path is None:
        path = ROOT / "sql_catalog" / "parquet_catalog.yaml"
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return [_parse_dataset(d) for d in data.get("datasets", [])]


def get_parquet_dir(path: Path | str | None = None) -> Path:
    """Resuelve la ruta absoluta al directorio de parquets."""
    if path is None:
        path = ROOT / "sql_catalog" / "parquet_catalog.yaml"
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    rel = data.get("parquet_dir", "data_pipeline/parquet")
    return (ROOT / rel).resolve()


# Peso de un hint de dataset (substring de id que el dominio asocia a la query).
# Alto frente al overlap léxico (1/token) para que p.ej. "DV01 fondos de
# pensiones" rankee ``dv01_spc_afp`` por encima de un dataset que solo comparte
# tokens genéricos.
_HINT_WEIGHT = 5


def search_datasets(
    entries: list[ParquetDataset],
    query: str,
    top_k: int = 8,
    segment: str | None = None,
    extra_scores: dict[str, float] | None = None,
) -> list[ParquetDataset]:
    """Descubrimiento de datasets: overlap léxico + alias del dominio.

    Pasos:
      1. Expande la query con sinónimos del dominio (``expand_query``): así
         "pensiones" alcanza datasets que solo dicen "afp".
      2. Overlap de tokens sobre ``id+name+description+segment+unit``.
      3. Boost por ``dataset_hints``: si la id del dataset contiene un hint que
         el dominio asocia a la query (ej. ``dv01``, ``_nr``, ``afp``).
      4. ``extra_scores`` (opcional): puntajes semánticos precalculados por id
         (catalog_index con embeddings) que se suman al score léxico.
    """
    # Import local para evitar acoplar el loader al paquete de dominio en import-time.
    from banks_rag.domain_knowledge.financial_aliases import (
        dataset_hints as _dataset_hints,
    )
    from banks_rag.domain_knowledge.financial_aliases import (
        expand_query as _expand_query,
    )

    if segment:
        entries = [e for e in entries if e.segment == segment]

    query_tokens = _tokenize(_expand_query(query))
    hints = _dataset_hints(query)
    extra_scores = extra_scores or {}

    # Un hint boostea si es ESPECÍFICO (substring tipo id, con "_": p.ej.
    # ``spread_swap``, ``dv01_spc_afp``) o si su token está en la query
    # expandida. Así los hints genéricos de un concepto amplio (``btp``,
    # ``ois``, ``spc`` de renta fija) no inflan TODOS los datasets de curva
    # cuando la pregunta es específica (p.ej. "spread swap-OIS").
    effective_hints = {
        h.lower() for h in hints
        if ("_" in h) or (h.lower() in query_tokens)
    }

    scored: list[tuple[float, int, ParquetDataset]] = []
    for idx, entry in enumerate(entries):
        # Los matches en id+name pesan doble: el nombre identifica el dataset,
        # mientras la descripción suele ENUMERAR conceptos vecinos (p.ej. las
        # familias spreads_* listan "DAP, prime, TADO, SOFR" y empataban con el
        # dataset cuyo nombre es exactamente "Spread DAP-SOFR por plazo").
        id_name_tokens = _tokenize(f"{entry.id} {entry.name}")
        text = f"{entry.id} {entry.name} {entry.description} {entry.segment} {entry.unit}"
        score = float(len(_tokenize(text) & query_tokens))
        score += len(id_name_tokens & query_tokens)
        if effective_hints:
            # Boost de hint a lo sumo UNA vez por dataset: un id que matchea
            # tanto el prefijo genérico (``spreads_``) como su hint específico
            # (``spreads_12m``) no debe duplicar el boost y enterrar al dataset
            # cuyo nombre responde literalmente la pregunta (``spreads_dap`` =
            # "Spread DAP-SOFR por plazo").
            id_lower = entry.id.lower()
            if any(h in id_lower for h in effective_hints):
                score += _HINT_WEIGHT
        score += extra_scores.get(entry.id, 0.0)
        # idx como tie-break estable (preserva el orden del catálogo).
        scored.append((score, -idx, entry))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [e for _, _, e in scored[: min(top_k, 20)]]


def get_dataset(entries: list[ParquetDataset], dataset_id: str) -> ParquetDataset | None:
    """Busca un dataset por id exacto."""
    for e in entries:
        if e.id == dataset_id:
            return e
    return None


# ── Internos ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def _parse_dataset(raw: dict) -> ParquetDataset:
    columns = [
        ColumnSpec(
            name=col["name"],
            type=col.get("type", "DOUBLE"),
            values=col.get("values", []),
        )
        for col in raw.get("columns", [])
    ]
    return ParquetDataset(
        id=raw["id"],
        file=raw["file"],
        name=raw["name"],
        description=raw.get("description", ""),
        segment=raw.get("segment", ""),
        unit=raw.get("unit", ""),
        date_range=raw.get("date_range"),
        columns=columns,
        chart_type=raw.get("chart_type") or "line",
        value_scale=float(raw.get("value_scale", 1.0) or 1.0),
        value_kind=str(raw.get("value_kind", "") or "").strip().lower(),
    )
