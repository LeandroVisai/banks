"""Serialización de figuras: JSON por sección (artefacto en disco) y el
payload COMPACTO que se embebe en el informe.

El payload compacto es lo que hace que el HTML pese ~4x menos sin cambiar
ni un gráfico. Tres codificaciones, todas revertidas por `_hidratar()` en
templates/chart.js antes del `Plotly.newPlot`:

    {"$c": [off, n]}      tramo contiguo del calendario maestro
    {"$i": "<b64 i32>"}   índices al calendario maestro (series con huecos)
    {"$f"/"$d": "<b64>"}  array numérico en float32 / float64

El calendario maestro (`cal`) son las fechas únicas de TODO el informe,
ordenadas y guardadas UNA sola vez. Medido sobre las 40 figuras: los ejes X
pasan de 3,25 MB (59% del payload) a 79 KB, porque 122 traces repetían el
mismo vector de fechas como "2019-07-30T00:00:00" — 21 bytes por punto, por
serie. Es la misma idea del registro de ejes de neo, aplicada a la
serialización de Plotly en vez de a un motor de dibujo propio.

Un efecto lateral que importa: la codificación la controla este archivo, no
plotly.py. Antes el HTML embebía el {dtype, bdata} que emite plotly.py 6.x,
que plotly.js 2.x NO sabe leer — con el CDN apuntando a 2.35.2 esos gráficos
salían en blanco. Ahora lo que viaja lo decodifica chart.js, así que la
versión de plotly.js del CDN deja de ser una dependencia oculta.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go

# Arrays más cortos que esto se dejan como JSON plano: el base64 no compensa
# y se pierde legibilidad al depurar los .json de outputs/figures/.
_MIN_LEN = 24

# Claves de trace cuyo valor es un array numérico largo. `x` se trata aparte
# (puede ser fecha, número o categoría) y `z` queda fuera a propósito: es una
# matriz 2D y los dos heatmaps del informe pesan menos de 20 KB juntos.
_NUM_KEYS = ("y", "open", "high", "low", "close", "customdata")

# "2019-07-30", "2019-07-30T00:00:00" y "2019-07-30T00:00:00.000000" son la
# misma fecha diaria: plotly.py emite las tres formas según de qué columna
# venga la serie (los gráficos S/R salían con microsegundos y por eso no se
# comprimían). Cualquier hora distinta de medianoche NO matchea y la serie
# se deja tal cual — es el caso de clp_intradia.
_FECHA_DIARIA_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[T ]00:00:00(?:\.0+)?)?$")


def serialize_figures(figs: dict[str, go.Figure]) -> dict[str, dict]:
    return {fig_id: json.loads(fig.to_json()) for fig_id, fig in figs.items()}


# ═════════════════════════════════════════════════════════════════════
# CODIFICACIÓN COMPACTA
# ═════════════════════════════════════════════════════════════════════

def _como_fechas(v) -> list[str] | None:
    """La lista de fechas 'YYYY-MM-DD' si `v` es un eje de fechas diarias.

    Devuelve None ante cualquier valor con hora real (el eje de
    `clp_intradia` son marcas intradía): truncarlas a día colapsaría todos
    los puntos de la jornada en uno solo.
    """
    if not isinstance(v, list) or len(v) < _MIN_LEN:
        return None
    fechas = []
    for s in v:
        m = _FECHA_DIARIA_RE.match(s) if isinstance(s, str) else None
        if m is None:
            return None
        fechas.append(m.group(1))
    return fechas


def _como_array(v) -> np.ndarray | None:
    """El array float64 detrás de `v`, venga como lista plana o como el
    {dtype, bdata} que ya emite plotly.py 6.x."""
    if isinstance(v, dict) and "bdata" in v and "dtype" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]).astype("f8")
    if not isinstance(v, list) or len(v) < _MIN_LEN:
        return None
    for e in v:
        if e is not None and (isinstance(e, bool) or not isinstance(e, (int, float))):
            return None
    return np.array([np.nan if e is None else e for e in v], dtype="f8")


def _codificar_numeros(a: np.ndarray) -> dict:
    """float32 salvo que pierda precisión relevante, y ahí float64.

    float32 da ~7 dígitos significativos y ahorra la mitad del espacio, de
    sobra para precios, índices y porcentajes. Pero algunas series son
    montos grandes (inventarios en toneladas, posiciones nocionales) donde
    los enteros dejan de ser exactos pasados los 16,7 millones — así que en
    vez de suponerlo se mide: se codifica, se decodifica, y si el error
    relativo supera 1e-6 se cae a float64.
    """
    f4 = a.astype("f4")
    finitos = np.isfinite(a)
    if finitos.any():
        escala = np.maximum(np.abs(a[finitos]), 1e-12)
        if np.max(np.abs(f4[finitos] - a[finitos]) / escala) > 1e-6:
            return {"$d": base64.b64encode(a.tobytes()).decode("ascii")}
    return {"$f": base64.b64encode(f4.tobytes()).decode("ascii")}


def _barrer_bdata(obj):
    """Reencoda cualquier {dtype, bdata} que haya quedado suelto en el trace.

    _NUM_KEYS cubre las claves grandes conocidas, pero plotly.py serializa
    así CUALQUIER array numérico — hoy además `text` (las etiquetas de las
    barras de carry trade), mañana el que traiga un gráfico nuevo. Sin esta
    pasada, ese resto obliga al CDN a ser plotly.js 3.x, que es justo la
    dependencia oculta que este módulo existe para eliminar.

    Los arrays 2D (los que traen "shape") se dejan intactos: aplanarlos
    rompería el heatmap, y hoy no hay ninguno.
    """
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            if "shape" in obj:
                return obj
            return _codificar_numeros(
                np.frombuffer(base64.b64decode(obj["bdata"]), dtype=obj["dtype"]).astype("f8")
            )
        return {k: _barrer_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_barrer_bdata(v) for v in obj]
    return obj


def _codificar_fechas(fechas: list[str], pos: dict[str, int]) -> dict:
    idx = [pos[f] for f in fechas]
    arranque = idx[0]
    if idx == list(range(arranque, arranque + len(idx))):
        return {"$c": [arranque, len(idx)]}
    return {"$i": base64.b64encode(np.array(idx, dtype="i4").tobytes()).decode("ascii")}


def compact_payload(
    figures_serialized: dict[str, dict],
    *,
    rebase_ids: Iterable[str] = (),
    nav_groups: list | None = None,
) -> dict:
    """{'cal': [...fechas únicas...], 'figs': {...}} — lo que se embebe.

    `nav_groups` deja fuera las figuras que no tienen slot. Una figura sin
    slot no se dibuja en ninguna parte, pero sin este filtro sus datos
    viajaban igual dentro del HTML: peso puro. Pasa cuando un builder se
    agrega a figures.py y se olvida el slot en layout.py, y también en
    testlocal, que comparte layout.py con el banco pero construye un
    conjunto distinto de figuras.

    `rebase_ids` son los gráficos base 100 (figures.REBASE_FIG_IDS). En esos
    la `y` no se manda: es el índice base 100 de `customdata`, que chart.js ya
    recalcula en cada cambio de ventana y siembra antes del primer dibujo.
    Mandar las dos era mandar la misma serie dos veces — son los cuatro
    gráficos de monedas, los más pesados del informe (10 series cada uno).

    No muta `figures_serialized`: los .json de outputs/figures/ se siguen
    escribiendo en formato Plotly normal, legibles y recargables con
    `load_figures()`.
    """
    rebase = set(rebase_ids)

    if nav_groups is not None:
        con_slot = {
            fig_id
            for ids in figures_by_section(nav_groups, figures_serialized).values()
            for fig_id in ids
        }
        figures_serialized = {k: v for k, v in figures_serialized.items() if k in con_slot}
    calendario: set[str] = set()
    for fig in figures_serialized.values():
        for trace in fig.get("data", []):
            fechas = _como_fechas(trace.get("x"))
            if fechas:
                calendario.update(fechas)

    cal = sorted(calendario)
    pos = {f: i for i, f in enumerate(cal)}

    figs: dict[str, dict] = {}
    for fig_id, fig in figures_serialized.items():
        traces = []
        for trace in fig.get("data", []):
            t = dict(trace)
            if fig_id in rebase and t.get("customdata") is not None:
                t.pop("y", None)
            fechas = _como_fechas(t.get("x"))
            if fechas:
                t["x"] = _codificar_fechas(fechas, pos)
            else:
                arr = _como_array(t.get("x"))
                if arr is not None:
                    t["x"] = _codificar_numeros(arr)
            for clave in _NUM_KEYS:
                arr = _como_array(t.get(clave))
                if arr is not None:
                    t[clave] = _codificar_numeros(arr)
            traces.append(_barrer_bdata(t))
        figs[fig_id] = {**fig, "data": traces}

    return {"cal": cal, "figs": figs}


# ═════════════════════════════════════════════════════════════════════
# ARTEFACTOS EN DISCO
# ═════════════════════════════════════════════════════════════════════

def _normalize_bdata(obj):
    """Decodifica recursivamente los {'dtype','bdata'} de plotly.py a listas
    planas — lo necesita `load_figures()` para reconstruir go.Figure."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            return np.frombuffer(base64.b64decode(obj["bdata"]), dtype=obj["dtype"]).tolist()
        return {k: _normalize_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_bdata(v) for v in obj]
    return obj


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_figures(figures_dir: Path, fig_ids: Iterable[str] | None = None) -> dict[str, go.Figure]:
    """Reconstruye go.Figure desde los JSON por sección de write_section_files()."""
    wanted = set(fig_ids) if fig_ids is not None else None
    figs: dict[str, go.Figure] = {}
    for section_file in figures_dir.glob("*.json"):
        section = read_json(section_file)
        for fig_id, fig_dict in section.items():
            if wanted is not None and fig_id not in wanted:
                continue
            figs[fig_id] = go.Figure(_normalize_bdata(fig_dict))
    return figs


def figures_by_section(nav_groups: list, available_fig_ids) -> dict[str, list[str]]:
    """Agrupa fig_ids por item["id"] de nav_groups (sin mapeo hardcodeado aparte)."""
    available = set(available_fig_ids)
    sections: dict[str, list[str]] = {}
    for group in nav_groups:
        for item in group["items"]:
            fig_ids = [
                slot["fig_id"] for slot in item["slots"]
                if slot.get("fig_id") in available
            ]
            if fig_ids:
                sections[item["id"]] = fig_ids
    return sections


def write_section_files(nav_groups: list, figures_serialized: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = figures_by_section(nav_groups, figures_serialized.keys())

    written = []
    for section_id, fig_ids in sections.items():
        out_file = out_dir / f"{section_id}.json"
        payload = {fig_id: figures_serialized[fig_id] for fig_id in fig_ids}
        out_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        written.append(out_file)
    return written
