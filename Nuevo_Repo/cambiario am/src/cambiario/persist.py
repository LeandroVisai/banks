from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go


def serialize_figures(figs: dict[str, go.Figure]) -> dict[str, dict]:
    return {fig_id: json.loads(fig.to_json()) for fig_id, fig in figs.items()}


def _normalize_bdata(obj):
    """Decodifica recursivamente cualquier {'dtype':..., 'bdata':...} de
    Plotly a una lista plana de números — ver nota del módulo."""
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
    """Reconstruye go.Figure a partir de los JSON por sección escritos por
    write_section_files(). Si `fig_ids` se pasa, sólo reconstruye esos
    (evita deserializar los ~40 gráficos cuando sólo se necesitan unos pocos,
    como en el correo condensado)."""
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
