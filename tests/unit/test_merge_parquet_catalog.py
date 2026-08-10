"""Fusión del diccionario auto-generado con nuestro catálogo curado.

``sql_catalog/diccionario_parquets_IA.yaml`` lo reescribe el pipeline cada vez
que cambia un parquet en las bases: es la verdad del esquema, pero no sabe nada
de lo que agregamos a mano (``cam_*``, ``value_scale``/``value_kind``/``facts``,
la de-duplicación de ids). Estos tests son el seguro de que volver a fusionar
NUNCA borre esa capa curada.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """Importa el script (vive en scripts/, fuera del paquete)."""
    path = _ROOT / "scripts" / "merge_parquet_catalog.py"
    spec = importlib.util.spec_from_file_location("merge_parquet_catalog", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["merge_parquet_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load_module()


def _ds(id_, file_=None, segment="ffmm", chart="line", cols=None, **extra):
    return {
        "id": id_,
        "file": file_ or f"{id_}.parquet",
        "chart_type": chart,
        "name": id_,
        "description": "desc",
        "segment": segment,
        "unit": "US$ Mill.",
        "date_range": ["2020-01-01", "2026-01-01"],
        "columns": cols or [{"name": "Fecha", "type": "TIMESTAMP"}],
        **extra,
    }


@pytest.mark.unit
class TestOverlayNoSePierde:
    """Lo nuestro (``value_scale``/``value_kind``/``facts``) se reinyecta."""

    def test_campos_curados_sobreviven(self, tmp_path: Path):
        dicc = {"datasets": [_ds("flujos_ffmm", chart="grouped_bar")]}
        ours = {"datasets": [_ds("flujos_ffmm", value_scale=0.001, value_kind="flow",
                                 facts={"category": "tipo_fondo"})]}
        merged, _ = mp.merge(dicc, ours, tmp_path)
        assert len(merged) == 1
        d = merged[0]
        assert d["value_scale"] == 0.001
        assert d["value_kind"] == "flow"
        assert d["facts"] == {"category": "tipo_fondo"}
        # ...y el esquema/chart_type sí se actualizan desde el diccionario.
        assert d["chart_type"] == "grouped_bar"

    def test_datasets_solo_nuestros_se_conservan_intactos(self, tmp_path: Path):
        """cam_* no existe en las bases del tablero: lo produce nuestro script."""
        mio = _ds("cam_clp_ohlc", chart="candlestick", segment="cambiario")
        merged, rep = mp.merge({"datasets": []}, {"datasets": [mio]}, tmp_path)
        assert merged == [mio]
        assert rep["conservados"] == ["cam_clp_ohlc"]


@pytest.mark.unit
class TestIdentidadYDeduplicacion:
    def test_id_con_sufijo_se_recupera_por_file_y_segment(self, tmp_path: Path):
        """El diccionario emite el id genérico; nosotros ya lo habíamos
        sufijado por segmento para que el loader no lo pise."""
        dicc = {"datasets": [_ds("posicion_spot_derivados", segment="ffmm")]}
        ours = {"datasets": [_ds("posicion_spot_derivados_ffmm",
                                 file_="posicion_spot_derivados.parquet", segment="ffmm")]}
        merged, _ = mp.merge(dicc, ours, tmp_path)
        assert [d["id"] for d in merged] == ["posicion_spot_derivados_ffmm"]

    def test_ids_duplicados_del_diccionario_no_se_emiten_dos_veces(self, tmp_path: Path):
        """Mismo id+file+segment repetido: el loader indexa por id, así que
        emitir los dos deja uno pisando al otro."""
        dup = _ds("cambiario_afp", segment="afp")
        otro = dict(dup, chart_type="stacked_bar")
        merged, rep = mp.merge({"datasets": [dup, otro]}, {"datasets": []}, tmp_path)
        assert [d["id"] for d in merged] == ["cambiario_afp"]
        assert rep["colisiones"]

    def test_catalogo_fusionado_no_repite_ids(self, tmp_path: Path):
        dicc = {"datasets": [_ds("a"), _ds("a"), _ds("b")]}
        merged, _ = mp.merge(dicc, {"datasets": [_ds("b"), _ds("c")]}, tmp_path)
        ids = [d["id"] for d in merged]
        assert len(ids) == len(set(ids))


@pytest.mark.unit
class TestSegmentCanonico:
    def test_etiqueta_de_seccion_no_pisa_el_segmento(self, tmp_path: Path):
        """El diccionario a veces trae la sección del tablero ("Stocks") donde
        ``parquet_report.py --segment ffmm`` necesita el segmento real."""
        dicc = {"datasets": [_ds("var_cartera_mensual_ffmm", segment="Stocks")]}
        ours = {"datasets": [_ds("var_cartera_mensual_ffmm", segment="ffmm")]}
        merged, rep = mp.merge(dicc, ours, tmp_path)
        assert merged[0]["segment"] == "ffmm"
        assert rep["segment_conservado"]

    def test_segmento_canonico_del_diccionario_se_respeta(self, tmp_path: Path):
        dicc = {"datasets": [_ds("x", segment="afp")]}
        ours = {"datasets": [_ds("x", segment="ffmm"), _ds("y", segment="afp")]}
        merged, _ = mp.merge(dicc, ours, tmp_path)
        assert merged[0]["segment"] == "afp"


@pytest.mark.unit
class TestColumnasPayload:
    def test_nunca_se_emite_enum_en_columnas_payload(self, tmp_path: Path):
        """``_sparkline_json`` lleva un blob por fila, no una categoría: el
        enum infla el catálogo y encima rompe el YAML."""
        cols = [{"name": "Activo", "type": "VARCHAR", "values": ["Chile"]},
                {"name": "_sparkline_json", "type": "VARCHAR", "values": ['[{"d": 1}]']},
                {"name": "_title_override", "type": "VARCHAR", "values": ["Bolsas · hoy"]}]
        merged, _ = mp.merge({"datasets": [_ds("cm_bolsas", cols=cols)]},
                             {"datasets": []}, tmp_path)
        out = {c["name"]: c for c in merged[0]["columns"]}
        assert out["Activo"]["values"] == ["Chile"]
        assert "values" not in out["_sparkline_json"]
        assert "values" not in out["_title_override"]

    def test_load_dicc_sanea_el_enum_json_roto(self, tmp_path: Path):
        """El generador dumpea el enum de payload con comillas sin escapar y
        deja el YAML inválido; hay que poder cargarlo igual."""
        p = tmp_path / "d.yaml"
        p.write_text(
            'parquet_dir: x\n'
            'datasets:\n'
            '  - id: cm_fx\n'
            '    file: cm_fx.parquet\n'
            '    chart_type: "market_monitor_table"\n'
            '    name: "FX"\n'
            '    description: d\n'
            '    segment: color_mercados\n'
            '    unit: "N/A"\n'
            '    date_range: []\n'
            '    columns:\n'
            '      - {name: Activo, type: VARCHAR}\n'
            '      - {name: _sparkline_json, type: VARCHAR, '
            'values: ["[{"d": "2021-08-06", "v": 1.0}]"]}\n',
            encoding="utf-8")
        with pytest.raises(yaml.YAMLError):          # crudo: no parsea
            yaml.safe_load(p.read_text(encoding="utf-8"))
        data = mp.load_dicc(p)                        # saneado: sí
        cols = data["datasets"][0]["columns"]
        assert [c["name"] for c in cols] == ["Activo", "_sparkline_json"]
        assert "values" not in cols[1]


@pytest.mark.unit
class TestSerializacion:
    def test_round_trip_con_acentos_comillas_y_overlay(self):
        ds = [_ds('raro', chart="line", value_scale=0.001, value_kind="flow",
                  facts={"filter": {"Institucion": "Total"}},
                  cols=[{"name": "Año", "type": "VARCHAR", "values": ["2026", "> 2036"]}])]
        ds[0]["name"] = 'Posición "neta" · señal'
        ds[0]["unit"] = "US$ Mill."
        text = mp.dump_yaml("data_pipeline/parquet", ds)
        back = yaml.safe_load(text)["datasets"][0]
        assert back["name"] == 'Posición "neta" · señal'
        assert back["value_scale"] == 0.001
        assert back["facts"] == {"filter": {"Institucion": "Total"}}
        assert back["columns"][0]["values"] == ["2026", "> 2036"]


@pytest.mark.unit
class TestCatalogoRealConservaLoCurado:
    """Guard sobre el catálogo de verdad: si alguien lo pisa con el
    diccionario crudo, estos números se van a cero y el test lo caza."""

    def test_cam_y_overlay_siguen_en_el_catalogo(self):
        from banks_rag.infrastructure.sql.parquet_catalog_loader import load_parquet_catalog

        entries = load_parquet_catalog()
        assert sum(1 for d in entries if d.id.startswith("cam_")) == 41
        assert sum(1 for d in entries if d.value_kind) >= 68
        assert sum(1 for d in entries if d.value_scale != 1.0) >= 7
        assert sum(1 for d in entries if d.facts_hints) >= 8

    def test_no_hay_ids_duplicados(self):
        from banks_rag.infrastructure.sql.parquet_catalog_loader import load_parquet_catalog

        ids = [d.id for d in load_parquet_catalog()]
        assert len(ids) == len(set(ids))
