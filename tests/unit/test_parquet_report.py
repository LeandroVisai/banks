"""Unit tests del informe descriptivo de parquets (sin DuckDB ni LLM real).

El MAP ya NO usa tools: Python calcula los hechos (``compute_facts``) y el LLM
hace UNA llamada por dataset. Los tests monkeypatchean ``compute_facts`` en el
módulo del informe para no depender del filesystem; el cálculo real se prueba
en test_parquet_facts.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import banks_rag.application.reporting.parquet_report as pr
from banks_rag.application.reporting.parquet_report import (
    _EMPTY_OVERVIEW,
    _NO_DATA_PARAGRAPH,
    DatasetSection,
    ParquetReport,
    _clean_paragraph,
    _parse_window,
    _synthesize_overview,
    _window_specs,
    generate_parquet_report,
    select_datasets,
)
from banks_rag.domain.agent import GenerationResult
from banks_rag.infrastructure.sql.parquet_catalog_loader import ColumnSpec, ParquetDataset


def _ds(dataset_id, *, segment="ffmm", file=None, unit="US$ Mill.", name=None, description="") -> ParquetDataset:
    return ParquetDataset(
        id=dataset_id, file=file or f"{dataset_id}.parquet", name=name or dataset_id,
        description=description, segment=segment, unit=unit,
        date_range=["2019-01-01", "2026-04-30"],
        columns=[ColumnSpec(name="Fecha", type="TIMESTAMP"), ColumnSpec(name="valor", type="DOUBLE")],
        chart_type="line",
    )


_FACTS_OK = {
    "shape": "timeseries_single", "dataset_id": "x", "unit": "US$ Mill.",
    "n_rows": 100, "date_col": "Fecha", "last_date": "2026-04-30",
    "value_cols": ["valor"], "category_cols": [],
    "windows_variation": [{"label": "última semana", "desde": "2026-04-23", "hasta": "2026-04-30",
                           "variacion": {"fecha_inicio_obs": "2026-04-23", "valor_inicio": 1.0,
                                         "fecha_fin_obs": "2026-04-30", "valor_fin": 2.0,
                                         "cambio_absoluto": 1.0, "variacion_pct": 100.0,
                                         "cambio_bps": 100.0, "minimo": 1.0, "maximo": 2.0,
                                         "n_observaciones": 5}}],
    "estadisticas": {"ultimo_valor": 2.0, "ultima_fecha": "2026-04-30", "media": 1.5,
                     "desviacion_estandar": 0.3, "minimo": 1.0, "maximo": 2.0,
                     "percentil_ultimo_valor": 95.0, "n_observaciones": 100},
    "anomalia": None,
}


@dataclass
class _MockLLM:
    responses: list[GenerationResult] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    async def generate(self, messages, *, tools=None, **kwargs):
        self.calls.append({
            "messages": [dict(m) for m in messages], "tools": tools,
            "temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"),
        })
        if not self.responses:
            return GenerationResult(text="default", n_tokens=5)
        return self.responses.pop(0)

    def count_text_tokens(self, text: str) -> int:
        return len(text) // 4


# ─────────────────────────────────────────────────────────────────────────────
# select_datasets
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSelectDatasets:
    def _catalog(self):
        return [
            _ds("flujos_ffmm", name="Flujos semanales por fondo",
                description="Flujos semanales de los fondos mutuos por tipo de fondo"),
            _ds("duracion_ffmm", name="Duración por tipo de fondo",
                description="Duración de los fondos mutuos", unit="Años"),
            _ds("lcr", segment="liquidez_bancaria", name="LCR",
                description="Liquidity coverage ratio del sistema bancario", unit="%"),
        ]

    def test_segment_canonical(self):
        sel = select_datasets(self._catalog(), segment="ffmm")
        assert [d.id for d in sel.datasets] == ["flujos_ffmm", "duracion_ffmm"]
        assert sel.selector_label == "ffmm"

    def test_segment_alias_fondos_mutuos(self):
        sel = select_datasets(self._catalog(), segment="fondos mutuos")
        assert [d.id for d in sel.datasets] == ["flujos_ffmm", "duracion_ffmm"]
        assert "fondos mutuos" in sel.selector_desc

    def test_segment_invalid_lists_valid_segments(self):
        with pytest.raises(ValueError, match="liquidez_bancaria"):
            select_datasets(self._catalog(), segment="inexistente")

    def test_ids_exact_file_and_basename(self):
        sel = select_datasets(self._catalog(),
                              dataset_ids=["duracion_ffmm", "flujos_ffmm.parquet", "ruta/al/lcr.parquet"])
        assert [d.id for d in sel.datasets] == ["duracion_ffmm", "flujos_ffmm", "lcr"]
        assert sel.missing_ids == ()

    def test_ids_dedupe_and_missing(self):
        sel = select_datasets(self._catalog(), dataset_ids=["flujos_ffmm", "flujos_ffmm.parquet", "no_existe"])
        assert [d.id for d in sel.datasets] == ["flujos_ffmm"]
        assert sel.missing_ids == ("no_existe",)

    def test_ids_all_missing_raises(self):
        with pytest.raises(ValueError, match="no_existe"):
            select_datasets(self._catalog(), dataset_ids=["no_existe", "tampoco"])

    def test_query_filters_unrelated(self):
        sel = select_datasets(self._catalog(), query="fondos mutuos")
        ids = {d.id for d in sel.datasets}
        assert "flujos_ffmm" in ids and "lcr" not in ids
        assert sel.selector_label == "fondos_mutuos"

    def test_query_no_match_raises(self):
        with pytest.raises(ValueError, match="coincide"):
            select_datasets(self._catalog(), query="zzzz qqqq")

    def test_requires_exactly_one_selector(self):
        with pytest.raises(ValueError, match="exactamente UNA"):
            select_datasets(self._catalog())
        with pytest.raises(ValueError, match="exactamente UNA"):
            select_datasets(self._catalog(), segment="ffmm", query="fondos")


# ─────────────────────────────────────────────────────────────────────────────
# Ventanas y saneo
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWindowsAndClean:
    def test_parse_window_labels(self):
        assert _parse_window("7d") == (7, "última semana")
        assert _parse_window("30d") == (30, "último mes")
        assert _parse_window("90d") == (90, "últimos 90 días")
        assert _parse_window("1y") == (365, "último año")

    def test_parse_window_invalid(self):
        with pytest.raises(ValueError, match="inválida"):
            _parse_window("ayer")

    def test_window_specs(self):
        assert _window_specs(["7d", "30d"]) == [("última semana", 7), ("último mes", 30)]

    def test_clean_paragraph_strips_markdown(self):
        raw = "# Título\n\n- punto uno\nTexto final.\n---\n"
        assert _clean_paragraph(raw) == "Título punto uno Texto final."


# ─────────────────────────────────────────────────────────────────────────────
# MAP — _describe_dataset (compute_facts monkeypatcheado + 1 llamada al LLM)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDescribeDataset:
    @pytest.mark.asyncio
    async def test_ok_single_llm_call_no_tools(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "compute_facts", lambda ds, pdir, w: dict(_FACTS_OK))
        llm = _MockLLM(responses=[GenerationResult(text="La serie subió a 2,0 al 2026-04-30.", n_tokens=10)])
        section = await pr._describe_dataset(
            _ds("flujos_ffmm"), llm=llm, window_specs=[("última semana", 7)], parquet_dir=tmp_path,
        )
        assert section.status == "ok"
        assert section.paragraph == "La serie subió a 2,0 al 2026-04-30."
        assert section.last_date == "2026-04-30"
        assert len(llm.calls) == 1
        assert llm.calls[0]["tools"] is None
        assert llm.calls[0]["temperature"] == 0.4
        assert llm.calls[0]["top_p"] == 0.80

    @pytest.mark.asyncio
    async def test_no_data_when_facts_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "compute_facts", lambda ds, pdir, w: None)
        llm = _MockLLM()
        section = await pr._describe_dataset(
            _ds("sin_parquet"), llm=llm, window_specs=[("última semana", 7)], parquet_dir=tmp_path,
        )
        assert section.status == "no_data"
        assert section.paragraph == _NO_DATA_PARAGRAPH
        assert llm.calls == []  # no llama al LLM si no hay datos

    @pytest.mark.asyncio
    async def test_no_data_when_facts_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "compute_facts", lambda ds, pdir, w: {"shape": "snapshot", "composition": None})
        llm = _MockLLM()
        section = await pr._describe_dataset(
            _ds("x"), llm=llm, window_specs=[("última semana", 7)], parquet_dir=tmp_path,
        )
        assert section.status == "no_data"
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_llm_failure_becomes_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pr, "compute_facts", lambda ds, pdir, w: dict(_FACTS_OK))

        class _BoomLLM:
            async def generate(self, *a, **k):
                raise RuntimeError("server down")

            def count_text_tokens(self, t):
                return 0

        section = await pr._describe_dataset(
            _ds("x"), llm=_BoomLLM(), window_specs=[("última semana", 7)], parquet_dir=tmp_path,
        )
        assert section.status == "error"


# ─────────────────────────────────────────────────────────────────────────────
# REDUCE + informe completo
# ─────────────────────────────────────────────────────────────────────────────


def _section(dataset_id, status="ok", paragraph="Texto.") -> DatasetSection:
    return DatasetSection(
        dataset_id=dataset_id, name=dataset_id, chart_type="line", unit="%",
        segment="ffmm", last_date="2026-04-30", paragraph=paragraph, status=status,
    )


@pytest.mark.unit
class TestReduceAndReport:
    @pytest.mark.asyncio
    async def test_synthesis_deterministic_sampling_no_tools(self):
        llm = _MockLLM(responses=[GenerationResult(text="Síntesis global.", n_tokens=8)])
        overview = await _synthesize_overview(
            [_section("a"), _section("b", status="no_data")], llm=llm, selector_desc="segmento ffmm",
        )
        assert overview == "Síntesis global."
        call = llm.calls[0]
        assert call["tools"] is None and call["temperature"] == 0.3 and call["top_p"] == 0.8
        user = call["messages"][1]["content"]
        assert "[a]" in user and "[b]" not in user
        assert "1 dataset(s)" in user

    @pytest.mark.asyncio
    async def test_synthesis_skips_llm_when_no_ok(self):
        llm = _MockLLM()
        overview = await _synthesize_overview([_section("a", status="no_data")], llm=llm, selector_desc="x")
        assert overview == _EMPTY_OVERVIEW
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_generate_end_to_end(self, tmp_path, monkeypatch):
        entries = [_ds("flujos_ffmm", name="Flujos"), _ds("dv01_ffmm", name="DV01")]

        def fake_facts(ds, pdir, w):
            return None if ds.id == "dv01_ffmm" else dict(_FACTS_OK)

        monkeypatch.setattr(pr, "compute_facts", fake_facts)
        llm = _MockLLM(responses=[
            GenerationResult(text="Flujos subieron 100%.", n_tokens=8),   # map flujos_ffmm
            GenerationResult(text="Síntesis del segmento.", n_tokens=8),  # reduce
        ])
        report = await generate_parquet_report(
            llm, segment="ffmm", windows=("7d", "30d"), entries=entries, parquet_dir=tmp_path,
        )
        assert [s.status for s in report.sections] == ["ok", "no_data"]
        assert report.overview_md == "Síntesis del segmento."

        md = report.to_markdown()
        assert md.startswith("# Informe descriptivo — segmento ffmm")
        assert "## 1. Flujos" in md and "## 2. DV01" in md
        assert "ok: 1, sin datos: 1" in md
        assert "Ventanas de análisis: 7d, 30d" in md

    @pytest.mark.asyncio
    async def test_invalid_window_fails_early(self):
        with pytest.raises(ValueError, match="inválida"):
            await generate_parquet_report(_MockLLM(), segment="ffmm", windows=("ayer",))

    def test_markdown_includes_missing_ids(self):
        report = ParquetReport(
            title="Informe descriptivo — datasets: a", selector_label="ids", selector_desc="datasets: a",
            generated_at="2026-06-11 10:00", windows=("7d",), overview_md="Síntesis.",
            sections=[_section("a")], missing_ids=("no_existe",),
        )
        assert "No hallados en el catálogo: no_existe" in report.to_markdown()
