"""Tests del clasificador determinista de tipo de gráfico (graficador con sentido).

El backend ya no deja que todo salga como línea: ``infer_chart_type`` y
``AgentState.add_series`` etiquetan la serie según la forma del dato (temporal →
``line``; categórico → ``bar``), y el frontend respeta ese ``chart_type``.
"""

from __future__ import annotations

import pytest

from banks_rag.domain.agent import AgentState
from banks_rag.domain.agent.agent_state import infer_chart_type


@pytest.mark.unit
class TestInferChartType:
    def test_temporal_series_is_line(self) -> None:
        points = [["2024-01-01", 1.0], ["2024-02-01", 2.0], ["2024-03-01", 3.0]]
        assert infer_chart_type(points, x_is_date=True) == "line"

    def test_categorical_series_is_bar(self) -> None:
        points = [["BTP", 70.0], ["BTU", 30.0]]
        assert infer_chart_type(points, x_is_date=False) == "bar"

    def test_few_temporal_points_is_bar(self) -> None:
        # 1-2 cifras sobre un eje temporal rinden mejor como barra que una
        # "línea" de dos vértices.
        points = [["2024-01-01", 1.0], ["2024-02-01", 2.0]]
        assert infer_chart_type(points, x_is_date=True) == "bar"


@pytest.mark.unit
class TestAddSeriesChartType:
    def test_temporal_rows_default_to_line(self) -> None:
        state = AgentState()
        rows = [
            {"date": "2024-01-01", "value": 10.0},
            {"date": "2024-02-01", "value": 11.0},
            {"date": "2024-03-01", "value": 12.0},
        ]
        state.add_series("ds:Valor", {"series_name": "X", "unit": "%"}, rows)
        s = state.series_used["ds:Valor"]
        assert s["chart_type"] == "line"
        assert s["points"] == [["2024-01-01", 10.0], ["2024-02-01", 11.0], ["2024-03-01", 12.0]]
        assert s["first_date"] == "2024-01-01"
        assert s["last_date"] == "2024-03-01"
        assert s["n_observations"] == 3

    def test_categorical_rows_become_bar(self) -> None:
        state = AgentState()
        rows = [
            {"category": "Nacional", "value": 46.4},
            {"category": "Extranjero", "value": 53.6},
        ]
        state.add_series("ds:comp", {"series_name": "Cartera", "unit": "%"}, rows)
        s = state.series_used["ds:comp"]
        assert s["chart_type"] == "bar"
        assert s["points"] == [["Nacional", 46.4], ["Extranjero", 53.6]]
        # Sin eje de fechas en una serie categórica.
        assert s["first_date"] is None
        assert s["last_date"] is None

    def test_explicit_chart_type_overrides_inference(self) -> None:
        state = AgentState()
        rows = [{"category": "A", "value": 1.0}, {"category": "B", "value": 2.0}]
        state.add_series("ds:g", {"series_name": "G"}, rows, chart_type="grouped_bar")
        assert state.series_used["ds:g"]["chart_type"] == "grouped_bar"

    def test_rows_without_value_are_not_plottable(self) -> None:
        state = AgentState()
        state.add_series("ds:empty", {"series_name": "E"}, [{"date": "2024-01-01"}])
        assert state.series_used["ds:empty"]["points"] == []
