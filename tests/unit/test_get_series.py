"""Tests de get_series: el atajo discover+execute en un paso (mocks aislados)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from banks_rag.application.agent.tools.get_series import get_series
from banks_rag.domain.agent import AgentState

_MOD = "banks_rag.application.agent.tools.get_series"


@pytest.mark.unit
class TestGetSeries:
    @pytest.mark.asyncio
    async def test_dataset_id_skips_discovery(self) -> None:
        exec_mock = AsyncMock(return_value={"dataset_id": "btp_curva", "n_rows": 3})
        disc_mock = AsyncMock()
        with patch(f"{_MOD}.execute_query", exec_mock), patch(f"{_MOD}.discover_query", disc_mock):
            out = await get_series(AgentState(), dataset_id="btp_curva")
        disc_mock.assert_not_called()
        exec_mock.assert_awaited_once()
        assert out["dataset_id"] == "btp_curva"
        assert "discovery" not in out  # no hubo búsqueda

    @pytest.mark.asyncio
    async def test_discovers_then_executes(self) -> None:
        disc_mock = AsyncMock(return_value={"results": [
            {"id": "spread_btp_spc"}, {"id": "spread_btp_ust"},
        ]})
        exec_mock = AsyncMock(return_value={"dataset_id": "spread_btp_spc", "n_rows": 5})
        with patch(f"{_MOD}.discover_query", disc_mock), patch(f"{_MOD}.execute_query", exec_mock):
            out = await get_series(AgentState(), query="spread BTP vs SPC 10 años")
        disc_mock.assert_awaited_once()
        # execute se llamó con el dataset top.
        assert exec_mock.await_args.kwargs["dataset_id"] == "spread_btp_spc"
        assert out["discovery"]["elegido"] == "spread_btp_spc"
        assert out["discovery"]["alternativas"] == ["spread_btp_ust"]

    @pytest.mark.asyncio
    async def test_error_when_neither_query_nor_id(self) -> None:
        out = await get_series(AgentState())
        assert "error" in out and "dataset_id" in out["error"]

    @pytest.mark.asyncio
    async def test_error_when_no_results(self) -> None:
        disc_mock = AsyncMock(return_value={"results": [], "message": "nada"})
        with patch(f"{_MOD}.discover_query", disc_mock):
            out = await get_series(AgentState(), query="algo inexistente")
        assert "error" in out
