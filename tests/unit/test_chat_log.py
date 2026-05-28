"""Tests del logger de turnos del agente (JSONL diario)."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from banks_rag.config import Settings, override_settings, reset_settings
from banks_rag.domain.agent import AgentResult
from banks_rag.infrastructure.observability import chat_log


def _result() -> AgentResult:
    return AgentResult(
        response="La TPM es 4,5%. [1]",
        iterations=2,
        tool_trace=[{"iteration": 1, "tool": "execute_query", "agent": "quant"}],
        chunks_seen=[{"ref": 1, "filename": "comunicado.pdf", "page_start": 1}],
        series_used=[{"series_id": "tpm", "series_name": "TPM"}],
        cited_refs=[1],
        finish_reason="stop",
        total_tokens=1830,
        latency_ms=4210,
        invalid_refs=[],
    )


@pytest.fixture
def settings_with_dir(tmp_path) -> Iterator[Settings]:
    """Settings con chat logging activo apuntando a un dir temporal."""
    s = Settings(chat_log_enabled=True, chat_log_dir=str(tmp_path / "chat_logs"))
    override_settings(s)
    yield s
    reset_settings()


@pytest.mark.unit
class TestLogChatTurn:
    def test_writes_one_jsonl_line(self, settings_with_dir) -> None:
        path = chat_log.log_chat_turn(
            "¿Cuál es la TPM?",
            history=[{"role": "user", "content": "hola"}],
            result=_result(),
            model="qwen",
            prompt_version="v3",
            request_id="abc123",
        )
        assert path is not None and path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["status"] == "ok"
        assert rec["user_message"] == "¿Cuál es la TPM?"
        assert rec["response"] == "La TPM es 4,5%. [1]"
        assert rec["model"] == "qwen"
        assert rec["request_id"] == "abc123"
        assert rec["cited_refs"] == [1]
        assert rec["tool_trace"][0]["tool"] == "execute_query"
        assert rec["chunks_seen"][0]["filename"] == "comunicado.pdf"

    def test_appends_multiple_turns_same_day(self, settings_with_dir) -> None:
        p1 = chat_log.log_chat_turn(
            "q1", [], _result(), model="qwen", prompt_version="v3",
        )
        p2 = chat_log.log_chat_turn(
            "q2", [], _result(), model="qwen", prompt_version="v3",
        )
        assert p1 == p2  # mismo archivo diario
        assert len(p1.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.unit
class TestLogChatError:
    def test_error_record(self, settings_with_dir) -> None:
        path = chat_log.log_chat_error(
            "pregunta", [], "LLM timeout",
            model="qwen", prompt_version="v3", request_id="r1",
        )
        rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["status"] == "error"
        assert rec["error"] == "LLM timeout"
        assert "response" not in rec


@pytest.mark.unit
class TestDisabledAndRobustness:
    def test_disabled_returns_none_and_writes_nothing(self, tmp_path) -> None:
        s = Settings(chat_log_enabled=False, chat_log_dir=str(tmp_path / "cl"))
        override_settings(s)
        try:
            assert chat_log.log_chat_turn(
                "q", [], _result(), model="qwen", prompt_version="v3",
            ) is None
            assert not (tmp_path / "cl").exists()
        finally:
            reset_settings()

    def test_never_raises_when_dir_unwritable(self, tmp_path) -> None:
        # chat_log_dir bajo un *archivo* → mkdir falla; debe degradar a None.
        blocker = tmp_path / "afile"
        blocker.write_text("x")
        s = Settings(chat_log_enabled=True, chat_log_dir=str(blocker / "sub"))
        override_settings(s)
        try:
            assert chat_log.log_chat_turn(
                "q", [], _result(), model="qwen", prompt_version="v3",
            ) is None
        finally:
            reset_settings()


@pytest.mark.unit
class TestHistorySummary:
    def test_drops_system_and_extra_fields(self) -> None:
        history = [
            {"role": "system", "content": "secret prompt"},
            {"role": "user", "content": "hola", "extra": "x"},
            {"role": "assistant", "content": "qué tal"},
        ]
        out = chat_log._history_summary(history)
        assert out == [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "qué tal"},
        ]
