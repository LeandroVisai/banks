"""Tests de LlamaCppEngine — sin instalar llama-cpp-python.

Todos los accesos a ``llama_cpp.Llama`` se mockean con ``unittest.mock``.
La suite cubre:

- _detect_family: Qwen3 → qwen (plantilla nativa GGUF), Gemma → gemma, default qwen.
- _parse_args: dict passthrough, string-JSON, doble-serialización, malformado.
- Texto sin tool calls → GenerationResult limpio.
- <think> strip antes de parsear.
- Native tool calls desde llama-cpp structured output.
- Text-embedded tool calls (<tool_call>...</tool_call>).
- Fallback texto cuando no hay tool calls.
- finish_reason propagado correctamente.
- n_tokens desde usage.completion_tokens.
- count_text_tokens / count_tokens delegados al modelo.
- load() falla con FileNotFoundError si el .gguf no existe.
- info() retorna campos esperados.
- from_settings() construye el engine con los parámetros correctos.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from banks_rag.infrastructure.llm.llama_cpp_engine import (
    LlamaCppEngine,
    _detect_family,
    _parse_args,
    _strip_think,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

MESSAGES = [{"role": "user", "content": "¿Cuál es la TPM?"}]
TOOLS = [{"type": "function", "function": {"name": "search_documents", "parameters": {}}}]


def _mock_response(
    content: str = "",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 42,
) -> dict:
    """Construye un response dict al estilo create_chat_completion."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"completion_tokens": completion_tokens},
    }


def _make_engine(tmp_path: Path) -> tuple[LlamaCppEngine, MagicMock]:
    """Crea un engine con un mock de Llama ya cargado."""
    gguf = tmp_path / "Qwen3.6-27B-Q4_K_XL.gguf"
    gguf.write_bytes(b"fake")

    mock_llama = MagicMock()
    mock_llama.tokenize.return_value = list(range(10))

    engine = LlamaCppEngine(str(gguf), n_ctx=512, n_gpu_layers=0)
    engine._model = mock_llama
    engine.loaded = True
    return engine, mock_llama


# ── Tests puros (sin I/O) ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestHelpers:
    def test_detect_family_qwen(self) -> None:
        assert _detect_family("models/Qwen3.6-27B-Q4_K_XL.gguf") == "qwen"

    def test_detect_family_gemma(self) -> None:
        assert _detect_family("models/gemma-4-26B-it.gguf") == "gemma"

    def test_detect_family_default(self) -> None:
        # Cualquier modelo no-gemma se trata como Qwen/ChatML-compatible.
        assert _detect_family("models/mistral-7b.gguf") == "qwen"

    def test_parse_args_dict_passthrough(self) -> None:
        assert _parse_args({"q": "TPM"}) == {"q": "TPM"}

    def test_parse_args_string_json(self) -> None:
        assert _parse_args('{"q": "inflación"}') == {"q": "inflación"}

    def test_parse_args_double_serialized(self) -> None:
        inner = json.dumps({"q": "test"})
        assert _parse_args(inner) == {"q": "test"}

    def test_parse_args_malformed_returns_empty(self) -> None:
        assert _parse_args("not json {{") == {}

    def test_parse_args_empty_string(self) -> None:
        assert _parse_args("{}") == {}

    def test_strip_think_closed_block(self) -> None:
        assert _strip_think("<think>razono</think>La TPM es 4,5%.") == "La TPM es 4,5%."

    def test_strip_think_dangling_close(self) -> None:
        # Qwen3 con plantilla nativa: solo emite el </think> de cierre.
        raw = "The user asks for X.\nDrafting...\n</think>\n\nEl dólar está en 904,45."
        assert _strip_think(raw) == "El dólar está en 904,45."

    def test_strip_think_no_thinking(self) -> None:
        assert _strip_think("Hola, ¿en qué te ayudo?") == "Hola, ¿en qué te ayudo?"

    def test_strip_think_multiline_closed(self) -> None:
        raw = "<think>\npaso 1\npaso 2\n</think>\nRespuesta final."
        assert _strip_think(raw) == "Respuesta final."


# ── Tests de generación ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerate:
    def _run(self, coro) -> object:
        # asyncio.run crea y cierra un loop propio: en Python 3.12
        # get_event_loop() ya no crea uno implícito en el MainThread y lanzaría
        # RuntimeError ("no current event loop").
        return asyncio.run(coro)

    def test_plain_text_response(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            content="La TPM es 5,5%.", finish_reason="stop", completion_tokens=8
        )
        result = self._run(engine.generate(MESSAGES))
        assert result.text == "La TPM es 5,5%."
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.n_tokens == 8

    def test_think_block_stripped(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            content="<think>razonando...</think>La TPM es 5,5%."
        )
        result = self._run(engine.generate(MESSAGES))
        assert "<think>" not in result.text
        assert result.text == "La TPM es 5,5%."

    def test_think_block_multiline_stripped(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            content="<think>\nPaso 1: buscar.\nPaso 2: responder.\n</think>\nRespuesta final."
        )
        result = self._run(engine.generate(MESSAGES))
        assert result.text == "Respuesta final."

    def test_native_tool_calls(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            content=None,
            tool_calls=[{
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "search_documents", "arguments": '{"query": "TPM 2024"}'},
            }],
            finish_reason="tool_calls",
        )
        result = self._run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "search_documents"
        assert tc.arguments == {"query": "TPM 2024"}
        assert result.finish_reason == "tool_calls"

    def test_native_tool_calls_multiple(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
            finish_reason="tool_calls",
        )
        result = self._run(engine.generate(MESSAGES, tools=TOOLS))
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "tool_a"
        assert result.tool_calls[1].name == "tool_b"

    def test_text_embedded_tool_call(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(
            content='<tool_call>{"name": "search_documents", "arguments": {"query": "inflación"}}</tool_call>'
        )
        result = self._run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search_documents"
        assert result.tool_calls[0].arguments == {"query": "inflación"}
        assert result.finish_reason == "tool_calls"

    def test_text_embedded_after_think_strip(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        raw = (
            "<think>necesito buscar</think>"
            '<tool_call>{"name": "search_documents", "arguments": {"query": "TPM"}}</tool_call>'
        )
        mock_llama.create_chat_completion.return_value = _mock_response(content=raw)
        result = self._run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search_documents"

    def test_tools_passed_to_model(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(content="ok")
        self._run(engine.generate(MESSAGES, tools=TOOLS))
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"

    def test_no_tools_kwargs_clean(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(content="ok")
        self._run(engine.generate(MESSAGES))
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_custom_temperature_forwarded(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.create_chat_completion.return_value = _mock_response(content="ok")
        self._run(engine.generate(MESSAGES, temperature=0.0))
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_raises_if_not_loaded(self, tmp_path: Path) -> None:
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        engine = LlamaCppEngine(str(gguf))
        with pytest.raises(RuntimeError, match="no cargado"):
            asyncio.run(engine.generate(MESSAGES))


# ── Tests de tokenización ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestTokenCounting:
    def test_count_text_tokens(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.tokenize.return_value = list(range(15))
        assert engine.count_text_tokens("hola mundo") == 15

    def test_count_tokens_with_messages(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        mock_llama.tokenize.return_value = list(range(20))
        n = engine.count_tokens(MESSAGES, tools=TOOLS)
        # 20 tokens crudos × 1.15 de margen por overhead del chat template.
        assert n == int(20 * 1.15)
        mock_llama.tokenize.assert_called_once()

    def test_count_tokens_returns_zero_when_unloaded(self, tmp_path: Path) -> None:
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"x")
        engine = LlamaCppEngine(str(gguf))
        assert engine.count_text_tokens("test") == 0
        assert engine.count_tokens(MESSAGES) == 0


# ── Tests de load / info ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestLoadAndInfo:
    def test_load_raises_if_gguf_missing(self, tmp_path: Path) -> None:
        import sys
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = MagicMock()
        with patch.dict(sys.modules, {"llama_cpp": mock_llama_module}):
            engine = LlamaCppEngine(str(tmp_path / "missing.gguf"))
            with pytest.raises(FileNotFoundError, match="no encontrado"):
                engine._sync_load()

    def test_load_idempotent(self, tmp_path: Path) -> None:
        engine, mock_llama = _make_engine(tmp_path)
        with patch("banks_rag.infrastructure.llm.llama_cpp_engine.LlamaCppEngine._sync_load") as m:
            asyncio.run(engine.load())
            m.assert_not_called()  # ya estaba loaded=True

    def test_info_fields(self, tmp_path: Path) -> None:
        engine, _ = _make_engine(tmp_path)
        info = engine.info()
        assert info["loaded"] is True
        assert info["family"] == "qwen"
        assert info["n_ctx"] == 512
        assert "model_path" in info
        assert "name" in info

    def test_from_settings(self, tmp_path: Path) -> None:
        gguf = tmp_path / "Qwen3.6-27B-Q4_K_XL.gguf"
        gguf.write_bytes(b"x")
        settings = SimpleNamespace(
            model_path_resolved=gguf,
            llm_n_ctx=8192,
            llm_n_gpu_layers=-1,
            llm_temperature=0.1,
            llm_top_p=0.95,
            llm_max_tokens=1024,
        )
        engine = LlamaCppEngine.from_settings(settings)
        assert engine._n_ctx == 8192
        assert engine._n_gpu_layers == -1
        assert engine._temperature == 0.1
        assert engine._max_tokens == 1024
        assert "qwen" == engine._family
