"""Tests de VLLMEngine — sin instalar openai ni conectarse a un servidor vLLM.

Patrón: inyectar ``_client`` directamente en el engine (bypass de ``load()``)
usando ``AsyncMock``, igual que ``test_llama_cpp_engine.py`` inyecta ``_model``.

La suite cubre:
- _parse_args: dict passthrough, string-JSON, malformado.
- load() idempotente (segunda llamada no re-crea el cliente).
- load() lanza RuntimeError si models.list() falla.
- generate() texto plano → GenerationResult limpio.
- generate() con reasoning_content (Qwen3 structured thinking).
- generate() con <think>...</think> embebido → strip por regex.
- generate() thinking desactivado → sin strip.
- generate() con tool_calls nativos estructurados.
- generate() con text-embedded <tool_call> fallback.
- generate() sin tools → tool_choice no incluido en kwargs.
- generate() con tools → tool_choice en kwargs.
- generate() lanza RuntimeError si no está cargado.
- count_text_tokens() retorna estimado positivo.
- count_text_tokens("") retorna 0.
- count_tokens() con tools > sin tools.
- count_tokens() aplica overhead de chat template.
- unload() cierra cliente y setea loaded=False.
- unload() es no-op si no está cargado.
- info() retorna campos esperados incluyendo backend="vllm".
- from_settings() mapea settings.vllm_* correctamente.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from banks_rag.infrastructure.llm.vllm_engine import VLLMEngine, _parse_args


# ── Helpers ───────────────────────────────────────────────────────────────────

MESSAGES = [{"role": "user", "content": "¿Cuál es la TPM?"}]
TOOLS = [{"type": "function", "function": {"name": "search_documents", "parameters": {}}}]


def _make_engine(**overrides) -> VLLMEngine:
    """Crea un VLLMEngine con valores por defecto razonables."""
    defaults = dict(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-32B-AWQ",
        api_key="none",
        enable_thinking=True,
        tool_choice="auto",
        timeout=30.0,
        temperature=0.2,
        top_p=0.9,
        max_tokens=512,
    )
    defaults.update(overrides)
    return VLLMEngine(**defaults)


def _make_loaded_engine(**overrides) -> tuple[VLLMEngine, AsyncMock]:
    """Crea un engine con cliente ya inyectado (bypass de load())."""
    engine = _make_engine(**overrides)
    mock_client = AsyncMock()
    engine._client = mock_client
    engine.loaded = True
    return engine, mock_client


def _completion_response(
    content: str = "",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 42,
    reasoning_content: str | None = None,
) -> MagicMock:
    """Construye un mock de ChatCompletion al estilo openai SDK."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    message.reasoning_content = reasoning_content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _run(coro) -> object:
    return asyncio.run(coro)


# ── Tests de _parse_args ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestParseArgs:
    def test_dict_passthrough(self) -> None:
        assert _parse_args({"query": "TPM"}) == {"query": "TPM"}

    def test_string_json(self) -> None:
        assert _parse_args('{"query": "inflación"}') == {"query": "inflación"}

    def test_malformed_returns_empty(self) -> None:
        assert _parse_args("not json {{") == {}

    def test_empty_dict_string(self) -> None:
        assert _parse_args("{}") == {}


# ── Tests de load() ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestLoad:
    def test_load_idempotente(self) -> None:
        """Segunda llamada a load() no re-crea el cliente."""
        engine, mock_client = _make_loaded_engine()
        original_client = engine._client
        _run(engine.load())
        assert engine._client is original_client
        mock_client.models.list.assert_not_called()

    def test_load_lanza_si_server_no_responde(self) -> None:
        """load() lanza RuntimeError si el servidor no está disponible."""
        engine = _make_engine()

        mock_openai_mod = MagicMock()
        mock_client_inst = AsyncMock()
        mock_client_inst.models.list = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_openai_mod.AsyncOpenAI.return_value = mock_client_inst

        mock_httpx_mod = MagicMock()
        mock_httpx_mod.Timeout.return_value = MagicMock()

        with (
            patch.dict(sys.modules, {"openai": mock_openai_mod, "httpx": mock_httpx_mod}),
        ):
            with pytest.raises(RuntimeError, match="No se pudo conectar"):
                _run(engine.load())

        assert not engine.loaded

    def test_load_exitoso_setea_loaded(self) -> None:
        """load() exitoso setea loaded=True."""
        engine = _make_engine()

        mock_openai_mod = MagicMock()
        mock_client_inst = AsyncMock()
        models_page = MagicMock()
        models_page.data = [MagicMock(id="Qwen/Qwen3-32B-AWQ")]
        mock_client_inst.models.list = AsyncMock(return_value=models_page)
        mock_openai_mod.AsyncOpenAI.return_value = mock_client_inst

        mock_httpx_mod = MagicMock()
        mock_httpx_mod.Timeout.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai_mod, "httpx": mock_httpx_mod}):
            _run(engine.load())

        assert engine.loaded


# ── Tests de generate() ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestGenerate:
    def test_texto_plano(self) -> None:
        engine, client = _make_loaded_engine()
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="La TPM es 5,5%.", completion_tokens=8)
        )
        result = _run(engine.generate(MESSAGES))
        assert result.text == "La TPM es 5,5%."
        assert result.tool_calls == []
        assert result.n_tokens == 8
        assert result.finish_reason == "stop"

    def test_think_strip_por_regex(self) -> None:
        """<think>...</think> embebido se elimina cuando enable_thinking=True."""
        engine, client = _make_loaded_engine(enable_thinking=True)
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(
                content="<think>razonamiento interno</think>La TPM es 5,5%."
            )
        )
        result = _run(engine.generate(MESSAGES))
        assert "<think>" not in result.text
        assert result.text == "La TPM es 5,5%."

    def test_think_strip_multilinea(self) -> None:
        engine, client = _make_loaded_engine(enable_thinking=True)
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(
                content="<think>\nPaso 1\nPaso 2\n</think>Respuesta final."
            )
        )
        result = _run(engine.generate(MESSAGES))
        assert result.text == "Respuesta final."

    def test_reasoning_content_preferido_sobre_regex(self) -> None:
        """Cuando vLLM separa reasoning_content + content, no hay regex strip."""
        engine, client = _make_loaded_engine(enable_thinking=True)
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(
                content="La TPM es 5,5%.",
                reasoning_content="razonando en detalle...",
            )
        )
        result = _run(engine.generate(MESSAGES))
        assert result.text == "La TPM es 5,5%."

    def test_thinking_desactivado_no_strip(self) -> None:
        """Con enable_thinking=False, los tags <think> se preservan."""
        engine, client = _make_loaded_engine(enable_thinking=False)
        raw = "<think>x</think>Respuesta."
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content=raw)
        )
        result = _run(engine.generate(MESSAGES))
        assert "<think>" in result.text

    def test_tool_calls_nativos(self) -> None:
        engine, client = _make_loaded_engine()
        tc_mock = MagicMock()
        tc_mock.id = "call_abc123"
        tc_mock.function.name = "search_documents"
        tc_mock.function.arguments = '{"query": "TPM 2024"}'
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(
                tool_calls=[tc_mock], finish_reason="tool_calls"
            )
        )
        result = _run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.has_tool_calls
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_documents"
        assert result.tool_calls[0].arguments == {"query": "TPM 2024"}
        assert result.tool_calls[0].id == "call_abc123"
        assert result.finish_reason == "tool_calls"

    def test_tool_call_id_generado_si_falta(self) -> None:
        """Si el servidor no retorna id, VLLMEngine genera uno."""
        engine, client = _make_loaded_engine()
        tc_mock = MagicMock()
        tc_mock.id = None  # sin id
        tc_mock.function.name = "search_documents"
        tc_mock.function.arguments = "{}"
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(tool_calls=[tc_mock])
        )
        result = _run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.tool_calls[0].id.startswith("call_")

    def test_fallback_text_embedded_tool_call(self) -> None:
        """Si no hay tool_calls nativos, parsea <tool_call> del texto."""
        engine, client = _make_loaded_engine()
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(
                content='<tool_call>{"name": "search_documents", "arguments": {"query": "test"}}</tool_call>'
            )
        )
        result = _run(engine.generate(MESSAGES, tools=TOOLS))
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search_documents"
        assert result.finish_reason == "tool_calls"

    def test_sin_tools_no_incluye_tool_choice(self) -> None:
        engine, client = _make_loaded_engine()
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="ok")
        )
        _run(engine.generate(MESSAGES))
        kwargs = client.chat.completions.create.call_args[1]
        assert "tool_choice" not in kwargs
        assert "tools" not in kwargs

    def test_con_tools_incluye_tool_choice(self) -> None:
        engine, client = _make_loaded_engine(tool_choice="auto")
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="ok")
        )
        _run(engine.generate(MESSAGES, tools=TOOLS))
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["tools"] == TOOLS

    def test_temperature_override(self) -> None:
        engine, client = _make_loaded_engine(temperature=0.5)
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="ok")
        )
        _run(engine.generate(MESSAGES, temperature=0.9))
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.9  # override prevalece

    def test_max_tokens_override(self) -> None:
        engine, client = _make_loaded_engine(max_tokens=100)
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="ok")
        )
        _run(engine.generate(MESSAGES, max_tokens=200))
        kwargs = client.chat.completions.create.call_args[1]
        assert kwargs["max_tokens"] == 200

    def test_lanza_si_no_cargado(self) -> None:
        engine = _make_engine()
        with pytest.raises(RuntimeError, match="no está cargado"):
            _run(engine.generate(MESSAGES))

    def test_raw_text_preservado(self) -> None:
        """raw_text contiene el output original antes del strip de <think>."""
        engine, client = _make_loaded_engine(enable_thinking=True)
        raw = "<think>interno</think>Respuesta."
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content=raw)
        )
        result = _run(engine.generate(MESSAGES))
        assert result.raw_text == raw
        assert result.text != raw

    def test_n_tokens_desde_usage(self) -> None:
        engine, client = _make_loaded_engine()
        client.chat.completions.create = AsyncMock(
            return_value=_completion_response(content="ok", completion_tokens=99)
        )
        result = _run(engine.generate(MESSAGES))
        assert result.n_tokens == 99


# ── Tests de token counting ───────────────────────────────────────────────────

@pytest.mark.unit
class TestTokenCounting:
    def test_count_text_tokens_positivo(self) -> None:
        engine = _make_engine()
        n = engine.count_text_tokens(
            "La tasa de política monetaria es del cinco coma cinco por ciento"
        )
        assert n > 0

    def test_count_text_tokens_string_vacio(self) -> None:
        engine = _make_engine()
        assert engine.count_text_tokens("") == 0

    def test_count_tokens_aplica_overhead(self) -> None:
        engine = _make_engine()
        messages = [{"role": "user", "content": "test hola mundo"}]
        n = engine.count_tokens(messages)
        raw = engine.count_text_tokens("user: test hola mundo")
        assert n == int(raw * 1.15)

    def test_count_tokens_con_tools_mayor_que_sin_tools(self) -> None:
        engine = _make_engine()
        base = engine.count_tokens(MESSAGES)
        with_tools = engine.count_tokens(MESSAGES, tools=TOOLS)
        assert with_tools > base

    def test_count_tokens_content_list(self) -> None:
        """Maneja mensajes con content como lista de bloques (multimodal)."""
        engine = _make_engine()
        messages = [{"role": "user", "content": [{"type": "text", "text": "hola"}]}]
        n = engine.count_tokens(messages)
        assert n > 0


# ── Tests de unload() ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUnload:
    def test_unload_cierra_cliente_y_setea_false(self) -> None:
        engine, mock_client = _make_loaded_engine()
        _run(engine.unload())
        mock_client.close.assert_called_once()
        assert not engine.loaded
        assert engine._client is None

    def test_unload_noop_si_no_cargado(self) -> None:
        engine = _make_engine()
        _run(engine.unload())  # no debe lanzar
        assert not engine.loaded


# ── Tests de info() ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestInfo:
    def test_campos_presentes(self) -> None:
        engine = _make_engine()
        info = engine.info()
        for campo in ("name", "model_path", "loaded", "backend", "base_url",
                      "enable_thinking", "tool_choice", "temperature", "top_p", "max_tokens"):
            assert campo in info, f"Campo ausente en info(): {campo}"

    def test_backend_es_vllm(self) -> None:
        engine = _make_engine()
        assert engine.info()["backend"] == "vllm"

    def test_loaded_false_inicial(self) -> None:
        engine = _make_engine()
        assert engine.info()["loaded"] is False

    def test_name_deriva_de_model(self) -> None:
        engine = _make_engine(model="Qwen/Qwen3-32B-AWQ")
        assert engine.name == "Qwen3-32B-AWQ"

    def test_name_sin_slash(self) -> None:
        engine = _make_engine(model="local-model")
        assert engine.name == "local-model"


# ── Tests de from_settings() ──────────────────────────────────────────────────

@pytest.mark.unit
class TestFromSettings:
    def test_mapea_settings_correctamente(self) -> None:
        settings = SimpleNamespace(
            vllm_base_url="http://localhost:8000/v1",
            vllm_model="Qwen/Qwen3-32B-AWQ",
            vllm_api_key="secret",
            vllm_enable_thinking=False,
            vllm_tool_choice="required",
            vllm_timeout=60.0,
            vllm_temperature=0.1,
            vllm_top_p=0.95,
            vllm_max_tokens=1024,
        )
        engine = VLLMEngine.from_settings(settings)
        assert engine._base_url == "http://localhost:8000/v1"
        assert engine._model == "Qwen/Qwen3-32B-AWQ"
        assert engine._api_key == "secret"
        assert engine._enable_thinking is False
        assert engine._tool_choice == "required"
        assert engine._timeout == 60.0
        assert engine._temperature == 0.1
        assert engine._top_p == 0.95
        assert engine._max_tokens == 1024

    def test_no_cargado_despues_de_from_settings(self) -> None:
        settings = SimpleNamespace(
            vllm_base_url="http://localhost:8000/v1",
            vllm_model="test",
            vllm_api_key="none",
            vllm_enable_thinking=True,
            vllm_tool_choice="auto",
            vllm_timeout=30.0,
            vllm_temperature=0.2,
            vllm_top_p=0.9,
            vllm_max_tokens=512,
        )
        engine = VLLMEngine.from_settings(settings)
        assert not engine.loaded


# ── Tests de conformidad con LLMEngine Protocol ───────────────────────────────

@pytest.mark.unit
class TestProtocol:
    def test_tiene_atributos_requeridos(self) -> None:
        engine = _make_engine()
        assert isinstance(engine.name, str)
        assert isinstance(engine.model_path, str)
        assert isinstance(engine.loaded, bool)

    def test_isinstance_llm_engine(self) -> None:
        from banks_rag.infrastructure.llm.base import LLMEngine
        engine = _make_engine()
        assert isinstance(engine, LLMEngine)
