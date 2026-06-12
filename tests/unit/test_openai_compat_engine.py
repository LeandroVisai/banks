"""Tests de OpenAICompatEngine — sin red ni servidor (httpx.MockTransport).

Cubre el CONTRATO compartido con LlamaCppEngine (mismos helpers de _common.py):

- load() espera /health con reintentos; 200 → loaded, agotados → LLMUnavailableError.
- generate(): tool calls nativos del servidor (estructurados OpenAI).
- generate(): tool calls embebidos en texto (<tool_call>...</tool_call>).
- generate(): strip de <think> cerrado y de cierre colgante (plantilla Qwen).
- family gemma: la conversación se aplana (sin roles system/tool, sin tools en
  el payload; tools inyectadas como texto en el primer user).
- Errores de red / 5xx → LLMUnavailableError; 4xx → RuntimeError.
- count_text_tokens vía /tokenize (con cache) y fallback heurístico.
- info() y from_settings().
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from banks_rag.infrastructure.llm.base import LLMUnavailableError
from banks_rag.infrastructure.llm.openai_compat_engine import OpenAICompatEngine

BASE_URL = "http://testserver:8081"

MESSAGES = [
    {"role": "system", "content": "Eres un analista del BCCh."},
    {"role": "user", "content": "¿Cuál es la TPM?"},
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Busca en el corpus.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


def _chat_body(
    content: str = "",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    completion_tokens: int = 42,
) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": 100},
    }


def _make_engine(
    handler,
    *,
    family: str = "qwen",
    sync_handler=None,
    **kwargs,
) -> OpenAICompatEngine:
    """Engine con transports mockeados (async para chat, sync para /tokenize)."""
    return OpenAICompatEngine(
        BASE_URL,
        family=family,
        transport=httpx.MockTransport(handler),
        sync_transport=httpx.MockTransport(sync_handler or handler),
        health_retries=3,
        health_interval_s=0.0,
        **kwargs,
    )


def _ok_handler_factory(chat_body: dict, captured: list[httpx.Request] | None = None):
    """Handler que responde /health, /props y /v1/chat/completions."""
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "model_path": "C:\\models\\Qwen3.6-27B-UD-Q6_K_XL.gguf",
                "total_slots": 4,
            })
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_body)
        if request.url.path == "/tokenize":
            payload = json.loads(request.content)
            n = max(1, len(payload.get("content", "")) // 5)
            return httpx.Response(200, json={"tokens": list(range(n))})
        return httpx.Response(404)
    return handler


async def _loaded_engine(chat_body: dict, captured=None, family: str = "qwen"):
    engine = _make_engine(_ok_handler_factory(chat_body, captured), family=family)
    await engine.load()
    return engine


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_ok_and_props(self) -> None:
        engine = _make_engine(_ok_handler_factory(_chat_body()))
        asyncio.run(engine.load())
        assert engine.loaded
        # /props actualiza el nombre con el basename del GGUF cargado.
        assert engine.name == "Qwen3.6-27B-UD-Q6_K_XL.gguf"

    def test_load_unavailable_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": "loading model"})
        engine = _make_engine(handler)
        with pytest.raises(LLMUnavailableError, match="no disponible"):
            asyncio.run(engine.load())
        assert not engine.loaded

    def test_generate_without_load_raises(self) -> None:
        engine = _make_engine(_ok_handler_factory(_chat_body()))
        with pytest.raises(RuntimeError, match="load"):
            asyncio.run(engine.generate(MESSAGES))


# ── Generación ────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_plain_text(self) -> None:
        engine = asyncio.run(_loaded_engine(_chat_body(content="La TPM es 5%. [1]")))
        result = asyncio.run(engine.generate(MESSAGES))
        assert result.text == "La TPM es 5%. [1]"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.n_tokens == 42

    def test_native_tool_calls(self) -> None:
        body = _chat_body(
            content="",
            tool_calls=[{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "arguments": '{"query": "TPM"}',
                },
            }],
            finish_reason="tool_calls",
        )
        engine = asyncio.run(_loaded_engine(body))
        result = asyncio.run(engine.generate(MESSAGES, tools=TOOLS))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_documents"
        assert result.tool_calls[0].arguments == {"query": "TPM"}
        assert result.finish_reason == "tool_calls"

    def test_text_embedded_tool_calls(self) -> None:
        content = (
            '<tool_call>\n{"name": "search_documents", '
            '"arguments": {"query": "TPM"}}\n</tool_call>'
        )
        engine = asyncio.run(_loaded_engine(_chat_body(content=content)))
        result = asyncio.run(engine.generate(MESSAGES, tools=TOOLS))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_documents"
        assert result.finish_reason == "tool_calls"

    def test_strip_think_closed_block(self) -> None:
        engine = asyncio.run(_loaded_engine(
            _chat_body(content="<think>razonando...</think>Respuesta final.")
        ))
        result = asyncio.run(engine.generate(MESSAGES))
        assert result.text == "Respuesta final."
        assert "<think>" in result.raw_text

    def test_strip_think_dangling_close(self) -> None:
        # Plantilla nativa de Qwen: el modelo arranca DENTRO del bloque think.
        engine = asyncio.run(_loaded_engine(
            _chat_body(content="primero pienso esto\n</think>\nRespuesta final.")
        ))
        result = asyncio.run(engine.generate(MESSAGES))
        assert result.text == "Respuesta final."

    def test_qwen_payload_passes_tools(self) -> None:
        captured: list[httpx.Request] = []
        engine = asyncio.run(_loaded_engine(_chat_body(content="ok"), captured))
        asyncio.run(engine.generate(MESSAGES, tools=TOOLS, temperature=0.6))
        chat_reqs = [r for r in captured if r.url.path == "/v1/chat/completions"]
        payload = json.loads(chat_reqs[0].content)
        assert payload["tools"] == TOOLS
        assert payload["tool_choice"] == "auto"
        assert payload["temperature"] == 0.6
        # Mensajes intactos (el servidor aplica la plantilla del GGUF).
        assert payload["messages"][0]["role"] == "system"

    def test_gemma_payload_is_flattened(self) -> None:
        captured: list[httpx.Request] = []
        engine = asyncio.run(_loaded_engine(
            _chat_body(content="ok"), captured, family="gemma",
        ))
        asyncio.run(engine.generate(MESSAGES, tools=TOOLS))
        chat_reqs = [r for r in captured if r.url.path == "/v1/chat/completions"]
        payload = json.loads(chat_reqs[0].content)
        # Sin tools estructuradas; system fusionado en el primer user.
        assert "tools" not in payload
        roles = [m["role"] for m in payload["messages"]]
        assert "system" not in roles
        assert "Herramientas disponibles" in payload["messages"][0]["content"]
        assert "Eres un analista del BCCh." in payload["messages"][0]["content"]

    def test_server_5xx_raises_unavailable(self) -> None:
        calls = {"n": 0}
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/props":
                return httpx.Response(404)
            calls["n"] += 1
            return httpx.Response(500, text="slot crashed")
        engine = _make_engine(handler)
        asyncio.run(engine.load())
        with pytest.raises(LLMUnavailableError, match="500"):
            asyncio.run(engine.generate(MESSAGES))

    def test_network_error_raises_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in ("/health", "/props"):
                return httpx.Response(200, json={})
            raise httpx.ConnectError("connection refused")
        engine = _make_engine(handler)
        asyncio.run(engine.load())
        with pytest.raises(LLMUnavailableError, match="ConnectError"):
            asyncio.run(engine.generate(MESSAGES))

    def test_client_4xx_raises_runtime(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in ("/health", "/props"):
                return httpx.Response(200, json={})
            return httpx.Response(400, text="invalid payload")
        engine = _make_engine(handler)
        asyncio.run(engine.load())
        with pytest.raises(RuntimeError, match="400"):
            asyncio.run(engine.generate(MESSAGES))


# ── Tokenización ──────────────────────────────────────────────────────────────

class TestTokenize:
    def test_count_via_server_and_cache(self) -> None:
        calls = {"tokenize": 0}
        def sync_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/tokenize":
                calls["tokenize"] += 1
                return httpx.Response(200, json={"tokens": [1, 2, 3, 4, 5]})
            return httpx.Response(404)
        engine = _make_engine(
            _ok_handler_factory(_chat_body()), sync_handler=sync_handler,
        )
        assert engine.count_text_tokens("hola mundo") == 5
        assert engine.count_text_tokens("hola mundo") == 5  # cache hit
        assert calls["tokenize"] == 1

    def test_fallback_heuristic_when_endpoint_missing(self) -> None:
        def sync_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)
        engine = _make_engine(
            _ok_handler_factory(_chat_body()), sync_handler=sync_handler,
        )
        text = "x" * 400
        assert engine.count_text_tokens(text) == 100  # 400 chars / 4
        # Tras el primer fallo no insiste contra el endpoint.
        assert not engine._tokenize_available

    def test_count_tokens_applies_overhead(self) -> None:
        def sync_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"tokens": list(range(100))})
        engine = _make_engine(
            _ok_handler_factory(_chat_body()), sync_handler=sync_handler,
        )
        n = engine.count_tokens(MESSAGES)
        assert n == int(100 * 1.15)

    def test_empty_text_is_zero(self) -> None:
        engine = _make_engine(_ok_handler_factory(_chat_body()))
        assert engine.count_text_tokens("") == 0


# ── Info / factory ────────────────────────────────────────────────────────────

class TestInfoAndFactory:
    def test_info_fields(self) -> None:
        engine = _make_engine(_ok_handler_factory(_chat_body()))
        info = engine.info()
        assert info["backend"] == "openai_compat"
        assert info["base_url"] == BASE_URL
        assert info["loaded"] is False
        assert info["family"] == "qwen"

    def test_from_settings(self) -> None:
        settings = SimpleNamespace(
            llm_base_url="http://127.0.0.1:9999",
            llm_family="gemma",
            llm_temperature=0.4,
            llm_top_p=0.8,
            llm_max_tokens=1024,
            llm_request_timeout_s=120.0,
            llm_server_api_key="",
        )
        engine = OpenAICompatEngine.from_settings(settings)
        assert engine.base_url == "http://127.0.0.1:9999"
        assert engine._family == "gemma"
        assert engine._temperature == 0.4
        assert engine._max_tokens == 1024

    def test_unload_closes_clients(self) -> None:
        engine = _make_engine(_ok_handler_factory(_chat_body()))
        asyncio.run(engine.load())
        asyncio.run(engine.unload())
        assert not engine.loaded
