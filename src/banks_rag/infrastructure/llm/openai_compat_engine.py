"""OpenAICompatEngine — adapter contra un servidor OpenAI-compatible (llama-server).

Implementa el Protocol ``LLMEngine`` hablando HTTP con un endpoint
``/v1/chat/completions`` (llama-server de llama.cpp hoy; TabbyAPI o un fork de
vLLM mañana, sin tocar la app). El servidor vive FUERA del proceso FastAPI:

  - **Concurrencia real**: sin ThreadPoolExecutor — cada ``generate()`` es una
    request httpx async. El paralelismo lo da el servidor (continuous batching,
    ``--parallel N``); el ``asyncio.gather`` de especialistas en
    ``conversation_loop`` deja de ser paralelismo ilusorio.
  - **Prefix/KV-cache reuse** (``--cache-reuse``): cada iteración del loop del
    especialista solo prefillea el delta, no todo el historial.
  - **Aislamiento operacional**: un crash/restart de la API no recarga el modelo.

Comportamiento idéntico al engine in-process (helpers compartidos en
``_common.py``):

  - family ``qwen``  → ``messages`` + ``tools`` van tal cual; el servidor (con
    ``--jinja``) aplica la plantilla del GGUF con tool-calling nativo.
  - family ``gemma`` → ``flatten_for_gemma`` + tools como texto, igual que hoy.
  - ``strip_think`` sobre el contenido (bloque cerrado y cierre colgante de Qwen);
    si el servidor separa el razonamiento en ``reasoning_content``, se ignora.
  - tool calls nativos primero; fallback ``parse_tool_calls`` sobre el texto.

Errores de red/timeout/5xx → ``LLMUnavailableError`` (la API la traduce a 503).

Tokenización: ``count_text_tokens`` usa ``POST /tokenize`` del servidor (cliente
sync con cache LRU; llamada localhost ~1ms). Si el endpoint no responde, cae a
la heurística ``len/4`` para no romper el presupuesto de contexto del agente.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

import httpx

from banks_rag.domain.agent import GenerationResult, ToolCall
from banks_rag.infrastructure.llm._common import (
    flatten_for_gemma,
    parse_args,
    strip_think,
)
from banks_rag.infrastructure.llm.base import LLMUnavailableError
from banks_rag.infrastructure.llm.tool_call_parser import parse_tool_calls

log = logging.getLogger(__name__)

# Heurística de respaldo cuando /tokenize no está disponible (≈4 chars/token
# para texto es/en mixto). Solo se usa como fallback: el path normal tokeniza
# con el tokenizer REAL del modelo vía el servidor.
_FALLBACK_CHARS_PER_TOKEN = 4.0


class OpenAICompatEngine:
    """Motor LLM contra un servidor OpenAI-compatible (implementa ``LLMEngine``).

    Parámetros:
        base_url: URL del servidor (ej. ``http://127.0.0.1:8081``). El path
            ``/v1/chat/completions`` se añade internamente.
        family: ``"qwen"`` (plantilla nativa del servidor, tools estructuradas)
            o ``"gemma"`` (aplanado manual + tools como texto).
        model_name: etiqueta enviada como ``model`` (llama-server la ignora,
            sirve para logs y para routers tipo llama-swap).
        temperature, top_p, max_tokens: defaults de generación (sobreescribibles
            por llamada, igual que el engine in-process).
        timeout_s: techo por request de generación. Debe superar el peor caso
            de síntesis bajo carga (decode de 4096 tokens compartiendo slots).
        api_key: si el servidor corre con ``--api-key`` (header Bearer).
        health_retries / health_interval_s: reintentos de ``load()`` mientras
            el servidor termina de cargar el GGUF.
        transport / sync_transport: transports httpx inyectables para tests
            (``httpx.MockTransport``) — la suite unit corre sin red ni servidor.
    """

    def __init__(
        self,
        base_url: str,
        *,
        family: str = "qwen",
        model_name: str = "default",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2_048,
        timeout_s: float = 300.0,
        api_key: str | None = None,
        health_retries: int = 60,
        health_interval_s: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = model_name
        # Protocol LLMEngine expone ``model_path``; para un backend remoto la
        # "ruta" es la URL del servidor (aparece en /readyz e info()).
        self.model_path = self.base_url
        self.loaded = False
        self._family = family if family in ("qwen", "gemma") else "qwen"
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._health_retries = health_retries
        self._health_interval_s = health_interval_s

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # llama-server (cpp-httplib) cierra conexiones keep-alive ociosas a los
        # ~5s y el default de httpx (keepalive_expiry=5.0) empata justo con ese
        # techo: un POST sobre un socket que el servidor acaba de cerrar queda
        # esperando headers hasta el ReadTimeout completo (visto 2026-06-10:
        # síntesis colgada 300s→503 con el servidor ocioso). Expirar antes en
        # el cliente fuerza una conexión fresca y elimina la ventana de carrera.
        limits = httpx.Limits(keepalive_expiry=2.0)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers=headers,
            limits=limits,
            transport=transport,
        )
        # Cliente sync corto para /tokenize (count_tokens es sync en el Protocol).
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers=headers,
            limits=limits,
            transport=sync_transport,
        )
        self._props: dict = {}
        self._tokenize_cache: dict[str, int] = {}
        self._tokenize_available = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Espera a que el servidor esté sano. Idempotente.

        llama-server responde 503 en ``/health`` mientras carga el modelo;
        se reintenta con backoff fijo hasta ``health_retries``.
        """
        if self.loaded:
            return
        t0 = time.monotonic()
        last_err: str = ""
        for attempt in range(self._health_retries):
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    self.loaded = True
                    await self._fetch_props()
                    log.info(
                        "LLM server sano en %s (%.1fs, modelo=%s)",
                        self.base_url, time.monotonic() - t0, self.name,
                    )
                    return
                last_err = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self._health_retries - 1:
                await asyncio.sleep(self._health_interval_s)
        raise LLMUnavailableError(
            f"llama-server no disponible en {self.base_url} tras "
            f"{self._health_retries} intentos ({last_err}). "
            f"¿Está corriendo deploy/start_llama_server.ps1?"
        )

    async def _fetch_props(self) -> None:
        """Lee ``GET /props`` (best-effort) para nombre real del modelo y n_ctx."""
        try:
            resp = await self._client.get("/props")
            if resp.status_code == 200:
                self._props = resp.json()
                model_path = (
                    self._props.get("model_path")
                    or self._props.get("default_generation_settings", {}).get("model")
                    or ""
                )
                if model_path:
                    self.name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
        except (httpx.HTTPError, ValueError):  # noqa: PERF203
            log.debug("GET /props no disponible — se mantiene model_name configurado")

    async def unload(self) -> None:
        """Cierra los clientes HTTP. El servidor (y su VRAM) NO se tocan."""
        self.loaded = False
        await self._client.aclose()
        self._sync_client.close()

    # ── Generación ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        if not self.loaded:
            raise RuntimeError(
                f"Engine {self.name!r} no inicializado. Llama load() primero."
            )

        # Gemma: aplanar conversación + tools como texto (idéntico al in-process).
        if self._family == "gemma":
            effective_messages = flatten_for_gemma(messages, tools)
            effective_tools = None
        else:
            effective_messages = messages
            effective_tools = tools

        payload: dict[str, Any] = {
            "model": self.name,
            "messages": effective_messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "top_p": top_p if top_p is not None else self._top_p,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if effective_tools:
            payload["tools"] = effective_tools
            # "auto" (NO "required"): el modelo decide si llama una tool o ya
            # responde — mismo razonamiento que el engine in-process (permite
            # terminar el loop del especialista en cuanto reúne evidencia).
            payload["tool_choice"] = "auto"

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Error de red contra {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code >= 500:
            raise LLMUnavailableError(
                f"llama-server respondió {resp.status_code}: {resp.text[:300]}"
            )
        if resp.status_code != 200:
            # 4xx: payload inválido (bug nuestro) — propagar con contexto.
            raise RuntimeError(
                f"chat/completions {resp.status_code}: {resp.text[:300]}"
            )

        body = resp.json()
        choice = body["choices"][0]
        message = choice["message"]
        finish_reason: str = choice.get("finish_reason") or "stop"
        n_tokens: int = (body.get("usage") or {}).get("completion_tokens", 0)
        raw_text: str = message.get("content") or ""

        # El razonamiento puede venir separado (reasoning_content, con --jinja y
        # reasoning-format) o embebido (<think>); strip_think cubre el embebido y
        # el cierre colgante. reasoning_content simplemente no se usa.
        cleaned = strip_think(raw_text)

        # 1. Tool calls nativos (estructurados por el servidor).
        native_calls: list[dict] = message.get("tool_calls") or []
        if native_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=tc["function"]["name"],
                    arguments=parse_args(tc["function"].get("arguments", "{}")),
                )
                for i, tc in enumerate(native_calls)
            ]
            return GenerationResult(
                text=cleaned,
                tool_calls=tool_calls,
                raw_text=raw_text,
                n_tokens=n_tokens,
                finish_reason=finish_reason,
            )

        # 2. Tool calls embebidos en texto (<tool_call>...</tool_call> / fenced JSON).
        tool_calls, final_text = parse_tool_calls(cleaned)
        return GenerationResult(
            text=final_text,
            tool_calls=tool_calls,
            raw_text=raw_text,
            n_tokens=n_tokens,
            finish_reason="tool_calls" if tool_calls else finish_reason,
        )

    # ── Conteo de tokens ──────────────────────────────────────────────────────

    # El chat template real (tokens de rol, separadores, formato de tools)
    # añade overhead que la concatenación en texto plano no refleja. Mismo
    # margen que el engine in-process.
    _CHAT_TEMPLATE_OVERHEAD = 1.15
    _TOKENIZE_CACHE_MAX = 512

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        """Estima tokens del prompt completo (concatenación + margen de template)."""
        parts: list[str] = []
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        for m in messages:
            parts.append(f"{m.get('role', '')}: {m.get('content', '')}")
        raw = self.count_text_tokens("\n".join(parts))
        return int(raw * self._CHAT_TEMPLATE_OVERHEAD)

    def count_text_tokens(self, text: str) -> int:
        """Cuenta tokens vía ``POST /tokenize`` del servidor (tokenizer real).

        Cache LRU-ish por hash del texto (los system prompts de los especialistas
        se repiten en cada iteración). Fallback heurístico si el endpoint falla:
        es preferible una estimación a romper el presupuesto del agente.
        """
        if not text:
            return 0
        key = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        cached = self._tokenize_cache.get(key)
        if cached is not None:
            return cached

        n: int | None = None
        if self._tokenize_available:
            try:
                resp = self._sync_client.post("/tokenize", json={"content": text})
                if resp.status_code == 200:
                    n = len(resp.json().get("tokens", []))
                else:
                    self._tokenize_available = False
            except (httpx.HTTPError, ValueError):
                # No insistir por request: el agente cuenta tokens en el camino
                # caliente y un servidor caído ya se reporta vía generate().
                self._tokenize_available = False
        if n is None:
            n = int(len(text) / _FALLBACK_CHARS_PER_TOKEN)

        if len(self._tokenize_cache) >= self._TOKENIZE_CACHE_MAX:
            self._tokenize_cache.pop(next(iter(self._tokenize_cache)))
        self._tokenize_cache[key] = n
        return n

    # ── Info / health ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "backend": "openai_compat",
            "base_url": self.base_url,
            "loaded": self.loaded,
            "family": self._family,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
            "server_props": {
                k: self._props[k]
                for k in ("total_slots", "n_ctx", "model_path")
                if k in self._props
            },
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatEngine":
        """Construye el engine desde el objeto Settings del servicio.

        Usa: ``llm_base_url``, ``llm_family``, ``llm_temperature``, ``llm_top_p``,
        ``llm_max_tokens``, ``llm_request_timeout_s``, ``llm_server_api_key``.
        """
        return cls(
            settings.llm_base_url,
            family=settings.llm_family,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_request_timeout_s,
            api_key=settings.llm_server_api_key or None,
        )
