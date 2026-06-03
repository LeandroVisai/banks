"""LlamaCppEngine — adapter llama-cpp-python que implementa el Protocol LLMEngine.

Soporta Qwen3.6-27B-UD-Q4_K_XL.gguf y Gemma4-26B-A4B-it.gguf sobre H100 offline.

Async-first: llama.cpp es blocking; todas las llamadas pesadas corren en un
ThreadPoolExecutor dedicado para no bloquear el event loop de FastAPI.

Tool calling:
  1. Si llama-cpp-python devuelve ``tool_calls`` estructurado (native) → se usa directo.
  2. Si el modelo emite ``<tool_call>{...}</tool_call>`` en texto → ``parse_tool_calls``.
  Qwen3 soporta ambos; Gemma 4 usa principalmente el texto.

Qwen3 thinking mode:
  Los bloques ``<think>...</think>`` se eliminan del texto final antes de
  entregarlo al usuario y al parser de tool calls.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from banks_rag.domain.agent import GenerationResult, ToolCall
from banks_rag.infrastructure.llm.tool_call_parser import parse_tool_calls

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(raw_text: str) -> str:
    """Quita el razonamiento (thinking) del texto entregado al usuario.

    Dos casos:
      1. Bloque cerrado ``<think>...</think>`` — formato clásico.
      2. **Cierre colgante** ``...razonamiento... </think> respuesta`` SIN apertura:
         es lo que emite Qwen3 con su plantilla nativa del GGUF, porque el
         template ya inyecta ``<think>`` al final del prompt, así que el modelo
         arranca DENTRO del bloque y solo emite el ``</think>`` de cierre. Sin
         este caso, todo el chain-of-thought (a menudo en inglés) se filtraba a
         la respuesta final.
    """
    text = _THINK_RE.sub("", raw_text)
    if "<think>" not in text and "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def _detect_family(model_path: str) -> str:
    """Infiere la familia del modelo desde el nombre del GGUF (solo basename).

    Devuelve ``"gemma"`` o ``"qwen"`` (este último es también el default para
    cualquier modelo ChatML-compatible). La familia decide CÓMO se construye el
    prompt:

    - ``qwen``  → se usa la **plantilla de chat embebida en el GGUF**
      (``chat_format=None`` al construir ``Llama``). Qwen3 trae en su metadata
      el template entrenado para tool-calling: renderiza ``tools``, los
      ``tool_calls`` del assistant y los resultados (rol ``tool``) como
      ``<tool_response>``. El ``chatml`` genérico de llama-cpp NO hace esto
      (ignora ``tools`` y no re-inyecta los resultados en el formato que Qwen
      reconoce), por lo que el modelo no "veía" lo que devolvían las tools y
      repetía la misma llamada sin avanzar (p. ej. ``discover_query`` en bucle
      sin llegar nunca a ``execute_query``).
    - ``gemma`` → aplanado manual a la alternancia user/model (ver
      ``_flatten_for_gemma`` y ``_sync_generate``).
    """
    p = Path(model_path).name.lower()
    if "gemma" in p:
        return "gemma"
    return "qwen"


def _parse_args(raw: str | dict) -> dict:
    """Parsea ``arguments`` de un tool call, tolerando string-JSON anidado."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_tools_as_text(tools: list[dict]) -> str:
    """Renderiza los schemas de tools como texto para modelos sin tool calling nativo.

    Gemma no recibe bien los schemas vía la API; se inyectan en el prompt como
    texto, con el formato <tool_call> exacto que ``parse_tool_calls`` reconoce.
    """
    lines = [
        "\n\n## Herramientas disponibles",
        "",
        "Para usar una herramienta, responde EXACTAMENTE en este formato "
        "(JSON dentro de etiquetas <tool_call>):",
        "",
        '<tool_call>',
        '{"name": "nombre_exacto", "arguments": {"arg": "valor"}}',
        '</tool_call>',
        "",
        "Herramientas (usa el nombre EXACTO):",
    ]
    for schema in tools:
        fn = schema["function"]
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        arg_parts = []
        for arg_name, arg_spec in props.items():
            mark = "" if arg_name in required else " (opcional)"
            arg_parts.append(f'{arg_name}{mark}')
        args_str = ", ".join(arg_parts) if arg_parts else "sin argumentos"
        lines.append(f'- `{fn["name"]}` — args: {args_str}. {fn["description"]}')
    lines.append("")
    lines.append(
        "NUNCA respondas una pregunta sustantiva sin llamar primero a una "
        "herramienta. Cuando ya tengas la información, responde en texto SIN "
        "etiquetas <tool_call>."
    )
    return "\n".join(lines)


def _flatten_for_gemma(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Aplana mensajes estilo OpenAI a la alternancia user/model de Gemma.

    Gemma no soporta los roles ``system`` ni ``tool``, ni ``assistant`` con
    ``tool_calls`` estructurados. Esta función:
      - fusiona los ``system`` (más las definiciones de tools como texto) en
        el primer mensaje ``user``;
      - renderiza los ``tool_calls`` del assistant como texto <tool_call>;
      - convierte los resultados de tools (rol ``tool``) en texto marcado,
        anexado al turno ``user`` para mantener la alternancia.
    """
    system_parts = [
        m["content"] for m in messages
        if m.get("role") == "system" and m.get("content")
    ]
    prefix = "\n\n".join(system_parts)
    if tools:
        prefix += _format_tools_as_text(tools)

    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            content = m.get("content", "") or ""
            if prefix:
                content = prefix + "\n\n---\n\n" + content
                prefix = ""
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            content = m.get("content", "") or ""
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                content += (
                    f'\n<tool_call>\n{{"name": "{fn.get("name")}", '
                    f'"arguments": {args}}}\n</tool_call>'
                )
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            tool_msg = f'[RESULTADO DE {m.get("name", "herramienta")}]\n{m.get("content", "")}'
            if out and out[-1]["role"] == "user":
                out[-1]["content"] += "\n\n" + tool_msg
            else:
                out.append({"role": "user", "content": tool_msg})

    if prefix:  # no había ningún user; mete el contexto como primer turno
        out.insert(0, {"role": "user", "content": prefix})
    return out


class LlamaCppEngine:
    """Motor de inferencia llama.cpp que implementa el Protocol ``LLMEngine``.

    Parámetros:
        model_path: ruta absoluta al archivo ``.gguf``.
        n_ctx: contexto máximo en tokens (default 16 384).
        n_gpu_layers: capas a offloadear en GPU; ``-1`` = todas (H100).
        temperature, top_p, max_tokens: defaults de generación (sobreescribibles
            por llamada en ``generate()``).
        family: ``"qwen"`` o ``"gemma"``; ``None`` = auto-detect desde el nombre
            del archivo. Qwen usa la plantilla nativa del GGUF (tool-calling
            correcto); ver ``_detect_family``.
        n_threads: workers del ThreadPoolExecutor (default 1; llama.cpp usa
            internamente sus propios threads para inferencia).
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 16_384,
        n_gpu_layers: int = -1,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 2_048,
        family: str | None = None,
        n_threads: int = 1,
    ) -> None:
        self.model_path = model_path
        self.name = Path(model_path).stem
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._family = family or _detect_family(model_path)
        self._model: Any = None
        # Una sola instancia del modelo no es concurrente: con n_threads=1
        # las requests a /v1/chat se serializan en este executor. Es el
        # comportamiento correcto (evita corromper el estado del modelo),
        # pero define el techo de throughput — escalar requiere réplicas
        # del proceso o una cola con back-pressure, no subir n_threads.
        self._executor = ThreadPoolExecutor(
            max_workers=n_threads,
            thread_name_prefix="llama_cpp",
        )
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Carga el modelo GGUF en GPU. Idempotente."""
        if self.loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_load)

    def _sync_load(self) -> None:
        # Windows: registra los directorios de DLLs de torch y llama_cpp antes
        # de importar, o el loader de Windows no los encuentra.
        import sys
        if sys.platform == "win32":
            import importlib.util
            import os
            try:
                import torch
                os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
            except Exception:
                pass
            try:
                spec = importlib.util.find_spec("llama_cpp")
                if spec and spec.origin:
                    os.add_dll_directory(os.path.dirname(spec.origin))
            except Exception:
                pass
            # DLLs del runtime CUDA cuando el wheel es CUDA pero torch es CPU-only:
            # los paquetes nvidia-*-cu12 (cuda_runtime, cublas, cuda_nvrtc) traen
            # cudart64_12 / cublas64_12 / nvrtc64 que ggml-cuda.dll necesita. En la
            # H100 los aporta torch-CUDA; en un box con torch-CPU + wheel CUDA hay
            # que exponerlos. Se añaden al search path de DLLs y al PATH (deps
            # transitivas). Inofensivo si los paquetes no están instalados.
            try:
                import glob
                nv_spec = importlib.util.find_spec("nvidia")
                if nv_spec and nv_spec.submodule_search_locations:
                    nv_base = os.path.dirname(nv_spec.submodule_search_locations[0])
                    nv_bins = glob.glob(os.path.join(nv_base, "nvidia", "*", "bin"))
                    for binp in nv_bins:
                        os.add_dll_directory(binp)
                    if nv_bins:
                        os.environ["PATH"] = (
                            os.pathsep.join(nv_bins) + os.pathsep + os.environ.get("PATH", "")
                        )
            except Exception:
                pass

        from llama_cpp import Llama  # importación lazy para tests sin el binario

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo GGUF no encontrado: {self.model_path}")

        # Qwen → chat_format=None: llama-cpp usa la plantilla embebida en el GGUF
        # (tool-calling nativo: tools, tool_calls y resultados con <tool_response>).
        # Gemma → "gemma" (la conversación igual se aplana en _sync_generate).
        llama_chat_format = None if self._family == "qwen" else "gemma"
        log.info(
            "Cargando %s (n_ctx=%d, n_gpu_layers=%d, family=%s, chat_format=%s)",
            self.name,
            self._n_ctx,
            self._n_gpu_layers,
            self._family,
            llama_chat_format or "GGUF-template",
        )
        t0 = time.monotonic()
        self._model = Llama(
            model_path=str(path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            chat_format=llama_chat_format,
            verbose=False,
        )
        self.loaded = True
        log.info("Modelo listo en %.1fs", time.monotonic() - t0)

    async def unload(self) -> None:
        """Libera el modelo y la VRAM. Idempotente."""
        if not self.loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_unload)

    def _sync_unload(self) -> None:
        self._model = None
        self.loaded = False
        log.info("Modelo %s descargado", self.name)

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
            raise RuntimeError(f"Modelo {self.name!r} no cargado. Llama load() primero.")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_generate,
                messages,
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        )

    def _sync_generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        # Gemma no soporta roles system/tool ni tool calling nativo confiable:
        # se aplana la conversación y las tools se inyectan como texto, dejando
        # que parse_tool_calls extraiga las <tool_call> de la respuesta.
        if self._family == "gemma":
            effective_messages = _flatten_for_gemma(messages, tools)
            effective_tools = None
        else:
            effective_messages = messages
            effective_tools = tools

        kwargs: dict[str, Any] = {
            "messages": effective_messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "top_p": top_p if top_p is not None else self._top_p,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if effective_tools:
            kwargs["tools"] = effective_tools
            # "auto" (NO "required"): el modelo decide si llama una tool o ya
            # responde. En el loop del subagente esto permite TERMINAR en cuanto
            # reúne evidencia — con "required" estaba forzado a emitir un tool
            # call en cada iteración no-final, agotando siempre el presupuesto y
            # disparando llamadas espurias. El guard anti-alucinación de
            # conversation_loop ya descarta cifras sin evidencia, así que "auto"
            # es a la vez más rápido y seguro.
            kwargs["tool_choice"] = "auto"

        response = self._model.create_chat_completion(**kwargs)
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason: str = choice.get("finish_reason") or "stop"
        n_tokens: int = response.get("usage", {}).get("completion_tokens", 0)
        raw_text: str = message.get("content") or ""

        # Qwen3 thinking mode: eliminar el razonamiento del texto entregado
        # (bloque cerrado o cierre colgante de la plantilla nativa).
        cleaned = _strip_think(raw_text)

        # 1. Native tool calls (llama-cpp structured output)
        native_calls: list[dict] = message.get("tool_calls") or []
        if native_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=tc["function"]["name"],
                    arguments=_parse_args(tc["function"].get("arguments", "{}")),
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

        # 2. Text-embedded tool calls (<tool_call>...</tool_call> o fenced JSON)
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
    # añade overhead que la concatenación en texto plano no refleja. Se
    # aplica un margen para no subestimar y arriesgar exceder n_ctx.
    _CHAT_TEMPLATE_OVERHEAD = 1.15

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> int:
        """Estima tokens del prompt completo (concatenación + margen de template)."""
        if not self.loaded or self._model is None:
            return 0
        parts: list[str] = []
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        for m in messages:
            parts.append(f"{m.get('role', '')}: {m.get('content', '')}")
        raw = self.count_text_tokens("\n".join(parts))
        return int(raw * self._CHAT_TEMPLATE_OVERHEAD)

    def count_text_tokens(self, text: str) -> int:
        """Cuenta tokens de texto plano usando el tokenizer del modelo."""
        if not self.loaded or self._model is None:
            return 0
        try:
            return len(self._model.tokenize(text.encode("utf-8", errors="replace")))
        except Exception:
            return 0

    # ── Info / health ─────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "name": self.name,
            "model_path": self.model_path,
            "loaded": self.loaded,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "family": self._family,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": self._max_tokens,
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "LlamaCppEngine":
        """Construye el engine desde el objeto Settings del servicio.

        Espera que ``settings`` tenga:
            ``llm_model_path``, ``llm_n_ctx``, ``llm_n_gpu_layers``,
            ``llm_temperature``, ``llm_top_p``, ``llm_max_tokens``.
        """
        model_path = getattr(settings, "model_path_resolved", None) or settings.llm_model_path
        return cls(
            str(model_path),
            n_ctx=settings.llm_n_ctx,
            n_gpu_layers=settings.llm_n_gpu_layers,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            max_tokens=settings.llm_max_tokens,
        )
