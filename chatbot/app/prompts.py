"""
Plantillas de prompt + ensamblaje del mensaje final.

El system prompt se versiona con `PROMPT_VERSION` para poder hacer A/B y
saber con qué prompt se generó cada respuesta (queda en logs).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import llm
from .settings import settings

log = logging.getLogger(__name__)


PROMPT_VERSION = "v1"

SYSTEM_INSTRUCTION = """\
Eres un analista experto en política monetaria y macroeconomía del Banco Central de Chile.
Respondes en español con precisión técnica y citas explícitas.

REGLAS:
1. Para argumentos cualitativos (decisiones, motivos, narrativa) usa SOLO la información \
del bloque <contexto>...</contexto>. Cita cada afirmación con el formato [N], donde N es \
el número del fragmento.
2. Para datos cuantitativos (tasas, niveles, variaciones) usa el bloque \
<datos_historicos>...</datos_historicos>. No inventes ni extrapoles cifras.
3. Si la información disponible es insuficiente, dilo explícitamente: \
"No tengo información suficiente en los documentos disponibles."
4. Estructura la respuesta de forma concisa: una conclusión breve seguida de los \
fragmentos que la respaldan.
5. NO inventes fechas, votaciones, nombres de consejeros ni cifras que no aparezcan \
literalmente en el contexto."""


# ─────────────────────────────────────────────────────────────────────────────
# Formateo de bloques
# ─────────────────────────────────────────────────────────────────────────────

def _format_chunk(idx: int, c: dict) -> str:
    filename = c.get("filename", "desconocido")
    page = c.get("page_start", "?")
    page_end = c.get("page_end")
    page_str = f"p.{page}" + (f"-{page_end}" if page_end and page_end != page else "")
    doc_type = c.get("doc_type_category", "")
    section = c.get("section_type", "")
    date = str(c.get("chunk_date") or c.get("document_date") or "")

    meta = f"[{idx}] {filename} · {page_str} · {doc_type} · {section}"
    if date:
        meta += f" · {date}"
    text = (c.get("text") or "").strip()
    return f"{meta}\n{text}"


def _format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Sin documentos relevantes para esta consulta.)"
    return "\n\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, start=1))


# ─────────────────────────────────────────────────────────────────────────────
# Token budget — recorta el contexto si excede el presupuesto
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_chunks_to_budget(
    chunks: list[dict],
    historical_block: str,
    history: list[dict],
    user_message: str,
    budget: int,
) -> list[dict]:
    """
    Si el prompt completo excede el presupuesto de tokens, retira chunks RAG
    de menor prioridad hasta caber. Devuelve la lista (posiblemente truncada).
    """
    if not chunks:
        return chunks

    chunks = list(chunks)
    while chunks:
        messages = build_messages(
            user_message=user_message,
            rag_chunks=chunks,
            historical_context=historical_block,
            history=history,
            _skip_truncate=True,
        )
        n_tokens = llm.count_tokens(messages)
        if n_tokens <= budget:
            return chunks
        # Retirar el chunk de menor importance (último, ya viene ranked)
        chunks.pop()

    return []


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def build_messages(
    user_message: str,
    rag_chunks: list[dict],
    historical_context: Optional[str],
    history: list[dict],
    *,
    _skip_truncate: bool = False,
) -> list[dict]:
    """Ensambla mensajes en formato OpenAI-chat."""
    if not _skip_truncate and llm.is_loaded():
        rag_chunks = _truncate_chunks_to_budget(
            chunks=rag_chunks,
            historical_block=historical_context or "",
            history=history,
            user_message=user_message,
            budget=settings.prompt_token_budget,
        )

    system_parts = [
        SYSTEM_INSTRUCTION,
        "\n\n<contexto>\n" + _format_rag_context(rag_chunks) + "\n</contexto>",
    ]
    if historical_context:
        system_parts.append(
            "\n\n<datos_historicos>\n" + historical_context + "\n</datos_historicos>"
        )

    messages: list[dict] = [{"role": "system", "content": "".join(system_parts)}]
    for turn in history:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Verificación de citas en la respuesta
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+)\]")


def verify_citations(response: str, n_sources: int) -> tuple[str, list[int]]:
    """
    Limpia citas inválidas de la respuesta (refs > n_sources se eliminan).
    Retorna (respuesta_limpia, lista_de_refs_usadas).
    """
    used: list[int] = []
    valid = set(range(1, n_sources + 1))

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""  # cita hallucinada — se elimina

    cleaned = _CITATION_RE.sub(_replace, response)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)  # arregla espacios sueltos
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip(), used
