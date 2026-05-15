"""Verificación y limpieza de citas ``[N]`` en respuestas del LLM.

El LLM puede emitir refs inventadas (números fuera del rango de chunks vistos).
Este módulo:

  1. Recorre todas las ocurrencias de ``[N]`` en el texto.
  2. Mantiene las que están en rango ``[1, n_chunks]``.
  3. Elimina las inválidas (junto con espacios/puntuación huérfana resultante).
  4. Retorna la lista de refs efectivamente usadas (sin duplicados, en orden de aparición).

Función pura: no muta ``AgentState`` ni hace I/O.
"""

from __future__ import annotations

import re

from banks_rag.domain.agent import AgentState

CITATION_RE = re.compile(r"\[(\d+)\]")


def verify_citations(response: str, state: AgentState) -> tuple[str, list[int], list[int]]:
    """Limpia citas inválidas y devuelve las usadas y las inválidas.

    Args:
        response: texto generado por el LLM con citas ``[N]``.
        state: estado del agente con ``chunks_seen`` (define el rango válido).

    Returns:
        ``(cleaned_text, used_refs, invalid_refs)``. ``used_refs`` está en
        orden de aparición; ``invalid_refs`` son las refs fuera de rango (sin
        duplicados) — señal de alucinación que el caller debe registrar.
    """
    n_chunks = len(state.chunks_seen)
    used: list[int] = []
    invalid: list[int] = []

    def _replace(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_chunks:
            if n not in used:
                used.append(n)
            return m.group(0)
        if n not in invalid:
            invalid.append(n)
        return ""

    cleaned = CITATION_RE.sub(_replace, response)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip(), used, invalid
