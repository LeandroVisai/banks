"""Tools de reuniones de política monetaria: compare_meetings + get_recent_policy_decisions.

Especializadas para el ``PolicyAnalyst`` y el ``DocumentAnalyst`` (Fase C).
Donde ``search_documents`` recupera fragmentos sueltos por relevancia
semántica, estas tools trabajan a nivel de documento-reunión: comparan dos
reuniones lado a lado, o reconstruyen la trayectoria reciente de decisiones.

Ambas seleccionan los chunks *clave* de cada documento — primero los marcados
``is_policy_decision``, luego por ``importance_score`` — en vez de volcar el
documento entero: una comparación útil es focalizada, no exhaustiva.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING, Any

from banks_rag.infrastructure.persistence import PostgresRepo

from .registry import register

if TYPE_CHECKING:
    from banks_rag.domain.agent import AgentState

_CHUNK_TEXT_MAX = 450      # recorte por chunk; la comparación necesita varios
_CHUNKS_FETCH_LIMIT = 60   # cuántos chunks traer del doc antes de seleccionar


def _repo() -> PostgresRepo:
    return PostgresRepo(prefix=os.getenv("RAG_TABLE_PREFIX", ""))


def _select_key_chunks(chunks: list[dict], max_n: int) -> list[dict]:
    """Selecciona los chunks más informativos de un documento.

    Prioridad: chunks marcados como decisión de política, luego por
    ``importance_score`` descendente. El subconjunto elegido se reordena por
    posición en el documento para que se lea en orden natural.
    """
    ranked = sorted(
        chunks,
        key=lambda c: (
            not bool(c.get("is_policy_decision")),       # decisiones primero
            -float(c.get("importance_score") or 0.0),    # luego más importantes
        ),
    )
    selected = ranked[:max_n]
    selected.sort(key=lambda c: c.get("position_in_doc") or 0)
    return selected


def _format_doc_block(
    state: AgentState, doc: dict, chunks: list[dict], max_chunks: int,
) -> list[dict]:
    """Selecciona y formatea los chunks clave de un documento, asignando refs [N]."""
    block: list[dict] = []
    for chunk in _select_key_chunks(chunks, max_chunks):
        # Inyecta metadata del doc para que add_chunk pueda exportar chunks_seen.
        chunk_with_meta = {
            **chunk,
            "filename": doc["filename"],
            "doc_type_category": doc.get("doc_type_category"),
            "document_date": doc.get("document_date"),
        }
        ref = state.add_chunk(chunk_with_meta)
        text = (chunk.get("text") or "").strip()
        if len(text) > _CHUNK_TEXT_MAX:
            text = text[: _CHUNK_TEXT_MAX - 3] + "..."
        block.append({
            "ref": ref,
            "text": text,
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "section": chunk.get("section_type"),
            "is_policy_decision": bool(chunk.get("is_policy_decision")),
            "importance": round(float(chunk.get("importance_score") or 0.0), 3),
        })
    return block


# ─────────────────────────────────────────────────────────────────────────────
# Extracción del nivel vigente de la TPM
#
# get_recent_policy_decisions devolvía solo prosa: el LLM tenía que extraer el
# número del texto y mezclaba/inventaba tasas entre Comunicados. Estos helpers
# atan un porcentaje al contexto de la TPM y exponen `latest_decision` con el
# nivel vigente, su acción y la `ref` para citarlo sin ambigüedad.
# ─────────────────────────────────────────────────────────────────────────────

# Contexto que identifica a la TPM (no a la inflación, el cobre, etc.).
_TPM_CTX = (
    r"(?:tasa de pol[ií]tica monetaria"
    r"|tasa de inter[ée]s de pol[ií]tica monetaria"
    r"|\btpm\b)"
)
# Porcentaje en la MISMA oración que la mención a la TPM ([^.\n] no cruza puntos).
_TPM_RATE_RE = re.compile(
    _TPM_CTX + r"[^.\n]{0,80}?(\d{1,2}(?:[.,]\d{1,2})?)\s*%",
    re.IGNORECASE,
)
# Variante con el porcentaje antes de la mención ("... en 5,50% la TPM").
_TPM_RATE_RE_REV = re.compile(
    r"(\d{1,2}(?:[.,]\d{1,2})?)\s*%[^.\n]{0,40}?" + _TPM_CTX,
    re.IGNORECASE,
)

# Acción de política. Stems sin acento para tolerar variantes; evita "sub"
# desnudo (matchearía "subyacente"). Orden: el primero que matchee gana.
_ACTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("recorte", re.compile(r"recort\w*|reduj\w*|reduc[ií]\w*|disminu\w*|baj[aeoó]\w*", re.IGNORECASE)),
    ("alza", re.compile(r"aument\w*|increment\w*|sub[ií]\w*|al alza", re.IGNORECASE)),
    ("mantener", re.compile(r"mant[eu]\w*|sin cambios?", re.IGNORECASE)),
]


def _detect_action(text: str, near: int) -> str | None:
    """Detecta la acción de política en la ventana alrededor de la tasa (la
    cláusula de la decisión), para no confundirse con otras oraciones."""
    window = text[max(0, near - 120): near + 120]
    for action, pat in _ACTION_PATTERNS:
        if pat.search(window):
            return action
    return None


def _normalize_chunk_text(text: str | None) -> str:
    """Quita el prefijo ``[IMAGE p.N]`` y colapsa saltos de línea/espacios.

    Necesario porque el OCR de los Comunicados-imagen mete saltos de línea que
    cortarían el match de la tasa (el regex no cruza ``\\n``)."""
    text = re.sub(r"\[IMAGE p\.\d+\]", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _find_tpm_rate(text: str) -> tuple[str, str | None] | None:
    """En un texto YA normalizado, ata un porcentaje al contexto TPM.

    Retorna ``(nivel, accion)`` (p. ej. ``("4,5", "mantener")``) o ``None``."""
    m = _TPM_RATE_RE.search(text) or _TPM_RATE_RE_REV.search(text)
    if m is None:
        return None
    return m.group(1).strip(), _detect_action(text, m.start())


def _extract_policy_rate(
    doc: dict, chunks: list[dict], state: AgentState,
) -> dict | None:
    """Extrae el nivel vigente de la TPM barriendo TODOS los chunks del Comunicado.

    No se fía de la etiqueta ``DECISION`` / ``is_policy_decision``: en la práctica
    el párrafo de apertura (que trae 'el Consejo acordó ... en X%') a veces queda
    mal clasificado como RIESGOS, y la etiqueta DECISION se la lleva un chunk
    ``[IMAGE p.1]`` truncado sin el número. Por eso barre todo el documento,
    prefiere el match acompañado de un verbo de acción (la decisión, no una
    proyección) y, entre esos, el de menor ``position_in_doc`` (la decisión va al
    inicio). Registra el chunk elegido en el ``state`` para poder citarlo.

    Retorna ``{"tpm_level": "4,5%", "action": "mantener", "ref": N}`` o ``None``
    (degradación elegante si no hay tasa en texto)."""
    best: tuple[bool, int, str, str | None, dict] | None = None
    for chunk in chunks:
        found = _find_tpm_rate(_normalize_chunk_text(chunk.get("text")))
        if found is None:
            continue
        level, action = found
        pos = int(chunk.get("position_in_doc") or 0)
        cand = (action is not None, pos, level, action, chunk)
        # Prioridad: con verbo de acción primero; a igualdad, menor posición.
        if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
            best = cand
    if best is None:
        return None

    _, _, level, action, chunk = best
    chunk_with_meta = {
        **chunk,
        "filename": doc["filename"],
        "doc_type_category": doc.get("doc_type_category"),
        "document_date": doc.get("document_date"),
    }
    return {
        "tpm_level": f"{level}%",
        "action": action,
        "ref": state.add_chunk(chunk_with_meta),
    }


# ─────────────────────────────────────────────────────────────────────────────
# compare_meetings
# ─────────────────────────────────────────────────────────────────────────────

_COMPARE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compare_meetings",
        "description": (
            "Compara dos documentos de reunión lado a lado (Comunicados o "
            "Minutas). Trae los fragmentos clave de cada uno — decisión, "
            "balance de riesgos, escenario, votación — para contrastar qué "
            "cambió entre una reunión y otra. Pasa los nombres de archivo "
            "(los obtienes de list_documents o search_documents)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename_a": {
                    "type": "string",
                    "description": "Nombre de archivo de la primera reunión.",
                },
                "filename_b": {
                    "type": "string",
                    "description": "Nombre de archivo de la segunda reunión.",
                },
                "max_chunks_per_doc": {
                    "type": "integer",
                    "description": "Fragmentos clave a traer de cada documento (default 6).",
                    "default": 6,
                    "minimum": 2,
                    "maximum": 12,
                },
            },
            "required": ["filename_a", "filename_b"],
        },
    },
}


@register("compare_meetings", _COMPARE_SCHEMA)
async def compare_meetings(
    state: AgentState,
    filename_a: str,
    filename_b: str,
    max_chunks_per_doc: int = 6,
) -> dict[str, Any]:
    max_chunks_per_doc = max(2, min(int(max_chunks_per_doc), 12))
    repo = _repo()

    async def _load(filename: str) -> tuple[dict | None, list[dict]]:
        doc = await asyncio.to_thread(repo.get_document_by_filename, filename)
        if not doc:
            return None, []
        chunks = await asyncio.to_thread(
            repo.get_chunks_by_document_id, doc["document_id"], limit=_CHUNKS_FETCH_LIMIT,
        )
        return doc, chunks

    (doc_a, chunks_a), (doc_b, chunks_b) = await asyncio.gather(
        _load(filename_a), _load(filename_b),
    )

    missing = [
        fn for fn, doc in ((filename_a, doc_a), (filename_b, doc_b)) if doc is None
    ]
    if missing:
        return {
            "error": (
                f"Documento(s) no encontrado(s): {missing}. Usa list_documents "
                "para ver los nombres de archivo disponibles."
            ),
        }

    return {
        "meeting_a": {
            "filename": doc_a["filename"],
            "doc_type": doc_a.get("doc_type_category"),
            "date": str(doc_a.get("document_date") or ""),
            "chunks": _format_doc_block(state, doc_a, chunks_a, max_chunks_per_doc),
        },
        "meeting_b": {
            "filename": doc_b["filename"],
            "doc_type": doc_b.get("doc_type_category"),
            "date": str(doc_b.get("document_date") or ""),
            "chunks": _format_doc_block(state, doc_b, chunks_b, max_chunks_per_doc),
        },
        "hint": (
            "Compara sección a sección (decisión, balance de riesgos, "
            "escenario, votación). Señala explícitamente qué cambió entre la "
            "reunión A y la B, y cita cada punto con [N]."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# get_recent_policy_decisions
# ─────────────────────────────────────────────────────────────────────────────

_RECENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_recent_policy_decisions",
        "description": (
            "Reconstruye la trayectoria reciente de la política monetaria del "
            "BCCh: trae los últimos Comunicados del corpus, del más reciente "
            "al más antiguo, con sus fragmentos de decisión. Úsala para "
            "preguntas sobre la postura actual del Consejo o la secuencia de "
            "decisiones de TPM."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Cuántos Comunicados recientes traer (default 3).",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 8,
                },
                "max_chunks_per_doc": {
                    "type": "integer",
                    "description": "Fragmentos clave por Comunicado (default 4).",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 8,
                },
            },
        },
    },
}


@register("get_recent_policy_decisions", _RECENT_SCHEMA)
async def get_recent_policy_decisions(
    state: AgentState,
    n: int = 3,
    max_chunks_per_doc: int = 4,
) -> dict[str, Any]:
    n = max(1, min(int(n), 8))
    max_chunks_per_doc = max(1, min(int(max_chunks_per_doc), 8))
    repo = _repo()

    # Consultar ambos tipos: COMUNICADO_RPM (nuevo esquema de carpetas)
    # y COMUNICADO (tipo legacy de documentos ingresados antes de la
    # migración a Datos_prueba/RPM/Comunicados_RPM/).
    docs_rpm, docs_legacy = await asyncio.gather(
        asyncio.to_thread(repo.list_documents, doc_type="COMUNICADO_RPM", limit=50),
        asyncio.to_thread(repo.list_documents, doc_type="COMUNICADO", limit=50),
    )
    seen: set[str] = set()
    docs: list[dict] = []
    for d in [*docs_rpm, *docs_legacy]:
        if d["document_id"] not in seen:
            seen.add(d["document_id"])
            docs.append(d)

    if not docs:
        return {
            "decisions": [],
            "n_decisions": 0,
            "message": (
                "No hay Comunicados de política monetaria en el corpus. "
                "Para responder sobre TPM debes indicar explícitamente que "
                "no tienes documentos del BCCh disponibles."
            ),
        }

    # Orden por fecha descendente; document_date es ISO, el orden lexicográfico
    # coincide con el cronológico. Los sin fecha quedan al final.
    docs_sorted = sorted(
        docs, key=lambda d: (d.get("document_date") or ""), reverse=True,
    )[:n]

    async def _load(doc: dict) -> tuple[dict, list[dict]]:
        chunks = await asyncio.to_thread(
            repo.get_chunks_by_document_id, doc["document_id"], limit=_CHUNKS_FETCH_LIMIT,
        )
        return doc, chunks

    loaded = await asyncio.gather(*(_load(doc) for doc in docs_sorted))

    decisions = [
        {
            "filename": doc["filename"],
            "date": str(doc.get("document_date") or ""),
            "year": doc.get("document_year"),
            "chunks": _format_doc_block(state, doc, chunks, max_chunks_per_doc),
        }
        for doc, chunks in loaded
    ]

    # Nivel VIGENTE: se extrae del Comunicado más reciente barriendo TODOS sus
    # chunks (no solo el marcado DECISION, que puede ser una imagen truncada).
    # Le da al LLM una cifra inequívoca en vez de obligarlo a parsear la prosa.
    latest_decision: dict | None = None
    if loaded:
        top_doc, top_chunks = loaded[0]
        rate = _extract_policy_rate(top_doc, top_chunks, state)
        latest_decision = {
            "filename": top_doc["filename"],
            "date": str(top_doc.get("document_date") or ""),
            "tpm_level": rate["tpm_level"] if rate else None,
            "action": rate["action"] if rate else None,
            "ref": rate["ref"] if rate else None,
        }

    return {
        "latest_decision": latest_decision,
        "decisions": decisions,
        "n_decisions": len(decisions),
        "hint": (
            "`latest_decision.tpm_level` es el nivel VIGENTE de la TPM, fijado en "
            "la reunión de `latest_decision.date` (cítalo con [latest_decision.ref]). "
            "Para 'la tasa actual' responde EXACTAMENTE ese valor; NO uses otra cifra "
            "ni combines niveles de Comunicados anteriores. Las `decisions` van del "
            "más reciente al más antiguo y son la trayectoria histórica. La fecha de "
            "una decisión es su `date`, NUNCA la fecha de hoy. Si "
            "`latest_decision.tpm_level` es null, lee el texto del chunk de decisión "
            "y cita el nivel que encuentres ahí."
        ),
    }
