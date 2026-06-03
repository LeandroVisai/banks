"""Modo análisis de documento adjunto (map-reduce).

Cuando un turno del chat trae un archivo adjunto (PDF/TXT/…), el agente NO rutea
a los especialistas de mercado/política (cuyas tools consultan el corpus
indexado y los parquets, no el archivo subido). En su lugar corre este flujo
dedicado de **analista económico-financiero**:

  1. **MAP** — el documento se lee POR LOTES de páginas que caben en el contexto
     del modelo (``build_batches``). Cada lote se resume/extrae en una llamada al
     LLM sin tools. Así se lee el documento ENTERO sin truncar ni reventar el
     contexto (el problema original: el texto se recortaba a 12k chars).
  2. **REDUCE** — una llamada final consolida las notas parciales en la respuesta
     (conclusión primero, citas por página).

Además se seleccionan los gráficos/figuras extraídos del PDF (``visuals``, ver
``infrastructure/uploads/upload_store``) más relevantes para mostrarlos en la UI.

``build_batches`` y ``select_visuals`` son funciones puras (sin LLM ni I/O):
fáciles de testear con un contador de tokens y datos sintéticos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from .prompts import DOC_MAP_PROMPT, DOC_REDUCE_PROMPT, apply_thinking

# Defaults (espejo de los settings ``upload_*``; ver config/settings.py). Quien
# llama (run_agent ← chat.py) los pasa desde Settings; aquí sirven de fallback.
DEFAULT_BATCH_TOKENS = 3500
DEFAULT_MAP_MAX_TOKENS = 1024
DEFAULT_REDUCE_MAX_TOKENS = 4096
DEFAULT_MAX_BATCHES = 24
DEFAULT_MAX_VISUALS = 6

# Sampling determinista (fiel para resumir/citar), como la síntesis del agente.
_TEMPERATURE = 0.3
_TOP_P = 0.8

# Marca con que el MAP indica que un lote no aporta nada relevante (se descarta).
_EMPTY_MARKER = "(sin contenido relevante)"


@dataclass
class DocAnalysisResult:
    """Resultado del flujo de análisis de documento."""

    response: str
    visuals: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"
    total_tokens: int = 0
    n_batches: int = 0
    partial_notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Batching (puro)
# ─────────────────────────────────────────────────────────────────────────────


def _pages_from_records(records: list[dict]) -> list[dict]:
    """Aplana las páginas de todos los documentos adjuntos.

    Cada item: ``{page, text, doc}``. Si un record no trae ``pages`` (formato
    viejo), cae a su ``text`` como una sola página."""
    out: list[dict] = []
    for rec in records or []:
        name = rec.get("name") or rec.get("upload_id") or "documento"
        pages = rec.get("pages")
        if pages:
            for pg in pages:
                txt = (pg.get("text") or "").strip()
                if txt:
                    out.append({"page": pg.get("page"), "text": txt, "doc": name})
        elif (rec.get("text") or "").strip():
            out.append({"page": None, "text": rec["text"].strip(), "doc": name})
    return out


def _split_text(text: str, batch_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Parte un texto que excede ``batch_tokens`` en ventanas que sí caben.

    Estima chars/token sobre el texto completo y corta por ventanas de chars
    (suficiente: el cálculo no necesita ser exacto, solo no desbordar)."""
    total_tok = max(count_tokens(text), 1)
    ratio = len(text) / total_tok                 # chars por token (aprox)
    window = max(200, int(batch_tokens * ratio * 0.9))
    return [text[i:i + window] for i in range(0, len(text), window)]


def build_batches(
    pages: list[dict],
    *,
    batch_tokens: int,
    count_tokens: Callable[[str], int],
    max_batches: int,
) -> list[str]:
    """Agrupa páginas en lotes de texto que caben en ``batch_tokens``.

    Greedy: acumula páginas hasta que agregar la siguiente excedería el budget;
    una página que por sí sola excede el budget se parte (``_split_text``). Se
    topa a ``max_batches`` (documentos gigantes se leen hasta ese techo)."""
    batches: list[str] = []
    current: list[str] = []
    current_tok = 0

    def flush() -> None:
        nonlocal current, current_tok
        if current:
            batches.append("\n\n".join(current))
            current = []
            current_tok = 0

    for pg in pages:
        text = (pg.get("text") or "").strip()
        if not text:
            continue
        marker = f"[pág. {pg['page']}]" if pg.get("page") else ""
        unit = f"{marker}\n{text}".strip()
        utok = count_tokens(unit)

        if utok > batch_tokens:
            flush()
            for sub in _split_text(unit, batch_tokens, count_tokens):
                batches.append(sub)
                if len(batches) >= max_batches:
                    return batches[:max_batches]
            continue

        if current and current_tok + utok > batch_tokens:
            flush()
            if len(batches) >= max_batches:
                return batches[:max_batches]
        current.append(unit)
        current_tok += utok

    flush()
    return batches[:max_batches]


# ─────────────────────────────────────────────────────────────────────────────
# Selección de gráficos a mostrar (puro)
# ─────────────────────────────────────────────────────────────────────────────


def select_visuals(
    records: list[dict],
    response_text: str,
    *,
    max_visuals: int,
) -> list[dict]:
    """Elige los gráficos del PDF a mostrar en la UI.

    Rankea por: mencionado en la respuesta (caption o "pág. N") > tiene caption >
    página ascendente. Devuelve ``[{caption, page, image_url, kind}, …]`` con
    ``image_url`` = ``/v1/uploads/{upload_id}/images/{file}``."""
    resp = (response_text or "").lower()
    scored: list[tuple[tuple, dict]] = []
    for rec in records or []:
        upload_id = rec.get("upload_id")
        for v in rec.get("visuals") or []:
            image_file = v.get("image_file")
            if not (upload_id and image_file):
                continue
            caption = v.get("caption") or ""
            page = v.get("page")
            mentioned = False
            if caption:
                head = caption.lower()[:24]
                mentioned = bool(head) and head in resp
            if not mentioned and page is not None:
                mentioned = (f"pág. {page}" in resp) or (f"página {page}" in resp)
            # Orden: mencionados primero, luego con caption, luego por página.
            key = (0 if mentioned else 1, 0 if caption else 1, page if page is not None else 10_000)
            scored.append((key, {
                "caption": caption or None,
                "page": page,
                "image_url": f"/v1/uploads/{upload_id}/images/{image_file}",
                "kind": v.get("kind") or "CHART",
            }))
    scored.sort(key=lambda t: t[0])
    return [item for _, item in scored[:max_visuals]]


# ─────────────────────────────────────────────────────────────────────────────
# Flujo map-reduce
# ─────────────────────────────────────────────────────────────────────────────


def _today_line() -> str:
    hoy = date.today()
    return (
        f"La fecha de hoy es {hoy.isoformat()}. Úsala solo para referencias "
        f"relativas del usuario; las fechas del documento provienen del propio "
        f"documento.\n\n"
    )


async def run_document_analysis(
    records: list[dict],
    user_message: str,
    *,
    llm,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
    map_max_tokens: int = DEFAULT_MAP_MAX_TOKENS,
    reduce_max_tokens: int = DEFAULT_REDUCE_MAX_TOKENS,
    max_batches: int = DEFAULT_MAX_BATCHES,
    max_visuals: int = DEFAULT_MAX_VISUALS,
) -> DocAnalysisResult:
    """Lee los documentos adjuntos por lotes (MAP) y consolida (REDUCE).

    Las llamadas al LLM van sin tools y sin thinking (resumir es barato). Las
    generaciones se serializan (llama.cpp es instancia única) — el MAP corre lote
    a lote en orden para mantener el orden del documento en las notas."""
    count_tokens = getattr(llm, "count_text_tokens", lambda s: max(1, len(s) // 4))
    pages = _pages_from_records(records)
    batches = build_batches(
        pages, batch_tokens=batch_tokens, count_tokens=count_tokens, max_batches=max_batches,
    )

    total_tokens = 0
    finish_reason = "stop"

    if not batches:
        return DocAnalysisResult(
            response=(
                "No pude extraer texto del archivo adjunto (¿PDF escaneado, "
                "vacío o protegido?). Súbelo en otro formato o pega el texto."
            ),
            visuals=select_visuals(records, "", max_visuals=max_visuals),
            finish_reason="empty",
        )

    # ── MAP ──────────────────────────────────────────────────────────────────
    map_system = apply_thinking(_today_line() + DOC_MAP_PROMPT, think=False)
    notes: list[str] = []
    for i, batch in enumerate(batches, start=1):
        user = (
            f"Consulta del usuario:\n{user_message}\n\n"
            f"Fragmento del documento ({i} de {len(batches)}):\n{batch}"
        )
        res = await llm.generate(
            [
                {"role": "system", "content": map_system},
                {"role": "user", "content": user},
            ],
            tools=None,
            temperature=_TEMPERATURE,
            top_p=_TOP_P,
            max_tokens=map_max_tokens,
        )
        total_tokens += res.n_tokens
        text = (res.text or "").strip()
        if text and _EMPTY_MARKER not in text:
            notes.append(f"### Sección {i}\n{text}")

    # ── REDUCE ─────────────────────────────────────────────────────────────────
    reduce_system = apply_thinking(_today_line() + DOC_REDUCE_PROMPT, think=False)
    if notes:
        notes_block = "\n\n".join(notes)
    else:
        # Ningún lote aportó: igual respondemos a partir del documento crudo
        # (la pregunta puede ser muy general). Usa el primer lote como contexto.
        notes_block = (
            "(Tus notas por sección quedaron vacías; trabaja con el inicio del "
            "documento.)\n\n" + batches[0]
        )
    reduce_user = (
        f"Consulta del usuario:\n{user_message}\n\n"
        f"Tus notas del documento (cubren {len(batches)} sección(es)):\n{notes_block}\n\n"
        "Redacta ahora el análisis final para el usuario."
    )
    reduced = await llm.generate(
        [
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": reduce_user},
        ],
        tools=None,
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        max_tokens=reduce_max_tokens,
    )
    total_tokens += reduced.n_tokens
    finish_reason = reduced.finish_reason
    response = (reduced.text or "").strip() or "No pude generar el análisis del documento."

    return DocAnalysisResult(
        response=response,
        visuals=select_visuals(records, response, max_visuals=max_visuals),
        finish_reason=finish_reason,
        total_tokens=total_tokens,
        n_batches=len(batches),
        partial_notes=notes,
    )
