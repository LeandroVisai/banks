"""Almacén de archivos subidos en el chat (contexto efímero, con TTL).

Flujo:
  1. ``save_upload(filename, content)`` valida, extrae el contenido a TEXTO
     (PDF/TXT → texto; CSV/Excel → esquema + primeras filas + estadística) y
     persiste un JSON ``data/uploads/<id>.json`` con la metadata + el texto.
  2. El endpoint ``/v1/upload`` devuelve ``upload_id`` + un resumen.
  3. En ``/v1/chat`` con ``attachments=[upload_id]``, ``load_upload`` recupera
     el texto y el agente lo inyecta como contexto del turno.

No se indexa nada (ni pgvector ni el catálogo de parquets): el contenido vive
solo en disco temporal y caduca por TTL. El texto se acota a ``MAX_TEXT_CHARS``
para no desbordar el contexto del LLM.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

from banks_rag.config.paths import DATA_UPLOADS_DIR

# Límites (defensivos): tamaño del archivo, texto inyectado y filas de tabla.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB
MAX_TEXT_CHARS = 12_000                 # texto efímero que entra al contexto
MAX_TABLE_ROWS = 50                     # primeras filas de una tabla
TTL_SECONDS = 24 * 3600                 # caducidad de los uploads

# Extensión → tipo lógico.
_DOC_EXTS = {".pdf", ".txt", ".md"}
_TABLE_EXTS = {".csv", ".xlsx", ".xls"}


class UploadError(ValueError):
    """Archivo inválido (tipo no soportado, vacío, demasiado grande, ilegible)."""


def _ext(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _clip(text: str) -> tuple[str, bool]:
    """Acota el texto a ``MAX_TEXT_CHARS``. Devuelve ``(texto, truncado)``."""
    if len(text) <= MAX_TEXT_CHARS:
        return text, False
    return text[:MAX_TEXT_CHARS], True


def _format_table(df: Any, *, source: str) -> str:
    """Texto compacto de una tabla para el contexto del LLM: forma, esquema,
    primeras filas (CSV) y estadística de las columnas numéricas."""
    import pandas as pd  # local: solo cuando hay un tabular

    n_rows, n_cols = df.shape
    schema = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    head_csv = df.head(MAX_TABLE_ROWS).to_csv(index=False).strip()
    parts = [
        f"Tabla adjunta «{source}» — {n_rows} filas × {n_cols} columnas.",
        f"Columnas: {schema}",
        "",
        f"Primeras {min(n_rows, MAX_TABLE_ROWS)} filas (CSV):",
        head_csv,
    ]
    num = df.select_dtypes(include="number")
    if not num.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            parts += ["", "Estadística (columnas numéricas):", num.describe().round(4).to_string()]
    if n_rows > MAX_TABLE_ROWS:
        parts.append(f"\n[Se muestran las primeras {MAX_TABLE_ROWS} de {n_rows} filas.]")
    return "\n".join(parts)


def _read_table(content: bytes, ext: str, filename: str) -> str:
    import pandas as pd

    buf = io.BytesIO(content)
    try:
        df = pd.read_csv(buf) if ext == ".csv" else pd.read_excel(buf)
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"No se pudo leer la tabla {filename!r}: {exc}") from exc
    if df.empty:
        raise UploadError(f"La tabla {filename!r} no tiene datos.")
    return _format_table(df, source=filename)


def _read_document(content: bytes, ext: str, filename: str) -> str:
    if ext == ".pdf":
        from banks_rag.infrastructure.extractors.pdf_extractor import extract_pages

        tmp = DATA_UPLOADS_DIR / f"_tmp_{uuid.uuid4().hex}.pdf"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(content)
        try:
            pages, warnings = extract_pages(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        text = "\n\n".join(f"[pág. {i + 1}]\n{p}" for i, p in enumerate(pages) if p.strip())
        if not text.strip():
            raise UploadError(
                f"No se extrajo texto de {filename!r} "
                f"(¿PDF escaneado o protegido? {', '.join(warnings) or 'sin detalle'})."
            )
        return text
    # txt / md
    try:
        return content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"No se pudo leer {filename!r}: {exc}") from exc


def extract_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Valida y extrae el contenido a texto. NO persiste (lo hace save_upload).

    Returns un dict con ``kind`` ('document'|'table'), ``name``, ``text``
    (acotado) y ``truncated``. Lanza ``UploadError`` si es inválido.
    """
    if not content:
        raise UploadError("Archivo vacío.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"Archivo demasiado grande ({len(content) // 1024} KB; "
            f"máx {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
        )
    ext = _ext(filename)
    if ext in _TABLE_EXTS:
        kind, raw = "table", _read_table(content, ext, filename)
    elif ext in _DOC_EXTS:
        kind, raw = "document", _read_document(content, ext, filename)
    else:
        raise UploadError(
            f"Tipo no soportado: {ext or '(sin extensión)'}. "
            "Permitidos: PDF, TXT, MD, CSV, XLSX, XLS."
        )
    text, truncated = _clip(raw)
    return {"kind": kind, "name": filename, "text": text, "truncated": truncated}


def _cleanup_old() -> None:
    """Borra uploads cuyo mtime excede el TTL. Best-effort (nunca lanza)."""
    try:
        cutoff = time.time() - TTL_SECONDS
        for p in DATA_UPLOADS_DIR.glob("*.json"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def save_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Extrae el contenido, persiste el JSON y devuelve la metadata + texto.

    El campo ``preview`` es un extracto corto para mostrar en la UI.
    """
    meta = extract_upload(filename, content)
    _cleanup_old()
    upload_id = f"up_{uuid.uuid4().hex[:16]}"
    record = {
        "upload_id": upload_id,
        "created_at": time.time(),
        **meta,
    }
    DATA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_UPLOADS_DIR / f"{upload_id}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8",
    )
    record["preview"] = meta["text"][:280]
    record["chars"] = len(meta["text"])
    return record


def load_upload(upload_id: str) -> dict[str, Any] | None:
    """Recupera un upload por id. ``None`` si no existe o caducó."""
    # Solo ids con el formato esperado (evita traversal con '../').
    if not upload_id or not upload_id.startswith("up_") or "/" in upload_id or "\\" in upload_id:
        return None
    path = DATA_UPLOADS_DIR / f"{upload_id}.json"
    if not path.exists():
        return None
    if path.stat().st_mtime < time.time() - TTL_SECONDS:
        path.unlink(missing_ok=True)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
