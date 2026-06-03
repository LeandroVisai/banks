"""Almacén de archivos subidos en el chat (contexto efímero, con TTL).

Flujo:
  1. ``save_upload(filename, content)`` valida, extrae el contenido a TEXTO
     (PDF/TXT → texto; CSV/Excel → esquema + primeras filas + estadística) y
     persiste un JSON ``data/uploads/<id>.json`` con la metadata + el texto.
  2. El endpoint ``/v1/upload`` devuelve ``upload_id`` + un resumen.
  3. En ``/v1/chat`` con ``attachments=[upload_id]``, ``load_upload`` recupera
     el texto y el agente lo inyecta como contexto del turno.

No se indexa nada (ni pgvector ni el catálogo de parquets): el contenido vive
solo en disco temporal y caduca por TTL.

Dos representaciones del contenido conviven en el record:

  - ``text``: extracto **acotado** a ``MAX_TEXT_CHARS`` — fallback liviano que se
    inyecta como contexto en flujos que no leen el archivo completo.
  - ``pages``: contenido **completo** por página (``[{page, text}, …]``) — lo que
    consume el modo análisis de documento (map-reduce) para leer el PDF ENTERO
    sin truncar. Para PDFs también se extraen los gráficos/figuras (``visuals``)
    recortados a PNG, que el frontend muestra junto a la respuesta.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

from banks_rag.config.paths import DATA_UPLOADS_DIR

# Campos comunes en JSONs de noticias webscrapeadas (es/en). Se buscan
# case-insensitive para formatear cada ítem de forma legible para el LLM.
_NEWS_FIELDS = {
    "title": ("title", "titulo", "título", "headline", "titular", "name"),
    "date": ("date", "fecha", "published", "published_at", "fecha_publicacion",
             "pubdate", "datetime", "timestamp"),
    "source": ("source", "fuente", "medio", "source_name", "publisher", "diario"),
    "body": ("content", "body", "texto", "contenido", "summary", "resumen",
             "description", "descripcion", "cuerpo", "articulo", "abstract"),
    "url": ("url", "link", "enlace", "href", "permalink"),
}

# Límites (defensivos): tamaño del archivo, texto inyectado y filas de tabla.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB
MAX_TEXT_CHARS = 12_000                 # texto efímero (fallback) que entra al contexto
MAX_TABLE_ROWS = 50                     # primeras filas de una tabla
TTL_SECONDS = 24 * 3600                 # caducidad de los uploads
# Techo de gráficos extraídos por PDF (evita inflar el disco temporal). La UI
# muestra solo los más relevantes (cap aparte, en el flujo de análisis).
MAX_EXTRACTED_VISUALS = 40

# Subdirectorio (dentro de DATA_UPLOADS_DIR) para los PNG de gráficos del PDF.
_UPLOAD_IMAGES_SUBDIR = "images"

# Extensión → tipo lógico.
_DOC_EXTS = {".pdf", ".txt", ".md"}
_TABLE_EXTS = {".csv", ".xlsx", ".xls"}
_JSON_EXTS = {".json"}


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


def _news_item_to_text(item: dict) -> str:
    """Formatea un objeto noticia (dict) a texto legible. Si no reconoce campos
    típicos, vuelca el dict en JSON indentado."""
    lower = {str(k).lower(): v for k, v in item.items()}

    def pick(kind: str) -> str:
        for key in _NEWS_FIELDS[kind]:
            v = lower.get(key)
            if v:
                return str(v).strip()
        return ""

    title, date, source = pick("title"), pick("date"), pick("source")
    body, url = pick("body"), pick("url")
    if not (title or body):  # no parece una noticia: vuelca el objeto
        return json.dumps(item, ensure_ascii=False, indent=2)
    parts = [f"### {title or '(sin título)'}"]
    meta = " · ".join(x for x in (date, source) if x)
    if meta:
        parts.append(meta)
    if body:
        parts.append(body)
    if url:
        parts.append(f"Fuente: {url}")
    return "\n".join(parts)


def _read_json(content: bytes, filename: str) -> str:
    """Convierte un JSON (típicamente noticias webscrapeadas) a texto efímero.

    Soporta: lista de noticias, dict con la lista anidada (``articles``/
    ``noticias``/``data``…), o un objeto suelto."""
    try:
        data = json.loads(content.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise UploadError(f"JSON inválido en {filename!r}: {exc}") from exc

    items: list | None = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Primera lista de objetos entre los valores (p. ej. {"articles": [...]}).
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                items = value
                break
        if items is None:
            return json.dumps(data, ensure_ascii=False, indent=2)

    if not items:
        raise UploadError(f"El JSON {filename!r} no tiene noticias/objetos para leer.")

    blocks = [f"{len(items)} ítem(s) en «{filename}»:"]
    for it in items[:MAX_TABLE_ROWS]:
        blocks.append(_news_item_to_text(it) if isinstance(it, dict) else str(it))
    if len(items) > MAX_TABLE_ROWS:
        blocks.append(f"[… {len(items) - MAX_TABLE_ROWS} ítem(s) más omitidos]")
    return "\n\n".join(blocks)


def _read_document(content: bytes, ext: str, filename: str) -> tuple[str, list[dict]]:
    """Devuelve ``(text, pages)``.

    ``text`` es el contenido completo unido con marcadores ``[pág. N]`` (luego
    ``extract_upload`` lo acota para el fallback). ``pages`` es la lista completa
    ``[{page, text}, …]`` con el número de página REAL (1-based) — la fuente que
    lee el modo análisis de documento sin truncar."""
    if ext == ".pdf":
        from banks_rag.infrastructure.extractors.pdf_extractor import extract_pages

        tmp = DATA_UPLOADS_DIR / f"_tmp_{uuid.uuid4().hex}.pdf"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(content)
        try:
            raw_pages, warnings = extract_pages(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        # Página REAL (i+1) preservada; se omiten las vacías para no gastar lotes.
        pages = [
            {"page": i + 1, "text": p}
            for i, p in enumerate(raw_pages)
            if p and p.strip()
        ]
        text = "\n\n".join(f"[pág. {pg['page']}]\n{pg['text']}" for pg in pages)
        if not text.strip():
            raise UploadError(
                f"No se extrajo texto de {filename!r} "
                f"(¿PDF escaneado o protegido? {', '.join(warnings) or 'sin detalle'})."
            )
        return text, pages
    # txt / md
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"No se pudo leer {filename!r}: {exc}") from exc
    return text, [{"page": 1, "text": text}]


def extract_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Valida y extrae el contenido a texto. NO persiste (lo hace save_upload).

    Returns un dict con ``kind`` ('document'|'table'), ``name``, ``text``
    (acotado, fallback), ``truncated``, ``pages`` (contenido COMPLETO por
    página ``[{page, text}, …]``) y ``char_count`` (tamaño total). Lanza
    ``UploadError`` si es inválido.
    """
    if not content:
        raise UploadError("Archivo vacío.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"Archivo demasiado grande ({len(content) // 1024} KB; "
            f"máx {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
        )
    ext = _ext(filename)
    pages: list[dict]
    if ext in _TABLE_EXTS:
        kind, raw = "table", _read_table(content, ext, filename)
        pages = [{"page": 1, "text": raw}]
    elif ext in _JSON_EXTS:
        kind, raw = "document", _read_json(content, filename)
        pages = [{"page": 1, "text": raw}]
    elif ext in _DOC_EXTS:
        kind = "document"
        raw, pages = _read_document(content, ext, filename)
    else:
        raise UploadError(
            f"Tipo no soportado: {ext or '(sin extensión)'}. "
            "Permitidos: PDF, TXT, MD, JSON, CSV, XLSX, XLS."
        )
    text, truncated = _clip(raw)
    return {
        "kind": kind,
        "name": filename,
        "text": text,
        "truncated": truncated,
        "pages": pages,
        "n_pages": len(pages),
        "char_count": sum(len(p["text"]) for p in pages),
    }


def _extract_pdf_visuals(content: bytes, upload_id: str) -> list[dict]:
    """Extrae y recorta los gráficos/figuras de un PDF a PNG (modo análisis de
    documento). Reusa la maquinaria de ingesta ``extract_visual_assets``.

    Los PNG se guardan en ``DATA_UPLOADS_DIR/images/<upload_id>/``. Devuelve la
    metadata ``[{asset_id, page, caption, kind, image_file}, …]`` (``image_file``
    es solo el nombre, no la ruta — el endpoint lo resuelve). Best-effort: si
    PyMuPDF no está o falla, devuelve ``[]`` sin abortar el upload."""
    try:
        from banks_rag.infrastructure.extractors.chart_detector import (
            extract_visual_assets,
        )
    except Exception:
        return []

    out_dir = DATA_UPLOADS_DIR / _UPLOAD_IMAGES_SUBDIR / upload_id
    tmp = DATA_UPLOADS_DIR / f"_tmp_{uuid.uuid4().hex}.pdf"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(content)
    try:
        assets = extract_visual_assets(tmp, upload_id, out_dir)
    except Exception:
        return []
    finally:
        tmp.unlink(missing_ok=True)

    visuals: list[dict] = []
    for a in assets[:MAX_EXTRACTED_VISUALS]:
        visuals.append({
            "asset_id": a.asset_id,
            "page": a.page,
            "caption": a.caption,
            "kind": a.kind,
            "image_file": Path(a.image_path).name,
        })
    return visuals


def _cleanup_old() -> None:
    """Borra uploads (JSON + PNG de gráficos) cuyo mtime excede el TTL.
    Best-effort (nunca lanza)."""
    import shutil

    try:
        cutoff = time.time() - TTL_SECONDS
        for p in DATA_UPLOADS_DIR.glob("*.json"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        images_root = DATA_UPLOADS_DIR / _UPLOAD_IMAGES_SUBDIR
        if images_root.exists():
            for d in images_root.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def save_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Extrae el contenido, persiste el JSON y devuelve la metadata + texto.

    Para PDFs extrae además los gráficos/figuras (``visuals``). El campo
    ``preview`` es un extracto corto para mostrar en la UI.
    """
    meta = extract_upload(filename, content)
    _cleanup_old()
    upload_id = f"up_{uuid.uuid4().hex[:16]}"

    visuals: list[dict] = []
    if _ext(filename) == ".pdf":
        visuals = _extract_pdf_visuals(content, upload_id)

    record = {
        "upload_id": upload_id,
        "created_at": time.time(),
        "visuals": visuals,
        **meta,
    }
    DATA_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_UPLOADS_DIR / f"{upload_id}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8",
    )
    record["preview"] = meta["text"][:280]
    # ``chars`` = tamaño total del documento (no del fallback acotado), para que
    # la UI muestre el peso real de lo que el agente leerá.
    record["chars"] = meta.get("char_count", len(meta["text"]))
    record["n_visuals"] = len(visuals)
    return record


def _valid_upload_id(upload_id: str) -> bool:
    """``True`` si el id tiene el formato esperado (evita traversal con '../')."""
    return bool(
        upload_id
        and upload_id.startswith("up_")
        and "/" not in upload_id
        and "\\" not in upload_id
    )


def upload_images_dir(upload_id: str) -> Path | None:
    """Directorio de PNG de gráficos de un upload, o ``None`` si el id es inválido."""
    if not _valid_upload_id(upload_id):
        return None
    return DATA_UPLOADS_DIR / _UPLOAD_IMAGES_SUBDIR / upload_id


def load_upload(upload_id: str) -> dict[str, Any] | None:
    """Recupera un upload por id. ``None`` si no existe o caducó."""
    # Solo ids con el formato esperado (evita traversal con '../').
    if not _valid_upload_id(upload_id):
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
