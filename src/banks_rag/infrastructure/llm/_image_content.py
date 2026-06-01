"""Construcción de contenido multimodal (imagen) para mensajes del LLM.

Qwen3.6 es multimodal: cargado con su proyector mtmd (``BANKS_LLM_MMPROJ_PATH``)
puede "ver" imágenes — p.ej. los gráficos/tablas extraídos de los IPoM
(``search_visuals`` expone su ``image_path``). Este módulo arma los bloques de
contenido estilo OpenAI (texto + ``image_url`` con data-URL base64) de forma
PURA y testeable, sin depender del binario llama.cpp ni del modelo.

Diseño defensivo: si ninguna imagen es válida, retorna el texto plano (``str``),
de modo que un modelo SOLO-texto nunca reciba bloques de imagen por error.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Límite defensivo de tamaño por imagen embebida (evita inflar el prompt).
_MAX_IMAGE_BYTES = 6 * 1024 * 1024


def _data_url(path: Path) -> str | None:
    """data-URL base64 de la imagen, o ``None`` si no se puede leer/es muy grande."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.warning("no se pudo leer imagen %s: %s", path, exc)
        return None
    if len(raw) > _MAX_IMAGE_BYTES:
        log.warning("imagen %s excede %d bytes; se omite", path, _MAX_IMAGE_BYTES)
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def build_image_content(text: str, image_paths: list[str] | None) -> str | list[dict[str, Any]]:
    """Arma el contenido de un mensaje con texto + imágenes.

    Returns:
        - ``str`` (el texto) si no hay imágenes válidas — seguro para modelos
          solo-texto.
        - lista de bloques ``[{type:text}, {type:image_url}, ...]`` si hay ≥1
          imagen legible.

    Las rutas inexistentes o ilegibles se omiten (best-effort).
    """
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for raw_path in image_paths or []:
        if not raw_path:
            continue
        p = Path(raw_path)
        if not p.is_file():
            log.debug("imagen no encontrada, se omite: %s", raw_path)
            continue
        url = _data_url(p)
        if url is not None:
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    if len(blocks) == 1:  # solo el texto → devolver texto plano
        return text
    return blocks
