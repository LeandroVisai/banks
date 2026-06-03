"""POST ``/v1/upload`` — sube un archivo como contexto efímero del chat.

Recibe el archivo en base64 (JSON, sin python-multipart), extrae su contenido a
texto (PDF/TXT → texto; CSV/Excel → tabla + estadística) y lo guarda con un
``upload_id``. El frontend luego manda ese id en ``attachments`` de ``/v1/chat``
y el agente lo inyecta como contexto. Nada se indexa: caduca por TTL.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from banks_rag.infrastructure.uploads import (
    UploadError,
    save_upload,
    upload_images_dir,
)
from banks_rag.interface.api.schemas import UploadRequest, UploadResponse

router = APIRouter()
log = logging.getLogger(__name__)


def _is_path_within(child: Path, parent: Path) -> bool:
    """``True`` si ``child`` está dentro de ``parent`` tras resolver ``..``."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@router.post("/v1/upload", response_model=UploadResponse, tags=["chat"])
async def upload(body: UploadRequest) -> UploadResponse:
    try:
        content = base64.b64decode(body.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"content_base64 inválido: {exc}") from exc

    try:
        record = await asyncio.to_thread(save_upload, body.filename, content)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Error procesando upload %s", body.filename)
        raise HTTPException(status_code=500, detail=f"No se pudo procesar el archivo: {exc}") from exc

    return UploadResponse(
        upload_id=record["upload_id"],
        kind=record["kind"],
        name=record["name"],
        chars=record["chars"],
        truncated=record["truncated"],
        preview=record["preview"],
    )


@router.get("/v1/uploads/{upload_id}/images/{asset_file}", tags=["chat"])
async def get_upload_image(upload_id: str, asset_file: str) -> FileResponse:
    """Sirve un PNG de gráfico extraído de un PDF adjunto (modo análisis de
    documento). Los PNG viven en ``data/uploads/images/<upload_id>/`` y caducan
    con el upload (TTL).

    Errors:
        404 — upload/archivo inexistente o caducado.
        403 — el path resuelto cae fuera del directorio del upload (traversal).
    """
    images_dir = upload_images_dir(upload_id)
    if images_dir is None:
        raise HTTPException(status_code=404, detail="upload_id inválido")

    img_path = images_dir / asset_file
    if not _is_path_within(img_path, images_dir):
        raise HTTPException(status_code=403, detail="ruta fuera del directorio permitido")
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    return FileResponse(
        path=str(img_path),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
