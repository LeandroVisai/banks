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

from fastapi import APIRouter, HTTPException

from banks_rag.infrastructure.uploads import UploadError, save_upload
from banks_rag.interface.api.schemas import UploadRequest, UploadResponse

router = APIRouter()
log = logging.getLogger(__name__)


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
