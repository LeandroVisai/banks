"""Endpoint ``GET /v1/images/{chunk_id}`` — sirve el PNG de un chunk visual.

Seguridad: el ``image_path`` viene de la BD (controlada por ingestion),
pero validamos que el archivo resuelto esté dentro de ``IMAGES_DIR``
para evitar path traversal si alguien manipuló registros directamente.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from banks_rag.config import paths

router = APIRouter()


def _is_path_within(child: Path, parent: Path) -> bool:
    """``True`` si ``child`` está dentro de ``parent`` tras resolver ``..``."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@router.get("/v1/images/{chunk_id}", tags=["images"])
async def get_image(request: Request, chunk_id: str) -> FileResponse:
    """Sirve el PNG asociado a un chunk visual.

    Errors:
        404 — chunk no existe, no es visual, o el archivo no se encuentra.
        403 — el image_path apunta fuera del images dir (defensa en profundidad).
    """
    deps = getattr(request.app.state, "deps", None)
    if deps is None or deps.repo is None:
        raise HTTPException(status_code=503, detail="Servicio no inicializado")

    row = await asyncio.to_thread(deps.repo.get_chunk_image, chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Chunk no encontrado")
    if not row.get("image_path"):
        raise HTTPException(
            status_code=404,
            detail=f"El chunk {chunk_id!r} no tiene imagen asociada",
        )

    img_path = Path(row["image_path"])
    if not img_path.is_absolute():
        img_path = paths.ROOT / img_path

    if not _is_path_within(img_path, paths.IMAGES_DIR):
        raise HTTPException(
            status_code=403,
            detail="image_path fuera del directorio permitido",
        )
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado en disco")

    return FileResponse(
        path=str(img_path),
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Chunk-Kind": row.get("kind") or "VISUAL",
            "X-Visual-Caption": row.get("visual_caption") or "",
        },
    )
