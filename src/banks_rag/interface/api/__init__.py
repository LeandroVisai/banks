"""FastAPI app + routes + middleware del servicio unificado."""

from .dependencies import AppState
from .main import create_app

__all__ = ["create_app", "AppState"]
