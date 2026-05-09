"""Configuración: paths estáticos + settings dinámicas (Pydantic)."""

from . import paths
from .settings import (
    Settings,
    get_settings,
    override_settings,
    reset_settings,
)

__all__ = [
    "paths",
    "Settings",
    "get_settings",
    "override_settings",
    "reset_settings",
]
