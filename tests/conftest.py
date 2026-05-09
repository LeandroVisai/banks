"""Configuración base de pytest.

Markers definidos en pyproject.toml:
- unit, integration, e2e, slow, requires_gpu, requires_models

Fixtures globales se agregarán acá conforme avance el refactor (Fase 1+).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla tests de variables de entorno externas que puedan filtrarse."""
    for var in ("PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("BANKS_ENV", "test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests que requieren GPU/modelos si no están disponibles."""
    skip_gpu = pytest.mark.skip(reason="GPU no disponible (set BANKS_TEST_GPU=1)")
    skip_models = pytest.mark.skip(reason="Modelos no descargados (set BANKS_TEST_MODELS=1)")

    has_gpu = os.environ.get("BANKS_TEST_GPU") == "1"
    has_models = os.environ.get("BANKS_TEST_MODELS") == "1"

    for item in items:
        if "requires_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "requires_models" in item.keywords and not has_models:
            item.add_marker(skip_models)
