"""Smoke test que valida que el paquete se importa correctamente."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    import banks_rag

    assert banks_rag.__version__ == "0.1.0"


@pytest.mark.unit
def test_layers_exist() -> None:
    """Las cuatro capas de Clean Architecture están presentes."""
    import banks_rag.application  # noqa: F401
    import banks_rag.domain  # noqa: F401
    import banks_rag.infrastructure  # noqa: F401
    import banks_rag.interface  # noqa: F401
