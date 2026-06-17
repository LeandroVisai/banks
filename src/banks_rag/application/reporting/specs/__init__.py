"""Specs de informes curados por familia (uno por archivo).

Cada spec es la fuente de verdad del ORDEN, los títulos y las secciones de ESE
informe (réplica de un informe real del BCCh). ``get_spec(family)`` los resuelve.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import FamilyReportSpec

from .afp_spec import AFP_SPEC
from .ffmm_spec import FFMM_SPEC
from .nr_spec import NR_SPEC

_SPECS: dict[str, FamilyReportSpec] = {
    FFMM_SPEC.family: FFMM_SPEC,
    NR_SPEC.family: NR_SPEC,
    AFP_SPEC.family: AFP_SPEC,
}


def get_spec(family: str) -> FamilyReportSpec | None:
    """Spec curado de una familia (``None`` si no hay)."""
    return _SPECS.get(family.strip().lower())


def available_families() -> list[str]:
    return sorted(_SPECS)


__all__ = ["AFP_SPEC", "FFMM_SPEC", "NR_SPEC", "available_families", "get_spec"]
