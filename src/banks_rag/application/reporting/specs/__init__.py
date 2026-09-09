"""Specs de informes curados por familia (uno por archivo).

Cada spec es la fuente de verdad del ORDEN, los títulos y las secciones de ESE
informe (réplica de un informe real del BCCh). ``get_spec(family)`` los resuelve.
"""

from __future__ import annotations

from banks_rag.application.reporting.report_spec import FamilyReportSpec

from .afp_spec import AFP_SPEC
from .cambiarioam_spec import CAMBIARIOAM_SPEC
from .dcv_spec import DCV_SPEC
from .ffmm_spec import FFMM_SPEC
from .fx_spec import FX_SPEC
from .nr_spec import NR_SPEC
from .rf_spec import RF_SPEC

_SPECS: dict[str, FamilyReportSpec] = {
    FFMM_SPEC.family: FFMM_SPEC,
    NR_SPEC.family: NR_SPEC,
    AFP_SPEC.family: AFP_SPEC,
    FX_SPEC.family: FX_SPEC,
    DCV_SPEC.family: DCV_SPEC,
    CAMBIARIOAM_SPEC.family: CAMBIARIOAM_SPEC,
    RF_SPEC.family: RF_SPEC,
}

# El ``segment`` del catálogo no siempre coincide con el ``family`` del spec
# (NR: segment ``no_residentes`` vs family ``nr``). El proceso de TEXTO
# (``generate_parquet_report``) resuelve el spec por SEGMENT; sin este alias caería
# al corte común e ignoraría ``share_weekly_cutoff``. Mantener sincronizado con los
# ``segment`` reales del catálogo.
_SEGMENT_ALIASES: dict[str, str] = {
    "no_residentes": NR_SPEC.family,
    "fx_diferencial": FX_SPEC.family,
    # El informe de renta fija se arma con datasets de varios segmentos (rf_tasas,
    # spc_ois, dcv, mercado_monetario); los dos de tasas apuntan a su spec.
    "rf_tasas": RF_SPEC.family,
    "rf_volumenes": RF_SPEC.family,
}


def get_spec(family: str) -> FamilyReportSpec | None:
    """Spec curado de una familia o ``segment`` del catálogo (``None`` si no hay)."""
    key = family.strip().lower()
    return _SPECS.get(_SEGMENT_ALIASES.get(key, key))


def available_families() -> list[str]:
    return sorted(_SPECS)


__all__ = [
    "AFP_SPEC", "CAMBIARIOAM_SPEC", "DCV_SPEC", "FFMM_SPEC", "FX_SPEC", "NR_SPEC",
    "RF_SPEC",
    "available_families", "get_spec",
]
