"""Guard de coherencia: el parquet_catalog.yaml refleja los parquets REALES.

El no-alucinar depende de que el catálogo sea fiel a los datos: si una columna
del YAML no existe en el parquet (o viceversa), el descubrimiento y las
analytics apuntarían a columnas inexistentes y el especialista quedaría
degradado. Este test compara las columnas declaradas con el schema real de cada
parquet (lectura del footer con pyarrow) y, para columnas categóricas con
``values``, que los valores enum declarados existan en el parquet.

Marcado ``integration`` (lee filesystem). Se salta si los parquets no están
presentes o si pyarrow no está instalado.
"""

from __future__ import annotations

import pytest

from banks_rag.infrastructure.sql.parquet_catalog_loader import (
    get_parquet_dir,
    load_parquet_catalog,
)

pq = pytest.importorskip("pyarrow.parquet")

_CATALOG = load_parquet_catalog()
_PARQUET_DIR = get_parquet_dir()
_PRESENT = [ds for ds in _CATALOG if (_PARQUET_DIR / ds.file).exists()]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _PRESENT, reason=f"No hay parquets en {_PARQUET_DIR}; nada que validar."
    ),
]


@pytest.mark.parametrize("ds", _PRESENT, ids=[d.id for d in _PRESENT])
def test_yaml_columns_match_parquet(ds) -> None:
    real_cols = set(pq.read_schema(_PARQUET_DIR / ds.file).names)
    yaml_cols = {c.name for c in ds.columns}
    assert yaml_cols == real_cols, (
        f"[{ds.id}] desajuste de columnas catálogo↔parquet — "
        f"solo en YAML: {sorted(yaml_cols - real_cols)}; "
        f"solo en parquet: {sorted(real_cols - yaml_cols)}"
    )


# Datasets con al menos una columna categórica declarando `values`.
_WITH_ENUMS = [ds for ds in _PRESENT if any(c.values for c in ds.columns)]


@pytest.mark.parametrize("ds", _WITH_ENUMS, ids=[d.id for d in _WITH_ENUMS])
def test_yaml_enum_values_exist_in_parquet(ds) -> None:
    """Los valores enum declarados deben existir en el parquet (los reales pueden
    ser un superconjunto; lo que importa es que el LLM no filtre por un valor
    declarado que no existe)."""
    import pyarrow.parquet as _pq

    table = _pq.read_table(_PARQUET_DIR / ds.file)
    for col in ds.columns:
        if not col.values:
            continue
        real = {str(v) for v in table.column(col.name).to_pylist() if v is not None}
        declared = {str(v) for v in col.values}
        missing = declared - real
        assert not missing, (
            f"[{ds.id}].{col.name}: valores declarados en YAML ausentes en el "
            f"parquet: {sorted(missing)}"
        )
