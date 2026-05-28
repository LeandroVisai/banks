# data_pipeline — Parquets de series macro/financieras

Aloja los **parquets crudos** de series macro/financieras del Data Warehouse que el
agente consulta de forma offline. Ya **no** hace extracción SQL en vivo ni genera
snapshots: los archivos en [`parquet/`](parquet/) son la única fuente de datos.

## Por qué parquet y no PostgreSQL ni queries en vivo

- **Sin internet en el servidor del agente**: el DW (`mesadineOLTP_`, `dbo.Base_DMN`, etc.) vive en otra red. El agente no debe depender de conectividad.
- **Portabilidad**: copiar archivos `.parquet` (~unos pocos MB cada uno) es trivial.
- **Velocidad**: lectura columnar + filtros vectorizados (DuckDB/pandas/pyarrow) son órdenes de magnitud más rápidos que ir a la base para series largas.
- **Una sola fuente de verdad**: el agente y cualquier otro consumidor leen los mismos parquets.

## Estructura

```
data_pipeline/
├── README.md                ← este archivo
├── __init__.py
├── requirements.txt         ← deps mínimas de lectura (pandas/pyarrow/numpy)
└── parquet/                 ← 107 parquets crudos del DW (la fuente)
    ├── <serie>.parquet
    └── ...
```

## Cómo lo consume el agente

El agente **no** lee este paquete directamente: lo hace a través del catálogo y de
las tools de `banks_rag`.

1. El catálogo [`sql_catalog/parquet_catalog.yaml`](../sql_catalog/parquet_catalog.yaml)
   describe cada dataset (`id`, `file`, `columns` con tipos/enums, `date_range`) y
   apunta a un archivo dentro de `parquet/`.
2. Las tools del agente (`discover_query`, `execute_query`, analytics) leen ese
   catálogo vía `infrastructure/sql/parquet_catalog_loader.py`.
3. La SQL **la arma siempre la tool** en
   [`_parquet_query.build_fetch_sql`](../src/banks_rag/application/agent/tools/_parquet_query.py)
   y la ejecuta DuckDB sobre el parquet. El LLM nunca escribe SQL: solo pasa
   `dataset_id`, columnas, fechas y filtros estructurados.

## Refrescar / agregar datos

La generación de los parquets ocurre **fuera de este repo**, en una máquina con
acceso al DW. El flujo es:

1. Regenerar el/los `.parquet` con las columnas reales de la serie.
2. Copiarlos a `data_pipeline/parquet/`.
3. Añadir/actualizar la entrada en `sql_catalog/parquet_catalog.yaml` (ver la guía
   "Agregar nuevos datasets al catálogo de parquets" en el [CLAUDE.md](../CLAUDE.md) raíz).

No hay que tocar código Python para agregar una serie nueva.
