"""
data_pipeline — Catálogo y acceso a series macro/financieras del Data Warehouse.

Carga las queries de querys/Monitor.py vía un catálogo declarativo
(series_catalog.yaml) y las ejecuta contra el SQL Server en tiempo real.

Módulos principales:
  dw_store      — consulta en vivo al DW (uso en producción, ambos chatbots)
  parquet_store — acceso a snapshots parquet (backup/offline)
  extract.py    — genera snapshots parquet desde el DW (uso manual/cron)
"""
__version__ = "2.0.0"
