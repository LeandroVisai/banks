"""
data_pipeline — Parquets crudos de series macro/financieras del Data Warehouse.

Ya NO contiene extracción SQL en vivo ni snapshots derivados: los parquets en
``parquet/`` son la única fuente. El agente los consulta vía el catálogo
``sql_catalog/parquet_catalog.yaml`` y las tools de ``banks_rag`` arman la SQL
(``application/agent/tools/_parquet_query.py``) — el LLM nunca escribe SQL.

Para refrescar los datos: regenerar los .parquet en otra máquina (con acceso al
DW) y copiarlos a ``parquet/``. Este paquete no abre conexiones al SQL Server.
"""
__version__ = "3.0.0"
