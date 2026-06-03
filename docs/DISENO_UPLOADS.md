# Subir archivos al chat — diseño y estado

Permite al usuario adjuntar archivos en el chat para que el agente los use.

## Estado actual: Fase 1+2 — contexto efímero (IMPLEMENTADO)

El contenido del archivo se extrae a **texto** y se inyecta como **contexto del
turno**. No se indexa nada (ni pgvector ni el catálogo de parquets); vive en
disco temporal con TTL.

Flujo:

```
Frontend (botón +)  → POST /v1/upload  (archivo en base64, sin python-multipart)
                        → upload_store.save_upload extrae:
                            PDF/TXT/MD → texto (extract_pages para PDF)
                            CSV/Excel  → esquema + primeras filas + estadística
                          guarda data/uploads/<id>.json (TTL 24h)
                        → {upload_id, kind, preview}
Frontend (chip)     → POST /v1/chat con attachments:[upload_id]
                        → chat.py: load_upload → bloque de contexto
                        → run_agent(attachments_context=...):
                            inyecta el bloque (marcado DATOS, no instrucciones)
                            en el task de los especialistas y en la síntesis
                        → nada persiste
```

Piezas:
- `src/banks_rag/infrastructure/uploads/upload_store.py` — extracción + store + TTL.
- `src/banks_rag/interface/api/routes/uploads.py` — `POST /v1/upload`.
- `ChatRequest.attachments`; `run_agent(attachments_context=...)`.
- Frontend: `interface2/assets/js/chat.js` (`_mountAttach`, chips), `api.upload`.

Límites: 10 MB/archivo, 5 archivos/turno, texto acotado a `MAX_TEXT_CHARS`,
tabla a `MAX_TABLE_ROWS`. Validación de tipo/tamaño/path-traversal.

## Fase avanzada (PENDIENTE — requiere H100 con pgvector + embedder Qwen)

### A. Dataset CSV/Excel SQL-consultable y graficable (efímero)
Hoy una tabla adjunta entra como **texto** (esquema + primeras filas). Para
tablas grandes o para **graficarlas** con las analytics tools, conviene
exponerla como un **dataset efímero consultable**:

- Convertir el CSV/Excel a un **parquet temporal** y construir un
  `ParquetDataset` con `file` = ruta absoluta del parquet (así
  `parquet_path()` lo usa directo; `build_fetch_sql` ya opera sobre parquets).
- Hacerlo visible a las tools **solo en ese turno** sin tocar el catálogo
  global. Camino recomendado: un `ContextVar` con los datasets efímeros que
  `fetch_rows_from_dataset` (en `_parquet_query.py`) y `discover_query` mergeen
  con `load_parquet_catalog()`. `load_parquet_catalog` no está memoizado, así
  que el merge es limpio.
- run_agent: setear el ContextVar al inicio del turno (heredado por los
  especialistas vía `asyncio.gather`) e inyectar en el contexto el `dataset_id`
  + esquema para que el LLM sepa que existe. Resetear al final.
- Routing: si hay tabla adjunta, garantizar al menos un especialista con tools
  cuantitativas (discover/execute/analytics).
- Beneficio: el agente consulta/filtra/calcula sobre la tabla y la **grafica**
  en la respuesta (reusa `series_used[].points`).

### B. Indexar permanente al corpus
Va **contra** el espíritu efímero, pero útil si se quiere que un documento
subido quede **buscable** a futuro:

- Pasar el archivo por el pipeline de ingesta (extract → enrich → vectorize →
  load) y guardarlo en pgvector como un documento más.
- **Requisitos**: PostgreSQL + pgvector y el **embedder Qwen3-VL-8B (4096-dim)**
  cargado (no disponible en el sandbox Mac). Re-embeber con el modelo del corpus.
- Considerar: namespacing/propiedad por usuario, borrado, y que la dimensión del
  embedding coincida con el corpus (4096).
