# Auditoría del Agente RAG — Hallazgos pendientes

**Fecha:** 2026-05-15
**Relación:** complementa `docs/AGENT_AUDIT.md` (auditoría completa, 32 hallazgos).

Este documento detalla los hallazgos **aún no implementados**. Los demás
27 hallazgos (3 críticos + 7 altos + 11 medios + 6 bajos) ya fueron corregidos
y la suite de 498 tests unitarios pasa. Lo que queda aquí requiere una **base
de datos / DuckDB en vivo para validarse**, es **trabajo de curación de
datos**, o es una **decisión de dependencias** — por eso se documenta en vez
de aplicarse a ciegas.

---

## Estado global

| Severidad | Total | Aplicados | Pendientes |
|---|---|---|---|
| 🔴 Crítico | 3 | 3 | 0 |
| 🟠 Alto | 8 | 7 | **1** (A7) |
| 🟡 Medio | 14 | 11 | **3** (M6, M8, M12) |
| 🟢 Bajo | 7 | 6 | **1** (B2) |

5 hallazgos pendientes. Ninguno bloquea el funcionamiento actual; todos mejoran
robustez/seguridad/observabilidad bajo carga real.

---

## A7 · 🟠 Alto · PostgreSQL síncrono sin pool de conexiones

**Archivos:** `src/banks_rag/infrastructure/persistence/postgres_repo.py` (todo el módulo); callers en `application/retrieval/hybrid_search.py` y `application/agent/tools/search_documents.py`.

**Problema:** `PostgresRepo` usa `psycopg2` síncrono; cada método abre y cierra una conexión nueva (`connect()`). No hay pool. El docstring del módulo promete un `AsyncPostgresRepo` "cuando se construya el FastAPI service" — el servicio ya existe (`interface/api/routes/chat.py`) pero esa clase no está en el repo. Bajo carga concurrente las conexiones se crean/destruyen una por request; si una ruta async llama al repo sync sin `to_thread`, bloquea el event loop.

**Por qué quedó pendiente:** migrar a un pool async requiere PostgreSQL en vivo para validar el comportamiento del pool, probar concurrencia real y confirmar que `hybrid_search` + `recall_queries` siguen devolviendo los mismos resultados. No es seguro hacerlo a ciegas.

**Plan de implementación:**
1. Agregar `psycopg[pool]` (psycopg3) a `pyproject.toml`.
2. Crear `AsyncPostgresRepo` con `psycopg_pool.AsyncConnectionPool` (tamaño configurable vía env, p. ej. `PG_POOL_MIN=2`, `PG_POOL_MAX=10`).
3. Portar las queries de `recall_queries.py` (`vector_recall`, `lexical_recall`, `date_importance_fallback`, `get_db_embedding_dim`) a variantes `async`.
4. Hacer `hybrid_search` async (o exponer `hybrid_search_async`) y que `search_documents`/`search_visuals` la usen directamente sin `asyncio.to_thread`.
5. Conservar `PostgresRepo` síncrono para los CLIs de ingesta (`banks-ingest`), que no corren bajo un event loop.
6. **Transición mínima si no se quiere el refactor completo ahora:** envolver toda llamada a `PostgresRepo` desde código async en `asyncio.to_thread` (ya se hace en `search_documents.py`, verificar que sea consistente en todas las tools).

**Validación:** levantar PostgreSQL, correr `tests/integration/`, y disparar N requests concurrentes a `/v1/chat` midiendo que las latencias no se acumulan linealmente.

---

## M8 · 🟡 Medio · DuckDB sin sandbox de filesystem

**Archivo:** `src/banks_rag/application/agent/tools/execute_query.py` — función `_run_duckdb` (~línea 135, `duckdb.connect(":memory:")`).

**Problema:** la conexión DuckDB permite por defecto `read_parquet`, `read_csv`, `COPY` y la instalación de extensiones sobre rutas arbitrarias del filesystem. Combinado con el patrón de interpolación de `render_sql` (hallazgo C3, ya mitigado con la whitelist de tipos), un SQL malicioso podría leer archivos fuera del directorio de snapshots.

**Por qué quedó pendiente:** deshabilitar `enable_external_access` a secas **rompería** las queries del catálogo, que legítimamente hacen `read_parquet('{snapshots_dir}/...')`. El sandbox correcto (`SET allowed_directories`) depende de la versión de DuckDB instalada y debe probarse contra las 23 queries reales. Sin DuckDB en vivo no se puede verificar que nada se rompe.

**Plan de implementación:** en `_run_duckdb`, justo tras `connect`, ejecutar:
```sql
SET autoinstall_known_extensions=false;
SET autoload_known_extensions=false;
SET allowed_directories=['<ruta absoluta de data_pipeline/snapshots>'];
SET enable_external_access=true;   -- restringido por allowed_directories
```
`allowed_directories` está disponible en DuckDB ≥ 0.10. Para versiones anteriores, alternativa: validar antes de ejecutar que el SQL solo referencie rutas bajo el snapshots_dir.

**Validación:** con DuckDB instalado, correr las 23 queries del catálogo y confirmar que siguen devolviendo filas; luego intentar un `read_csv('/etc/passwd')` y confirmar que DuckDB lo bloquea.

---

## M6 · 🟡 Medio · Golden set pequeño y evaluación de routing incompleta

**Archivos:** `data/golden_set/retrieval.jsonl` (30 casos), `sql_routing.jsonl` (30), `generation.jsonl` (15); `src/banks_rag/application/evaluation/retrieval_metrics.py` — `evaluate_sql_routing` (~línea 177).

**Problema:**
- El golden set es pequeño (75 casos para 5 tipos de documento).
- Los 30 casos de `sql_routing` son todos `expected_route: "sql"` — sin negativos, no se mide el falso positivo de routing.
- `evaluate_sql_routing` solo verifica que la rama elegida sea `sql`; **nunca compara la query elegida contra `expected_query_id`** (el campo existe en el `.jsonl` pero no se usa).

**Por qué quedó pendiente:** ampliar el golden set es trabajo de **curación de datos**, no de código. Evaluar el `query_id` correcto requiere una función nueva que corra `discover_query` y cuya salida amerita revisión manual de resultados reales.

**Plan de implementación:**
1. Ampliar los `.jsonl`: más casos de retrieval; en `sql_routing` agregar casos `expected_route: "rag"` y `"visual"` para medir falsos positivos; ampliar `generation`.
2. Agregar `evaluate_query_discovery(cases, discover_fn)` en `retrieval_metrics.py`: corre `discover_query` por caso y compara el top-1 contra `expected_query_id`. **No requiere BD** — `discover_query` solo usa el catálogo YAML (`load_catalog`).
3. (Opcional, mayor esfuerzo) curar `relevant_chunk_ids` por caso de retrieval para computar **recall real** además del `hit_rate_at_k` actual.

**Validación:** correr `make eval` y revisar `eval_report.md`.

---

## M12 · 🟡 Medio · Credencial PostgreSQL por defecto y sin SSL

**Archivo:** `src/banks_rag/infrastructure/persistence/postgres_repo.py` (~línea 113, `os.getenv("PGPASSWORD", "postgres")`).

**Problema:** el password por defecto es `"postgres"` — una credencial real y adivinable. No se configura `sslmode`, así que la conexión puede ir en claro.

**Por qué quedó pendiente:** cambiar la lógica de conexión es sensible al entorno de despliegue. El entorno de desarrollo actual usa autenticación local (peer/trust) sin password; forzar fallos o SSL a ciegas rompería el setup local del usuario.

**Plan de implementación:**
1. Cambiar el default de `"postgres"` a `""` (cadena vacía → psycopg usa peer auth local, no un password real adivinable).
2. En producción (cuando `PGHOST` no sea `localhost`/`127.0.0.1`), **fallar con error claro** si `PGPASSWORD` no está seteado.
3. Añadir `sslmode` configurable vía env `PGSSLMODE` (default `prefer`; documentar `require` para producción).

**Validación:** probar conexión local (peer auth, sin password) y conexión remota con password + `sslmode=require`.

---

## B2 · 🟢 Bajo · `dispatch` no valida argumentos contra el JSON-schema

**Archivo:** `src/banks_rag/application/agent/tools/registry.py` — función `dispatch` (~línea 85).

**Problema:** `dispatch` solo detecta argumentos inválidos cuando Python lanza `TypeError` al hacer `**arguments`. Un argumento de tipo incorrecto pero aceptado por la firma (p. ej. `k="cinco"` donde la tool hace `int(k)` internamente, o un enum fuera de rango) no se valida contra el JSON-schema declarado de la tool.

**Por qué quedó pendiente:** validar contra JSON-schema requiere agregar la dependencia `jsonschema` al proyecto — es una decisión de dependencias que conviene consultar.

**Plan de implementación:**
1. Agregar `jsonschema` a `pyproject.toml`.
2. En `dispatch`, antes de invocar la tool, validar `arguments` contra el `parameters` del schema de la tool (accesible vía `signature_hint(name)` o el `TOOL_SCHEMAS`).
3. Si la validación falla, devolver el mismo formato de error estructurado que hoy se usa para `TypeError` (con `expected_signature`), para que el LLM pueda auto-corregirse.

**Validación:** tests unitarios en `tests/unit/test_tool_registry.py` con argumentos de tipo y enum incorrectos.

---

## Resumen

| ID | Severidad | Bloqueante para… | Requiere |
|---|---|---|---|
| A7 | Alto | rendimiento bajo carga concurrente | PostgreSQL en vivo |
| M8 | Medio | hardening de seguridad | DuckDB en vivo |
| M6 | Medio | confianza en la evaluación | curación de datos |
| M12 | Medio | hardening de credenciales | acceso al entorno de despliegue |
| B2 | Bajo | robustez del tool-calling | decisión de dependencia (`jsonschema`) |

Prioridad sugerida cuando haya entorno con BD: **A7 → M8 → M12 → M6 → B2**.
