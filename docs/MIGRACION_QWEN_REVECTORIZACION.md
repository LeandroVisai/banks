# Runbook — Migración a la infraestructura definitiva (Qwen + re-vectorización)

Guía para llevar el servidor **Windows H100 (offline)** a la infraestructura
definitiva tras el overhaul de agentes (Fase 0 + Fase 1). Complementa
`DEPLOYMENT_H100.md` (deploy base) y `H100_VALIDATION_CHECKLIST.md`.

> **¿Por qué es obligatorio re-vectorizar?**
> El embedder por defecto cambió a **Qwen3-VL-Embedding-8B**. Un índice vectorial
> creado con otro modelo **no es compatible**: aunque ambos sean 4096-dim, los
> espacios vectoriales difieren y la similitud coseno entre modelos distintos no
> tiene sentido. Por eso hay que re-embeber todo el corpus con el embedder nuevo
> y reconstruir el índice HNSW. Además, la extracción visual ahora es **solo de
> IPoM**, lo que cambia qué se ingesta → requiere re-extracción (no basta
> re-vectorizar).

Todo lo que sigue es **offline**: los modelos ya están en `models/`. No hay
descargas ni red.

---

## 0. Pre-checks (antes de tocar nada)

```powershell
# Desde la raíz del repo, con el venv activado
.\.venv\Scripts\Activate.ps1

# 0.1 Modelos presentes en models/ (convención <owner>--<name> o <name>)
Get-ChildItem models\ | Select-Object Name
#   Debe verse: Qwen3-VL-Embedding-8B  (carpeta del embedder)
#               Qwen3.6-27B-...gguf    (archivo GGUF del LLM)

# 0.2 PyMuPDF instalado (necesario para extraer los gráficos de los IPoM)
python -c "import fitz; print('pymupdf OK', fitz.__doc__.splitlines()[0])"

# 0.3 Tests unitarios verdes (sin BD ni modelos)
$env:PYTHONPATH="src"; python -m pytest tests/unit/ -q
```

## 1. Configuración (`.env` en la raíz del repo)

Copiar de `.env.example` y ajustar. Claves de la migración:

```ini
# LLM definitivo
BANKS_LLM_FAMILY=qwen
BANKS_LLM_MODEL_PATH=Qwen3.6-27B-UD-Q4_K_XL.gguf   # relativo a models/

# Embedder definitivo (resuelve a models/Qwen3-VL-Embedding-8B/)
RAG_EMBEDDING_MODEL=Qwen3-VL-Embedding-8B
RAG_VISUAL_IMG_WEIGHT=0.7

# Descubrimiento semántico del catálogo (ON en producción; OFF cae a léxico+alias)
BANKS_CATALOG_SEMANTIC=true
```

> El nombre exacto del `.gguf` y de la carpeta del embedder deben coincidir con
> lo que hay en `models/`. `BANKS_LLM_MODEL_PATH` se resuelve contra `models/`.

## 2. Re-extracción + re-vectorización del corpus

Limpia el índice viejo y reconstruye con el embedder nuevo. `persist reset`
hace DROP + recrea schema e índices HNSW con la dimensión correcta.

```powershell
.\.venv\Scripts\Activate.ps1

# 2.1 Pipeline completo: extract -> enrich -> vectorize -> persist(load)
#     (extract aplica el gating visual IPoM-only; vectorize usa Qwen3-VL-Embedding)
banks-ingest extract   --data-root Datos_prueba\
banks-ingest enrich
banks-ingest vectorize --batch-size 4
#   En [vectorize] confirma: "Modelo: Qwen3-VL-Embedding-8B (dim=4096, ... multimodal=True)"

# 2.2 Recrear schema + índices con la dim nueva, luego cargar
banks-ingest persist reset      # DROP CASCADE + setup (destructivo: borra el índice viejo)
banks-ingest persist load
```

Alternativa en un solo paso (sin reset explícito; úsalo solo si la BD está
vacía o ya tiene la dim correcta):

```powershell
banks-ingest full --data-root Datos_prueba\ --batch-size 4
```

> Si quieres conservar el índice viejo en paralelo (rollback), usa
> `RAG_TABLE_PREFIX=qwenvl_` en `vectorize`/`persist` para escribir en tablas
> separadas y apuntar la API a ese prefijo.

## 3. Verificación post-migración

```powershell
.\.venv\Scripts\Activate.ps1; $env:PYTHONPATH="src"

# 3.1 Dimensión del embedder = 4096
python -c "from banks_rag.infrastructure.embeddings import build_default_embedder as b; e=b(); e.encode_text(['x']); print('dim', e.dim, 'multimodal', e.is_multimodal)"

# 3.2 Stats del corpus cargado
banks-ingest persist stats

# 3.3 Visuales SOLO de IPoM (Comunicados/Minutas RPM deben dar 0 chunks VISUAL)
psql -d rag_banco -c "SELECT d.doc_type_category, count(*) FROM chunks c JOIN documents d ON c.document_id=d.document_id WHERE c.kind='VISUAL' GROUP BY 1 ORDER BY 2 DESC;"
#   Esperado: solo IPOM aparece con conteo > 0; el resto, ausente.

# 3.4 Descubrimiento de datasets (las 3 preguntas que el modelo alucinaba en los logs)
python -c "from banks_rag.infrastructure.sql.parquet_catalog_loader import load_parquet_catalog, search_datasets as s; c=load_parquet_catalog(); [print(q, '->', [e.id for e in s(c,q,top_k=3)]) for q in ['DV01 de los fondos de pensiones','composicion de cartera AFP','posicion de no residentes']]"
#   Esperado: dv01_spc_afp / allocation_int_nac / posicion_nr_* en el top-3.
```

### 3.5 Arrancar la API y probar end-to-end

```powershell
uvicorn banks_rag.interface.api.main:create_app --factory --host 127.0.0.1 --port 8080
```

```powershell
# Liveness / readiness (el LLM tarda en cargar; espera a readyz=ok)
curl.exe -s http://localhost:8080/healthz
curl.exe -s http://localhost:8080/readyz

# Chat: DV01 de fondos de pensiones — DEBE consultar dv01_spc_afp/dv01_ffmm y
# NO inventar cifras. Revisa en la respuesta: series_used no vacío y
# ungrounded_numbers vacío.
curl.exe -s http://localhost:8080/v1/chat -H "Content-Type: application/json" `
  -d '{\"message\": \"Tienes informacion del DV01 de los fondos de pensiones?\"}'

# Dato inexistente: debe responder "no disponible", sin cifras inventadas.
curl.exe -s http://localhost:8080/v1/chat -H "Content-Type: application/json" `
  -d '{\"message\": \"Cual es la tasa de mora de las tarjetas de credito retail?\"}'
```

### 3.6 Replay de los logs del frontend (regresión real)

Re-ejecuta las preguntas de `data/logs_intermedios/logsdefronted.jsonl` contra la
API nueva y verifica, por turno:
- las cifras provienen de tools (`series_used` poblado, `ungrounded_numbers` vacío);
- `list_documents` no arroja error SQL;
- el ruteo va al especialista de mercado correcto (campo `agent` en `tool_trace`).

## 4. Evaluación / gate

```powershell
banks-eval all --output eval_report.md
# Gate CI de recall@5:
make eval-ci   # (o el equivalente: banks-eval ci)
```

## 5. Señales de que algo quedó inconsistente

| Síntoma | Causa probable | Acción |
|---|---|---|
| Búsquedas devuelven basura / similitud baja | Corpus vectorizado con otro embedder | Re-vectorizar (paso 2) con `RAG_EMBEDDING_MODEL` correcto |
| `persist load` salta chunks por "dimensión inválida" | Schema con dim != 4096 | `banks-ingest persist reset` y recargar |
| `search_visuals` trae visuales de Comunicados/Minutas | Índice viejo (pre IPoM-gating) | Re-extraer + re-vectorizar |
| Chat inventa cifras (ungrounded_numbers no vacío) | discover_query no halló el dataset | Verificar paso 3.4; revisar alias en `domain_knowledge/financial_aliases.py` |
| `BANKS_CATALOG_SEMANTIC=true` no mejora ranking | Embedder no carga (cae a léxico+alias) | Ver logs WARNING "Embedder no disponible"; verificar `models/` |

---

**Rollback rápido**: poner `BANKS_LLM_FAMILY` y `RAG_EMBEDDING_MODEL` a los
valores previos y apuntar a las tablas anteriores (o `RAG_TABLE_PREFIX` del set
viejo). El código nuevo es compatible hacia atrás salvo por el espacio vectorial:
lo único que NO se puede mezclar es un índice de un embedder con queries de otro.
