# Banks RAG — Sistema RAG + Agente Multimodal · Banco Central de Chile

Sistema de análisis con IA para documentos financieros del BCCh: Comunicados, Minutas del Consejo, Fed Statements, research JPMorgan y Monitor PM. Combina **retrieval híbrido + cross-encoder reranking** sobre un corpus multimodal (texto + gráficos de PDFs) con un **agente de tool-calling** que discierne autónomamente cuándo buscar documentos, ejecutar queries SQL del catálogo curado, o devolver gráficos.

**Diseñado para ejecutarse en GPU H100 80GB sin internet**, con modelos y dependencias pre-descargadas.

---

## Arquitectura

```
Documentos (PDFs + Excel Monitor PM)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  Pipeline ingesta (banks-ingest)                        │
│  extract → enrich → vectorize (Qwen3-VL-Embedding-8B)  │
│  → persist (PostgreSQL 16 + pgvector HNSW cosine)       │
└──────────────────┬──────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │ hybrid_search   │ vector + BM25 → RRF → MMR → reranker
          └────────┬────────┘  (jina-reranker-v3, activo por defecto)
                   │
      ┌────────────────────────────────────────────────────────┐
      │ Router determinista (router-v1)                        │
      │  selecciona ≥1 especialista por vocabulario            │
      └──────────┬─────────────────────────────────────────────┘
                 │ despacha en paralelo
      ┌──────────▼──────────────────────────────────────────────┐
      │ 8 Sub-agentes especialistas (LlamaCppEngine)            │
      │                                                         │
      │  Mercado Cambiario (FX)  · No Residentes                │
      │  AFP · Fondos Mutuos (FFMM)                            │
      │  Renta Fija · Liquidez & Balance                       │
      │  Analista Documentos · Analista Política Monetaria     │
      │                                                         │
      │  Tools cuantitativos: discover_query · execute_query   │
      │    compute_variation · compute_spread                  │
      │    compute_composition · compute_aggregate             │
      │    get_series_stats · detect_anomaly                   │
      │                                                         │
      │  Tools documentales: search_documents · search_visuals │
      │    list_documents · get_document_chunks                │
      │    compare_meetings · get_recent_policy_decisions      │
      └────────────┬────────────────────────────────────────────┘
                   │
          ┌────────▼────────┐
          │   FastAPI API   │ /v1/chat · /v1/search · /metrics
          └─────────────────┘
```

### Capas (Clean Architecture)

| Capa | Paquete | Responsabilidad |
|---|---|---|
| Domain | `banks_rag.domain` | Dataclasses puras: `Document`, `Chunk`, `SearchResult`, `ParsedQuery` |
| Application | `banks_rag.application` | Casos de uso: ingesta, retrieval, agente, evaluación |
| Infrastructure | `banks_rag.infrastructure` | Adaptadores: embeddings, LLM, PostgreSQL, reranker, observabilidad |
| Interface | `banks_rag.interface` | FastAPI + CLI (`banks-ingest`, `banks-search`, `banks-eval`) |

---

## Quickstart

### Instalación

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"      # producción
pip install -e ".[dev]"      # desarrollo + tests
```

### Variables de entorno mínimas

```bash
export PGHOST=localhost PGDATABASE=rag_banco PGUSER=<user> PGPASSWORD=<pass>
export BANKS_LLM_FAMILY=mock   # o 'qwen' con BANKS_LLM_MODEL_PATH=<ruta.gguf>
```

**Windows (dev box / RTX 3080):** usar `run_local_cpu.ps1` — fija variables de entorno y levanta la API directamente.

```powershell
.\run_local_cpu.ps1
```

### Ingesta

```bash
banks-ingest run --source Datos_prueba/
```

### Búsqueda

```bash
banks-search "política monetaria 2024" --k 5

# Vía API
uvicorn banks_rag.interface.api.main:create_app --factory --port 8080
curl http://localhost:8080/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "TPM actual", "k": 5}'
```

### Frontend (interface2)

`interface2/` es un dashboard estático (HTML + JS + ApexCharts) que la API monta en `/`. **Mismo origen que el backend** — abrir directamente `http://localhost:8080/`, no servir aparte.

```bash
# Con el paquete instalado (pip install -e ".[api]")
python -m banks_rag.interface.api.main

# Sin instalar (sandbox/dev rápido)
PYTHONPATH=src python3 -m banks_rag.interface.api.main
```

Notas:
- La pill superior derecha pasa por `SIN CONEXIÓN → LLM CARGANDO → OK` mientras `llama-cpp` carga. Los charts del catálogo no esperan al LLM; sólo el panel "Agente IA" sí.
- Los charts usan `/v1/query/{dataset_id}` (DuckDB sobre parquets) y los KPIs son hardcoded en `kpi.js` — no requieren PostgreSQL para renderizar.
- Bajo el campo de chat hay un **segmented control de razonamiento** (`Rápido` / `Análisis` / `Profundo`) que el usuario elige por consulta: envía `thinking_mode` (`off`/`adaptive`/`on`) en el request; el server cae a `BANKS_THINKING_MODE` si no se especifica. Se persiste en `localStorage`.
- El agente **grafica las series temporales** que analiza: cuando una respuesta usó datos de un parquet, debajo del texto se renderiza un mini-gráfico de área (ApexCharts) con los puntos que el agente consultó. Backend: `series_used[].points` en `/v1/chat`.
- **Adjuntar archivos** (botón `+`): PDF/TXT/CSV/Excel se suben a `/v1/upload` (base64) y se inyectan como **contexto efímero** del turno — el agente los analiza sin indexarlos. Ver [`docs/DISENO_UPLOADS.md`](docs/DISENO_UPLOADS.md).
- **Leer en voz** (🔊): cada respuesta tiene un botón TTS (voz del navegador, offline).

### Reporte de noticias JARVIS (informe diario + audio)

Paquete **aislado** `src/jarvis_news/` (separado de `banks_rag` por seguridad:
procesa datos externos scrapeados). Analiza un JSON de noticias en
`data_pipeline/Noticias_scrapping/` y genera un reporte estructurado con las más
importantes del día (prioriza por tema del dominio × alcance, resume vía
map-reduce — los ~100 artículos no caben en una sola ventana). Opcionalmente
produce un **audio con la voz de JARVIS** en español e inglés (motor `sapi`: voz
del SO + efecto DSP, sin descargar modelos).

```bash
# CLI: texto + audio (ES y EN)
python scripts/jarvis_news_report.py --audio --top-n 25

# API
curl -X POST http://localhost:8080/v1/news-report \
  -H "Content-Type: application/json" \
  -d '{"top_n": 25}'   # json_path opcional; por defecto el más reciente
```

Tarda (varias llamadas al LLM serializado); no es un endpoint interactivo.
Instalación, `.env`, voz JARVIS offline y roadmap de integración al agente:
[`docs/SETUP_JARVIS.md`](docs/SETUP_JARVIS.md).

### Verificación de sub-agentes

```bash
# Verifica los 8 especialistas contra el LLM del .env (local o H100)
python scripts/verify_subagents.py

# Solo un especialista
python scripts/verify_subagents.py --subagent fx

# Con techo de iteraciones
python scripts/verify_subagents.py --max-iters 6
```

### Evaluación

```bash
make eval          # golden set completo → eval_report.md
make eval-ci       # gate CI (falla si recall@5 cae >5%)
```

---

## Modelos

| Rol | Modelo | Tamaño | Ubicación |
|---|---|---|---|
| Embedding texto + imagen | `Qwen/Qwen3-VL-Embedding-8B` | ~16GB | `models/Qwen--Qwen3-VL-Embedding-8B/` |
| Reranker | `jinaai/jina-reranker-v3` | ~1.1GB | `models/jinaai--jina-reranker-v3/` |
| LLM principal | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | ~16GB | `models/` |
| LLM alternativo | `gemma-4-26B-A4B-it.gguf` | ~14GB | `models/` |

El pipeline auto-detecta `models/<owner>--<name>/` antes de descargar desde HuggingFace.

---

## Catálogo de parquets (107 datasets)

El agente consulta datos del Monitor PM vía DuckDB sobre parquets offline. Los 107 datasets están todos consultables — los 7 snapshots sin columna de fecha (`posicion_rfl_afp`, `cambiario_afp`, `attribution`, `dcv_composicion_ffmm`, `fixing_*`, `variacion_dcv_afp`) usan `build_fetch_sql` sin filtro temporal; los de serie temporal usan `date_column()` estricta para analytics.

| Segmento | Descripción |
|---|---|
| `mercado_cambiario` | USD/CLP, forwards, microestructura FX, índice monedas LATAM |
| `posiciones_cambiarias` | RFL AFP, posición cambiaria AFP, no residentes |
| `fondos_pension` | AFP composición, FFMM composición, DCV, atribución |
| `renta_fija_chile` | Curvas BTP/BTU, SPC CLP/UF, spreads crédito |
| `instrumentos_bcch` | PDBC, TIB, DAP vs swap CLP |
| `renta_fija_eeuu` | Curva UST, OIS SOFR, SOFR plazos, PRIME USD |
| `liquidez_bancaria` | LCR/NSFR por banco, ratio liquidez/obligaciones |
| `balance_bancario` | Depósitos, colocaciones, fixing, variación DCV |
| `tasas_internacionales` | SOFR, PRIME, OIS, spread onshore DAP USD, TADO |
| `politica_monetaria` | Expectativas TPM (MiPr) |
| `commodities` | Precio cobre |

Parquets en `data_pipeline/parquet/`. Esquemas completos (id, columnas, tipos, enums, `date_range`) en [`sql_catalog/parquet_catalog.yaml`](sql_catalog/parquet_catalog.yaml). La SQL la arma siempre la tool (`_parquet_query.build_fetch_sql`); el LLM solo elige `dataset_id` + filtros — nunca escribe SQL.

---

## Tests

```bash
PYTHONPATH=src pytest tests/unit/ -q    # 742 tests, <2s, sin BD ni modelos
PYTHONPATH=src pytest tests/ -q         # + integración (requiere PostgreSQL)
```

---

## Observabilidad

- **Logs**: JSON estructurado con `request_id`, `session_id`, `tool_call_id` (`BANKS_LOG_JSON=true`).
- **Métricas**: `GET /metrics` (Prometheus). Dashboard en [`deploy/grafana/banks_rag_dashboard.json`](deploy/grafana/banks_rag_dashboard.json).
- **Tracing**: OpenTelemetry OTLP opcional (`BANKS_TRACING=otlp`).
- **Rate limit**: token bucket por API key (`BANKS_RATE_LIMIT_RPM=60`).
- **Auth**: `BANKS_API_KEYS=key1,key2`. `/healthz` y `/metrics` son públicos.

---

## Deploy en H100

Ver [`docs/DEPLOYMENT_H100.md`](docs/DEPLOYMENT_H100.md): systemd, nginx, logrotate, backup, Prometheus, troubleshooting.

Para el setup inicial del runtime en el servidor (variables de entorno, venv, CUDA DLLs, activar el venv) ver [`docs/SETUP_SERVIDOR.txt`](docs/SETUP_SERVIDOR.txt).

---

## Estructura del repo

```
banks/
├── src/banks_rag/
│   ├── domain/          # dataclasses puras
│   ├── application/
│   │   ├── agent/       # router-v1, 8 sub-agentes, tools, prompts
│   │   ├── ingestion/   # extract → enrich → vectorize → load
│   │   ├── retrieval/   # hybrid_search, query_parser, fusion, reranker
│   │   └── evaluation/  # retrieval_metrics, ragas_runner, golden set
│   ├── infrastructure/  # adaptadores (embeddings, llm, postgres, reranker, observability)
│   └── interface/       # FastAPI + CLI
├── tests/
│   ├── unit/            # 742 tests, sin BD ni modelos reales
│   └── integration/     # requieren PostgreSQL + pgvector
├── scripts/
│   └── verify_subagents.py   # verificación e2e de los 8 sub-agentes
├── sql_catalog/         # parquet_catalog.yaml — 107 datasets con esquema
├── data_pipeline/       # parquet/ — 107 datasets crudos del DW (única fuente)
├── data/
│   ├── golden_set/      # retrieval.jsonl, sql_routing.jsonl, generation.jsonl
│   └── chat_logs/       # JSONL diario de turnos del agente (gitignored)
├── deploy/
│   ├── systemd/
│   ├── nginx/
│   └── grafana/         # dashboard JSON
├── docs/
│   ├── REFACTOR_PLAN.md
│   ├── DEPLOYMENT_H100.md
│   ├── SETUP_JARVIS.md       # analizador de noticias + voz JARVIS (TTS offline)
│   └── SETUP_SERVIDOR.txt    # guía de entorno H100 (venv, CUDA, .env)
├── run_local_cpu.ps1    # script Windows dev box (fija env vars + levanta API)
├── pyproject.toml       # única fuente de deps y entry points
└── Makefile
```







# Tarea: añadir edición in-place + exportar versión final a un informe HTML

Quiero agregar a un HTML autocontenido (informe generado) una **barra flotante de edición**
con 3 botones, de modo que se pueda corregir el texto en el navegador y exportar una copia limpia.

## Requisitos de los 3 botones
1. **Editar texto** — toggle: activa/desactiva `contentEditable` en los elementos de texto
   (`p, h3, h4, h5, li, td, th, figcaption, .lead`), resaltándolos con un borde punteado.
   NO debe tocar gráficos/charts (excluir lo que esté dentro de su contenedor).
2. **Guardar borrador** — guarda el archivo CON el editor incluido (para seguir editando luego).
3. **Guardar versión final** — pide carpeta/nombre y guarda una copia **limpia**: sin la barra
   y sin el script del editor, de modo que al abrir esa copia NO aparezcan botones. Conserva
   el resto (índice/scrollspy, gráficos, tablas, contenido con las ediciones).

## Puntos técnicos CRÍTICOS (sin esto no funciona o queda sucio)
- **La barra se crea por JS** (`document.createElement`), nunca en el markup. Antes de serializar
  haz `bar.remove()`, serializa `document.documentElement.outerHTML`, y vuelve a `appendChild`.
  Así la barra nunca queda en el archivo guardado.
- **Separa los scripts**: el scrollspy/otros van en un `<script>` normal (se conservan); el editor
  va en su PROPIO `<script id="__editor__">`. La "versión final" elimina solo ese script por id
  con un `String.replace` sobre el HTML serializado:
  `html.replace(/[ \t]*<script id="__editor__">[\s\S]*?<\/script>\s*/, '\n')`.
- **Gotcha del cierre de script**: dentro del código del editor, cualquier `</script>` literal
  (incluido el de la propia regex de arriba) debe escribirse como `<\/script>` (con backslash),
  o el navegador cierra el `<script>` antes de tiempo y la regex de strip matchea el lugar
  equivocado.
- **Guardado**: usa la File System Access API (`window.showSaveFilePicker` → `createWritable`)
  cuando exista (Chrome/Edge/Opera → escribe directo sobre el archivo elegido). Si no existe
  (Safari/Firefox), **fallback a descarga** vía `Blob` + `<a download>`. Maneja `AbortError`
  (usuario canceló el picker) sin romper.
- "Guardar borrador" reusa un handle guardado (no vuelve a preguntar). "Guardar versión final"
  pide ubicación nueva cada vez (handle=null) para que se elija la carpeta.
- Oculta la barra al imprimir: `@media print{#__editbar{display:none!important}}`.

## Código de referencia (el bloque del editor, adaptable)
```html
<script id="__editor__">
(function(){
  var SEL='p,h3,h4,h5,li,td,th,figcaption,.lead', draftHandle=null, editing=false;
  var bar=document.createElement('div'); bar.id='__editbar';
  bar.style.cssText='position:fixed;right:18px;bottom:18px;z-index:99999;display:flex;gap:7px;'+
    'background:#0f2942;border-radius:11px;padding:9px 11px;font:13px system-ui';
  function mk(t,bg){var b=document.createElement('button');b.textContent=t;
    b.style.cssText='border:0;border-radius:8px;padding:9px 12px;color:#fff;cursor:pointer;background:'+bg;return b;}
  var bEdit=mk('Editar texto','#0e7c86'), bDraft=mk('Guardar borrador','#3a6079'),
      bFinal=mk('Guardar versión final','#b8860b');
  [bEdit,bDraft,bFinal].forEach(function(e){bar.appendChild(e);});
  document.body.appendChild(bar);

  function setEditable(on){document.querySelectorAll('main '+SEL).forEach(function(el){
    if(el.closest('.chart, .fullbleed')) return;           // excluir gráficos
    el.contentEditable=on; el.style.outline=on?'1px dashed #b8860b':''; el.style.outlineOffset='3px';});}
  bEdit.onclick=function(){editing=!editing; setEditable(editing);
    bEdit.textContent=editing?'Terminar edición':'Editar texto';};

  function serialize(final){
    editing=false; setEditable(false); bar.remove();
    var html='<!DOCTYPE html>\n'+document.documentElement.outerHTML;
    document.body.appendChild(bar);
    if(final) html=html.replace(/[ \t]*<script id="__editor__">[\s\S]*?<\/script>\s*/,'\n');
    return html;
  }
  function download(html,name){var a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([html],{type:'text/html'})); a.download=name; a.click();}
  async function write(html,handle,name){
    if(window.showSaveFilePicker){ try{
      var h=handle||await showSaveFilePicker({suggestedName:name,
        types:[{accept:{'text/html':['.html']}}]});
      var w=await h.createWritable(); await w.write(html); await w.close(); return h;
    }catch(e){ if(e.name==='AbortError') return 'abort'; download(html,name); } }
    else download(html,name);
  }
  bDraft.onclick=async function(){var r=await write(serialize(false),draftHandle,'informe.html');
    if(r&&r!=='abort') draftHandle=r;};
  bFinal.onclick=function(){write(serialize(true),null,'informe_VERSION_FINAL.html');};
})();
</script>
```

## Importante sobre el flujo
Si el HTML lo genera un script (Python u otro), agrega este bloque al final del `<body>` en el
GENERADOR (no editando el HTML a mano), para que sobreviva a las regeneraciones. Avisa que las
ediciones hechas en el navegador viven solo en el HTML: si se regenera desde el generador, se pisan.
 