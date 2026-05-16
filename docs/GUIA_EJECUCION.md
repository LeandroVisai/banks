# Guía de ejecución — Banks RAG

Runbook paso a paso para levantar **todo el sistema de cero**: desde la
instalación, pasando por el pipeline de datos, hasta el agente funcionando en
su interfaz web. Cada paso indica el comando exacto, qué hace y qué deberías
ver.

> Todos los comandos se corren desde la raíz del repo:
> `cd ~/Documents/GitHub/Proyecto/banks`

---

## 0. Requisitos previos

Antes de empezar necesitas tener instalado:

| Requisito | Para qué | Cómo verificar |
|---|---|---|
| **Python 3.12+** | correr el paquete | `python3 --version` |
| **PostgreSQL 16** corriendo | almacenar el corpus + embeddings | `pg_isready` |
| **Extensión pgvector** | búsqueda vectorial (índice HNSW) | se instala en el Paso 3 |
| **Documentos fuente** | el corpus a indexar | carpeta `Datos_prueba/` con los PDFs |
| **Modelos** (opcional) | LLM y embeddings reales | carpeta `models/` — ver Paso 2 |

Si PostgreSQL no está corriendo, arráncalo (en macOS con Homebrew):
```bash
brew services start postgresql@16
```

---

## 1. Instalar el paquete

Instala `banks_rag` en modo editable. Esto deja disponibles los comandos
`banks-ingest`, `banks-search`, `banks-chat` y `banks-eval`.

```bash
pip install -e .
```

Equivalente: `make install`. Para incluir herramientas de test/lint usa
`make install-dev`; para la evaluación con RAGAS, `make install-eval`.

**Qué esperar:** termina con `Successfully installed banks_rag-0.1.0`.

Verifica que los comandos quedaron disponibles:
```bash
banks-ingest --help
```

---

## 2. Configurar las variables de entorno

El sistema se configura por variables de entorno. Para una corrida local
mínima, exporta estas en tu terminal:

```bash
# ── Base de datos ──
export PGHOST=localhost
export PGDATABASE=rag_banco
export PGUSER=$(whoami)        # tu usuario del sistema
export PGPASSWORD=             # vacío = autenticación local (peer)

# ── Modelo de lenguaje del agente ──
export BANKS_LLM_FAMILY=mock   # 'mock' = sin GPU, para probar el sistema
```

**Sobre `BANKS_LLM_FAMILY`:**
- `mock` — el agente responde con un LLM simulado. **Úsalo para verificar que
  todo el pipeline funciona** sin necesitar GPU ni descargar modelos. Los
  gráficos, el catálogo SQL y la búsqueda funcionan con datos reales; solo las
  respuestas conversacionales del agente son simuladas.
- `qwen` o `gemma` — LLM real. Requiere además:
  ```bash
  export BANKS_LLM_MODEL_PATH=/ruta/al/modelo.gguf
  ```
  Los modelos viven en `models/<owner>--<name>/`; si la carpeta existe se usa,
  si no, se descarga de HuggingFace (necesita internet).

> Consejo: pon estos `export` en un archivo `.env.local` y haz `source .env.local`
> al abrir la terminal, para no repetirlos.

---

## 3. Crear la base de datos y el esquema

Un solo comando crea la base de datos, instala la extensión `pgvector`, y crea
las tablas + los 18 índices (HNSW para vectores, GIN para JSONB/full-text).
Es **idempotente** — puedes correrlo más de una vez sin problema.

```bash
banks-ingest persist setup
```

**Qué hace:** crea la BD `rag_banco`, ejecuta `CREATE EXTENSION vector`, y
monta el esquema (`documents`, `chunks`) con todos los índices.

**Qué esperar:** mensajes de creación de tablas e índices, sin errores.

> Si necesitas empezar de cero borrando todo: `banks-ingest persist reset`
> (destructivo — hace DROP CASCADE y vuelve a crear el esquema).

---

## 4. Correr el pipeline de datos

El pipeline tiene 4 etapas: **extract → enrich → vectorize → persist**.
Convierte los PDFs de `Datos_prueba/` en un corpus semántico consultable.

### Opción A — todo de una vez (recomendado)

```bash
banks-ingest full
```

Esto corre las 4 etapas en orden. Es lo que necesitas la mayoría de las veces.

### Opción B — etapa por etapa (para depurar o re-correr una sola)

```bash
banks-ingest extract     # PDF/Excel → chunks.json + documents.json
banks-ingest enrich      # añade taxonomía semántica → chunks_enriched.json
banks-ingest vectorize   # genera embeddings (texto + imagen) → chunks_vectorized.json
banks-ingest persist load   # carga el corpus vectorizado a PostgreSQL
```

**Qué hace cada etapa:**
1. **extract** — lee los PDFs (Comunicados, Minutas, Fed, research) y el Excel
   del Monitor PM; los parte en fragmentos (`chunks`).
2. **enrich** — clasifica cada fragmento (variables económicas, secciones,
   entidades) con la taxonomía del dominio.
3. **vectorize** — genera los embeddings de 4096 dimensiones con el modelo
   Qwen3-VL (texto + imágenes de gráficos). **Esta etapa es la más lenta** y
   usa GPU si está disponible.
4. **persist load** — inserta documentos y fragmentos en PostgreSQL.

**Qué esperar:** cada etapa imprime su progreso y un resumen al final
(nº de documentos, nº de chunks).

### Verificar que la ingesta funcionó

```bash
banks-ingest persist stats
```

Muestra las métricas del corpus cargado (nº de documentos, chunks, etc.) sin
modificar nada. Si ves números mayores a 0, la ingesta fue exitosa.

Prueba rápida de búsqueda desde la terminal:
```bash
banks-search "política monetaria 2024" --k 5
```

---

## 5. Levantar el servicio (API + interfaz del agente)

La API de FastAPI expone el agente **y además sirve la interfaz web** del
mismo origen. Un solo comando levanta todo:

```bash
make serve
```

Equivale a `python -m banks_rag.interface.api.main`. Levanta el servidor en
**http://localhost:8080**.

> **Si NO instalaste el paquete** (saltaste el Paso 1), usa:
> `PYTHONPATH=src python3 -m banks_rag.interface.api.main`

**Qué esperar:** `Uvicorn running on http://0.0.0.0:8080`.

Verifica que respondió, en otra terminal:
```bash
curl http://localhost:8080/healthz
```
Debe devolver un JSON con `"status":"ok"`.

> El endpoint `/readyz` puede tardar 1-2 minutos en pasar a `ok` mientras el
> LLM carga (solo con `BANKS_LLM_FAMILY=qwen/gemma`). Con `mock` es inmediato.

Para desarrollo con recarga automática al cambiar código: `make serve-dev`.

---

## 6. Usar el agente — el resultado final

Con el servicio corriendo, tienes **dos formas** de hablar con el agente.

### A. Interfaz web (el "template" donde vive el agente)

Abre en el navegador:

```
http://localhost:8080/
```

Esta es la **Terminal de Análisis Financiero** (`interface2/`). Desde el
sidebar puedes:
- **Agente GOEM** — chatear con el agente RAG a pantalla completa.
- **Resumen General / Mercados / Mercados en Línea** — dashboards de datos.
- **S1–S16** — secciones temáticas de mercado.

El botón **"Agente IA"** (arriba a la derecha) abre el chat en un panel lateral
desde cualquier sección.

### B. Agente por terminal

Sin navegador, directo desde la consola:

```bash
banks-chat
```

Abre una sesión de chat con el agente en la terminal. Útil para pruebas
rápidas o uso por scripts.

**Probar el agente vía API directamente:**
```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué decidió el Consejo del BCCh en su última reunión?", "history": []}'
```

---

## 7. Evaluación (opcional)

Para medir la calidad del sistema contra el golden set:

```bash
make eval            # golden set completo → eval_report.md
make eval-retrieval  # solo métricas de retrieval (hit-rate@k, MRR, nDCG)
make eval-routing    # solo métricas de routing SQL
```

`make eval-ci` corre el gate de CI (falla si la métrica cae más de 5% vs el
baseline). Requiere haber instalado las dependencias de evaluación
(`make install-eval`).

---

## Resumen — la secuencia mínima de cero a agente

```bash
cd ~/Documents/GitHub/Proyecto/banks

# 1. Instalar
pip install -e .

# 2. Variables de entorno
export PGHOST=localhost PGDATABASE=rag_banco PGUSER=$(whoami) PGPASSWORD=
export BANKS_LLM_FAMILY=mock

# 3. Base de datos
banks-ingest persist setup

# 4. Pipeline de datos
banks-ingest full
banks-ingest persist stats          # verificar

# 5. Levantar el servicio
make serve

# 6. Abrir el agente en el navegador:
#    http://localhost:8080/
```

---

## Comandos de referencia rápida

| Comando | Qué hace |
|---|---|
| `make help` | lista todos los targets del Makefile |
| `make install` / `install-dev` / `install-eval` | instalar el paquete (base / +tests / +RAGAS) |
| `banks-ingest persist setup` | crear BD + esquema + índices |
| `banks-ingest full` | pipeline completo de datos |
| `banks-ingest persist stats` | métricas del corpus cargado |
| `banks-search "consulta" --k 5` | búsqueda desde terminal |
| `make serve` | levantar API + interfaz en :8080 |
| `make serve-dev` | igual, con recarga automática |
| `banks-chat` | agente por terminal |
| `make eval` | evaluación contra el golden set |
| `make test` | suite de tests |
| `make clean` | limpiar archivos temporales |

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `pg_isready` falla | PostgreSQL no está corriendo | `brew services start postgresql@16` |
| `banks-ingest: command not found` | el paquete no se instaló | repetir el Paso 1 (`pip install -e .`) |
| `persist setup` falla con error de permisos | el usuario PG no puede crear BD | usar un `PGUSER` con permisos, o crear la BD a mano |
| La interfaz carga pero los gráficos dicen "Sin conexión" | la API no está corriendo en :8080 | revisar que `make serve` siga activo |
| El agente responde genérico / "no tengo información" | `BANKS_LLM_FAMILY=mock` (LLM simulado) | configurar `qwen`/`gemma` + `BANKS_LLM_MODEL_PATH` |
| La interfaz carga sin estilos / scripts | caché del navegador | recargar con `Cmd+Shift+R` |
| `/readyz` queda en `loading` | el LLM real aún está cargando | esperar 1-2 min; no afecta a los gráficos |

---

_Para el despliegue en servidor H100 (systemd, nginx, Prometheus) ver
`docs/DEPLOYMENT_H100.md`._
