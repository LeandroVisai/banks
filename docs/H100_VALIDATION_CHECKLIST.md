# Checklist de Validación H100

Ejecutar en orden después de `setup_h100.sh`. Tachar cada ítem al completarlo.

---

## Paso 1 — Prerequisitos

- [ ] `nvidia-smi` muestra la H100 con driver ≥ 535
- [ ] `nvcc --version` muestra CUDA ≥ 12.1
- [ ] `python3.11 --version` disponible
- [ ] PostgreSQL 16 corriendo: `systemctl status postgresql`
- [ ] pgvector instalado: `psql -c "SELECT * FROM pg_extension WHERE extname='vector';"`

---

## Paso 2 — Instalación

- [ ] `setup_h100.sh` completó sin errores
- [ ] `.venv/bin/banks-ingest --help` responde
- [ ] `.venv/bin/banks-eval --help` responde
- [ ] `/etc/banks_rag.env` editado (contraseñas, API keys, ruta del .gguf)

---

## Paso 3 — Modelos transferidos

Transferir desde workstation antes de iniciar el servicio:

```bash
# Desde la workstation:
rsync -avz --progress models/jinaai--jina-reranker-v3/ user@h100:/opt/banks_rag/models/jinaai--jina-reranker-v3/
rsync -avz --progress models/Qwen--Qwen3-VL-Embedding-8B/ user@h100:/opt/banks_rag/models/Qwen--Qwen3-VL-Embedding-8B/
rsync -avz --progress models/Qwen3.6-27B-UD-Q4_K_XL.gguf user@h100:/opt/banks_rag/models/
rsync -avz --progress data_pipeline/snapshots/           user@h100:/opt/banks_rag/data_pipeline/snapshots/
rsync -avz --progress Datos_prueba/                      user@h100:/opt/banks_rag/Datos_prueba/
```

- [ ] `ls models/` muestra los 3 directorios/archivos de modelo
- [ ] `ls data_pipeline/snapshots/*.parquet | wc -l` muestra ≥ 14 parquets

---

## Paso 4 — Base de datos

```bash
# Crear BD y usuario (como postgres):
sudo -u postgres psql <<'SQL'
CREATE USER banks_rag WITH PASSWORD '<contraseña del .env>';
CREATE DATABASE rag_banco OWNER banks_rag;
\c rag_banco
CREATE EXTENSION IF NOT EXISTS vector;
SQL

# Verificar:
psql -U banks_rag -d rag_banco -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
```

- [ ] Extension `vector` visible en la BD
- [ ] Ingesta inicial completada:
  ```bash
  systemctl start banks-ingest
  journalctl -u banks-ingest -f   # esperar "ingesta completa"
  ```
- [ ] `psql -U banks_rag -d rag_banco -c "SELECT COUNT(*) FROM chunks;"` muestra > 0

---

## Paso 5 — Servicio API

```bash
systemctl start banks-api
systemctl status banks-api   # debe mostrar "active (running)"
journalctl -u banks-api -f   # esperar "API iniciando"
```

- [ ] `systemctl status banks-api` → active (running)
- [ ] `curl http://localhost:8080/healthz` → `{"status":"ok",...}`
- [ ] `curl http://localhost:8080/readyz` → status `ok` o `loading` (esperar hasta `ok`)

---

## Paso 6 — Validación automática

```bash
BANKS_API_KEY=<tu-key> bash scripts/validate_h100.sh
```

- [ ] Check 1 (liveness): PASS
- [ ] Check 2 (readiness): PASS
- [ ] Check 3 (métricas): PASS
- [ ] Check 4 (búsqueda RAG): PASS
- [ ] Check 5 (routing golden set): PASS
- [ ] Pregunta narrativa 1 — decisión Consejo: responde con cita
- [ ] Pregunta narrativa 2 — proyecciones inflación: responde con cita
- [ ] Pregunta narrativa 3 — riesgos externos: responde con cita
- [ ] Pregunta cuantitativa 1 — USD/CLP: muestra datos de parquet
- [ ] Pregunta cuantitativa 2 — precio cobre: muestra datos de parquet

---

## Paso 7 — Swap de modelos

Verificar que el switch `qwen ↔ gemma` funciona:

```bash
# Cambiar a Gemma
sed -i 's/BANKS_LLM_FAMILY=qwen/BANKS_LLM_FAMILY=gemma/' /etc/banks_rag.env
sed -i 's|Qwen3.6-27B.*\.gguf|gemma-4-26B-A4B-it.gguf|' /etc/banks_rag.env
systemctl restart banks-api

# Esperar readyz=ok y probar
curl http://localhost:8080/readyz
BANKS_API_KEY=<key> curl http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué es la política monetaria?"}'

# Volver a Qwen
sed -i 's/BANKS_LLM_FAMILY=gemma/BANKS_LLM_FAMILY=qwen/' /etc/banks_rag.env
sed -i 's|gemma-4-26B.*\.gguf|Qwen3.6-27B-UD-Q4_K_XL.gguf|' /etc/banks_rag.env
systemctl restart banks-api
```

- [ ] Gemma responde coherentemente
- [ ] Qwen responde coherentemente tras swap

---

## Paso 8 — Nginx y acceso externo

```bash
cp deploy/nginx/banks.conf /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/banks.conf /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

- [ ] `curl -k https://banks.internal/healthz` → ok (desde red interna)
- [ ] HTTP redirige a HTTPS

---

## Paso 9 — Reporte final

```bash
BANKS_API_KEY=<key> banks-eval all \
  --output "docs/H100_VALIDATION_$(date +%Y-%m-%d).md" \
  --update-baseline
```

- [ ] `docs/H100_VALIDATION_<fecha>.md` generado
- [ ] `eval_baseline.json` guardado (sirve de baseline para futuros `make eval-ci`)
- [ ] Commitear el reporte: `git add docs/H100_VALIDATION_*.md eval_baseline.json && git commit -m "chore: validación H100 $(date +%Y-%m-%d)"`

---

## Troubleshooting rápido

| Síntoma | Causa | Acción |
|---|---|---|
| `readyz` → `loading` por más de 2 min | LLM muy grande para la VRAM | Reducir `BANKS_LLM_N_GPU_LAYERS` a 20-30 |
| `readyz` → `degraded` | BD inaccesible | `systemctl status postgresql`; verificar `.env` |
| `503` en `/v1/chat` | LLM no cargó | Ver `journalctl -u banks-api -n 50` |
| `401` | API key incorrecta | Verificar `BANKS_API_KEYS` en `.env` |
| `429` | Rate limit | Aumentar `BANKS_RATE_LIMIT_RPM` |
| GPU no detectada por llama.cpp | CUDA mal linkeado | Recompilar `llama-cpp-python` con `CMAKE_ARGS="-DLLAMA_CUBLAS=on"` |
| OOM al cargar modelo | VRAM insuficiente | Usar `.gguf` Q3 o Q2; reducir `n_gpu_layers` |
