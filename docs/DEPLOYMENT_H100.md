# Deployment en H100 — Guía paso a paso

## Requisitos previos

| Componente | Versión mínima |
|---|---|
| Ubuntu / RHEL | 22.04 LTS / 9 |
| CUDA | 12.1+ |
| Python | 3.11+ |
| PostgreSQL + pgvector | 16 + 0.7 |
| nvidia-driver | 535+ |

---

## 1. Preparar el entorno Python

```bash
# Clonar el repo (o rsync desde workstation)
git clone <repo-url> /opt/banks_rag
cd /opt/banks_rag

# Crear venv aislado
python3.11 -m venv .venv
source .venv/bin/activate

# Instalar wheels offline (copiar primero desde workstation)
# En workstation: pip download -r requirements.txt -d dist/wheels/
pip install --no-index --find-links dist/wheels/ -e ".[api]"

# Verificar GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 2. Variables de entorno

Crear `/etc/banks_rag.env`:

```bash
# PostgreSQL
PGHOST=localhost
PGPORT=5432
PGUSER=banks_rag
PGPASSWORD=postgres
PGDATABASE=rag_banco
RAG_TABLE_PREFIX=

# LLM
BANKS_LLM_FAMILY=qwen
BANKS_LLM_MODEL_PATH=models/Qwen--Qwen3-8B-Instruct-Q4_K_M.gguf
BANKS_LLM_N_CTX=16384
BANKS_LLM_N_GPU_LAYERS=-1
BANKS_LLM_TEMPERATURE=0.2
BANKS_LLM_MAX_TOKENS=2048

# API
BANKS_API_HOST=127.0.0.1
BANKS_API_PORT=8080
BANKS_API_KEYS=<key1>,<key2>
BANKS_LOG_LEVEL=INFO
BANKS_LOG_JSON=true
BANKS_RATE_LIMIT_RPM=60
BANKS_RATE_LIMIT_BURST=10

# Tracing (opcional, default off)
BANKS_TRACING=off
# BANKS_TRACING=otlp
# BANKS_OTLP_ENDPOINT=http://localhost:4317
```

## 3. PostgreSQL + pgvector

```bash
# Instalar pgvector (como root)
apt-get install postgresql-16-pgvector

# Crear BD y usuario
sudo -u postgres psql <<'SQL'
CREATE USER banks_rag WITH PASSWORD '<contraseña>';
CREATE DATABASE rag_banco OWNER banks_rag;
\c rag_banco
CREATE EXTENSION IF NOT EXISTS vector;
SQL

# Cargar schema y datos (primera vez)
source .venv/bin/activate && source /etc/banks_rag.env
banks-ingest run --source Datos_prueba/
```

## 4. Systemd units

### 4.1 API principal

Copiar `deploy/systemd/banks-api.service` a `/etc/systemd/system/`:

```ini
[Unit]
Description=Banks RAG API
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=exec
User=banks_rag
Group=banks_rag
WorkingDirectory=/opt/banks_rag
EnvironmentFile=/etc/banks_rag.env
ExecStart=/opt/banks_rag/.venv/bin/python -m banks_rag.interface.api.main
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=banks-api

# Límites de recursos
LimitNOFILE=65536
# CUDA necesita acceso a /dev/nvidia*
DeviceAllow=/dev/nvidia* rw
DeviceAllow=/dev/nvidiactl rw

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now banks-api
systemctl status banks-api
```

### 4.2 Worker de ingesta (one-shot)

```bash
# Re-ingesta manual cuando llegan documentos nuevos
systemctl start banks-ingest
# O directamente:
source .venv/bin/activate
banks-ingest run --source /datos/nuevos/
```

## 5. Nginx reverse proxy

Copiar `deploy/nginx/banks.conf` a `/etc/nginx/sites-available/`:

```nginx
upstream banks_api {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name banks.internal;

    ssl_certificate     /etc/ssl/certs/banks.crt;
    ssl_certificate_key /etc/ssl/private/banks.key;

    # Seguridad
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header Strict-Transport-Security "max-age=31536000" always;

    # Métricas solo desde red interna (Prometheus)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://banks_api;
    }

    location / {
        proxy_pass http://banks_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
        proxy_send_timeout 30s;

        # Streaming (para SSE en futuras versiones)
        proxy_buffering off;
    }
}

server {
    listen 80;
    server_name banks.internal;
    return 301 https://$host$request_uri;
}
```

```bash
ln -s /etc/nginx/sites-available/banks.conf /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## 6. Log rotation

Crear `/etc/logrotate.d/banks`:

```
/var/log/banks/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        systemctl kill -s HUP banks-api
    endscript
}
```

## 7. Backup de la BD

Script diario `/opt/banks_rag/scripts/backup_db.sh`:

```bash
#!/bin/bash
set -euo pipefail
source /etc/banks_rag.env

BACKUP_DIR=/backups/banks_rag
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"

pg_dump -Fc rag_banco > "$BACKUP_DIR/rag_banco_${DATE}.dump"

# Retener últimos 14 días
find "$BACKUP_DIR" -name "*.dump" -mtime +14 -delete

echo "Backup completado: $BACKUP_DIR/rag_banco_${DATE}.dump"
```

```bash
# Cron diario a las 2 AM
echo "0 2 * * * banks_rag /opt/banks_rag/scripts/backup_db.sh >> /var/log/banks/backup.log 2>&1" | crontab -
```

## 8. Monitoreo con Prometheus + Grafana

```yaml
# prometheus.yml (scrape config)
scrape_configs:
  - job_name: banks_rag
    scrape_interval: 15s
    static_configs:
      - targets: ['127.0.0.1:8080']
    metrics_path: /metrics
```

Importar el dashboard en Grafana:
```bash
# Dashboard JSON en deploy/grafana/banks_rag_dashboard.json
curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @deploy/grafana/banks_rag_dashboard.json
```

## 9. Validación post-deploy

```bash
# 1. Liveness
curl -s http://localhost:8080/healthz | jq .status

# 2. Readiness
curl -s http://localhost:8080/readyz | jq .

# 3. Métricas Prometheus
curl -s http://localhost:8080/metrics | grep banks_

# 4. Búsqueda (requiere API key)
curl -s http://localhost:8080/v1/search \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "política monetaria 2024", "k": 3}' | jq .hits[0].text

# 5. Chat agentic
curl -s http://localhost:8080/v1/chat \
  -H "X-API-Key: <key>" \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cuál es la TPM actual?"}' | jq .

# 6. Evaluar el pipeline
banks-eval all --output /tmp/eval_report.md
cat /tmp/eval_report.md
```

## 10. Troubleshooting rápido

| Síntoma | Causa probable | Acción |
|---|---|---|
| `/readyz` retorna `loading` | LLM no terminó de cargar | Esperar 30-60s; ver `journalctl -u banks-api -f` |
| `/readyz` retorna `degraded` | BD inaccesible | `systemctl status postgresql` |
| 503 en `/v1/chat` | LLM no configurado | Verificar `BANKS_LLM_MODEL_PATH` |
| 401 en endpoints | API key incorrecta | Verificar `BANKS_API_KEYS` |
| 429 en requests | Rate limit activo | Aumentar `BANKS_RATE_LIMIT_RPM` o distribuir carga |
| GPU no detectada | Driver / CUDA | `nvidia-smi`; reinstalar drivers |
| OOM en GPU | Modelo demasiado grande | Reducir `BANKS_LLM_N_GPU_LAYERS` |
