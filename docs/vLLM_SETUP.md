# Guía de instalación: vLLM como backend de inferencia

Esta guía cubre todo lo necesario para migrar el servidor de inferencia de **llama-cpp-python** (GGUF) a **vLLM** (HuggingFace safetensors) en el H100.

## Resumen de la migración

| Aspecto | llama-cpp-python (anterior) | vLLM (nuevo) |
|---|---|---|
| Formato de modelo | `.gguf` (quantizado GGUF) | HuggingFace safetensors (bf16, AWQ, GPTQ) |
| Throughput | 1 request a la vez | Batching continuo (10-20× más) |
| Tool calling | Texto (`<tool_call>` tags) | Nativo OpenAI API (structured output) |
| Proceso | Embebido en FastAPI | Proceso separado, API HTTP en `:8000` |
| Rollback | `BANKS_LLM_FAMILY=qwen` | `BANKS_LLM_FAMILY=vllm` |

---

## 1. Requisitos de hardware y sistema

- **GPU**: NVIDIA H100 80 GB (o A100 80 GB mínimo para modelos 32B en bf16)
- **RAM sistema**: 64 GB+ recomendado
- **CUDA**: 12.1 o superior
- **Python**: 3.12+
- **Espacio en disco**: 60-80 GB por modelo (más si se descarga en bf16 completo)

Verificar versión de CUDA:
```bash
nvidia-smi
nvcc --version
```

---

## 2. Instalar vLLM

vLLM **no** está en el `pyproject.toml` del proyecto (es un proceso separado). Instalarlo en el mismo venv o en uno dedicado:

```bash
# Opción A: mismo venv del proyecto
source /opt/banks_rag/.venv/bin/activate
pip install "vllm>=0.6.0"

# Opción B: venv separado (más limpio)
python -m venv /opt/vllm_venv
source /opt/vllm_venv/bin/activate
pip install "vllm>=0.6.0"
```

Verificar instalación:
```bash
python -c "import vllm; print(vllm.__version__)"
vllm --version
```

> **Nota**: vLLM instala sus propias versiones de torch, transformers y otras dependencias. Si se usa el mismo venv que el proyecto, puede haber conflictos de versiones. Se recomienda venv separado para producción.

---

## 3. Dependencias nuevas en el proyecto

Las siguientes dependencias se agregaron a `pyproject.toml` para el cliente HTTP que habla con vLLM:

```bash
# Re-instalar el proyecto para incluir openai y httpx
cd /opt/banks_rag
pip install -e .
```

Paquetes nuevos:
- `openai>=1.54.0` — cliente AsyncOpenAI que habla con el servidor vLLM
- `httpx>=0.27.0` — timeout configurable para el cliente

---

## 4. Modelos recomendados

### Opción A: Qwen3-32B-AWQ (recomendado para H100)

El mejor balance calidad/VRAM. Tool calling excelente con el parser `hermes`.

```bash
# Con internet en el servidor
vllm serve Qwen/Qwen3-32B-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-reasoning \
  --api-key none \
  --disable-log-requests
```

| Propiedad | Valor |
|---|---|
| VRAM aprox. | ~35 GB (AWQ int4) |
| Context window | 32768 tokens |
| Tool calling | Hermes-style (nativo) |
| Thinking mode | Sí (`reasoning_content`) |

### Opción B: Qwen3-32B-Instruct (bf16 completo)

Máxima calidad, requiere más VRAM:

```bash
vllm serve Qwen/Qwen3-32B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-reasoning \
  --api-key none \
  --disable-log-requests
```

| Propiedad | Valor |
|---|---|
| VRAM aprox. | ~65 GB (bf16) |
| Context window | 32768 tokens |

### Opción C: Qwen2.5-72B-Instruct-AWQ (máxima calidad)

Para cuando se necesite la mejor calidad posible:

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --api-key none \
  --disable-log-requests
```

| Propiedad | Valor |
|---|---|
| VRAM aprox. | ~40 GB (AWQ int4) |
| Context window | 16384 tokens |

### Opción D: Gemma 3 27B

Si se prefiere Gemma, usar el parser `pythonic`:

```bash
vllm serve google/gemma-3-27b-it \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic \
  --api-key none \
  --disable-log-requests
```

| Propiedad | Valor |
|---|---|
| VRAM aprox. | ~54 GB (bf16) |
| Tool calling | Pythonic-style (nativo en vLLM) |
| Thinking mode | No |

> **`--tool-call-parser` por familia de modelo**:
> - `hermes` → Qwen3, Qwen2.5, Mistral, Llama-3 (Hermes-trained)
> - `pythonic` → Gemma 3/4
> - `llama3_json` → Llama-3 base sin fine-tune Hermes

---

## 5. Descarga offline (H100 sin internet)

Si el servidor H100 no tiene acceso a internet, descargar el modelo en una máquina con acceso y transferirlo:

```bash
# En máquina con internet:
pip install huggingface_hub
huggingface-cli login  # si el modelo requiere aceptar licencia

# Descargar modelo completo
huggingface-cli download Qwen/Qwen3-32B-AWQ \
  --local-dir /tmp/Qwen3-32B-AWQ/

# Transferir al H100 (ejemplo con rsync)
rsync -avz /tmp/Qwen3-32B-AWQ/ usuario@h100:/opt/banks_rag/models/Qwen--Qwen3-32B-AWQ/
```

Luego lanzar vLLM apuntando al path local:

```bash
vllm serve /opt/banks_rag/models/Qwen--Qwen3-32B-AWQ/ \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-reasoning \
  --api-key none
```

> **Convención de directorios**: usar `models/<owner>--<name>/` (igual que sentence-transformers). Ejemplo: `models/Qwen--Qwen3-32B-AWQ/`.

---

## 6. Configurar el proyecto para usar vLLM

### Variables de entorno

Agregar/modificar en `/etc/banks_rag.env`:

```bash
# ── LLM backend ──────────────────────────────────────────────────────────────
BANKS_LLM_FAMILY=vllm

# ── vLLM client settings ─────────────────────────────────────────────────────
BANKS_VLLM_BASE_URL=http://localhost:8000/v1
BANKS_VLLM_MODEL=Qwen/Qwen3-32B-AWQ        # debe coincidir con el --model del server
BANKS_VLLM_API_KEY=none
BANKS_VLLM_ENABLE_THINKING=true             # activa strip de <think> y reasoning_content
BANKS_VLLM_TOOL_CHOICE=auto                 # "auto" | "required"
BANKS_VLLM_TIMEOUT=120.0                    # segundos de timeout por request
BANKS_VLLM_TEMPERATURE=0.2
BANKS_VLLM_TOP_P=0.9
BANKS_VLLM_MAX_TOKENS=2048

# ── Rollback instantáneo ──────────────────────────────────────────────────────
# Para volver a llama.cpp, comentar BANKS_LLM_FAMILY=vllm y descomentar:
# BANKS_LLM_FAMILY=qwen
# BANKS_LLM_MODEL_PATH=models/Qwen3.6-27B-Q4_K_XL.gguf
```

### Verificar que el servidor vLLM responde

```bash
# Health check del servidor vLLM
curl http://localhost:8000/health

# Listar modelos disponibles
curl http://localhost:8000/v1/models | python -m json.tool

# Test de generación rápida
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "messages": [{"role": "user", "content": "Hola, ¿cuál es la TPM?"}],
    "max_tokens": 100
  }' | python -m json.tool
```

---

## 7. Systemd: gestión del servidor vLLM

Se incluye el archivo `deploy/systemd/banks-vllm.service`. Instalarlo en el H100:

```bash
# Copiar el servicio
sudo cp deploy/systemd/banks-vllm.service /etc/systemd/system/

# Recargar systemd
sudo systemctl daemon-reload

# Habilitar y arrancar
sudo systemctl enable banks-vllm
sudo systemctl start banks-vllm

# Ver logs
sudo journalctl -u banks-vllm -f
```

El servicio `banks-api` (FastAPI) ya maneja la reconexión automática: si vLLM no está disponible al arrancar la API, loguea el error y el endpoint `/v1/chat` retorna `503` hasta que vLLM esté listo.

Orden de arranque recomendado:
```bash
sudo systemctl start banks-vllm   # primero — tarda 1-3 min en cargar el modelo
sudo systemctl start banks-api    # después
```

---

## 8. Monitoreo de VRAM durante inferencia

```bash
# Monitor en tiempo real
watch -n 2 nvidia-smi

# Ver memoria por proceso
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Logs del servidor vLLM
sudo journalctl -u banks-vllm -f --since "1 hour ago"
```

---

## 9. Diferencias operacionales vs llama-cpp

| Aspecto | llama-cpp | vLLM |
|---|---|---|
| Tiempo de arranque | ~30s | 1-3 min (carga el modelo completo) |
| Primer token | ~2-5s | ~1-3s |
| Requests simultáneos | 1 (bloqueante) | N (batching continuo) |
| VRAM layout | Layers en GPU, contexto en CPU | Todo en GPU (PagedAttention) |
| Streaming | No implementado | Nativo (disponible para implementar) |
| Tokenizer en código | Sí (via llama_cpp) | No (estimado por words) |
| Formato modelo | GGUF (Q4, Q5, Q8) | HF safetensors (bf16, AWQ, GPTQ) |

---

## 10. Rollback a llama-cpp

Si se necesita volver a llama-cpp:

```bash
# Cambiar env var (sin tocar código)
sed -i 's/^BANKS_LLM_FAMILY=vllm/BANKS_LLM_FAMILY=qwen/' /etc/banks_rag.env

# Reiniciar la API
sudo systemctl restart banks-api

# Opcional: apagar el servidor vLLM para liberar VRAM
sudo systemctl stop banks-vllm
```

La `LlamaCppEngine` queda intacta en el código. El rollback es operacional, sin deployment de código.
