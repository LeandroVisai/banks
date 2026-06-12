# run_h100.ps1 — arranca la API en el servidor H100 (Windows, backend openai_compat).
#
# Prerequisito: llama-server corriendo en puerto 8081 (deploy\start_llama_server.ps1).
# Las variables BANKS_* salen del .env (pydantic-settings las lee automáticamente).
# Solo PG* y RAG_* van aquí porque settings.py las lee con os.getenv(), no del .env.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File run_h100.ps1

$env:RAG_EMBEDDING_MODEL  = "Qwen3-VL-Embedding-8B"   # carpeta models\Qwen3-VL-Embedding-8B\
$env:RAG_VISUAL_IMG_WEIGHT = "0.7"
$env:PGHOST     = "localhost"
$env:PGPORT     = "5432"
$env:PGDATABASE = "rag_banco"
$env:PGUSER     = "postgres"
$env:PGPASSWORD = "postgres"

$env:PYTHONPATH = "src"

Write-Host "Levantando API en http://0.0.0.0:8080 (backend=openai_compat, llama-server en :8081)..."
python -m uvicorn banks_rag.interface.api.main:create_app --factory --host 0.0.0.0 --port 8080
