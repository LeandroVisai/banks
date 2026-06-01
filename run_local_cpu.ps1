# Arranca la API del agente para pruebas locales (Windows dev box, RTX 3080).
# El modelo y n_gpu_layers salen del .env (BANKS_LLM_*); con n_gpu_layers=-1
# corre en GPU. Ver SETUP_SERVIDOR.txt para el detalle del entorno.
#
# Por qué estas env vars en el proceso (no en .env): settings.py solo lee del
# .env las claves con prefijo BANKS_. Las PG* y RAG_EMBEDDING_MODEL se leen con
# os.getenv() del entorno del proceso, así que deben fijarse aquí.
#
# CLAVE: RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
#   1) coincide con el corpus actual en Postgres (384-dim — se ingestó con el
#      fallback, no con Qwen3-VL-8B); y
#   2) es ~0.5GB, así deja RAM para el LLM 27B IQ2 (cargar el embedder de 8B
#      (~16GB) + el LLM no cabría en los ~19GB libres).

$env:RAG_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
$env:PGHOST     = "localhost"
$env:PGPORT     = "5432"
$env:PGDATABASE = "rag_banco"
$env:PGUSER     = "postgres"
$env:PGPASSWORD = "postgres"

# PYTHONPATH=src para resolver el paquete sin instalar el proyecto.
$env:PYTHONPATH = "src"

Write-Host "Levantando API en http://localhost:8080 (LLM 27B IQ2 en CPU — la 1a carga tarda)..."
python -m uvicorn banks_rag.interface.api.main:create_app --factory --host 127.0.0.1 --port 8080
