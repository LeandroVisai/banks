



PostgreSQL 14
Es la base de datos donde se almacenarán los datos del proyecto. Se necesita la versión 14 específicamente por compatibilidad con pgvector. Para Windows se descarga el instalador .exe desde la página oficial, y para Linux están disponibles los paquetes .deb o .rpm según la distribución. Hay que seleccionar la versión 14.x en la tabla de descargas.
[Windows installer](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)



pgvector
Extensión de PostgreSQL que permite almacenar y buscar vectores de embeddings directamente en la base de datos — es lo que conecta los modelos de embedding con el storage. Se descarga el código fuente desde GitHub (sección Releases, archivo .tar.gz o .zip de la última versión). Si el servidor corre Ubuntu, también existe el paquete postgresql-14-pgvector descargable como .deb directamente desde el repositorio apt de PostgreSQL.
[GitHub Releases](https://github.com/andreiramani/pgvector_pgsql_windows/releases/tag/0.8.2_14.20)


Qwen3-Embedding-8B
Es el modelo que genera los vectores de embeddings que pgvector va a indexar. Se elige el 8B sobre el 4B porque ocupa el primer lugar en el MTEB multilingual leaderboard y el H100 80GB lo aguanta sin problema — no tiene sentido sacrificar calidad de embeddings cuando el hardware lo permite. Hay que descargar todos los archivos de la página: los shards .safetensors más los archivos de config y tokenizer. Peso total aproximado: 16 GB. Si van a correrlo con llama.cpp, usar la versión GGUF en cambio.
[Pesos completos (HF) · GGUF (llama.cpp)](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/main)


llama.cpp
Motor de inferencia que corre los modelos en formato GGUF directamente sobre GPU sin necesidad de Python ni frameworks adicionales. Se descargan los binarios precompilados desde la sección Releases del repo oficial en GitHub. Para Windows con CUDA buscar el archivo que diga bin-win-cuda-cu12.x-x64.zip, para Linux buscar el equivalente ubuntu-x64. Siempre bajar la versión más reciente con soporte CUDA 12.x.
https://github.com/ggml-org/llama.cpp/releases/tag/b9049

unsloth/Qwen3.6-27B-GGUF
Versión cuantizada de Qwen3.6-27B en formato GGUF, optimizada por Unsloth con su sistema Dynamic 2.0 para máxima calidad por tamaño. Se recomienda bajar el archivo Qwen3.6-27B-UD-Q4_K_XL.gguf (~16 GB) que es un solo archivo — el mejor balance calidad/tamaño de la colección según sus propios benchmarks. Para correrlo se usa llama.cpp (ítem 4).
[Ver todos los archivos](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main)
Descarga directa Q4_K_XL https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-Q4_K_XL.gguf


unsloth/gemma-4-26B-A4B-it-GGUF
Versión cuantizada de Gemma 4 en formato GGUF. Es un modelo MoE (Mixture of Experts) con solo 4B parámetros activos durante inferencia — corre casi tan rápido como un modelo de 4B puro aunque tiene 26B parámetros en total, con contexto de 256K. El nombre correcto del repo incluye -it- (instruction-tuned). Descargar el archivo .gguf preferido desde la página del repo.
HuggingFace (archivos GGUF) https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/main

Revisables:

WSL2 + Ubuntu — solo si el servidor corre Windows
WSL2 es necesario únicamente si el servidor corre Windows Server 2019 o 2022, ya que vLLM solo corre en Linux. Si el servidor ya corre Linux, esto no aplica. Para instalación offline se descarga el instalador de WSL2 desde la documentación oficial de Microsoft más el paquete de distribución Ubuntu (disponible como .AppxBundle en la misma página).
Instalación offline WSL2 (Microsoft)
https://learn.microsoft.com/en-us/windows/wsl/install-manual


vLLM — Python 3.12, CUDA 12.8
Motor de inferencia especializado en GPU, optimizado para modelos grandes en producción con una H100. Solo corre en Linux. El wheel se descarga desde PyPI (archivo .whl, patrón vllm-[versión]-cp38-abi3-manylinux_2_35_x86_64.whl) o desde el índice oficial de wheels cu128 para forzar compatibilidad con CUDA 12.8 específicamente, ya que el default de vLLM v0.20+ apunta a CUDA 13.0.
PyPI (.whl) https://pypi.org/project/vllm/#files



Qwen3.6-27B FP8 — para vLLM
Versión cuantizada en FP8 de Qwen3.6-27B, diseñada específicamente para correr con vLLM en una sola GPU H100 80GB. La cuantización FP8 mantiene calidad casi idéntica al modelo original según los propios benchmarks de Qwen. Hay que descargar todos los archivos de la página. Peso total: ~30.9 GB.
HuggingFace (todos los archivos) https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/main


Gemma 4 26B A4B IT — para vLLM
Modelo MoE de Google con 26B parámetros totales pero solo 4B activos durante inferencia, lo que lo hace extremadamente eficiente en VRAM para su tamaño. Contexto de 256K tokens y soporte multimodal. Hay que descargar todos los archivos de la página de HuggingFace incluyendo config, tokenizer y shards.
HuggingFace (todos los archivos) https://huggingface.co/google/gemma-4-26b-a4b-it/tree/main

