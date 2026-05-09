"""banks_rag — Sistema RAG + agente multimodal para banco central.

Capas (Clean Architecture):

    interface/      → entry points (CLI + FastAPI)
    application/    → casos de uso, orquestación
    domain/         → entidades, value objects, reglas puras
    infrastructure/ → adaptadores: postgres, llamacpp, embeddings, etc.

Reglas de dependencia: ``domain`` y ``application`` jamás importan de
``infrastructure``. La infra implementa Protocols definidos en domain/application.
"""

__version__ = "0.1.0"
