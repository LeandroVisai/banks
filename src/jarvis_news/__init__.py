"""jarvis_news — generador de noticias + TTS (voz JARVIS), AISLADO del core.

Vive en su propio paquete (no dentro de ``banks_rag``) por separación: procesa
**datos externos scrapeados** (JSON de noticias) y produce audio. Reusa el LLM
del core (se lo inyecta la app vía ``app.state.deps.llm``) pero su lógica es
autónoma. La app FastAPI monta su router con ``include_router``.

Expone:
  - report.generate_news_report — reporte diario por map-reduce.
  - tts.build_default_tts — TTS Piper (voz JARVIS).
  - api.router — endpoints /v1/news-report y /v1/tts.
"""

from . import api, report, tts

__all__ = ["api", "report", "tts"]
