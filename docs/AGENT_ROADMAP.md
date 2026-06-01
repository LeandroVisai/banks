# Roadmap del Agente — `branch/AgenteCreadoPorClaude`

Proyecto autónomo (Claude) para llevar el agente del BCCh a "analista senior":
informes completos de mercado, lectura de documentos y gráficos, y graficado de
series desde los parquets. Diseñado para **H100 + Qwen3.6-27B-Q4** (modelo
multimodal reciente y muy capaz).

## Diagnóstico de partida (validado esta sesión)

- **Infra sólida**: 107/107 parquets consultables, routing determinista por
  especialista correcto, analytics OK, `plot_series` (Vega-Lite) funcionando.
- **Debilidad #1 — fiabilidad del tool-use**: modelos chicos (7B/IQ2)
  **alucinan identificadores** (columna `usdclp` vs `CLP`; dataset
  `spread_btp_spc_plazo` inexistente) y no encadenan `discover→execute`.
  Los tools se endurecieron (alias de args, `query` opcional/browse) pero falta
  resolución de VALORES.

## Fases

- [x] **Fase 0** — `plot_series` (Vega-Lite desde parquets). *(branch qwenllamacpp)*
- [ ] **Fase 1 — Resolución difusa de `dataset_id` y columnas.** Si el modelo
  pide un id/columna que no existe, se resuelve al más cercano (difflib) con una
  nota "interpreté X como Y", en vez de fallar. Ataca la alucinación que rompió
  6/8 especialistas. *(determinista, testeable)*
- [ ] **Fase 2 — `get_series` (tool de alto nivel).** Colapsa
  `discover_query`+`execute_query` en una sola llamada (descripción NL → dataset
  + filas + stats). Menos pasos = menos puntos de fallo (afp/fondos_mutuos hacían
  loop en discover).
- [ ] **Fase 3 — Modo informe.** `run_report`: outline → síntesis por sección
  (mercado por segmento, política, riesgos) → ensamble con gráficos embebidos y
  citas. El "analista senior" que lee el gerente. (FinRobot-style CoT.)
- [~] **Fase 4 — discover ranking. DIFERIDA.** Se probó ponderar id/name e IDF
  para que los 4/107 fueran descubribles por su nombre exacto, pero rompía casos
  golden reales (p.ej. "LCR del sistema bancario" sacaba `lcr` del top-3). Los
  golden (queries reales) priman sobre el 4/107-por-nombre-exacto, y `get_series`
  ya mitiga la incertidumbre devolviendo alternativas. Revertido.
- [~] **Fase 5 — Visión (mtmd). FUNDACIÓN.** Qwen3.6 es multimodal. Se deja la
  base SEGURA y testeable: `BANKS_LLM_MMPROJ_PATH` (config), builder de bloques
  de imagen (base64 data-URL) y `LlamaCppEngine.supports_vision` — todo guardado:
  con mmproj vacío (default) el camino de texto NO cambia. La carga del chat
  handler mtmd y el cableado tool→imagen→generate se validan en la H100 (necesita
  el archivo mmproj del modelo). Ver `_image_content.py`.
- [ ] **Fase 6 — CLI banks-report.** Generar informes desde la terminal.

## Notas de ejecución

- No hay H100 en el box de desarrollo (RTX 3080 + Qwen2.5-7B): la validación
  LLM-driven local es indicativa; la calidad real se valida con el 27B en la
  H100 (`scripts/verify_subagents.py`). Todo lo testeable se cubre con unit tests
  deterministas (sin BD/modelo).
- Invariante respetado: el LLM elige QUÉ; la tool arma el CÓMO (SQL, spec de
  gráfico, resolución de ids).
