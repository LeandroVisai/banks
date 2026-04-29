# Búsqueda Avanzada — Filtros Granulares

Desde ahora el sistema RAG soporta **filtros más precisos** para búsquedas granulares. En lugar de solo filtrar por año, ahora puedes filtrar por:

- **Mes** (enero, febrero, ..., diciembre)
- **Institución** (BANCO_CENTRAL_CHILE, FEDERAL_RESERVE, etc.)
- **Entidades mencionadas** (BANCO_CENTRAL_CHILE, FEDERAL_RESERVE, JPM, etc.)
- **Tags** (DECISION_POLITICA, FORWARD_LOOKING, DATOS_NUMERICOS, VARIABLE_CRITICA)
- **Importance score** (umbral mínimo y máximo 0-1)
- **Section confidence** (umbral mínimo de confianza)
- **Exclusiones** (excluir tipos de documento, instituciones, boilerplate legal)
- **Mejor filtrado de boilerplate** (copyright, disclaimers, avisos legales)

---

## Instalación & Requisitos

**No hay dependencias nuevas**. El sistema usa:
- PostgreSQL + pgvector (ya existente)
- psycopg2 (ya instalado)
- Python 3.7+

---

## Uso

### Opción 1: Menú interactivo (recomendado para principiantes)

```bash
python3 run.py advanced-search
```

Te pedirá:
1. Consulta (ej: "política monetaria")
2. Filtros opcionales (ej: "--year 2022 --month enero,febrero")
3. Número de resultados k

### Opción 2: Línea de comando (rápido)

```bash
# Búsqueda con filtros básicos
python3 04_advanced_search.py "tasa de interés" \
  --year 2022 \
  --month enero,febrero \
  --institution BANCO_CENTRAL_CHILE \
  -k 5

# Excluir boilerplate y Monitor PM
python3 04_advanced_search.py "inflación" \
  --exclude-boilerplate \
  --exclude-doc-types MONITOR_PM \
  -k 10

# Solo decisiones con variables críticas
python3 04_advanced_search.py "política monetaria" \
  --tags DECISION_POLITICA,VARIABLE_CRITICA \
  --min-importance 0.8 \
  -k 3

# Salida en JSON para integración
python3 04_advanced_search.py "commodities" \
  --year 2023 \
  --exclude-boilerplate \
  --json
```

### Opción 3: Archivo de configuración JSON

Crea `mi_busqueda.json`:

```json
{
  "year": [2022, 2023],
  "month": [1, 2, 3],
  "institution": ["BANCO_CENTRAL_CHILE"],
  "entities": ["BANCO_CENTRAL_CHILE", "FEDERAL_RESERVE"],
  "tags": ["DECISION_POLITICA", "VARIABLE_CRITICA"],
  "min_importance": 0.7,
  "min_section_confidence": 0.5,
  "exclude_boilerplate": true,
  "exclude_doc_types": ["MONITOR_PM"],
  "exclude_institutions": []
}
```

Luego usa:

```bash
python3 04_advanced_search.py --config mi_busqueda.json "política monetaria"
```

**Guardar configuración**:
```bash
python3 04_advanced_search.py "inflación" \
  --year 2022 \
  --exclude-boilerplate \
  --save-config mi_filtros.json
```

---

## Parámetros Disponibles

### Filtros de Tiempo

| Parámetro | Ejemplo | Descripción |
|-----------|---------|-------------|
| `--year` | `2022` o `2022,2023` | Año(s) del documento |
| `--month` | `enero,febrero` o `1,2,3` | Mes(es) — nombres o números 1-12 |

### Filtros de Institución / Entidad

| Parámetro | Ejemplo | Descripción |
|-----------|---------|-------------|
| `--institution` | `BANCO_CENTRAL_CHILE` | Incluir solo esta institución |
| `--entity` | `FEDERAL_RESERVE,JPM` | Entidades mencionadas en el contenido |
| `--exclude-institutions` | `JPMORGAN` | Excluir esta institución |

### Filtros de Tipo de Documento

| Parámetro | Ejemplo | Descripción |
|-----------|---------|-------------|
| `--doc-type` | `COMUNICADO,MINUTA` | Tipos: COMUNICADO, MINUTA, FED_STATEMENT, REPORTE_RESEARCH, MONITOR_PM |
| `--exclude-doc-types` | `MONITOR_PM` | Excluir tipos de documento |

### Filtros de Contenido

| Parámetro | Ejemplo | Descripción |
|-----------|---------|-------------|
| `--tags` | `DECISION_POLITICA,VARIABLE_CRITICA` | Tags: DECISION_POLITICA, FORWARD_LOOKING, DATOS_NUMERICOS, VARIABLE_CRITICA |
| `--min-importance` | `0.7` | Score mínimo de importancia (0-1) |
| `--max-importance` | `0.9` | Score máximo de importancia (0-1) |
| `--min-section-confidence` | `0.6` | Confianza mínima en clasificación de sección (0-1) |
| `--exclude-boilerplate` | — | Excluir copyright, avisos legales, disclaimers |

### Parámetros de Búsqueda

| Parámetro | Ejemplo | Descripción |
|-----------|---------|-------------|
| `-k`, `--top-k` | `10` | Número de resultados (default: 5) |
| `--json` | — | Salida en JSON (con metadatos completos) |
| `--no-mmr` | — | Desactivar reranking MMR (diversidad) |

---

## Ejemplos Prácticos

### 1. Buscar decisiones de política monetaria (2022-2023)

```bash
python3 04_advanced_search.py "política monetaria" \
  --year 2022,2023 \
  --tags DECISION_POLITICA \
  --min-importance 0.8 \
  -k 5
```

### 2. Análisis de inflación excluyendo ruido

```bash
python3 04_advanced_search.py "inflación expectativas" \
  --year 2023 \
  --month enero,febrero,marzo \
  --exclude-boilerplate \
  --exclude-doc-types MONITOR_PM \
  --json \
  -k 10
```

### 3. Comunicados del Banco Central (no minutos, sin legal)

```bash
python3 04_advanced_search.py "tasas de interés" \
  --doc-type COMUNICADO \
  --institution BANCO_CENTRAL_CHILE \
  --exclude-boilerplate \
  --min-section-confidence 0.7 \
  -k 3
```

### 4. Comparar visiones BCCh vs Federal Reserve sobre commodities

**Crear config1.json**:
```json
{
  "institution": ["BANCO_CENTRAL_CHILE"],
  "exclude_boilerplate": true
}
```

**Crear config2.json**:
```json
{
  "institution": ["FEDERAL_RESERVE"],
  "exclude_boilerplate": true
}
```

**Ejecutar ambas**:
```bash
echo "=== BCCh ===" && python3 04_advanced_search.py --config config1.json "commodities precios" -k 3 --json
echo "=== Fed ===" && python3 04_advanced_search.py --config config2.json "commodities precios" -k 3 --json
```

### 5. Investigación de riesgos de sector específico

```bash
python3 04_advanced_search.py "riesgos vulnerabilidades liquidez" \
  --tags VARIABLE_CRITICA \
  --year 2022,2023 \
  --min-importance 0.65 \
  --exclude-boilerplate \
  -k 15 \
  --json > resultados_riesgos.json
```

---

## Entendiendo los Scores

Cada resultado incluye:

| Campo | Rango | Qué significa |
|-------|-------|---------------|
| `importance_score` | 0-1 | Relevancia curada (variables críticas, datos, decisiones) |
| `section_confidence` | 0-1 | Confianza en que la sección fue clasificada correctamente |
| `text_score` | 0+ | Relevancia léxica (full-text search) |
| `vector_score` | 0-1 | Relevancia semántica (embedding similarity) |

**Nota**: Si `importance_score = 0.0`, generalmente significa que fue detectado como boilerplate/legal.

---

## Mejoras Implementadas

### 1. **Detección mejorada de boilerplate**

Se agregaron patrones adicionales para detectar:
- Copyright notices (`© YYYY`, "all rights reserved", "derechos reservados")
- Confidentiality warnings ("confidential", "mensaje confidencial")
- Forward-looking disclaimers
- Footer/contact info repetitivo
- Proprietary information notices

Todos estos chunks reciben `importance_score = 0.0` y pueden excluirse con `--exclude-boilerplate`.

### 2. **Filtrado por mes**

Perfecto para análisis temporal fino (ej: inflación en "enero" vs "febrero").

### 3. **Filtrado por institución**

Todos los documentos etiquetan la `institution` de origen. Usa esto para:
- Solo BCCh: `--institution BANCO_CENTRAL_CHILE`
- Solo Fed: `--institution FEDERAL_RESERVE`
- Excluir JPMorgan: `--exclude-institutions JPMORGAN`

### 4. **Filtrado por entidades mencionadas**

Los chunks que mencionan ciertas entidades (CENTRAL_BANK_OF_CHILE, FED, etc.) pueden filtrarse.

### 5. **Umbrales de confianza**

- `--min-importance X`: Solo chunks con contenido curado relevante
- `--min-section-confidence Y`: Solo chunks bien clasificados

---

## Integración con Otros Sistemas

### Obtener JSON para aplicaciones

```bash
python3 04_advanced_search.py "tu consulta" \
  --year 2023 \
  --json \
  -k 10
```

Salida:
```json
[
  {
    "chunk_id": "abc-123",
    "filename": "comunicado1.pdf",
    "section_type": "DECISION",
    "importance_score": 0.92,
    "text": "...",
    "tags": ["DECISION_POLITICA", "VARIABLE_CRITICA"],
    ...
  },
  ...
]
```

---

## Preguntas Frecuentes

**P: ¿Cómo encuentro solo decisiones de política?**  
R: `--tags DECISION_POLITICA --min-importance 0.8`

**P: ¿Puedo buscar en solo Comunicados del BCCh?**  
R: `--doc-type COMUNICADO --institution BANCO_CENTRAL_CHILE`

**P: ¿Cómo excluyo todo el ruido legal (copyright, disclaimers)?**  
R: `--exclude-boilerplate`

**P: ¿Qué filtros debo usar para una búsqueda bien calibrada?**  
R: Comienza con `--min-importance 0.65 --min-section-confidence 0.6` y ajusta según resultados.

**P: ¿Puedo guardar mi configuración de filtros?**  
R: Sí, usa `--save-config mi_config.json` en tu búsqueda, luego reutiliza con `--config mi_config.json`.

---

## Troubleshooting

### Error: "No se conecta a PostgreSQL"

Verifica que PostgreSQL esté corriendo y las variables de entorno:
```bash
export PGHOST=localhost
export PGUSER=postgres
export PGDATABASE=rag_banco
python3 04_advanced_search.py "test"
```

### Error: "Sin resultados" incluso con consultas simples

Reduce los filtros o quita `--min-importance` / `--min-section-confidence`:
```bash
python3 04_advanced_search.py "inflación"  # sin filtros
```

### Los resultados incluyen boilerplate

Agrega `--exclude-boilerplate`:
```bash
python3 04_advanced_search.py "tasa" --exclude-boilerplate
```

---

## Próximas Mejoras Posibles

- [ ] Filtrado por rango numérico (ej: "encontrar chunks que mencionen inflación > 5%")
- [ ] Búsqueda por regex en texto
- [ ] Filtrado por author/autor del documento
- [ ] Exportar a formato Excel con tabla pivote
- [ ] Integración con LLM local (Ollama) para análisis automático

---

## Resumen de Comandos Clave

```bash
# Menú interactivo (fácil)
python3 run.py advanced-search

# Búsqueda rápida línea de comando
python3 04_advanced_search.py "consulta" --filtro valor -k 10

# Con configuración guardada
python3 04_advanced_search.py --config filtros.json "consulta"

# Guardar filtros para reutilizar
python3 04_advanced_search.py "consulta" --filtro valor --save-config filtros.json

# Salida JSON para programación
python3 04_advanced_search.py "consulta" --json -k 20
```

---

**Última actualización**: 29 de abril de 2026  
**Sistema**: RAG Banco Central con filtros granulares  
**Archivo**: ADVANCED_SEARCH.md
