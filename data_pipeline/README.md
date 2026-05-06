# data_pipeline — Series macro/financieras para los chatbots

Convierte las queries de [`querys/Monitor.py`](../querys/Monitor.py) en parquets que ambos chatbots (`chatbot/` y `chatbot_calling_tool/`) pueden leer sin tocar el data warehouse.

## Por qué parquet y no PostgreSQL ni queries en vivo

- **Sin internet en el servidor del chatbot**: el DW (`mesadineOLTP_`, `dbo.Base_DMN`, etc.) vive en otra red. No queremos que el chatbot dependa de conectividad.
- **Portabilidad**: copiar 8–10 archivos `.parquet` (~unos pocos MB cada uno) es trivial.
- **Velocidad**: lectura columnar + filtros vectorizados con pandas/pyarrow son órdenes de magnitud más rápidos que ir a PostgreSQL para series largas.
- **Una sola fuente de verdad**: ambos chatbots y cualquier otro consumidor leen el mismo snapshot.

## Estructura

```
data_pipeline/
├── README.md                       ← este archivo
├── series_catalog.yaml             ← catálogo declarativo de TODAS las series
├── extract.py                      ← runner: catálogo → parquet
├── parquet_store.py                ← API de lectura usada por los chatbots
├── requirements.txt
└── snapshots/                      ← output, gitignored
    ├── tasas_clp.parquet
    ├── curva_spc_clp.parquet
    ├── curva_spc_uf.parquet
    ├── bonos_clp.parquet
    ├── bonos_uf.parquet
    ├── ust.parquet
    ├── ois_sofr.parquet
    ├── expectativas_tpm.parquet
    ├── fx.parquet
    ├── commodities.parquet
    ├── forwards_clp.parquet
    ├── tasas_usd_mn.parquet
    ├── liquidez_mx.parquet
    └── microestructura.parquet
```

## El catálogo (`series_catalog.yaml`)

Define ~80 series organizadas en 14 categorías. Cada entrada:

```yaml
categories:
  bonos_clp:
    description: "Bonos del Tesoro y BCCh en pesos"
    parquet_file: bonos_clp.parquet
    series:
      - id: btp_5y
        name: "BTP 5Y"
        unit: "% anual"
        frequency: diario
        source: BCCh
        variable: TASAS_LARGO_PLAZO
        tenor: 5Y
        sql_table: dbo.Base_DMN
        sql_column: "[BONOS $ 5Y]"
        description: "..."
```

Los campos `sql_table` y `sql_column` corresponden directamente a las queries de `Monitor.py`. `extract.py` los usa para construir las consultas.

### Inventario por categoría

| Categoría | Series | Tabla DW principal |
|---|---|---|
| `tasas_clp` | spreads DAP/Prime/Swap, PDBC, TIB | `dbo.Base_DMN`, `mesadineOLTP_.Interbancario` |
| `curva_spc_clp` | swap cámara CLP 3M–10Y | `dbo.bbg_spc_clp` |
| `curva_spc_uf` | swap cámara UF 3M–1Y | `dbo.bbg_spc_uf` |
| `bonos_clp` | BTP 1Y–30Y | `dbo.Base_DMN` |
| `bonos_uf` | BTU 2Y–30Y | `dbo.Base_DMN` |
| `ust` | UST 2Y–30Y | `DACE.dbo.bbg_rfi*_gen` |
| `ois_sofr` | OIS SOFR 3M–24M | `DACE.dbo.bbg_OIS_SOFR` |
| `expectativas_tpm` | spreads MIPR 3M–24M | `DACE.dbo.zero_curve_MIPR` |
| `fx` | USD/CLP, DXY, MXN, BRL, COP, PEN, KRW, AUD, NZD, canastas | `mesadineOLTP_.mercado.divisas`, `dbo.bbg_monedas` |
| `commodities` | Cobre | `dbo.bbg_commodities` |
| `forwards_clp` | FWD USD/CLP 30–360D | `dbo.Base_DMN` |
| `tasas_usd_mn` | TADO, SOFR, Prime USD, DAP USD, Spread On-Shore | `dbo.Base_DMN` |
| `liquidez_mx` | LCR, NSFR, Ratio Liquidez/Obligaciones, IHH | `mesadineOLTP_.c49`, `Outputs.Bancos` |
| `microestructura` | Bid-ask, volatilidad, monto transado | `Inputs.Spot.Datatec_*` |

## Cómo funciona la extracción

### Modo `dw` (extracción real)

Para correr en una máquina **con conectividad al SQL Server**:

```bash
pip install -r requirements.txt
pip install pyodbc sqlalchemy

# Credenciales en el entorno
export DW_SERVER=tu-servidor-sql.empresa.local
export DW_DATABASE=mesadineOLTP_
export DW_TRUSTED=yes        # o DW_USER / DW_PASSWORD

python extract.py --mode dw --since 2018-01-01
```

`extract.py` lee el YAML, traduce cada serie a un `SELECT Fecha, columna FROM tabla WHERE ...`, normaliza los valores (multiplica por 100 si la unidad es bp pero el dato viene como fracción), y escribe un parquet por categoría.

Las series con `derivation:` (agregaciones complejas como LCR, IHH, TIB ponderada) se omiten en este pase — se gestionan con scripts dedicados que replican la lógica de pandas de `Monitor.py`.

```bash
# Solo una o pocas categorías
python extract.py --mode dw --category bonos_clp --category fx
```

### Modo `mock` (offline / desarrollo)

Genera datos sintéticos plausibles (random walk con reversión a la media) usando rangos realistas según la unidad. **No requiere conexión al DW** — perfecto para desarrollar y probar los chatbots sin depender del data warehouse.

```bash
python extract.py --mode mock
```

Cada serie obtiene ~1500 días hábiles de datos sintéticos. El seed deriva del `series_id` así que es reproducible.

## Sincronización al servidor del chatbot

Después de generar los parquets en una máquina con DW, copiarlos al servidor del chatbot:

```bash
rsync -av snapshots/ usuario@servidor-chatbot:/path/banks/data_pipeline/snapshots/
```

Los chatbots se enteran del refresh automáticamente al expirar el LRU cache de `parquet_store._load_parquet` (o pueden llamar `parquet_store.clear_cache()` vía un endpoint).

## API de lectura — `parquet_store.py`

Lo único que importan los chatbots:

```python
from data_pipeline import parquet_store

# Listar todas las series con su cobertura
parquet_store.list_series()

# Filtrado
parquet_store.list_series(category="bonos_clp")
parquet_store.list_series(variable="TASAS_LARGO_PLAZO")
parquet_store.list_series(tenor="5Y")

# Inspección de una serie
parquet_store.get_series_meta("btp_5y")
# → {"id": "btp_5y", "name": "BTP 5Y", "unit": "% anual", ...}

# Datos
df = parquet_store.fetch_series(
    "btp_5y",
    date_from=date(2024, 1, 1),
    date_to=date(2024, 12, 31),
)
# → DataFrame [date, value]

# Formato tabla para inyectar al prompt
text = parquet_store.format_table(meta, df)
# →
# BTP 5Y  [fuente: BCCh | diario | unidad: % anual]
# -----------------------------------------------
#   2024-01-02  6.12
#   2024-01-03  6.10
#   ...
```

## Cómo lo consumen los chatbots

| Chatbot | Cómo usa parquet_store |
|---|---|
| `chatbot/` (clásico) | `app/sql_context.py` reemplaza la lectura de PostgreSQL `historical_data` por `parquet_store.fetch_series` |
| `chatbot_calling_tool/` (agentic) | `app/tools/historical_series.py` expone `list_historical_series` y `get_historical_series` que delegan a `parquet_store` |

En ambos casos los chatbots solo leen — la extracción es un proceso independiente.

## Agregar una nueva serie

1. Editar `series_catalog.yaml`, agregar el item dentro de la categoría correspondiente:
   ```yaml
   - id: nueva_serie
     name: "Mi nueva serie"
     unit: "% anual"
     frequency: diario
     source: Bloomberg
     sql_table: dbo.alguna_tabla
     sql_column: NombreColumna
   ```
2. Re-correr `python extract.py --mode dw --category <categoria>` (solo esa categoría).
3. Copiar el parquet actualizado al servidor del chatbot.
4. (Opcional) `parquet_store.clear_cache()` o reiniciar.

Nada más cambia: los chatbots descubren la serie automáticamente al recargar el catálogo.

## Para series con `derivation:` (lógica compleja)

Algunas series no son simples `SELECT col FROM tabla` — vienen de agregaciones que `Monitor.py` hace en pandas. Por ejemplo:

- **TIB ponderada**: `weighted_avg(Tasa, Monto) groupby Fecha`
- **LCR/NSFR**: agregación con filtros por categoría sistémico/no-sistémico
- **IHH liquidez**: cálculo Herfindahl–Hirschman

Estas se marcan en el YAML con `derivation:` (descripción) en lugar de `sql_column:`. Para implementarlas, agregar un handler en `extract.py:_run_derived_series` (TODO) o un script ad-hoc en `scripts/`.

Inicialmente quedan vacías en los parquets — los chatbots las muestran como "sin datos" hasta que se popule.
