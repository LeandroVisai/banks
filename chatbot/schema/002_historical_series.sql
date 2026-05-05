-- =============================================================================
-- 002_historical_series.sql
-- Series de tiempo macroeconómicas. Sin vectorización: se consultan por SQL
-- directo y se inyectan al prompt como tabla legible.
-- =============================================================================

CREATE TABLE IF NOT EXISTS historical_series (
    series_id         TEXT        PRIMARY KEY,
    series_name       TEXT        NOT NULL,
    unit              TEXT        NOT NULL,
    frequency         TEXT        NOT NULL CHECK (frequency IN ('diario','mensual','trimestral','anual')),
    source            TEXT        NOT NULL,
    economic_variable TEXT,
    description       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS historical_data (
    series_id  TEXT        NOT NULL REFERENCES historical_series(series_id) ON DELETE CASCADE,
    date       DATE        NOT NULL,
    value      NUMERIC(18, 6),
    notes      TEXT,
    PRIMARY KEY (series_id, date)
);

CREATE INDEX IF NOT EXISTS idx_historical_data_series_date
    ON historical_data (series_id, date DESC);

-- Catálogo inicial — alineado con taxonomy.py del pipeline padre.
INSERT INTO historical_series
    (series_id, series_name, unit, frequency, source, economic_variable, description)
VALUES
    ('tpm',                 'Tasa de Política Monetaria',          '% anual',     'mensual',    'BCCh',      'TASA_INTERES',                'Tasa rectora del BCCh'),
    ('ipc_anual',           'IPC variación anual',                 'var % a/a',   'mensual',    'INE',       'INFLACION',                   'IPC vs 12 meses atrás'),
    ('ipc_mensual',         'IPC variación mensual',               'var % m/m',   'mensual',    'INE',       'INFLACION',                   'IPC vs mes anterior'),
    ('ipcx_anual',          'IPC subyacente (IPCX)',               'var % a/a',   'mensual',    'INE',       'INFLACION',                   'IPC sin alimentos ni energía'),
    ('usdclp_spot',         'Tipo de cambio USD/CLP',              'CLP por USD', 'diario',     'BCCh',      'TIPO_CAMBIO',                 'Paridad observada'),
    ('pib_trimestral',      'PIB real',                            'var % a/a',   'trimestral', 'BCCh',      'PIB',                         'Crecimiento anual del trimestre'),
    ('imacec_mensual',      'IMACEC',                              'var % a/a',   'mensual',    'BCCh',      'PIB',                         'Indicador Mensual de Actividad Económica'),
    ('bcu_5y',              'BCU 5 años (real, UF)',               '% anual',     'diario',     'BCCh',      'TASAS_LARGO_PLAZO',           'Bono BCCh 5y en UF'),
    ('btp_5y',              'BTP 5 años (nominal)',                '% anual',     'diario',     'BCCh',      'TASAS_LARGO_PLAZO',           'Bono BCCh 5y en pesos'),
    ('exp_inflacion_12m',   'Expectativa inflación 12m',           '% anual',     'mensual',    'BCCh',      'EXPECTATIVAS_INFLACIONARIAS', 'EEE — horizonte 12m'),
    ('exp_inflacion_24m',   'Expectativa inflación 24m',           '% anual',     'mensual',    'BCCh',      'EXPECTATIVAS_INFLACIONARIAS', 'EEE — horizonte 24m'),
    ('exp_tpm_12m',         'Expectativa TPM 12m',                 '% anual',     'mensual',    'BCCh',      'TASA_INTERES',                'EEE — TPM en 12m'),
    ('precio_cobre',        'Precio cobre (LME)',                  'USD/lb',      'diario',     'LME',       'COMMODITIES',                 'Spot LME'),
    ('precio_petroleo_wti', 'Precio petróleo WTI',                 'USD/barril',  'diario',     'NYMEX',     'COMMODITIES',                 'Spot WTI'),
    ('fed_funds_rate',      'Fed Funds Rate (objetivo)',           '% anual',     'mensual',    'Fed',       'TASA_INTERES',                'Tasa objetivo Fed'),
    ('cds_chile_5y',        'CDS Chile 5y',                        'pb',          'diario',     'Bloomberg', 'RIESGO_CREDITO',              'Credit Default Swap soberano'),
    ('desempleo',           'Tasa de desempleo (trimestre móvil)', '%',           'mensual',    'INE',       'MERCADO_LABORAL',             'Tasa de desocupación INE')
ON CONFLICT (series_id) DO NOTHING;
