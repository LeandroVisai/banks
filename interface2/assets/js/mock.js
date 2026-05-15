/* ─────────────────────────────────────────────────────────────────────────
   MOCK DATA — generador de series sintéticas para charts sin parquet aún
   Reproducible con seed para que los gráficos se mantengan estables.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

// Mulberry32 PRNG (seeded)
const seededRng = (seed) => {
    let t = seed >>> 0;
    return () => {
        t = (t + 0x6D2B79F5) >>> 0;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
};

/**
 * Genera una serie temporal sintética tipo random walk con tendencia/volatilidad.
 *
 * @param {object} opts
 * @param {number} opts.seed     - PRNG seed (entero)
 * @param {number} opts.days     - cantidad de días (default 365)
 * @param {number} opts.start    - valor inicial (default 100)
 * @param {number} opts.vol      - volatilidad diaria (default 0.012)
 * @param {number} opts.trend    - drift diario en % (default 0)
 * @param {number} opts.precision- decimales a redondear (default 2)
 * @returns {Array<{date: string, value: number}>}
 */
BCCh.mockSeries = ({
    seed = 42,
    days = 365,
    start = 100,
    vol = 0.012,
    trend = 0,
    precision = 2,
} = {}) => {
    const rng = seededRng(seed);
    const out = [];
    let v = start;
    const today = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        // Skip weekends to imitate financial calendar
        if (d.getDay() === 0 || d.getDay() === 6) continue;
        const shock = (rng() * 2 - 1) * vol;
        v = v * (1 + shock + trend / 100);
        out.push({
            date: d.toISOString().slice(0, 10),
            value: Number(v.toFixed(precision)),
        });
    }
    return out;
};

/**
 * Genera múltiples series (multi-line / multi-tenor).
 *
 * @param {object} opts
 * @param {Array<string>} opts.names   - nombres de las series
 * @param {number}        opts.seed    - seed base; cada serie usa seed+i
 * @param {object}        opts.config  - opciones comunes para mockSeries
 * @returns {Array<{name: string, points: Array}>}
 */
BCCh.mockMultiSeries = ({ names, seed = 100, config = {} } = {}) => {
    return names.map((name, i) => ({
        name,
        points: BCCh.mockSeries({
            ...config,
            seed: seed + i * 13,
            start: (config.start || 100) * (1 + (i - names.length / 2) * 0.05),
        }),
    }));
};

/**
 * Genera datos para una curva de tasas (eje X = tenor, no fecha).
 *
 * @param {object} opts
 * @param {Array<string>} opts.tenors  - ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y']
 * @param {number}        opts.seed
 * @param {number}        opts.base    - tasa base (% nominal)
 * @returns {Array<{tenor: string, value: number}>}
 */
BCCh.mockCurve = ({ tenors, seed = 7, base = 5.0 } = {}) => {
    const rng = seededRng(seed);
    const slopes = [-0.4, -0.2, 0.0, 0.15, 0.35, 0.6, 0.9, 1.1];
    return tenors.map((t, i) => ({
        tenor: t,
        value: Number((base + slopes[i % slopes.length] + (rng() - 0.5) * 0.3).toFixed(2)),
    }));
};
