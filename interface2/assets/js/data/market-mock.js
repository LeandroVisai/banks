/* ─────────────────────────────────────────────────────────────────────────
   MARKET MOCK — datos sintéticos para la sección "Mercados en Línea".
   Cubre lo que el catálogo SQL del proyecto NO tiene: equity local,
   índices globales, commodities globales, monedas cruzadas, indicadores.
   Valores semilla realistas (mercado Chile) + jitter determinístico por
   día para que la pantalla se vea viva pero estable dentro de la sesión.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

// PRNG determinístico (mulberry32) — semilla = fecha del día
const _mulberry32 = (a) => () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const _todaySeed = () => {
    const d = new Date();
    return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
};
const rng = _mulberry32(_todaySeed());
const _signed = (range) => (rng() * 2 - 1) * range;

// items: [label, baseValue, basePct, {dim?, decimals?, badge?}]
const _table = (items) => items.map((it) => {
    const [label, base, basePct, opts = {}] = it;
    return {
        label,
        value: base * (1 + _signed(0.0015)),
        pct: basePct + _signed(0.18),
        dim: opts.dim || false,
        badge: opts.badge || false,
        decimals: opts.decimals !== undefined ? opts.decimals : 2,
    };
});

// ── Índices bursátiles ──────────────────────────────────────────────────
const indices = () => _table([
    ["IPSA",         10359.15, -1.17],
    ["IGPA",         52423.60, -1.07],
    ["S&P 500",       7431.42, -0.93],
    ["S&P 100",       3705.37, -1.00],
    ["NASDAQ 100",   29580.30,  0.73, { dim: true }],
    ["NASDAQ COMP",  26635.22,  0.88, { dim: true }],
    ["S&P 40 MILA",    698.52,  2.48, { dim: true }],
    ["BOVESPA",     175966.75, -1.35],
    ["MERVAL",     2747310.00,  0.33, { dim: true }],
    ["COLCAP",        2130.41,  0.40],
    ["FTSE 100",     10208.30, -1.59],
    ["DAX",          24007.19, -1.84],
    ["NIKKEI 225",   61409.29, -1.99, { dim: true }],
    ["HANG SENG",    25962.73, -1.62, { dim: true }],
    ["CSI 300",       4859.59, -1.12, { dim: true }],
]);

// ── Principales acciones (IPSA) ─────────────────────────────────────────
const acciones = () => _table([
    ["AGUAS-A",      314.18,  -0.58],
    ["ANDINA-B",    4204.60,  -1.30],
    ["BCI",        55911.00,  -2.34],
    ["BSANTANDER",    68.07,  -1.49],
    ["CAP",         6700.00,  -1.00],
    ["CCU",         5241.00,  -0.01],
    ["CENCOMALLS",  2335.00,  -2.71],
    ["CENCOSUD",    2117.00,  -0.38],
    ["CHILE",        161.20,  -1.40],
    ["CMPC",        1066.60,   0.15],
    ["COPEC",       6890.00,  -0.92],
    ["FALABELLA",   5350.60,  -1.67],
]);

// ── Mayores alzas ───────────────────────────────────────────────────────
const alzas = () => _table([
    ["CONCHATORO",   854.00,  2.89],
    ["SMU",          198.40,  1.74],
    ["CMPC",        1066.60,  0.15],
    ["IB01",         120.52,  0.13],
    ["ENTEL",       3587.00,  0.08],
]);

// ── Mayores bajas ───────────────────────────────────────────────────────
const bajas = () => _table([
    ["ADBECL",    220000.00, -5.77],
    ["VAPORES",       44.90, -12.05],
    ["ECL",         1682.00, -2.89],
    ["ENELAM",        76.00, -2.69],
    ["SQM-B",      76300.00, -2.43],
    ["LTM",           21.60, -2.26],
    ["ORO BLANCO",    11.26, -2.09],
    ["BCI",        56093.00, -2.03],
    ["FALABELLA",   5350.60, -1.67],
    ["RIPLEY",       352.61, -1.62],
]);

// ── Más transadas (monto MM$) ───────────────────────────────────────────
const transadas = () => _table([
    ["SQM-B",       1622.70, -2.43],
    ["CHILE",        828.14, -1.40],
    ["BSANTANDER",   508.59, -1.49],
    ["LTM",          424.15, -2.26],
    ["VAPORES",      353.25, -12.05],
    ["CMPC",         116.84,  0.15],
    ["CENCOSUD",      82.60, -0.38],
    ["FALABELLA",     79.94, -1.67],
    ["BCI",           70.30, -2.03],
    ["PARAUCO",       53.89, -0.32],
]);

// ── Indicadores económicos ──────────────────────────────────────────────
const indicadores = () => _table([
    ["UF",              40340.86, 0.04],
    ["UTM",             70588.00, 1.00],
    ["Dólar Observado",   891.00, 0.20],
    ["IPC (abril)",       134.10, 1.30],
]);

// ── Commodities globales (cobre viene del catálogo, real) ───────────────
const commoditiesExtra = () => _table([
    ["Petróleo WTI",     103.38,  1.73, { decimals: 2 }],
    ["Petróleo Brent",   105.72,  0.09, { decimals: 2 }],
    ["Oro oz.",         4528.01, -2.66, { decimals: 2 }],
    ["Plata oz.",         76.48, -8.43, { decimals: 2 }],
    ["Gas Natural",        2.95,  0.68, { decimals: 2 }],
    ["Harina Pescado",  1932.77,  5.24, { decimals: 2 }],
]);

// ── Monedas cruzadas (USD/CLP viene del catálogo, real) ─────────────────
const monedasExtra = () => _table([
    ["Euro (EUR/USD)",        1.1631, -0.33, { decimals: 4 }],
    ["Real (USD/BRL)",        5.0467,  1.26, { decimals: 4 }],
    ["Peso Mex. (USD/MXN)",  17.3450,  0.65, { decimals: 4 }],
    ["Dólar Aus. (AUD/USD)",  0.7155, -0.91, { decimals: 4 }],
    ["Yen (USD/JPY)",       158.5815,  0.14, { decimals: 4 }],
    ["Libra (GBP/USD)",       1.3360, -0.32, { decimals: 4 }],
    ["Peso Col. (USD/COP)",3815.1000,  0.67, { decimals: 2 }],
    ["Sol Per. (USD/PEN)",    3.4208, -0.02, { decimals: 4 }],
]);

// ── Tasas de interés (renta fija, estilo BTG) ───────────────────────────
const tasas = () => _table([
    ["TIP 30-89 CLP",   4.20, 0.00, { decimals: 2 }],
    ["TIP 1 año UF",    0.20, 0.00, { decimals: 2 }],
    ["BCP 2 años",      5.00, 0.00, { decimals: 2 }],
    ["BCP 5 años",      5.33, 0.00, { decimals: 2 }],
    ["BCP 10 años",     5.53, -0.02, { decimals: 2 }],
    ["BCU 5 años",      1.95, 0.00, { decimals: 2 }],
    ["BCU 10 años",     2.27, 0.03, { decimals: 2 }],
    ["US Treasury 2",   4.00, 0.02, { decimals: 2 }],
    ["US Treasury 5",   4.13, 0.01, { decimals: 2 }],
    ["US Treasury 10",  4.47, 0.01, { decimals: 2 }],
]);

// ── Serie histórica IPSA para el hero chart ─────────────────────────────
const ipsaSeries = (n = 60) => {
    const out = [];
    let v = 11320;
    for (let i = 0; i < n; i++) {
        v = v * (1 + _signed(0.006) - 0.0009);  // leve tendencia bajista
        out.push(Number(v.toFixed(2)));
    }
    return out;
};

// ── Serie histórica por instrumento (determinística, termina en endValue) ─
const _hashStr = (s) => {
    let hsh = 0;
    for (let i = 0; i < s.length; i++) {
        hsh = (Math.imul(hsh, 31) + s.charCodeAt(i)) | 0;
    }
    return hsh >>> 0;
};

const seriesFor = (label, endValue, n = 60) => {
    if (!Number.isFinite(endValue)) return [];
    const rnd = _mulberry32((_hashStr(label) ^ _todaySeed()) >>> 0);
    const out = [Number(endValue.toFixed(4))];
    let v = endValue;
    for (let i = 1; i < n; i++) {
        v = v / (1 + (rnd() * 2 - 1) * 0.013);  // camina hacia atrás
        out.unshift(Number(v.toFixed(4)));
    }
    return out;
};

BCCh.MarketMock = {
    indices, acciones, alzas, bajas, transadas,
    indicadores, commoditiesExtra, monedasExtra, tasas, ipsaSeries, seriesFor,
};

}());
