/* ─────────────────────────────────────────────────────────────────────────
   MARKET HELPERS — cálculos derivados sobre rows del catalog.
   Todas las funciones son puras: reciben rows ya devueltas por
   BCCh.API.query() y producen métricas para los componentes Bloomberg.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const STATUS_FLAT_EPS = 1e-9;

const _values = (rows, col) => rows
    .map((r) => r[col])
    .filter((v) => v !== null && v !== undefined && Number.isFinite(v));

const last = (rows, col) => {
    const v = _values(rows, col);
    return v.length ? v[v.length - 1] : null;
};

const prev = (rows, col) => {
    const v = _values(rows, col);
    return v.length >= 2 ? v[v.length - 2] : null;
};

const chgAbs = (rows, col) => {
    const a = last(rows, col);
    const b = prev(rows, col);
    if (a === null || b === null) return null;
    return a - b;
};

const chgPct = (rows, col) => {
    const a = last(rows, col);
    const b = prev(rows, col);
    if (a === null || b === null || b === 0) return null;
    return ((a - b) / Math.abs(b)) * 100;
};

const sliceLast = (rows, col, n = 30) => {
    const v = _values(rows, col);
    return v.slice(-n);
};

const range52w = (rows, col) => {
    const v = _values(rows, col);
    if (!v.length) return { lo: null, hi: null, cur: null };
    const window = v.slice(-252);
    return {
        lo: Math.min(...window),
        hi: Math.max(...window),
        cur: v[v.length - 1],
    };
};

const statusFromChg = (chg) => {
    if (chg === null || chg === undefined) return "flat";
    if (Math.abs(chg) < STATUS_FLAT_EPS) return "flat";
    return chg > 0 ? "up" : "down";
};

const slope = (rows, colShort, colLong) => {
    const a = last(rows, colShort);
    const b = last(rows, colLong);
    if (a === null || b === null) return null;
    return b - a;
};

// ── Formatters ──────────────────────────────────────────────────────────

const _LOCALE = "es-CL";
const _NF_2 = new Intl.NumberFormat(_LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const _NF_3 = new Intl.NumberFormat(_LOCALE, { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const _NF_INT = new Intl.NumberFormat(_LOCALE, { maximumFractionDigits: 0 });

const formatNum = (v, decimals = 2) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    if (decimals === 0) return _NF_INT.format(v);
    if (decimals === 3) return _NF_3.format(v);
    return _NF_2.format(v);
};

const formatPct = (v, decimals = 2) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    const sign = v > 0 ? "+" : "";
    return `${sign}${formatNum(v, decimals)}%`;
};

const formatBp = (v) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    const sign = v > 0 ? "+" : "";
    return `${sign}${formatNum(v, 1)}`;
};

const formatSigned = (v, decimals = 2) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    const sign = v > 0 ? "+" : "";
    return `${sign}${formatNum(v, decimals)}`;
};

BCCh.MKT = {
    last, prev, chgAbs, chgPct, sliceLast, range52w, statusFromChg, slope,
    formatNum, formatPct, formatBp, formatSigned,
};

}());
