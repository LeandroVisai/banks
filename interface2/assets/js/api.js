/* ─────────────────────────────────────────────────────────────────────────
   API CLIENT — wrappers REST con cache LRU + manejo de errores
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

// Cache simple con TTL en memoria
const _cache = new Map();
const _cacheGet = (k) => {
    const e = _cache.get(k);
    if (!e) return null;
    if (Date.now() - e.t > BCCh.CONFIG.CACHE_TTL) {
        _cache.delete(k);
        return null;
    }
    return e.v;
};
const _cacheSet = (k, v) => _cache.set(k, { v, t: Date.now() });

const _headers = () => {
    const h = { "Content-Type": "application/json" };
    if (BCCh.CONFIG.API_KEY) h["X-API-Key"] = BCCh.CONFIG.API_KEY;
    return h;
};

class APIError extends Error {
    constructor(status, message, body) {
        super(message);
        this.status = status;
        this.body = body;
    }
}

const _fetch = async (path, opts = {}) => {
    const url = `${BCCh.CONFIG.API_BASE}${path}`;
    let res;
    try {
        res = await fetch(url, opts);
    } catch (e) {
        throw new APIError(0, `Sin conexión a ${url}`, null);
    }
    if (!res.ok) {
        let body = null;
        try { body = await res.json(); } catch { /* ignore */ }
        const msg = body?.detail || `HTTP ${res.status}`;
        throw new APIError(res.status, msg, body);
    }
    return res.json();
};

BCCh.API = {
    APIError,

    // ── Health ──────────────────────────────────────────────────────────
    healthz: () => _fetch("/healthz"),
    readyz:  () => _fetch("/readyz"),

    // ── Chat ────────────────────────────────────────────────────────────
    // thinkingMode: "off" | "adaptive" | "on" | null (null → default del server)
    chat: (message, history = [], thinkingMode = null) => _fetch("/v1/chat", {
        method: "POST",
        headers: _headers(),
        body: JSON.stringify(
            thinkingMode ? { message, history, thinking_mode: thinkingMode }
                         : { message, history },
        ),
    }),

    // ── Search ──────────────────────────────────────────────────────────
    search: (query, k = 5, filters = {}) => _fetch("/v1/search", {
        method: "POST",
        headers: _headers(),
        body: JSON.stringify({ query, k, filters }),
    }),

    // ── Catalog ─────────────────────────────────────────────────────────
    catalog: async () => {
        const cached = _cacheGet("catalog");
        if (cached) return cached;
        const data = await _fetch("/v1/catalog");
        _cacheSet("catalog", data);
        return data;
    },

    // ── Query (datos para gráficos) ─────────────────────────────────────
    query: async (queryId, params = {}) => {
        const qs = new URLSearchParams();
        if (params.fecha_inicio) qs.set("fecha_inicio", params.fecha_inicio);
        if (params.fecha_fin)    qs.set("fecha_fin", params.fecha_fin);
        if (params.limit)        qs.set("limit", String(params.limit));

        const key = `query:${queryId}?${qs.toString()}`;
        const cached = _cacheGet(key);
        if (cached) return cached;

        const path = `/v1/query/${encodeURIComponent(queryId)}${qs.toString() ? "?" + qs : ""}`;
        const data = await _fetch(path);
        _cacheSet(key, data);
        return data;
    },

    // ── Cache control ───────────────────────────────────────────────────
    clearCache: () => _cache.clear(),
};
