/* ─────────────────────────────────────────────────────────────────────────
   CONFIG — API base URL, paleta BCCh, formateadores, helpers globales
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

BCCh.CONFIG = {
    // API base: si el frontend se sirve desde la propia FastAPI, '' = same-origin
    API_BASE: window.location.port === "8080" ? "" : "http://localhost:8080",
    API_KEY: localStorage.getItem("bcch_api_key") || "",

    // Paleta BCCh (en orden de prioridad para series)
    PALETTE: [
        "#BF9C69",  // gold
        "#4472C4",  // blue
        "#70AD47",  // green
        "#C00000",  // red
        "#FFC000",  // yellow
        "#ED7D31",  // orange
        "#57257D",  // violet
        "#44546A",  // blue-gray
        "#7F7F7F",  // gray
    ],

    THEME: {
        bg:        "#001730",
        bgElev:    "#002347",
        text:      "#D4CFBE",
        textDim:   "#7F7F7F",
        textMute:  "#44546A",
        border:    "#1a3a5c",
        gold:      "#BF9C69",
        green:     "#70AD47",
        red:       "#C00000",
    },

    // Rangos temporales (días desde hoy)
    RANGES: {
        "1D":  1,
        "1W":  7,
        "1M":  30,
        "3M":  90,
        "6M":  180,
        "1Y":  365,
        "YTD": "ytd",
        "Max": 3650,
    },

    // Cache TTL (ms) para /v1/query
    CACHE_TTL: 5 * 60 * 1000,
};

// ── Formateadores ─────────────────────────────────────────────────────────

BCCh.FMT = {
    number: (v, decimals = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "—";
        return v.toLocaleString("es-CL", {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals,
        });
    },

    percent: (v, decimals = 2) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "—";
        return `${v.toFixed(decimals)}%`;
    },

    bps: (v) => {
        if (v === null || v === undefined || Number.isNaN(v)) return "—";
        const sign = v >= 0 ? "+" : "";
        return `${sign}${v.toFixed(0)} pb`;
    },

    date: (d) => {
        if (!d) return "—";
        const dt = (d instanceof Date) ? d : new Date(d);
        return dt.toLocaleDateString("es-CL", { day: "2-digit", month: "2-digit", year: "numeric" });
    },

    dateTime: (d) => {
        const dt = (d instanceof Date) ? d : new Date(d);
        return dt.toLocaleString("es-CL", { dateStyle: "short", timeStyle: "short" });
    },

    clock: (d) => {
        const dt = d || new Date();
        return dt.toLocaleTimeString("es-CL", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    },
};

// ── Helpers DOM ───────────────────────────────────────────────────────────

BCCh.$  = (sel, root = document) => root.querySelector(sel);
BCCh.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

BCCh.h = (tag, attrs = {}, children = []) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") el.className = v;
        else if (k === "html") el.innerHTML = v;
        else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
        else if (k.startsWith("data-")) el.setAttribute(k, v);
        else el[k] = v;
    }
    const list = Array.isArray(children) ? children : [children];
    for (const c of list) {
        if (c == null || c === false) continue;
        el.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
};

// ── Date helpers ──────────────────────────────────────────────────────────

BCCh.dateFromRange = (rangeKey) => {
    const today = new Date();
    const r = BCCh.CONFIG.RANGES[rangeKey];
    if (r === "ytd") {
        return new Date(today.getFullYear(), 0, 1);
    }
    const d = new Date(today);
    d.setDate(d.getDate() - r);
    return d;
};

BCCh.isoDate = (d) => d.toISOString().slice(0, 10);
