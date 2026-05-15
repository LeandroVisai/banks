/* ─────────────────────────────────────────────────────────────────────────
   KPI STRIP — indicadores económicos al top de la pantalla.
   En esta versión usa datos mock; cuando los parquets se conecten a Monitor PM
   las KPIs se calcularán desde /v1/query/{id} del último punto.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

const KPIS = [
    { label: "TPM",      value: "5.50%",  delta: "−25 pb",  dir: "down" },
    { label: "USD/CLP",  value: "942.3",  delta: "+1.2%",   dir: "up" },
    { label: "IPC 12m",  value: "4.2%",   delta: "+0.1 pp", dir: "up" },
    { label: "Cobre",    value: "4.12",   delta: "−0.8%",   dir: "down" },
    { label: "IPSA",     value: "7,284",  delta: "+0.3%",   dir: "up" },
];

BCCh.renderKPIStrip = (container = BCCh.$("#kpi-strip")) => {
    container.innerHTML = "";
    const { h } = BCCh;
    KPIS.forEach((k) => {
        const dirCls = `kpi-cell__delta--${k.dir}`;
        const arrow  = k.dir === "up" ? "▲" : (k.dir === "down" ? "▼" : "—");
        container.appendChild(h("div", { "class": "kpi-cell" }, [
            h("span", { "class": "kpi-cell__label" }, k.label),
            h("span", { "class": "kpi-cell__value" }, k.value),
            h("span", { "class": `kpi-cell__delta ${dirCls}` }, [
                h("span", {}, arrow),
                h("span", {}, k.delta),
            ]),
        ]));
    });
};
