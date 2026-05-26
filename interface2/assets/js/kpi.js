/* ─────────────────────────────────────────────────────────────────────────
   KPI STRIP — indicadores económicos al top de la pantalla.
   Lee datos reales desde /v1/kpis (parquets). Los valores no disponibles
   muestran "—" en lugar de datos inventados.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

async function _fetchKPIs() {
    try {
        const res = await fetch("/v1/kpis", { cache: "no-store" });
        if (!res.ok) return null;
        const data = await res.json();
        return data.kpis ?? null;
    } catch {
        return null;
    }
}

BCCh.renderKPIStrip = async (container = BCCh.$("#kpi-strip")) => {
    container.innerHTML = "";
    const { h } = BCCh;

    const kpis = await _fetchKPIs();
    if (!kpis) {
        container.appendChild(
            h("div", { "class": "kpi-cell" }, [
                h("span", { "class": "kpi-cell__label" }, "KPIs no disponibles"),
            ])
        );
        return;
    }

    kpis.forEach((k) => {
        const dir = k.dir ?? "neu";
        const dirCls = `kpi-cell__delta--${dir}`;
        const arrow = dir === "up" ? "▲" : (dir === "down" ? "▼" : "—");
        const titleAttr = k.fecha ? `Dato al ${k.fecha}` : "";
        container.appendChild(h("div", { "class": "kpi-cell", title: titleAttr }, [
            h("span", { "class": "kpi-cell__label" }, k.label),
            h("span", { "class": "kpi-cell__value" }, k.value),
            h("span", { "class": `kpi-cell__delta ${dirCls}` }, [
                h("span", {}, arrow),
                h("span", {}, k.delta),
            ]),
        ]));
    });
};
