/* ─────────────────────────────────────────────────────────────────────────
   SPARKLINE MINI — micro-chart inline (~72×20px) sin ejes ni tooltip.
   Wrapper de ApexCharts en modo sparkline. Cache de instancias para
   permitir update sin re-mount.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const _instances = new WeakMap();

const _color = (deltaSign) => {
    if (deltaSign > 0)  return getComputedStyle(document.documentElement).getPropertyValue("--c-up").trim()   || "#4F8C5D";
    if (deltaSign < 0)  return getComputedStyle(document.documentElement).getPropertyValue("--c-down").trim() || "#B23A2E";
    return getComputedStyle(document.documentElement).getPropertyValue("--c-flat").trim() || "#8a8478";
};

const _options = (series, deltaSign) => ({
    chart: {
        type: "line",
        sparkline: { enabled: true },
        animations: { enabled: false },
        toolbar: { show: false },
        zoom: { enabled: false },
    },
    series: [{ data: series }],
    stroke: { width: 1.4, curve: "straight" },
    colors: [_color(deltaSign)],
    tooltip: { enabled: false },
    grid: { show: false, padding: { left: 0, right: 0, top: 0, bottom: 0 } },
});

const mount = (el, { series, deltaSign = 0 }) => {
    if (!el || !window.ApexCharts) return null;
    if (!series || series.length < 2) {
        el.innerHTML = '<span style="color:var(--text-mute);font-size:10px">—</span>';
        return null;
    }

    const existing = _instances.get(el);
    if (existing) {
        try { existing.destroy(); } catch (_) { /* ignore */ }
        _instances.delete(el);
    }

    el.innerHTML = "";
    const chart = new ApexCharts(el, _options(series, deltaSign));
    chart.render();
    _instances.set(el, chart);
    return chart;
};

const destroy = (el) => {
    if (!el) return;
    const chart = _instances.get(el);
    if (chart) {
        try { chart.destroy(); } catch (_) { /* ignore */ }
        _instances.delete(el);
    }
};

BCCh.SparklineMini = { mount, destroy };

}());
