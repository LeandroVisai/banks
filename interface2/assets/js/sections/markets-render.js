/* ─────────────────────────────────────────────────────────────────────────
   MARKETS RENDER — sección "Mercados" estilo Bloomberg-dense.
   Orquesta los 6 paneles POC: FX/Forwards, Política Monetaria, BTP CLP,
   UST, Cobre, Tasas Monetarias CLP.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const { h, MKT, MarketPanel, API } = BCCh;

// ── Helpers de panel ────────────────────────────────────────────────────

const _tickerRow = (label, rows, col, opts = {}) => {
    const last = MKT.last(rows, col);
    const chg  = MKT.chgAbs(rows, col);
    const pct  = MKT.chgPct(rows, col);
    const spark = MKT.sliceLast(rows, col, 30);
    const range = MKT.range52w(rows, col);
    return {
        ticker: label,
        last,
        chg,
        pct: opts.chgFormat === "bp" ? null : pct,  // bp: ocultar pct
        sparkline: spark,
        range52w: range,
        status: MKT.statusFromChg(chg),
        chgFormat: opts.chgFormat || "num",
        decimals: opts.decimals !== undefined ? opts.decimals : 2,
    };
};

const _colsHeader = (showPct = true) => [
    "TICKER", "LAST", showPct ? "CHG" : "CHG(bp)", showPct ? "% CHG" : "", "30D", "52W",
];

// ── PANEL 1 · FX · USD/CLP & Forwards ───────────────────────────────────

const buildPanelFX = async (panel) => {
    panel.setBusy(true);
    try {
        const [spot, fwd] = await Promise.all([
            API.query("usdclp_historico", { limit: 400 }),
            API.query("forwards_clp_curva", { limit: 400 }),
        ]);
        const rows = [
            _tickerRow("USD/CLP",   spot.rows, "usdclp",      { decimals: 2 }),
            _tickerRow("FWD 30d",   fwd.rows,  "fwd_clp_30",  { chgFormat: "bp", decimals: 2 }),
            _tickerRow("FWD 90d",   fwd.rows,  "fwd_clp_90",  { chgFormat: "bp", decimals: 2 }),
            _tickerRow("FWD 180d",  fwd.rows,  "fwd_clp_180", { chgFormat: "bp", decimals: 2 }),
            _tickerRow("FWD 360d",  fwd.rows,  "fwd_clp_360", { chgFormat: "bp", decimals: 2 }),
        ];
        panel.setRows(rows, { colsHeader: _colsHeader(true) });
        const lastDate = spot.rows[spot.rows.length - 1]?.fecha;
        panel.setFooter(`${spot.unit || "CLP/USD"} · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── PANEL 2 · Política Monetaria CL · MIPR + SPC CLP ────────────────────

const buildPanelPolicy = async (panel) => {
    panel.setBusy(true);
    try {
        const [mipr, spc] = await Promise.all([
            API.query("expectativas_tpm_mipr", { limit: 400 }),
            API.query("spc_clp_curva", { limit: 400 }),
        ]);
        panel.setSubsections([
            {
                label: "TPM IMPLÍCITA · MIPR (cambio bp acum.)",
                rows: [
                    _tickerRow("3M",  mipr.rows, "spread_mipr_3m",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("6M",  mipr.rows, "spread_mipr_6m",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("9M",  mipr.rows, "spread_mipr_9m",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("12M", mipr.rows, "spread_mipr_12m", { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("24M", mipr.rows, "spread_mipr_24m", { chgFormat: "bp", decimals: 1 }),
                ],
            },
            {
                label: "CURVA SWAP CLP · SPC",
                rows: [
                    _tickerRow("SPC 3M",  spc.rows, "spc_3m_clp",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("SPC 1Y",  spc.rows, "spc_1y_clp",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("SPC 5Y",  spc.rows, "spc_5y_clp",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("SPC 10Y", spc.rows, "spc_10y_clp", { chgFormat: "bp", decimals: 2 }),
                ],
            },
        ]);
        const lastDate = mipr.rows[mipr.rows.length - 1]?.fecha;
        panel.setFooter(`bp · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── PANEL 3 · Curva Soberana CLP (BTP) ──────────────────────────────────

const buildPanelBTP = async (panel) => {
    panel.setBusy(true);
    try {
        const data = await API.query("curva_btp_clp", { limit: 400 });
        const rows = ["btp_1y","btp_2y","btp_5y","btp_7y","btp_10y","btp_15y","btp_20y","btp_30y"]
            .map((c) => _tickerRow(c.replace("btp_","").toUpperCase(), data.rows, c, { chgFormat: "bp", decimals: 2 }));
        // Slope 2s10s
        const s2s10s = MKT.slope(data.rows, "btp_2y", "btp_10y");
        rows.push({
            ticker: "2s10s",
            last: s2s10s,
            chg: null, pct: null, sparkline: null, range52w: null,
            status: MKT.statusFromChg(s2s10s), chgFormat: "num", decimals: 2,
        });
        panel.setRows(rows, { colsHeader: _colsHeader(false) });
        const lastDate = data.rows[data.rows.length - 1]?.fecha;
        panel.setFooter(`% nominal · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── PANEL 4 · UST · Curva EEUU ──────────────────────────────────────────

const buildPanelUST = async (panel) => {
    panel.setBusy(true);
    try {
        const data = await API.query("curva_ust", { limit: 400 });
        const rows = ["ust_2y","ust_5y","ust_10y","ust_20y","ust_30y"]
            .map((c) => _tickerRow(c.replace("ust_","UST ").toUpperCase(), data.rows, c, { chgFormat: "bp", decimals: 2 }));
        const s2s10s = MKT.slope(data.rows, "ust_2y", "ust_10y");
        rows.push({
            ticker: "2s10s",
            last: s2s10s,
            chg: null, pct: null, sparkline: null, range52w: null,
            status: MKT.statusFromChg(s2s10s), chgFormat: "num", decimals: 2,
        });
        panel.setRows(rows, { colsHeader: _colsHeader(false) });
        const lastDate = data.rows[data.rows.length - 1]?.fecha;
        panel.setFooter(`% · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── PANEL 5 · Cobre & Drivers ───────────────────────────────────────────

const buildPanelCobre = async (panel) => {
    panel.setBusy(true);
    try {
        const data = await API.query("precio_cobre", { limit: 400 });
        const last = MKT.last(data.rows, "cobre_usd_lb");
        const chg  = MKT.chgAbs(data.rows, "cobre_usd_lb");
        const pct  = MKT.chgPct(data.rows, "cobre_usd_lb");
        const range = MKT.range52w(data.rows, "cobre_usd_lb");
        const series = MKT.sliceLast(data.rows, "cobre_usd_lb", 252);
        const status = MKT.statusFromChg(chg);

        panel.setKpiTiles([
            { label: "LAST USD/lb",   value: MKT.formatNum(last, 3), status, sub: MKT.formatPct(pct) },
            { label: "CHG vs ayer",   value: MKT.formatSigned(chg, 3), status, sub: MKT.formatPct(pct) },
            { label: "52W RANGE",     value: `${MKT.formatNum(range.lo,2)}–${MKT.formatNum(range.hi,2)}`, sub: "USD/lb" },
        ]);

        // Append chart below tiles
        const chartEl = h("div", { "class": "market-panel__chart-inline" });
        panel.bodyEl.appendChild(chartEl);
        if (window.ApexCharts && series.length > 1) {
            const color = getComputedStyle(document.documentElement).getPropertyValue("--c-gold").trim() || "#BF9C69";
            const chart = new ApexCharts(chartEl, {
                chart: { type: "area", height: 120, toolbar: { show: false }, animations: { enabled: false } },
                series: [{ name: "Cu", data: series }],
                stroke: { width: 1.6, curve: "smooth" },
                colors: [color],
                fill: { type: "gradient", gradient: { opacityFrom: 0.25, opacityTo: 0 } },
                grid: { borderColor: "rgba(0,0,0,0.05)", strokeDashArray: 2, padding: { left: 0, right: 0, top: 0, bottom: 0 } },
                xaxis: { labels: { show: false }, axisTicks: { show: false }, axisBorder: { show: false } },
                yaxis: {
                    labels: {
                        style: { fontSize: "10px" },
                        formatter: (v) => (Number.isFinite(v) ? v.toFixed(2) : v),
                    },
                },
                tooltip: { x: { show: false }, y: { formatter: (v) => (Number.isFinite(v) ? v.toFixed(3) + " USD/lb" : v) } },
                dataLabels: { enabled: false },
            });
            chart.render();
            panel._chartInstance = chart;
        }

        const lastDate = data.rows[data.rows.length - 1]?.fecha;
        panel.setFooter(`USD/lb · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── PANEL 6 · Tasas Monetarias CLP ──────────────────────────────────────

const buildPanelTasasCLP = async (panel) => {
    panel.setBusy(true);
    try {
        const [dap, pdbc, tib] = await Promise.all([
            API.query("spreads_dap_swap_clp", { limit: 400 }),
            API.query("pdbc_bolsa", { limit: 400 }),
            API.query("tib_mercado", { limit: 400 }),
        ]);
        panel.setSubsections([
            {
                label: "DAP-SWAP & PRIME-SWAP (bp)",
                rows: [
                    _tickerRow("DAP 1M",   dap.rows, "spread_dap_swap_1m_clp",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("DAP 3M",   dap.rows, "spread_dap_swap_3m_clp",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("DAP 6M",   dap.rows, "spread_dap_swap_6m_clp",  { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("DAP 12M",  dap.rows, "spread_dap_swap_12m_clp", { chgFormat: "bp", decimals: 1 }),
                    _tickerRow("PRIME 3M", dap.rows, "spread_prime_swap_3m_clp",{ chgFormat: "bp", decimals: 1 }),
                ],
            },
            {
                label: "PDBC BOLSA",
                rows: [
                    _tickerRow("7D",   pdbc.rows, "pdbc_bolsa_7d",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("30D",  pdbc.rows, "pdbc_bolsa_30",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("90D",  pdbc.rows, "pdbc_bolsa_90",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("180D", pdbc.rows, "pdbc_bolsa_180", { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("360D", pdbc.rows, "pdbc_bolsa_360", { chgFormat: "bp", decimals: 2 }),
                ],
            },
            {
                label: "TIB MERCADO",
                rows: [
                    _tickerRow("Tasa",  tib.rows, "tib_tasa",  { chgFormat: "bp", decimals: 2 }),
                    _tickerRow("Monto", tib.rows, "tib_monto", { chgFormat: "num", decimals: 2 }),
                ],
            },
        ]);
        const lastDate = dap.rows[dap.rows.length - 1]?.fecha;
        panel.setFooter(`bp · diario · al ${lastDate || "—"}`);
    } catch (e) {
        panel.setError(e.message || "Error de carga");
    } finally {
        panel.setBusy(false);
    }
};

// ── Render orchestrator ─────────────────────────────────────────────────

const PANELS = [
    { title: "FX · USD/CLP & FORWARDS",       count: 5, span: 1, build: buildPanelFX },
    { title: "POLÍTICA MONETARIA CL",         count: 9, span: 1, build: buildPanelPolicy },
    { title: "COBRE & DRIVERS",               count: 1, span: 1, build: buildPanelCobre },
    { title: "CURVA SOBERANA CLP · BTP",      count: 9, span: 1, build: buildPanelBTP },
    { title: "UST · CURVA EEUU",              count: 6, span: 1, build: buildPanelUST },
    { title: "TASAS MONETARIAS CLP",          count: 12, span: 1, build: buildPanelTasasCLP },
];

const renderMarkets = (section, container) => {
    container.innerHTML = "";
    container.classList.add("market-grid");

    const panelInstances = PANELS.map((p) => {
        const panel = new BCCh.MarketPanel({
            title: p.title, count: p.count, span: p.span, variant: "tickers",
        });
        panel.mount(container);
        return { panel, build: p.build };
    });

    // Trigger all builds in parallel (Promise.allSettled — un panel roto no bloquea otros)
    Promise.allSettled(panelInstances.map(({ panel, build }) => build(panel)));

    return panelInstances;
};

BCCh.renderMarkets = renderMarkets;

}());
