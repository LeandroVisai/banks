/* ─────────────────────────────────────────────────────────────────────────
   CHARTS — ApexCharts renderer + temporality + expand modal + brush
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {
const { $, h } = BCCh;

// ── Base config ApexCharts (light theme institucional) ───────────────────

const baseChartConfig = (type = "line", { compact = false, height = 240 } = {}) => ({
    chart: {
        type,
        height,
        background: "transparent",
        foreColor: "#7F8C8D",
        fontFamily: "Inter, system-ui, sans-serif",
        toolbar: {
            show: !compact,
            tools: { download: true, selection: true, zoom: true, zoomin: true, zoomout: true, pan: true, reset: true },
            autoSelected: "zoom",
        },
        zoom: { enabled: true, type: "x" },
        animations: { enabled: true, easing: "easeout", speed: 300, animateGradually: { enabled: false } },
    },
    colors: [
        "#BF9C69", "#001730", "#4472C4", "#70AD47", "#C00000",
        "#ED7D31", "#57257D", "#44546A", "#FFC000",
    ],
    stroke: { curve: "smooth", width: 2 },
    grid: {
        borderColor: "#e7e1d8",
        strokeDashArray: 3,
        xaxis: { lines: { show: false } },
        yaxis: { lines: { show: true } },
        padding: { top: 0, right: 10, bottom: 0, left: 10 },
    },
    xaxis: {
        type: "datetime",
        axisBorder: { color: "#d6d0c9" },
        axisTicks: { color: "#d6d0c9" },
        labels: { style: { colors: "#7F8C8D", fontSize: "10px", fontFamily: "IBM Plex Mono" } },
    },
    yaxis: {
        axisBorder: { show: false },
        axisTicks: { color: "#d6d0c9" },
        labels: {
            style: { colors: "#7F8C8D", fontSize: "10px", fontFamily: "IBM Plex Mono" },
            formatter: (v) => BCCh.FMT.number(v, 2),
        },
    },
    tooltip: {
        theme: "light",
        x: { format: "dd MMM yyyy" },
        style: { fontSize: "11px", fontFamily: "IBM Plex Mono" },
    },
    legend: {
        position: "top",
        horizontalAlign: "right",
        fontSize: "11px",
        fontFamily: "Inter",
        labels: { colors: "#7F8C8D" },
        markers: { width: 8, height: 8, radius: 1 },
        itemMargin: { horizontal: 8 },
    },
    dataLabels: { enabled: false },
});

// ── Builders ──────────────────────────────────────────────────────────────

const buildSeriesFromCatalog = (queryResp, chart) => {
    const rows = queryResp.rows || [];
    if (!rows.length) return [];

    const cols = queryResp.columns || Object.keys(rows[0]);
    const dateCol = chart.xField || cols[0];
    const valueCols = chart.yFields || cols.slice(1);

    const sorted = [...rows].sort((a, b) => {
        const da = new Date(a[dateCol]).getTime();
        const db = new Date(b[dateCol]).getTime();
        return da - db;
    });

    return valueCols.map((col) => ({
        name: col,
        data: sorted
            .filter((r) => r[col] !== null && r[col] !== undefined && !Number.isNaN(Number(r[col])))
            .map((r) => [new Date(r[dateCol]).getTime(), Number(r[col])]),
    })).filter((s) => s.data.length > 0);
};

const buildSeriesFromMock = (chart) => {
    const cfg = chart.mockConfig || {};
    if (chart.dataSource.mockType === "multi") {
        const seriesNames = chart.dataSource.mockSeries || ["A", "B"];
        const series = BCCh.mockMultiSeries({
            names: seriesNames,
            seed: chart.dataSource.seed || 42,
            config: cfg,
        });
        return series.map((s) => ({
            name: s.name,
            data: s.points.map((p) => [new Date(p.date).getTime(), p.value]),
        }));
    }
    const points = BCCh.mockSeries({ seed: chart.dataSource.seed || 42, ...cfg });
    return [{
        name: chart.title,
        data: points.map((p) => [new Date(p.date).getTime(), p.value]),
    }];
};

const applyChartTypeOpts = (cfg, chart) => {
    if (chart.type === "area") {
        cfg.fill = {
            type: "gradient",
            gradient: { shadeIntensity: 1, opacityFrom: 0.5, opacityTo: 0.05 },
        };
    }
    if (chart.type === "bar") {
        cfg.plotOptions = { bar: { borderRadius: 2, columnWidth: "60%" } };
        cfg.stroke = { width: 0 };
    }
    if (chart.yFormat === "percent") {
        cfg.yaxis.labels.formatter = (v) => BCCh.FMT.percent(v, 2);
        cfg.tooltip.y = { formatter: (v) => BCCh.FMT.percent(v, 2) };
    } else if (chart.yFormat === "bps") {
        cfg.yaxis.labels.formatter = (v) => `${v.toFixed(0)} pb`;
        cfg.tooltip.y = { formatter: (v) => `${v.toFixed(0)} pb` };
    }
    return cfg;
};

// ── Chart Card ────────────────────────────────────────────────────────────

class ChartCard {
    constructor(chart, parent) {
        this.chart = chart;
        this.parent = parent;
        this.apex = null;
        this.currentRange = chart.defaultRange || "1Y";
        this.lastSeries = null;     // cache para abrir modal sin refetch
        this.lastMeta   = null;
        this.el = this._buildShell();
        parent.appendChild(this.el);
    }

    _buildShell() {
        const isMock = this.chart.dataSource?.type === "mock";

        const tempButtons = (this.chart.temporality || ["1M", "3M", "6M", "1Y", "YTD", "Max"])
            .map((r) => h("button", {
                "class": "temporality__btn",
                "data-range": r,
                "data-active": String(r === this.currentRange),
                onClick: () => this.setRange(r),
            }, r));

        const tempCtl = h("div", { "class": "temporality", role: "tablist" }, tempButtons);

        const badge = isMock
            ? h("span", { "class": "badge badge--demo" }, "DEMO")
            : null;

        const titleWrap = h("div", { "class": "chart-card__title-wrap" }, [
            h("h3", { "class": "chart-card__title" }, this.chart.title),
            badge,
        ]);

        const expandBtn = h("button", {
            "class": "chart-card__expand",
            title: "Pantalla completa",
            "aria-label": "Pantalla completa",
            onClick: () => BCCh.ChartModal.open(this),
        });
        expandBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

        const actions = h("div", { "class": "chart-card__actions" }, [tempCtl, expandBtn]);

        const header = h("div", { "class": "chart-card__header" }, [titleWrap, actions]);

        const body = h("div", { "class": "chart-card__body" });
        this._bodyEl = body;
        this._showSkeleton();

        const footer = h("div", { "class": "chart-card__footer" }, [
            h("span", {}, this.chart.unit || ""),
            h("span", {}, "—"),
        ]);
        this._footerEl = footer;

        return h("div", {
            "class": "chart-card",
            "data-chart-id": this.chart.id,
            "data-span": String(this.chart.span || 1),
        }, [header, body, footer]);
    }

    _showSkeleton() {
        this._bodyEl.innerHTML = "";
        this._bodyEl.appendChild(h("div", { "class": "chart-skeleton" }));
    }

    _showError(msg) {
        this._bodyEl.innerHTML = "";
        this._bodyEl.appendChild(h("div", { "class": "chart-error" }, [
            h("div", { "class": "chart-error__icon" }, "!"),
            h("div", {}, msg),
        ]));
    }

    setRange(range) {
        this.currentRange = range;
        BCCh.$$(".temporality__btn", this.el).forEach((b) => {
            b.dataset.active = String(b.dataset.range === range);
        });
        this.load();
    }

    async fetchData() {
        const ds = this.chart.dataSource;
        if (ds.type === "catalog") {
            const fromD = BCCh.dateFromRange(this.currentRange);
            const data = await BCCh.API.query(ds.query_id, {
                fecha_inicio: BCCh.isoDate(fromD),
                fecha_fin: BCCh.isoDate(new Date()),
                limit: 500,
            });
            return {
                series: buildSeriesFromCatalog(data, this.chart),
                meta: data,
            };
        }
        if (ds.type === "mock") {
            return {
                series: buildSeriesFromMock(this.chart),
                meta: { unit: this.chart.unit, frequency: "MOCK" },
            };
        }
        throw new Error(`dataSource.type desconocido: ${ds.type}`);
    }

    async load() {
        try {
            this._showSkeleton();
            const { series, meta } = await this.fetchData();
            this.lastSeries = series;
            this.lastMeta   = meta;

            if (!series.length || !series[0].data?.length) {
                this._showError("Sin datos para el rango seleccionado");
                this._footerEl.children[0].textContent = this.chart.unit || "";
                this._footerEl.children[1].textContent = "0 puntos";
                return;
            }

            if (typeof ApexCharts === "undefined") {
                this._showError("ApexCharts no cargó — revisa la conexión al CDN");
                return;
            }

            const cfg = baseChartConfig(this.chart.type || "line", { height: 240 });
            cfg.series = series;
            applyChartTypeOpts(cfg, this.chart);

            // Si es bar y x datetime, mantenemos datetime (bar funciona con timestamps).
            this._bodyEl.innerHTML = "";
            if (this.apex) { try { this.apex.destroy(); } catch (_) {} this.apex = null; }
            this.apex = new ApexCharts(this._bodyEl, cfg);
            await this.apex.render();

            const lastPoint = series[0].data[series[0].data.length - 1];
            const asOf = lastPoint ? BCCh.FMT.date(new Date(lastPoint[0])) : "—";
            this._footerEl.children[0].textContent = `${meta.unit || this.chart.unit || ""} · ${meta.frequency || ""}`;
            this._footerEl.children[1].textContent = `al ${asOf}`;
        } catch (e) {
            console.warn(`[chart ${this.chart.id}]`, e);
            this._showError(e.message || "Error de carga");
        }
    }

    destroy() {
        if (this.apex) { try { this.apex.destroy(); } catch (_) {} this.apex = null; }
        this.el.remove();
    }
}

BCCh.ChartCard = ChartCard;

// ── Render section ────────────────────────────────────────────────────────

BCCh.renderSection = async (section, container) => {
    container.innerHTML = "";
    if (section.kind === "chat" && BCCh.mountInlineChat) {
        BCCh.mountInlineChat(container);
        return [];
    }
    const cards = (section.charts || []).map((c) => new ChartCard(c, container));
    await Promise.allSettled(cards.map((c) => c.load()));
    return cards;
};

// ─────────────────────────────────────────────────────────────────────────
// CHART MODAL — pantalla completa con brush/range selector
// ─────────────────────────────────────────────────────────────────────────

const ChartModal = {
    modalEl: null,
    mainApex: null,
    brushApex: null,
    currentCard: null,

    init() {
        this.modalEl   = $("#chart-modal");
        this.titleEl   = $("#chart-modal-title");
        this.subEl     = $("#chart-modal-page");
        this.mainEl    = $("#chart-modal-main");
        this.brushEl   = $("#chart-modal-brush");
        this.tempWrap  = $("#chart-modal-temporality");
        this.closeBtn  = $("#chart-modal-close");
        this.overlayEl = $("#chart-modal-overlay");

        this.closeBtn.addEventListener("click", () => this.close());
        this.overlayEl.addEventListener("click", () => this.close());
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && this.modalEl.classList.contains("open")) this.close();
        });
    },

    async open(card) {
        this.currentCard = card;
        this.titleEl.textContent = card.chart.title;
        this.subEl.textContent =
            (card.lastMeta?.unit || card.chart.unit || "") +
            (card.lastMeta?.frequency ? ` · ${card.lastMeta.frequency}` : "");

        // Temporality buttons en el modal
        this.tempWrap.innerHTML = "";
        const ranges = card.chart.temporality || ["1M", "3M", "6M", "1Y", "YTD", "Max"];
        const tempCtl = h("div", { "class": "temporality", role: "tablist" },
            ranges.map((r) => h("button", {
                "class": "temporality__btn",
                "data-range": r,
                "data-active": String(r === card.currentRange),
                onClick: async () => {
                    card.setRange(r);
                    BCCh.$$(".temporality__btn", this.tempWrap).forEach((b) => {
                        b.dataset.active = String(b.dataset.range === r);
                    });
                    await this._renderModalCharts();
                },
            }, r))
        );
        this.tempWrap.appendChild(tempCtl);

        this.modalEl.classList.add("open");
        await this._renderModalCharts();
    },

    async _renderModalCharts() {
        const card = this.currentCard;
        if (!card) return;

        let series = card.lastSeries;
        if (!series) {
            const r = await card.fetchData();
            series = r.series;
        }

        // Limpieza previa
        if (this.mainApex) { try { this.mainApex.destroy(); } catch (_) {} this.mainApex = null; }
        if (this.brushApex) { try { this.brushApex.destroy(); } catch (_) {} this.brushApex = null; }
        this.mainEl.innerHTML = "";
        this.brushEl.innerHTML = "";

        if (!series.length || !series[0].data?.length) {
            this.mainEl.appendChild(h("div", { "class": "chart-error" }, [
                h("div", { "class": "chart-error__icon" }, "!"),
                h("div", {}, "Sin datos para mostrar"),
            ]));
            return;
        }

        // Main chart (id sincronizado para que brush lo controle)
        const mainCfg = baseChartConfig(card.chart.type || "line", { height: "100%" });
        mainCfg.series = series;
        mainCfg.chart.id = "modal-main";
        mainCfg.chart.zoom = { autoScaleYaxis: true };
        applyChartTypeOpts(mainCfg, card.chart);

        // En el modal, rendereo más alto
        const computeHeight = () => Math.max(this.mainEl.clientHeight, 380);
        mainCfg.chart.height = computeHeight();

        this.mainApex = new ApexCharts(this.mainEl, mainCfg);
        await this.mainApex.render();

        // Brush chart (controla el zoom del main)
        const fullData = series[0].data;
        const initialFrom = fullData[Math.max(0, fullData.length - 60)][0]; // ~últimas 60 obs
        const initialTo   = fullData[fullData.length - 1][0];

        const brushCfg = {
            chart: {
                id: "modal-brush",
                height: 90,
                type: "area",
                brush: { enabled: true, target: "modal-main", autoScaleYaxis: true },
                selection: { enabled: true, xaxis: { min: initialFrom, max: initialTo } },
                background: "transparent",
                toolbar: { show: false },
                fontFamily: "Inter, sans-serif",
            },
            colors: ["#BF9C69"],
            stroke: { curve: "smooth", width: 1 },
            fill: { type: "gradient", gradient: { opacityFrom: 0.35, opacityTo: 0 } },
            series: [{ name: series[0].name, data: series[0].data }],
            xaxis: {
                type: "datetime",
                tooltip: { enabled: false },
                labels: { style: { colors: "#7F8C8D", fontSize: "10px", fontFamily: "IBM Plex Mono" } },
            },
            yaxis: { tickAmount: 2, labels: { show: false } },
            grid: { show: false },
            dataLabels: { enabled: false },
            legend: { show: false },
        };

        this.brushApex = new ApexCharts(this.brushEl, brushCfg);
        await this.brushApex.render();
    },

    close() {
        this.modalEl.classList.remove("open");
        if (this.mainApex)  { try { this.mainApex.destroy(); } catch (_) {} this.mainApex = null; }
        if (this.brushApex) { try { this.brushApex.destroy(); } catch (_) {} this.brushApex = null; }
        this.mainEl.innerHTML = "";
        this.brushEl.innerHTML = "";
        this.currentCard = null;
    },
};

BCCh.ChartModal = ChartModal;
}());
