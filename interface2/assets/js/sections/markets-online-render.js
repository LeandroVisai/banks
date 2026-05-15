/* ─────────────────────────────────────────────────────────────────────────
   MERCADOS EN LÍNEA — sección estilo LarrainVial / BTG, paleta clara BCCh.
   Hero chart interactivo: selector de tenor (1M…3A) + rango de fechas
   Desde/Hasta seleccionable + eje datetime. 9 tablas densas; click en
   cualquier fila grafica ese instrumento. USD/CLP y Cobre traen serie
   real del catálogo; el resto, sintética.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const { h, MKT, API, QuoteTable, MarketMock } = BCCh;

const TENORS = [
    { id: "1M", days: 22 },
    { id: "3M", days: 66 },
    { id: "6M", days: 132 },
    { id: "1A", days: 252 },
    { id: "3A", days: 756 },
];
const DEFAULT_TENOR = "6M";
const FULL_LEN = 756;          // 3 años de puntos para el slice
const DAY_MS = 86400000;

let _heroChart = null;
let _tables = [];
const _hero = {};              // refs DOM
let _current = null;           // { label, tag, series, dates, decimals }
let _tenor = DEFAULT_TENOR;    // id de tenor activo, o null si rango custom
let _customRange = null;       // { from, to } timestamps, o null

// ── Fechas ──────────────────────────────────────────────────────────────

const _synthDates = (n) => {
    const out = [];
    let d = new Date();
    let count = 0;
    while (count < n) {
        const dow = d.getDay();
        if (dow !== 0 && dow !== 6) { out.unshift(d.getTime()); count += 1; }
        d = new Date(d.getTime() - DAY_MS);
    }
    return out;
};

const _toInputDate = (ts) => {
    const d = new Date(ts);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
};

const _fromInputDate = (str) => {
    if (!str) return null;
    const [y, m, d] = str.split("-").map(Number);
    if (!y || !m || !d) return null;
    return new Date(y, m - 1, d).getTime();
};

// ── Hero ────────────────────────────────────────────────────────────────

const _mountHero = (container) => {
    _hero.indexEl = h("span", { "class": "mo-hero__index" }, "IPSA");
    _hero.tagEl   = h("span", { "class": "mo-hero__tag" }, "MERCADO LOCAL · CHILE");
    _hero.statsEl = h("div",  { "class": "mo-hero__stats" });

    const header = h("div", { "class": "mo-hero__header" }, [
        h("div", { "class": "mo-hero__title" }, [_hero.indexEl, _hero.tagEl]),
        _hero.statsEl,
    ]);

    _hero.tenorEl = h("div", { "class": "mo-hero__tenors" });
    TENORS.forEach((t) => {
        _hero.tenorEl.appendChild(h("button", {
            "class": "mo-tenor" + (t.id === _tenor ? " mo-tenor--active" : ""),
            "data-tenor": t.id,
            onClick: () => _setTenor(t.id),
        }, t.id));
    });

    _hero.dateFrom = h("input", { "type": "date", "class": "mo-date__input", "aria-label": "Desde" });
    _hero.dateTo   = h("input", { "type": "date", "class": "mo-date__input", "aria-label": "Hasta" });
    _hero.dateFrom.addEventListener("change", _applyCustomRange);
    _hero.dateTo.addEventListener("change", _applyCustomRange);

    const dates = h("div", { "class": "mo-hero__dates" }, [
        h("span", { "class": "mo-date__label" }, "Desde"),
        _hero.dateFrom,
        h("span", { "class": "mo-date__label" }, "Hasta"),
        _hero.dateTo,
    ]);

    const toolbar = h("div", { "class": "mo-hero__toolbar" }, [_hero.tenorEl, dates]);
    _hero.chartEl = h("div", { "class": "mo-hero__chart" });
    container.appendChild(h("div", { "class": "mo-hero" }, [header, toolbar, _hero.chartEl]));
};

const _setTenorButtons = () => {
    _hero.tenorEl.querySelectorAll(".mo-tenor").forEach((b) => {
        b.classList.toggle("mo-tenor--active", _tenor !== null && b.dataset.tenor === _tenor);
    });
};

const _setTenor = (id) => {
    _tenor = id;
    _customRange = null;
    _setTenorButtons();
    if (_current) _renderHero();
};

const _applyCustomRange = () => {
    if (!_current) return;
    let from = _fromInputDate(_hero.dateFrom.value);
    let to   = _fromInputDate(_hero.dateTo.value);
    if (from === null || to === null) return;
    if (from > to) { const t = from; from = to; to = t; }
    _customRange = { from, to: to + DAY_MS - 1 };
    _tenor = null;
    _setTenorButtons();
    _renderHero();
};

const _visibleSlice = () => {
    const { series, dates } = _current;
    if (_customRange) {
        const idx = [];
        dates.forEach((ts, i) => {
            if (ts >= _customRange.from && ts <= _customRange.to) idx.push(i);
        });
        if (idx.length >= 2) {
            return { slice: idx.map((i) => series[i]), dslice: idx.map((i) => dates[i]) };
        }
        // Rango sin suficientes puntos → cae a tenor por defecto
        _customRange = null;
        _tenor = DEFAULT_TENOR;
        _setTenorButtons();
    }
    const tenorDef = TENORS.find((t) => t.id === _tenor) || TENORS[2];
    const n = Math.min(tenorDef.days, series.length);
    return { slice: series.slice(-n), dslice: dates.slice(-n) };
};

const _renderHero = () => {
    if (!_current) return;
    const { label, tag, decimals } = _current;
    const { slice, dslice } = _visibleSlice();
    if (slice.length < 2) return;

    const firstV = slice[0];
    const lastV  = slice[slice.length - 1];
    const chgPts = lastV - firstV;
    const chgPct = firstV ? (chgPts / firstV) * 100 : 0;
    const status = chgPts > 0 ? "up" : chgPts < 0 ? "down" : "flat";

    _hero.indexEl.textContent = label;
    _hero.tagEl.textContent = tag || "";

    _hero.statsEl.innerHTML = "";
    [
        ["ÚLTIMO",    MKT.formatNum(lastV, decimals),     ""],
        ["VAR. PTS.", MKT.formatSigned(chgPts, decimals), status],
        ["VAR. %",    MKT.formatPct(chgPct, 2),           status],
    ].forEach(([lbl, val, st]) => {
        _hero.statsEl.appendChild(h("div", { "class": "mo-hero__stat" + (st ? " tick-" + st : "") }, [
            h("span", { "class": "mo-hero__stat-label" }, lbl),
            h("span", { "class": "mo-hero__stat-value mono" }, val),
        ]));
    });

    // Sincroniza los inputs de fecha con el rango efectivamente mostrado
    _hero.dateFrom.value = _toInputDate(dslice[0]);
    _hero.dateTo.value   = _toInputDate(dslice[dslice.length - 1]);

    if (window.ApexCharts) {
        const color = getComputedStyle(document.documentElement)
            .getPropertyValue(status === "up" ? "--c-up" : status === "down" ? "--c-down" : "--c-gold")
            .trim() || "#BF9C69";
        if (_heroChart) { try { _heroChart.destroy(); } catch (_) {} }
        _heroChart = new ApexCharts(_hero.chartEl, {
            chart: { type: "area", height: 240, toolbar: { show: false }, animations: { enabled: false } },
            series: [{ name: label, data: slice.map((v, i) => [dslice[i], v]) }],
            stroke: { width: 1.8, curve: "smooth" },
            colors: [color],
            fill: { type: "gradient", gradient: { opacityFrom: 0.22, opacityTo: 0 } },
            grid: { borderColor: "rgba(0,0,0,0.05)", strokeDashArray: 3 },
            xaxis: {
                type: "datetime",
                labels: { style: { fontSize: "10px", colors: "var(--text-mute)" }, datetimeUTC: false },
                axisTicks: { show: false },
                axisBorder: { show: false },
            },
            yaxis: {
                labels: {
                    style: { fontSize: "10px" },
                    formatter: (v) => MKT.formatNum(v, decimals >= 2 ? 0 : decimals),
                },
            },
            dataLabels: { enabled: false },
            tooltip: {
                x: { format: "dd MMM yyyy" },
                y: { formatter: (v) => MKT.formatNum(v, decimals) },
            },
        });
        _heroChart.render();
    }
};

// ── Selección de instrumento ────────────────────────────────────────────

const _setInstrument = ({ label, tag, series, dates, decimals }) => {
    const full = series.slice(-FULL_LEN);
    const fullDates = (dates && dates.length === full.length)
        ? dates
        : _synthDates(full.length);
    _current = {
        label,
        tag: (tag || "").toUpperCase(),
        series: full,
        dates: fullDates,
        decimals: decimals !== undefined ? decimals : 2,
    };

    // Límites del date-picker = rango disponible del instrumento
    _hero.dateFrom.min = _hero.dateTo.min = _toInputDate(fullDates[0]);
    _hero.dateFrom.max = _hero.dateTo.max = _toInputDate(fullDates[fullDates.length - 1]);

    // Clampea un rango custom previo al nuevo instrumento
    if (_customRange) {
        _customRange.from = Math.max(_customRange.from, fullDates[0]);
        _customRange.to   = Math.min(_customRange.to, fullDates[fullDates.length - 1]);
    }
    _renderHero();
};

const _select = (rowData, table, rowEl) => {
    _tables.forEach((t) => t.clearActive());
    if (rowEl) rowEl.classList.add("quote-table__row--active");
    const decimals = rowData.decimals !== undefined ? rowData.decimals : 2;
    const series = rowData._series && rowData._series.length > 1
        ? rowData._series
        : MarketMock.seriesFor(rowData.label, rowData.value, FULL_LEN);
    _setInstrument({
        label: rowData.label,
        tag: table ? table.title : "",
        series,
        dates: rowData._dates,
        decimals,
    });
};

// ── Tablas con datos reales del catálogo ────────────────────────────────

const _fillMonedas = async (table) => {
    table.setBusy(true);
    try {
        const spot = await API.query("usdclp_historico", { limit: 400 });
        const valid = spot.rows.filter((r) => Number.isFinite(r.usdclp));
        table.setRows([
            {
                label: "Dólar (USD/CLP)",
                value: MKT.last(spot.rows, "usdclp"),
                pct: MKT.chgPct(spot.rows, "usdclp"),
                decimals: 2,
                _series: valid.map((r) => r.usdclp),
                _dates: valid.map((r) => new Date(r.fecha).getTime()),
            },
            ...MarketMock.monedasExtra(),
        ]);
    } catch (e) {
        table.setRows(MarketMock.monedasExtra());
    } finally {
        table.setBusy(false);
    }
};

const _fillCommodities = async (table) => {
    table.setBusy(true);
    try {
        const cu = await API.query("precio_cobre", { limit: 400 });
        const valid = cu.rows.filter((r) => Number.isFinite(r.cobre_usd_lb));
        table.setRows([
            {
                label: "Cobre A Cash",
                value: MKT.last(cu.rows, "cobre_usd_lb"),
                pct: MKT.chgPct(cu.rows, "cobre_usd_lb"),
                decimals: 4,
                _series: valid.map((r) => r.cobre_usd_lb),
                _dates: valid.map((r) => new Date(r.fecha).getTime()),
            },
            ...MarketMock.commoditiesExtra(),
        ]);
    } catch (e) {
        table.setRows(MarketMock.commoditiesExtra());
    } finally {
        table.setBusy(false);
    }
};

// ── Render orchestrator ─────────────────────────────────────────────────

const renderMarketsOnline = (section, container) => {
    container.innerHTML = "";
    container.classList.add("markets-online");
    _tables = [];
    _heroChart = null;
    _current = null;
    _tenor = DEFAULT_TENOR;
    _customRange = null;

    const heroWrap = h("div", { "class": "mo-cell mo-cell--hero" });
    container.appendChild(heroWrap);
    _mountHero(heroWrap);

    const tIndices = new QuoteTable({
        title: "Índices", count: 15, valueLabel: "PUNTOS", scroll: true, onSelect: _select,
    });
    tIndices.mount(container);
    tIndices.setRows(MarketMock.indices());
    _tables.push(tIndices);

    const defs = [
        { t: new QuoteTable({ title: "Principales Acciones", count: 12, valueLabel: "PRECIO", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.acciones()) },
        { t: new QuoteTable({ title: "Mayores Alzas", count: 5, valueLabel: "PRECIO", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.alzas()) },
        { t: new QuoteTable({ title: "Mayores Bajas", count: 10, valueLabel: "PRECIO", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.bajas()) },
        { t: new QuoteTable({ title: "Más Transadas", count: 10, valueLabel: "MM $", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.transadas()) },
        { t: new QuoteTable({ title: "Indicadores", count: 4, valueLabel: "VALOR", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.indicadores()) },
        { t: new QuoteTable({ title: "Commodities", count: 7, valueLabel: "VALOR (USD)", onSelect: _select }),
          fill: (tb) => _fillCommodities(tb) },
        { t: new QuoteTable({ title: "Monedas", count: 9, valueLabel: "VALOR", onSelect: _select }),
          fill: (tb) => _fillMonedas(tb) },
        { t: new QuoteTable({ title: "Tasas de Interés", count: 10, valueLabel: "TASA %", onSelect: _select }),
          fill: (tb) => tb.setRows(MarketMock.tasas()) },
    ];

    defs.forEach(({ t, fill }) => {
        t.mount(container);
        _tables.push(t);
        fill(t);
    });

    // Hero inicial: IPSA
    _setInstrument({
        label: "IPSA",
        tag: "MERCADO LOCAL · CHILE",
        series: MarketMock.seriesFor("IPSA", 10828, FULL_LEN),
        decimals: 2,
    });

    return [];
};

BCCh.renderMarketsOnline = renderMarketsOnline;

}());
