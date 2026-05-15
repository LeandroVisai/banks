/* ─────────────────────────────────────────────────────────────────────────
   MARKET PANEL — contenedor densificable estilo Bloomberg.
   Header (título + nº + opciones) + cuerpo (rows o chart) + footer opcional.
   API: new MarketPanel({title, subtitle, count, span, variant}).mount(el)
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const { h } = BCCh;

class MarketPanel {
    constructor({ title, subtitle = "", count = null, span = 1, variant = "tickers", footer = "" } = {}) {
        this.title = title;
        this.subtitle = subtitle;
        this.count = count;
        this.span = span;
        this.variant = variant;
        this.footer = footer;
        this._rows = [];
        this._chartInstance = null;
        this.el = null;
        this.bodyEl = null;
        this.footerEl = null;
    }

    _buildShell() {
        const headerLeft = h("span", { "class": "market-panel__title" }, this.title);
        const headerMid = this.count !== null
            ? h("span", { "class": "market-panel__count" }, String(this.count))
            : h("span", { "class": "market-panel__count" });
        const headerRight = h("span", { "class": "market-panel__opts" }, "» Options");

        const header = h("div", { "class": "market-panel__header" }, [headerLeft, headerMid, headerRight]);
        const subtitle = this.subtitle
            ? h("div", { "class": "market-panel__subtitle" }, this.subtitle)
            : null;
        const body = h("div", { "class": "market-panel__body" });
        const footer = h("div", { "class": "market-panel__footer mono" }, this.footer || "");

        const cls = `market-panel market-panel--${this.variant}` + (this.span > 1 ? " market-panel--span" + this.span : "");
        const panel = h("div", { "class": cls }, subtitle ? [header, subtitle, body, footer] : [header, body, footer]);

        this.el = panel;
        this.bodyEl = body;
        this.footerEl = footer;
        return panel;
    }

    mount(container) {
        if (!this.el) this._buildShell();
        container.appendChild(this.el);
        return this;
    }

    setBusy(isBusy) {
        if (!this.el) return;
        this.el.dataset.busy = isBusy ? "true" : "false";
    }

    setError(msg) {
        if (!this.bodyEl) return;
        this._destroyAllSparklines();
        this.bodyEl.innerHTML = "";
        this.bodyEl.appendChild(h("div", { "class": "market-panel__error" }, [
            h("div", { "class": "market-panel__error-icon" }, "!"),
            h("div", { "class": "market-panel__error-msg" }, msg || "Error de carga"),
        ]));
    }

    setRows(rows, opts = {}) {
        if (!this.bodyEl) return;
        this._destroyAllSparklines();
        this.bodyEl.innerHTML = "";

        if (opts.colsHeader) {
            this.bodyEl.appendChild(this._buildColsHeader(opts.colsHeader));
        }

        const list = h("div", { "class": "market-panel__rows" });
        rows.forEach((rowProps) => {
            const node = BCCh.TickerRow.render(rowProps);
            list.appendChild(node);
        });
        this.bodyEl.appendChild(list);

        // Mount sparklines after attached (need layout)
        requestAnimationFrame(() => {
            list.querySelectorAll(".ticker-row").forEach((node, i) => {
                // Find corresponding row's mountSparkline (stored on the node by closure)
                if (node._mountSparkline) node._mountSparkline();
            });
        });

        this._rows = list;
    }

    setSubsections(sections) {
        // sections: [{label, rows}]
        if (!this.bodyEl) return;
        this._destroyAllSparklines();
        this.bodyEl.innerHTML = "";
        sections.forEach((s) => {
            this.bodyEl.appendChild(h("div", { "class": "market-panel__subhead" }, s.label));
            const list = h("div", { "class": "market-panel__rows" });
            s.rows.forEach((r) => list.appendChild(BCCh.TickerRow.render(r)));
            this.bodyEl.appendChild(list);
        });
        requestAnimationFrame(() => {
            this.bodyEl.querySelectorAll(".ticker-row").forEach((node) => {
                if (node._mountSparkline) node._mountSparkline();
            });
        });
    }

    setChart(apexOptions) {
        if (!this.bodyEl || !window.ApexCharts) return;
        this._destroyAllSparklines();
        this.bodyEl.innerHTML = "";
        const chartEl = h("div", { "class": "market-panel__chart" });
        this.bodyEl.appendChild(chartEl);
        if (this._chartInstance) {
            try { this._chartInstance.destroy(); } catch (_) {}
        }
        this._chartInstance = new ApexCharts(chartEl, apexOptions);
        this._chartInstance.render();
    }

    setKpiTiles(tiles) {
        // tiles: [{label, value, status?, sub?}]
        if (!this.bodyEl) return;
        this._destroyAllSparklines();
        this.bodyEl.innerHTML = "";
        const wrap = h("div", { "class": "market-panel__kpis" });
        tiles.forEach((t) => {
            const cls = "market-kpi" + (t.status ? " tick-" + t.status : "");
            wrap.appendChild(h("div", { "class": cls }, [
                h("div", { "class": "market-kpi__label" }, t.label),
                h("div", { "class": "market-kpi__value mono" }, t.value),
                h("div", { "class": "market-kpi__sub mono" }, t.sub || ""),
            ]));
        });
        this.bodyEl.appendChild(wrap);
    }

    setFooter(text) {
        if (this.footerEl) this.footerEl.textContent = text;
    }

    _buildColsHeader(cols) {
        // cols: ["TICKER", "LAST", "CHG", "%CHG", "30D", "52W"]
        return h("div", { "class": "market-panel__cols" }, cols.map((c) =>
            h("span", { "class": "market-panel__col" }, c)
        ));
    }

    _destroyAllSparklines() {
        if (!this._rows) return;
        this._rows.querySelectorAll && this._rows.querySelectorAll(".ticker-row").forEach((node) => {
            if (node._destroySparkline) node._destroySparkline();
        });
    }

    unmount() {
        this._destroyAllSparklines();
        if (this._chartInstance) {
            try { this._chartInstance.destroy(); } catch (_) {}
            this._chartInstance = null;
        }
        if (this.el && this.el.parentNode) this.el.parentNode.removeChild(this.el);
    }
}

BCCh.MarketPanel = MarketPanel;

}());
