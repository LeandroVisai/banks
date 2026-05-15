/* ─────────────────────────────────────────────────────────────────────────
   QUOTE TABLE — tabla densa estilo LarrainVial / BTG Mercados en Línea.
   3 columnas: nombre · valor · var%. Filas clickeables (onSelect) para
   alimentar el hero chart. Soporta badge y filas dim.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const { h } = BCCh;
const { formatNum, formatPct } = BCCh.MKT;

const _status = (pct) => {
    if (pct === null || pct === undefined || !Number.isFinite(pct)) return "flat";
    if (pct > 0) return "up";
    if (pct < 0) return "down";
    return "flat";
};

// Valores grandes (índices, UF, acciones caras) sin decimales para no
// desbordar la columna; decimales finos (monedas, cobre) se respetan.
const _autoDecimals = (value, declared) => {
    if (typeof value !== "number") return declared;
    if (Math.abs(value) >= 10000) return 0;
    return declared;
};

class QuoteTable {
    constructor({ title, count = null, valueLabel = "VALOR", pctLabel = "VAR.%",
                  scroll = false, span = 1, onSelect = null } = {}) {
        this.title = title;
        this.count = count;
        this.valueLabel = valueLabel;
        this.pctLabel = pctLabel;
        this.scroll = scroll;
        this.span = span;
        this.onSelect = onSelect;
        this.el = null;
        this.bodyEl = null;
        this._rowEls = [];
    }

    _buildShell() {
        const header = h("div", { "class": "quote-table__header" }, [
            h("span", { "class": "quote-table__title" }, this.title),
            this.count !== null
                ? h("span", { "class": "quote-table__count" }, String(this.count))
                : h("span", { "class": "quote-table__count" }),
        ]);
        const cols = h("div", { "class": "quote-table__cols" }, [
            h("span", { "class": "qt-col-name" }, "NOMBRE"),
            h("span", { "class": "qt-col-val"  }, this.valueLabel),
            h("span", { "class": "qt-col-pct"  }, this.pctLabel),
        ]);
        const body = h("div", {
            "class": "quote-table__body" + (this.scroll ? " quote-table__body--scroll" : ""),
        });
        this.bodyEl = body;

        const cls = "quote-table" + (this.span > 1 ? " quote-table--span" + this.span : "");
        this.el = h("div", { "class": cls }, [header, cols, body]);
        return this.el;
    }

    mount(container) {
        if (!this.el) this._buildShell();
        container.appendChild(this.el);
        return this;
    }

    setBusy(isBusy) {
        if (this.el) this.el.dataset.busy = isBusy ? "true" : "false";
    }

    setError(msg) {
        if (!this.bodyEl) return;
        this.bodyEl.innerHTML = "";
        this._rowEls = [];
        this.bodyEl.appendChild(h("div", { "class": "quote-table__error" }, msg || "Error de carga"));
    }

    setRows(rows) {
        if (!this.bodyEl) return;
        this.bodyEl.innerHTML = "";
        this._rowEls = [];

        rows.forEach((r) => {
            const status = _status(r.pct);
            const dec = _autoDecimals(r.value, r.decimals !== undefined ? r.decimals : 2);
            const valueStr = typeof r.value === "string" ? r.value : formatNum(r.value, dec);
            const pctStr = (r.pct === null || r.pct === undefined || !Number.isFinite(r.pct))
                ? "—"
                : formatPct(r.pct, 2);

            const pctEl = h("span", {
                "class": r.badge ? `qt-badge qt-badge--${status}` : `quote-table__pct qt-${status}`,
            }, pctStr);

            const rowCls = "quote-table__row tick-" + status
                + (r.dim ? " quote-table__row--dim" : "")
                + (this.onSelect ? " quote-table__row--clickable" : "");

            const rowEl = h("div", { "class": rowCls }, [
                h("span", { "class": "quote-table__label" }, r.label),
                h("span", { "class": "quote-table__value mono" }, valueStr),
                pctEl,
            ]);

            if (this.onSelect) {
                rowEl.addEventListener("click", () => this.onSelect(r, this, rowEl));
            }

            this.bodyEl.appendChild(rowEl);
            this._rowEls.push(rowEl);
        });
    }

    clearActive() {
        this._rowEls.forEach((el) => el.classList.remove("quote-table__row--active"));
    }

    unmount() {
        if (this.el && this.el.parentNode) this.el.parentNode.removeChild(this.el);
    }
}

BCCh.QuoteTable = QuoteTable;

}());
