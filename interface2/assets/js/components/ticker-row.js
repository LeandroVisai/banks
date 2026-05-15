/* ─────────────────────────────────────────────────────────────────────────
   TICKER ROW — fila densa: ticker · last · chg · chg% · sparkline · range
   También expone RangeBar para el bullet 52w-low / cur / 52w-high.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {

const { h } = BCCh;
const { formatNum, formatSigned, formatPct, formatBp } = BCCh.MKT;

// ── Range bar (bullet) ──────────────────────────────────────────────────

const rangeBarHTML = ({ lo, hi, cur }) => {
    if (lo === null || hi === null || cur === null || hi === lo) {
        return '<span class="range-bar range-bar--empty"></span>';
    }
    const pct = Math.max(0, Math.min(100, ((cur - lo) / (hi - lo)) * 100));
    return `<span class="range-bar" title="${formatNum(lo,2)} – ${formatNum(hi,2)}">
        <span class="range-bar__cur" style="left:${pct.toFixed(1)}%"></span>
    </span>`;
};

// ── Ticker row ──────────────────────────────────────────────────────────

const renderTickerRow = ({
    ticker, last, chg, pct, sparkline, range52w,
    status = "flat",
    chgFormat = "num",  // "num" | "bp"
    decimals = 2,
}) => {
    const chgStr = chgFormat === "bp" ? formatBp(chg) : formatSigned(chg, decimals);
    const pctStr = pct === null || pct === undefined ? "" : formatPct(pct, 2);

    const sparkEl = h("span", { "class": "ticker-row__spark sparkline-mini" });
    const rangeWrap = document.createElement("span");
    rangeWrap.className = "ticker-row__range";
    rangeWrap.innerHTML = range52w ? rangeBarHTML(range52w) : "";

    const row = h("div", { "class": `ticker-row tick-${status}` }, [
        h("span", { "class": "ticker-row__sym" }, ticker),
        h("span", { "class": "ticker-row__last mono" }, formatNum(last, decimals)),
        h("span", { "class": "ticker-row__chg mono"  }, chgStr),
        h("span", { "class": "ticker-row__pct mono"  }, pctStr),
        sparkEl,
        rangeWrap,
    ]);

    // SparklineMini se monta tras attach al DOM (necesita layout para medir)
    row._mountSparkline = () => {
        if (sparkline && sparkline.length > 1) {
            const sign = (last !== null && sparkline.length >= 2)
                ? Math.sign(last - sparkline[0])
                : 0;
            BCCh.SparklineMini.mount(sparkEl, { series: sparkline, deltaSign: sign });
        }
    };
    row._destroySparkline = () => BCCh.SparklineMini.destroy(sparkEl);

    return row;
};

BCCh.TickerRow = { render: renderTickerRow, rangeBarHTML };

}());
