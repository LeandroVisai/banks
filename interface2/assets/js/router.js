/* ─────────────────────────────────────────────────────────────────────────
   ROUTER — hash-based (#/s1, #/s2, ...) — bidireccional con sidebar
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

class Router {
    constructor({ onChange }) {
        this.onChange = onChange;
        window.addEventListener("hashchange", () => this._fire());
    }

    start(defaultRoute = "overview") {
        if (!window.location.hash || window.location.hash === "#" || window.location.hash === "#/") {
            window.location.hash = `#/${defaultRoute}`;
        } else {
            this._fire();
        }
    }

    go(sectionId) {
        const target = `#/${sectionId}`;
        if (window.location.hash !== target) {
            window.location.hash = target;
        } else {
            this._fire();
        }
    }

    _current() {
        const m = window.location.hash.match(/^#\/(.+)$/);
        return m ? m[1] : null;
    }

    _fire() {
        const id = this._current();
        if (id && this.onChange) this.onChange(id);
    }
}

BCCh.Router = Router;
