/* ─────────────────────────────────────────────────────────────────────────
   APP — Bootstrap: KPI, sidebar nav (accordion), router, modal, status.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

(function () {
    const { $, h } = BCCh;

    // ── Reloj ─────────────────────────────────────────────────────────
    const clockEl = $("#clock");
    if (clockEl) {
        const tick = () => { clockEl.textContent = BCCh.FMT.clock(); };
        tick();
        setInterval(tick, 1000);
    }

    // ── Sidebar nav (accordion por grupo) ─────────────────────────────
    const navEl = $("#sidebar-nav");

    // Estado abierto/cerrado por grupo persistido en localStorage
    const groupState = {};
    try {
        const saved = JSON.parse(localStorage.getItem("bcch_nav_groups") || "{}");
        Object.assign(groupState, saved);
    } catch (_) {}

    const isOpen = (groupId) => groupState[groupId] !== false; // default: abierto

    const persistGroupState = () => {
        localStorage.setItem("bcch_nav_groups", JSON.stringify(groupState));
    };

    const buildNav = () => {
        navEl.innerHTML = "";
        BCCh.SECTION_GROUPS.forEach((group) => {
            const open = isOpen(group.id);

            const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            chevron.setAttribute("class", "nav-chevron");
            chevron.setAttribute("viewBox", "0 0 24 24");
            chevron.setAttribute("fill", "none");
            chevron.setAttribute("stroke", "currentColor");
            chevron.setAttribute("stroke-width", "2.5");
            chevron.innerHTML = `<polyline points="6 9 12 15 18 9"/>`;

            const header = h("div", { "class": "nav-group-header" }, [
                h("span", {}, group.label),
                chevron,
            ]);

            const items = h("div", { "class": "nav-group-items" });
            group.ids.forEach((id) => {
                const sec = BCCh.getSection(id);
                if (!sec) return;
                const code = id.startsWith("s") ? id.toUpperCase() : "•";
                items.appendChild(h("button", {
                    "class": "nav-item",
                    "data-section-id": id,
                    onClick: () => router.go(id),
                }, [
                    h("span", { "class": "nav-item__code mono" }, code),
                    h("span", { "class": "nav-item__label" }, sec.title),
                ]));
            });

            const groupEl = h("div", {
                "class": "nav-group",
                "data-group-id": group.id,
                "data-open": String(open),
            }, [header, items]);

            header.addEventListener("click", () => {
                const cur = groupEl.dataset.open === "true";
                groupEl.dataset.open = String(!cur);
                groupState[group.id] = !cur;
                persistGroupState();
            });

            navEl.appendChild(groupEl);
        });
    };

    const setActiveNav = (id) => {
        BCCh.$$(".nav-item", navEl).forEach((b) => {
            b.dataset.active = String(b.dataset.sectionId === id);
        });
        // Asegurar que el grupo que contiene el item activo esté abierto
        const activeGroup = BCCh.SECTION_GROUPS.find((g) => g.ids.includes(id));
        if (activeGroup) {
            const grpEl = navEl.querySelector(`[data-group-id="${activeGroup.id}"]`);
            if (grpEl && grpEl.dataset.open === "false") {
                grpEl.dataset.open = "true";
                groupState[activeGroup.id] = true;
                persistGroupState();
            }
        }
    };

    // ── Sidebar search (filtra items por nombre) ───────────────────────
    const searchInput = $("#nav-search");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            const q = e.target.value.trim().toLowerCase();
            BCCh.$$(".nav-item", navEl).forEach((it) => {
                const label = it.querySelector(".nav-item__label").textContent.toLowerCase();
                const code  = it.querySelector(".nav-item__code").textContent.toLowerCase();
                const match = !q || label.includes(q) || code.includes(q);
                it.style.display = match ? "" : "none";
            });
        });
    }

    // ── Section render ────────────────────────────────────────────────
    const titleEl    = $("#section-title");
    const subtitleEl = $("#section-subtitle");
    const eyebrowEl  = $("#section-eyebrow");
    const gridEl     = $("#chart-grid");

    let currentCards = [];

    const loadSection = async (id) => {
        const sec = BCCh.getSection(id);
        if (!sec) {
            eyebrowEl.textContent  = "ERROR";
            titleEl.textContent    = "Sección no encontrada";
            subtitleEl.textContent = id;
            gridEl.innerHTML = "";
            return;
        }
        currentCards.forEach((c) => c.destroy());
        currentCards = [];

        eyebrowEl.textContent  = (sec.group || "").toUpperCase() + (sec.id !== "overview" ? ` · ${sec.id.toUpperCase()}` : "");
        titleEl.textContent    = sec.title;
        subtitleEl.textContent = sec.subtitle || "";

        setActiveNav(id);
        currentCards = await BCCh.renderSection(sec, gridEl);
    };

    // ── Router ────────────────────────────────────────────────────────
    const router = new BCCh.Router({ onChange: loadSection });

    // ── API status indicator ─────────────────────────────────────────
    const updateAPIStatus = async () => {
        const pill  = $("#api-status");
        if (!pill) return;
        const label = pill.querySelector(".status-pill-label");
        try {
            const r = await BCCh.API.readyz();
            if (r.status === "ok") {
                pill.className = "status-pill status-pill--ok";
                label.textContent = "API LISTA";
            } else if (r.status === "loading") {
                pill.className = "status-pill status-pill--load";
                label.textContent = "LLM CARGANDO";
            } else {
                pill.className = "status-pill status-pill--err";
                label.textContent = "DEGRADADO";
            }
        } catch (e) {
            pill.className = "status-pill status-pill--err";
            label.textContent = "SIN CONEXIÓN";
        }
    };

    // ── Bootstrap ─────────────────────────────────────────────────────
    try { BCCh.renderKPIStrip(); } catch (e) { console.error("[boot] KPIStrip:", e); }
    try { buildNav(); } catch (e) { console.error("[boot] buildNav:", e); }
    try { BCCh.ChartModal.init(); } catch (e) { console.error("[boot] ChartModal:", e); }
    try { BCCh.chatPanel = new BCCh.ChatPanel(); } catch (e) { console.error("[boot] ChatPanel:", e); }
    updateAPIStatus();
    setInterval(updateAPIStatus, 30000);
    router.start("overview");
})();
