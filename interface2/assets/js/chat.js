/* ─────────────────────────────────────────────────────────────────────────
   CHAT PANEL — agente IA lateral colapsable. Conecta a /v1/chat.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {
const { $, h } = BCCh;

const SUGGESTIONS = [
    "¿Qué decidió el Consejo del BCCh en su última reunión?",
    "¿Cuáles son las proyecciones de inflación del BCCh?",
    "¿Qué riesgos externos menciona el Consejo en las minutas?",
    "¿Cuánto vale el USD/CLP actualmente?",
    "Dame el precio histórico del cobre",
];

const DOC_CLASSES = {
    MINUTAS: "minutas",
    COMUNICADO: "comunicado",
    IPOM: "ipom",
    IEF: "ief",
    REPORTE_RESEARCH: "research",
    MONITOR_PM: "monitor_pm",
};

const ROUTE_CLASSES = {
    rag: "route-pill--rag",
    sql: "route-pill--sql",
    visual: "route-pill--visual",
};

class ChatPanel {
    constructor() {
        this.panel    = $("#chat-panel");
        this.toggle   = $("#chat-toggle");
        this.close    = $("#chat-close");
        this.messages = $("#chat-messages");
        this.form     = $("#chat-form");
        this.input    = $("#chat-input");
        this.send     = this.form.querySelector(".chat-input-send");
        this.suggestionsEl = $("#chat-suggestions");

        this.history = [];   // [{role, content}]
        this.open    = localStorage.getItem("bcch_chat_open") === "true";

        this._bindEvents();
        this._renderSuggestions();
        this._applyOpenState();
    }

    _bindEvents() {
        this.toggle.addEventListener("click", () => this.setOpen(!this.open));
        this.close.addEventListener("click",  () => this.setOpen(false));
        this.form.addEventListener("submit", (e) => {
            e.preventDefault();
            const msg = this.input.value.trim();
            if (msg) this.ask(msg);
        });
    }

    _applyOpenState() {
        this.panel.dataset.open = String(this.open);
        document.body.dataset.chatOpen = String(this.open);
        this.toggle.dataset.active = String(this.open);
        localStorage.setItem("bcch_chat_open", String(this.open));
    }

    setOpen(value) {
        this.open = value;
        this._applyOpenState();
        if (value) setTimeout(() => this.input.focus(), 250);
    }

    _renderSuggestions() {
        this.suggestionsEl.innerHTML = "";
        SUGGESTIONS.forEach((q) => {
            this.suggestionsEl.appendChild(h("button", {
                "class": "chat-suggestion",
                onClick: () => this.ask(q),
            }, q));
        });
    }

    async ask(message) {
        // Limpia empty state
        const empty = this.messages.querySelector(".chat-empty");
        if (empty) empty.remove();

        // Renderiza mensaje user
        this._appendUserMsg(message);
        this.input.value = "";
        this.send.disabled = true;

        // Loading multi-step
        const loadingEl = this._appendLoading();

        try {
            const t0 = performance.now();
            const data = await BCCh.API.chat(message, this.history.slice(-8));
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

            this.history.push({ role: "user", content: message });
            this.history.push({ role: "assistant", content: data.response });

            loadingEl.remove();
            this._appendAssistantMsg(data, elapsed);

        } catch (e) {
            loadingEl.remove();
            this._appendError(e);
        } finally {
            this.send.disabled = false;
            this._scrollToBottom();
        }
    }

    _appendUserMsg(content) {
        const msg = h("div", { "class": "chat-msg chat-msg--user" }, [
            h("div", { "class": "chat-msg__role" }, "TÚ"),
            h("div", { "class": "chat-msg__content" }, [
                h("p", {}, content),
            ]),
        ]);
        this.messages.appendChild(msg);
        this._scrollToBottom();
    }

    _appendAssistantMsg(data, elapsed) {
        const response   = data.response || "Sin respuesta.";
        const citedRefs  = data.cited_refs || [];
        const chunksSeen = data.chunks_seen || [];
        const seriesUsed = data.series_used || [];

        // Determinar route a partir de tool_trace (si execute_query → sql, search → rag)
        let route = "rag";
        if (data.tool_trace?.some((t) => t.tool === "execute_query")) route = "sql";

        const meta = h("div", { "class": "chat-msg__meta" }, [
            h("span", { "class": `route-pill ${ROUTE_CLASSES[route] || ""}` }, route.toUpperCase()),
            h("span", { "class": "chat-msg__meta-time mono" }, `${elapsed}s · ${data.iterations || 0} iter`),
        ]);

        const contentDiv = h("div", { "class": "chat-msg__content" });
        contentDiv.innerHTML = this._markdown(response);

        const msg = h("div", { "class": "chat-msg chat-msg--assistant" }, [
            h("div", { "class": "chat-msg__role" }, "AGENTE"),
            contentDiv,
            meta,
        ]);

        // Sources
        const sources = this._buildSources(citedRefs, chunksSeen, seriesUsed);
        if (sources) msg.appendChild(sources);

        this.messages.appendChild(msg);
        this._scrollToBottom();
    }

    _buildSources(citedRefs, chunksSeen, seriesUsed) {
        if (!citedRefs.length && !seriesUsed.length) return null;

        const items = [];

        citedRefs.forEach((refNum) => {
            const chunk = chunksSeen.find((c) => c.ref === refNum);
            if (!chunk) return;
            const cls = DOC_CLASSES[chunk.doc_type] || "";
            items.push(h("div", { "class": `src-card src-card--${cls}` }, [
                h("div", { "class": "src-card__title" },
                    `[${refNum}] ${chunk.doc_type || "—"}${chunk.page_start ? " · pág. " + chunk.page_start : ""}`),
                h("div", { "class": "src-card__meta" },
                    `${chunk.filename || ""} · ${chunk.section || ""}`),
                h("div", { "class": "src-card__score mono" },
                    `importancia ${(chunk.importance || 0).toFixed(2)}`),
            ]));
        });

        seriesUsed.forEach((s) => {
            items.push(h("div", { "class": "src-card src-card--monitor_pm" }, [
                h("div", { "class": "src-card__title" }, `${s.series_name}`),
                h("div", { "class": "src-card__meta" },
                    `${s.unit} · ${s.frequency} · ${s.n_observations} obs`),
                h("div", { "class": "src-card__score mono" },
                    `${s.first_date} → ${s.last_date}`),
            ]));
        });

        if (!items.length) return null;

        return h("div", { "class": "src-list" }, [
            h("div", { "class": "src-list__label" }, `${items.length} FUENTE(S)`),
            ...items,
        ]);
    }

    _appendLoading() {
        const steps = [
            { id: "analyze", label: "Analizando consulta…" },
            { id: "search",  label: "Buscando en corpus / catálogo SQL…" },
            { id: "generate",label: "Generando respuesta…" },
        ];
        const stepEls = steps.map((s, i) => h("div", {
            "class": "chat-loading__step",
            "data-step": s.id,
            "data-active": i === 0 ? "true" : "false",
        }, [
            h("span", { "class": "chat-loading__bullet" }, "›"),
            h("span", {}, s.label),
        ]));

        const el = h("div", { "class": "chat-loading" }, stepEls);
        this.messages.appendChild(el);
        this._scrollToBottom();

        // Avance automático cada 1.5s para feedback (no real, sólo UX)
        let idx = 0;
        const interval = setInterval(() => {
            const s = stepEls[idx];
            if (!s) return;
            s.dataset.active = "false";
            s.dataset.done = "true";
            idx += 1;
            if (idx < stepEls.length) {
                stepEls[idx].dataset.active = "true";
            } else {
                clearInterval(interval);
            }
        }, 1500);

        el._interval = interval;
        const origRemove = el.remove.bind(el);
        el.remove = () => {
            clearInterval(interval);
            origRemove();
        };
        return el;
    }

    _appendError(err) {
        let label = err.message || "Error desconocido";
        if (err.status === 401) label = "API Key incorrecta — configura en localStorage.bcch_api_key";
        if (err.status === 429) label = "Rate limit alcanzado. Espera unos segundos.";
        if (err.status === 0)   label = "Sin conexión con la API";

        const msg = h("div", { "class": "chat-msg chat-msg--assistant" }, [
            h("div", { "class": "chat-msg__role" }, "ERROR"),
            h("div", { "class": "chat-msg__content" }, [
                h("p", {}, label),
            ]),
        ]);
        this.messages.appendChild(msg);
    }

    _scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }

    _markdown(text) {
        // Markdown mínimo: párrafos, **bold**, `code`, [n] refs
        if (!text) return "";
        const escape = (s) => s
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        return escape(text)
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/`(.+?)`/g, "<code>$1</code>")
            .replace(/\[(\d+)\]/g, '<sup style="color:var(--c-gold)">[$1]</sup>')
            .split(/\n\n+/)
            .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
            .join("");
    }
}

BCCh.ChatPanel = ChatPanel;
}());
