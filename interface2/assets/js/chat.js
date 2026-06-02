/* ─────────────────────────────────────────────────────────────────────────
   CHAT — agente IA. Dos vistas comparten ChatController:
     • ChatPanel        → panel lateral colapsable (HTML estático en index.html)
     • mountInlineChat  → vista full-width embebida en sección "Agente GOEM"
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

const THINKING_KEY = "bcch_thinking_mode";   // localStorage
const THINKING_MODES = ["off", "adaptive", "on"];           // valores válidos
const THINKING_MENU_ORDER = ["on", "adaptive", "off"];      // orden visual: profundo→rápido
const THINKING_LABELS = {
    off: "Rápido",
    adaptive: "Análisis",
    on: "Profundo",
};
const THINKING_DESCS = {
    off: "Respuestas veloces, sin razonamiento",
    adaptive: "Razona en los datos · recomendado",
    on: "Razonamiento en todo, máxima calidad",
};

class ChatController {
    constructor({ messages, form, input, sendBtn, suggestions }) {
        this.messages = messages;
        this.form = form;
        this.input = input;
        this.send = sendBtn;
        this.suggestionsEl = suggestions;

        this.history = [];
        // off | adaptive (default) | on. Persistido entre sesiones y compartido
        // por ambas vistas del chat (panel lateral + inline) vía localStorage.
        const saved = localStorage.getItem(THINKING_KEY);
        this.thinkingMode = THINKING_MODES.includes(saved) ? saved : "adaptive";

        this._bindForm();
        this._mountModeToggle();
        this._renderSuggestions();
    }

    _mountModeToggle() {
        if (!this.form || !this.send) return;
        // Dropdown (estilo selector de modelos): trigger con el modo actual +
        // menú con título/descripción/check por modo. Descubrible y accesible.
        this.modeWrap = h("div", { "class": "chat-mode" });

        this.modeTrigger = h("button", {
            type: "button",
            "class": "chat-mode__trigger",
            "aria-haspopup": "true",
            "aria-expanded": "false",
            "aria-label": "Modo de razonamiento del agente",
            onClick: (e) => { e.stopPropagation(); this._toggleMenu(); },
        }, [
            h("span", { "class": "chat-mode__trigger-label" }, ""),
            h("span", { "class": "chat-mode__chevron", "aria-hidden": "true" }, "▾"),
        ]);
        this.modeTriggerLabel = this.modeTrigger.querySelector(".chat-mode__trigger-label");

        this.modeMenu = h("div", { "class": "chat-mode__menu", "role": "menu", hidden: true });
        this.modeItems = THINKING_MENU_ORDER.map((mode) => {
            const item = h("button", {
                type: "button",
                "class": "chat-mode__item",
                "data-mode": mode,
                "role": "menuitemradio",
                onClick: () => this._setMode(mode),
            }, [
                h("span", { "class": "chat-mode__item-main" }, [
                    h("span", { "class": "chat-mode__item-title" }, THINKING_LABELS[mode]),
                    h("span", { "class": "chat-mode__item-desc" }, THINKING_DESCS[mode]),
                ]),
                h("span", { "class": "chat-mode__check", "aria-hidden": "true" }, "✓"),
            ]);
            return item;
        });
        this.modeMenu.append(...this.modeItems);

        // Cerrar con Escape o click fuera.
        this.modeMenu.addEventListener("keydown", (e) => {
            if (e.key === "Escape") { this._closeMenu(); this.modeTrigger.focus(); }
        });
        if (!ChatController._outsideBound) {
            document.addEventListener("click", () => {
                document.querySelectorAll(".chat-mode.is-open").forEach((el) => {
                    el.classList.remove("is-open");
                    el.querySelector(".chat-mode__trigger")?.setAttribute("aria-expanded", "false");
                    el.querySelector(".chat-mode__menu")?.setAttribute("hidden", "");
                });
            });
            ChatController._outsideBound = true;
        }

        this.modeWrap.append(this.modeTrigger, this.modeMenu);
        // A la izquierda del botón Enviar, en la misma fila que el input.
        this.form.insertBefore(this.modeWrap, this.send);
        this._applyModeUI();
    }

    _toggleMenu() {
        this.modeWrap.classList.contains("is-open") ? this._closeMenu() : this._openMenu();
    }
    _openMenu() {
        this.modeWrap.classList.add("is-open");
        this.modeTrigger.setAttribute("aria-expanded", "true");
        this.modeMenu.hidden = false;
    }
    _closeMenu() {
        this.modeWrap.classList.remove("is-open");
        this.modeTrigger.setAttribute("aria-expanded", "false");
        this.modeMenu.hidden = true;
    }

    _setMode(mode) {
        this.thinkingMode = mode;
        localStorage.setItem(THINKING_KEY, mode);
        this._applyModeUI();
        this._closeMenu();
        this.modeTrigger.focus();
    }

    _applyModeUI() {
        if (this.modeTriggerLabel) this.modeTriggerLabel.textContent = THINKING_LABELS[this.thinkingMode];
        if (this.modeTrigger) this.modeTrigger.dataset.mode = this.thinkingMode;
        (this.modeItems || []).forEach((item) => {
            const active = item.dataset.mode === this.thinkingMode;
            item.classList.toggle("is-active", active);
            item.setAttribute("aria-checked", String(active));
        });
    }

    _bindForm() {
        this.form.addEventListener("submit", (e) => {
            e.preventDefault();
            const msg = this.input.value.trim();
            if (msg) this.ask(msg);
        });
    }

    _renderSuggestions() {
        if (!this.suggestionsEl) return;
        this.suggestionsEl.innerHTML = "";
        SUGGESTIONS.forEach((q) => {
            this.suggestionsEl.appendChild(h("button", {
                "class": "chat-suggestion",
                onClick: () => this.ask(q),
            }, q));
        });
    }

    async ask(message) {
        const empty = this.messages.querySelector(".chat-empty");
        if (empty) empty.remove();

        this._appendUserMsg(message);
        this.input.value = "";
        if (this.send) this.send.disabled = true;
        if (this.modeTrigger) this.modeTrigger.disabled = true;

        const loadingEl = this._appendLoading();

        try {
            const t0 = performance.now();
            const data = await BCCh.API.chat(message, this.history.slice(-8), this.thinkingMode);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

            this.history.push({ role: "user", content: message });
            this.history.push({ role: "assistant", content: data.response });

            loadingEl.remove();
            this._appendAssistantMsg(data, elapsed);
        } catch (e) {
            loadingEl.remove();
            this._appendError(e);
        } finally {
            if (this.send) this.send.disabled = false;
            if (this.modeTrigger) this.modeTrigger.disabled = false;
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

        let route = "rag";
        if (data.tool_trace?.some((t) => t.tool === "execute_query")) route = "sql";

        const meta = h("div", { "class": "chat-msg__meta" }, [
            h("span", { "class": `route-pill ${ROUTE_CLASSES[route] || ""}` }, route.toUpperCase()),
            h("span", { "class": "chat-msg__meta-time mono" }, `${elapsed}s · ${data.iterations || 0} iter`),
        ]);
        // Botón para leer la respuesta en voz alta (TTS del navegador, offline).
        if (window.speechSynthesis) {
            const ttsBtn = h("button", {
                type: "button",
                "class": "chat-tts",
                title: "Leer en voz alta",
                "aria-label": "Leer la respuesta en voz alta",
            }, "🔊");
            ttsBtn.addEventListener("click", () => this._speak(response, ttsBtn));
            meta.appendChild(ttsBtn);
        }

        const contentDiv = h("div", { "class": "chat-msg__content" });
        contentDiv.innerHTML = this._markdown(response);

        const msg = h("div", { "class": "chat-msg chat-msg--assistant" }, [
            h("div", { "class": "chat-msg__role" }, "AGENTE"),
            contentDiv,
            meta,
        ]);

        // Gráficos de las series temporales que el agente analizó (antes de
        // las fuentes, para que el dato cuantitativo sea lo primero que se ve).
        this._renderSeriesCharts(seriesUsed, msg);

        const sources = this._buildSources(citedRefs, chunksSeen, seriesUsed);
        if (sources) msg.appendChild(sources);

        this.messages.appendChild(msg);
        this._scrollToBottom();
    }

    _renderSeriesCharts(seriesUsed, msgEl) {
        // Solo series temporales reales (≥2 puntos). ApexCharts y baseChartConfig
        // deben estar cargados; si no, se omite en silencio.
        if (!window.ApexCharts || !BCCh.baseChartConfig) return;
        const plottable = (seriesUsed || []).filter((s) => (s.points || []).length >= 2);
        if (!plottable.length) return;

        const MAX_CHARTS = 3;   // no saturar la respuesta
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        plottable.slice(0, MAX_CHARTS).forEach((s, idx) => {
            const head = h("div", { "class": "chat-chart__head" }, [
                h("span", { "class": "chat-chart__title" }, s.series_name || s.series_id),
                h("span", { "class": "chat-chart__unit mono" }, s.unit || ""),
            ]);
            const body = h("div", { "class": "chat-chart__body" });
            const card = h("div", { "class": "chat-chart" }, [head, body]);
            msgEl.appendChild(card);

            const data = s.points
                .map((p) => [new Date(p[0]).getTime(), Number(p[1])])
                .filter((p) => !Number.isNaN(p[0]) && !Number.isNaN(p[1]));
            if (data.length < 2) { card.remove(); return; }

            // El primero más alto; los apilados, compactos (no enterrar las citas).
            const height = idx === 0 ? 160 : 132;
            const cfg = BCCh.baseChartConfig("area", { compact: true, height });
            cfg.series = [{ name: s.series_name || s.series_id, data }];
            cfg.fill = {
                type: "gradient",
                gradient: { shadeIntensity: 0.2, opacityFrom: 0.25, opacityTo: 0.02, stops: [0, 100] },
            };
            // El reveal lo hace el CSS (.chat-chart); Apex no anima (evita doble
            // animación) y respeta prefers-reduced-motion.
            cfg.chart.animations = { enabled: false };
            void reduce;
            try {
                const apex = new window.ApexCharts(body, cfg);
                apex.render();
            } catch (e) {
                card.remove();   // si ApexCharts falla, la respuesta de texto queda intacta
            }
        });
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

    _speak(text, btn) {
        const synth = window.speechSynthesis;
        if (!synth) return;
        // Toggle: si este botón ya está leyendo, detener.
        if (this._ttsBtn === btn) { synth.cancel(); return; }
        synth.cancel();   // corta cualquier lectura previa
        // Texto plano: sin markdown ni marcadores de cita [N].
        const plain = String(text)
            .replace(/\*\*(.+?)\*\*/g, "$1")
            .replace(/`(.+?)`/g, "$1")
            .replace(/\[\d+(?:\s*,\s*\d+)*\]/g, "")
            .replace(/[#*_>]/g, "")
            .trim();
        if (!plain) return;
        const u = new SpeechSynthesisUtterance(plain);
        u.lang = "es-CL";
        const voice = synth.getVoices().find((v) => /^es\b|^es[-_]/i.test(v.lang));
        if (voice) u.voice = voice;
        const reset = () => { btn.classList.remove("is-speaking"); this._ttsBtn = null; };
        u.onend = reset;
        u.onerror = reset;
        btn.classList.add("is-speaking");
        this._ttsBtn = btn;
        synth.speak(u);
    }

    _markdown(text) {
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

class ChatPanel {
    constructor() {
        this.panel    = $("#chat-panel");
        this.toggle   = $("#chat-toggle");
        this.close    = $("#chat-close");
        this.open     = localStorage.getItem("bcch_chat_open") === "true";

        const form = $("#chat-form");
        this.controller = new ChatController({
            messages:    $("#chat-messages"),
            form,
            input:       $("#chat-input"),
            sendBtn:     form.querySelector(".chat-input-send"),
            suggestions: $("#chat-suggestions"),
        });

        this.toggle.addEventListener("click", () => this.setOpen(!this.open));
        this.close.addEventListener("click",  () => this.setOpen(false));
        this._applyOpenState();
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
        if (value) setTimeout(() => this.controller.input.focus(), 250);
    }
}

BCCh.mountInlineChat = (container) => {
    container.innerHTML = "";

    const suggestions = h("div", { "class": "chat-suggestions" });
    const empty = h("div", { "class": "chat-empty" }, [
        h("div", { "class": "chat-empty-icon" }, "◈"),
        h("div", { "class": "chat-empty-label" }, "Consultas sugeridas"),
        suggestions,
    ]);
    const messages = h("div", { "class": "chat-inline__messages" }, [empty]);

    const input = h("input", {
        type: "text",
        "class": "chat-input-field",
        placeholder: "Pregunta al agente…",
        "aria-label": "Pregunta",
        autocomplete: "off",
    });
    const sendBtn = h("button", {
        type: "submit",
        "class": "chat-input-send",
        "aria-label": "Enviar",
    }, "Enviar");
    const form = h("form", { "class": "chat-input chat-inline__form", autocomplete: "off" }, [
        input, sendBtn,
    ]);

    const wrapper = h("div", { "class": "chat-inline" }, [messages, form]);
    container.appendChild(wrapper);

    return new ChatController({ messages, form, input, sendBtn, suggestions });
};

BCCh.ChatPanel = ChatPanel;
BCCh.ChatController = ChatController;
}());
