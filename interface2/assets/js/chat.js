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

// Ventana deslizante de memoria: el modelo ve las últimas N consultas (pares
// pregunta+respuesta) INCLUYENDO la actual. Como la pregunta del turno se manda
// aparte, se envían los últimos (N-1)*2 mensajes previos → al llegar a la
// consulta N+1, la 1.ª se cae de la ventana. El hilo en pantalla conserva todo.
const CONTEXT_WINDOW_QUERIES = 5;
const CONTEXT_WINDOW_MESSAGES = (CONTEXT_WINDOW_QUERIES - 1) * 2;

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
        this.attachments = [];   // [{upload_id, name, kind}] del turno en curso
        // off | adaptive (default) | on. Persistido entre sesiones y compartido
        // por ambas vistas del chat (panel lateral + inline) vía localStorage.
        const saved = localStorage.getItem(THINKING_KEY);
        this.thinkingMode = THINKING_MODES.includes(saved) ? saved : "adaptive";

        this._bindForm();
        this._mountAttach();
        this._mountModeToggle();
        this._renderSuggestions();

        // Conversación persistida: el store es la fuente de verdad compartida
        // por la sección inline y el panel lateral. Hidrata la conversación
        // activa (re-renderiza sus turnos) al construirse.
        this.emptyEl = this.messages ? this.messages.querySelector(".chat-empty") : null;
        this.conversationId = BCCh.ChatStore.activeId();
        this._hydrate();
    }

    _mountAttach() {
        if (!this.form || !this.input) return;
        // Contenedor de chips (fila propia sobre el input).
        this.chipsEl = h("div", { "class": "chat-attachments" });
        this.form.insertBefore(this.chipsEl, this.form.firstChild);

        // Botón 📎 + input file oculto, a la izquierda del campo de texto.
        this.fileInput = h("input", {
            type: "file",
            "class": "chat-attach__input",
            accept: ".pdf,.txt,.md,.json,.csv,.xlsx,.xls",
            multiple: true,
        });
        this.fileInput.addEventListener("change", () => {
            this._handleFiles(this.fileInput.files);
            this.fileInput.value = "";   // permite re-subir el mismo archivo
        });
        this.attachBtn = h("button", {
            type: "button",
            "class": "chat-attach",
            title: "Adjuntar archivo (PDF, TXT, CSV, Excel)",
            "aria-label": "Adjuntar archivo",
            onClick: () => this.fileInput.click(),
        }, "+");
        this.form.insertBefore(this.attachBtn, this.input);
        this.form.appendChild(this.fileInput);   // oculto vía CSS
    }

    async _handleFiles(fileList) {
        for (const file of Array.from(fileList || [])) {
            if (this.attachments.length >= 5) break;   // tope del backend
            const chip = this._addChip(file.name, true);
            try {
                const b64 = await this._fileToBase64(file);
                const rec = await BCCh.API.upload(file.name, b64);
                this.attachments.push({ upload_id: rec.upload_id, name: rec.name, kind: rec.kind });
                chip.dataset.kind = rec.kind;
                chip.classList.remove("is-loading");
                chip._uploadId = rec.upload_id;
            } catch (e) {
                chip.classList.remove("is-loading");
                chip.classList.add("is-error");
                chip.title = e.message || "No se pudo subir";
            }
        }
    }

    _fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const r = new FileReader();
            r.onload = () => resolve(String(r.result).split(",")[1] || "");
            r.onerror = () => reject(new Error("No se pudo leer el archivo"));
            r.readAsDataURL(file);
        });
    }

    _addChip(name, loading) {
        const remove = h("button", { type: "button", "class": "chat-chip__x", "aria-label": "Quitar" }, "×");
        const chip = h("div", { "class": "chat-chip" + (loading ? " is-loading" : "") }, [
            h("span", { "class": "chat-chip__name" }, name),
            remove,
        ]);
        remove.addEventListener("click", () => {
            if (chip._uploadId) {
                this.attachments = this.attachments.filter((a) => a.upload_id !== chip._uploadId);
            }
            chip.remove();
        });
        this.chipsEl.appendChild(chip);
        return chip;
    }

    _clearAttachments() {
        this.attachments = [];
        if (this.chipsEl) this.chipsEl.innerHTML = "";
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
            // Permite enviar solo con adjuntos (mensaje por defecto).
            if (msg || this.attachments.length) this.ask(msg || "Analiza el archivo adjunto.");
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

    // ── Conversaciones persistidas (store compartido) ──────────────────────

    /** Restaura el estado vacío (icono + sugerencias) en el panel de mensajes. */
    _showEmpty() {
        this.messages.innerHTML = "";
        if (this.emptyEl) {
            this.messages.appendChild(this.emptyEl);
            this._renderSuggestions();
        }
    }

    /** Re-renderiza los turnos de la conversación activa y rehace `history`. */
    _hydrate() {
        if (!this.messages) return;
        const conv = BCCh.ChatStore.get(this.conversationId);
        const turns = (conv && conv.turns) || [];
        // `history` (solo role/content) alimenta la ventana de contexto.
        this.history = turns.map((t) => (
            t.role === "user"
                ? { role: "user", content: t.content }
                : { role: "assistant", content: (t.data && t.data.response) || "" }
        ));
        if (!turns.length) { this._showEmpty(); return; }
        this.messages.innerHTML = "";
        turns.forEach((t) => {
            if (t.role === "user") this._appendUserMsg(t.content, t.attachmentNames || []);
            else if (t.role === "assistant") this._appendAssistantMsg(t.data, t.elapsed);
        });
        this._scrollToBottom();
    }

    /** Inicia una conversación nueva y vacía (la anterior queda en el historial). */
    newConversation() {
        this.conversationId = BCCh.ChatStore.create().id;
        this.history = [];
        this._clearAttachments();
        this._showEmpty();
        if (this.input) this.input.focus();
    }

    /** Carga una conversación existente por id (cambia la activa y re-renderiza). */
    loadConversation(id) {
        BCCh.ChatStore.setActive(id);
        this.conversationId = id;
        this._clearAttachments();
        this._hydrate();
    }

    /** Re-sincroniza con la conversación activa del store (lo usa el panel al abrir). */
    reloadActive() {
        this.conversationId = BCCh.ChatStore.activeId();
        this._hydrate();
    }

    async ask(message) {
        const empty = this.messages.querySelector(".chat-empty");
        if (empty) empty.remove();

        // La conversación activa pudo cambiar en otra vista (panel/inline).
        if (!BCCh.ChatStore.get(this.conversationId)) {
            this.conversationId = BCCh.ChatStore.activeId();
        }

        // Captura y consume los adjuntos de este turno (efímero).
        const attachmentIds = this.attachments.map((a) => a.upload_id).filter(Boolean);
        const attachmentNames = this.attachments.map((a) => a.name);

        this._appendUserMsg(message, attachmentNames);
        BCCh.ChatStore.appendUserTurn(this.conversationId, message, attachmentNames);
        this.input.value = "";
        this._clearAttachments();
        if (this.send) this.send.disabled = true;
        if (this.modeTrigger) this.modeTrigger.disabled = true;
        if (this.attachBtn) this.attachBtn.disabled = true;

        const loadingEl = this._appendLoading();

        try {
            const t0 = performance.now();
            // Ventana deslizante: solo las últimas CONTEXT_WINDOW_QUERIES consultas
            // de contexto (la pregunta actual va aparte). El hilo visible es completo.
            const ctx = this.history.slice(-CONTEXT_WINDOW_MESSAGES);
            const data = await BCCh.API.chat(message, ctx, this.thinkingMode, attachmentIds);
            const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

            this.history.push({ role: "user", content: message });
            this.history.push({ role: "assistant", content: data.response });
            BCCh.ChatStore.appendAssistantTurn(this.conversationId, data, elapsed);

            loadingEl.remove();
            this._appendAssistantMsg(data, elapsed);
        } catch (e) {
            loadingEl.remove();
            this._appendError(e);
        } finally {
            if (this.send) this.send.disabled = false;
            if (this.modeTrigger) this.modeTrigger.disabled = false;
            if (this.attachBtn) this.attachBtn.disabled = false;
            this._scrollToBottom();
        }
    }

    _appendUserMsg(content, attachmentNames = []) {
        const children = [h("p", {}, content)];
        if (attachmentNames.length) {
            children.push(h("div", { "class": "chat-msg__files" },
                attachmentNames.map((n) => h("span", { "class": "chat-msg__file" }, `📎 ${n}`))));
        }
        const msg = h("div", { "class": "chat-msg chat-msg--user" }, [
            h("div", { "class": "chat-msg__role" }, "TÚ"),
            h("div", { "class": "chat-msg__content" }, children),
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

        // Gráficos extraídos de los PDFs adjuntos (modo análisis de documento).
        this._renderAttachmentVisuals(data.attachment_visuals || [], msg);

        const sources = this._buildSources(citedRefs, chunksSeen, seriesUsed);
        if (sources) msg.appendChild(sources);

        this.messages.appendChild(msg);
        this._scrollToBottom();
    }

    _renderSeriesCharts(seriesUsed, msgEl) {
        // Series con ≥2 puntos. ApexCharts y baseChartConfig deben estar
        // cargados; si no, se omite en silencio. El TIPO de gráfico lo decide el
        // backend (chart_type, derivado del chart_type canónico del dataset en
        // el parquet_catalog): "bar"/"grouped_bar"/"stacked_bar" para datos
        // categóricos o temporales cortos y "line"/"area" para series
        // temporales — así no todo sale como línea.
        if (!window.ApexCharts || !BCCh.baseChartConfig) return;
        const plottable = (seriesUsed || []).filter((s) => (s.points || []).length >= 2);
        if (!plottable.length) return;

        const MAX_CHARTS = 3;   // no saturar la respuesta
        plottable.slice(0, MAX_CHARTS).forEach((s, idx) => {
            const head = h("div", { "class": "chat-chart__head" }, [
                h("span", { "class": "chat-chart__title" }, s.series_name || s.series_id),
                h("span", { "class": "chat-chart__unit mono" }, s.unit || ""),
            ]);
            const body = h("div", { "class": "chat-chart__body" });
            const card = h("div", { "class": "chat-chart" }, [head, body]);
            msgEl.appendChild(card);

            // El primero más alto; los apilados, compactos (no enterrar las citas).
            const height = idx === 0 ? 160 : 132;
            const type = s.chart_type || "line";
            const isBar = type === "bar" || type === "grouped_bar" || type === "stacked_bar";

            let cfg;
            if (isBar) {
                // Eje categórico: x = etiqueta (no fecha), y = valor → barras.
                const categories = [];
                const values = [];
                (s.points || []).forEach((p) => {
                    const v = Number(p[1]);
                    if (Number.isNaN(v)) return;
                    categories.push(String(p[0]));
                    values.push(v);
                });
                if (!values.length) { card.remove(); return; }
                cfg = BCCh.baseChartConfig("bar", { compact: true, height });
                cfg.xaxis = {
                    ...(cfg.xaxis || {}),
                    type: "category",
                    categories,
                    labels: { style: { colors: "#7F8C8D", fontSize: "10px", fontFamily: "IBM Plex Mono" } },
                };
                cfg.plotOptions = { bar: { borderRadius: 2, columnWidth: "55%", distributed: values.length <= 12 } };
                cfg.stroke = { width: 0 };
                cfg.legend = { show: false };
                cfg.series = [{ name: s.series_name || s.series_id, data: values }];
            } else {
                // Serie temporal: x = timestamp, área (estética de una sola serie).
                const data = (s.points || [])
                    .map((p) => [new Date(p[0]).getTime(), Number(p[1])])
                    .filter((p) => !Number.isNaN(p[0]) && !Number.isNaN(p[1]));
                if (data.length < 2) { card.remove(); return; }
                cfg = BCCh.baseChartConfig("area", { compact: true, height });
                cfg.series = [{ name: s.series_name || s.series_id, data }];
                cfg.fill = {
                    type: "gradient",
                    gradient: { shadeIntensity: 0.2, opacityFrom: 0.25, opacityTo: 0.02, stops: [0, 100] },
                };
            }
            // El reveal lo hace el CSS (.chat-chart); Apex no anima (evita doble
            // animación) y respeta prefers-reduced-motion.
            cfg.chart.animations = { enabled: false };
            try {
                const apex = new window.ApexCharts(body, cfg);
                apex.render();
            } catch (e) {
                card.remove();   // si ApexCharts falla, la respuesta de texto queda intacta
            }
        });
    }

    _renderAttachmentVisuals(visuals, msgEl) {
        // Galería de gráficos/figuras extraídos del PDF adjunto (modo análisis
        // de documento). Las imágenes las sirve el backend vía image_url.
        if (!visuals || !visuals.length) return;
        const base = (BCCh.CONFIG && BCCh.CONFIG.API_BASE) || "";
        const cards = visuals.map((v) => {
            const img = h("img", {
                "class": "chat-doc-visual__img",
                src: `${base}${v.image_url}`,
                alt: v.caption || "Gráfico del documento",
                loading: "lazy",
            });
            const cap = h("div", { "class": "chat-doc-visual__cap" },
                [v.caption || "Figura", v.page ? `  ·  pág. ${v.page}` : ""].join(""));
            const fig = h("figure", { "class": "chat-doc-visual" }, [img, cap]);
            // Si la imagen no carga (PDF sin esa figura / TTL vencido), se oculta.
            img.addEventListener("error", () => fig.remove());
            return fig;
        });
        const wrap = h("div", { "class": "chat-doc-visuals" }, [
            h("div", { "class": "chat-doc-visuals__head" }, "Gráficos del documento"),
            h("div", { "class": "chat-doc-visuals__grid" }, cards),
        ]);
        msgEl.appendChild(wrap);
    }

    _buildSources(citedRefs, chunksSeen, seriesUsed) {
        if (!citedRefs.length && !seriesUsed.length) return null;

        const items = [];

        citedRefs.forEach((refNum) => {
            const chunk = chunksSeen.find((c) => c.ref === refNum);
            if (!chunk) return;
            const cls = DOC_CLASSES[chunk.doc_type] || "";
            const isNews = chunk.doc_type === "NOTICIA";

            let titleLine, metaLine, scoreLine;
            if (isNews) {
                // Noticias: titular + medio (institution) + fecha
                titleLine = h("div", { "class": "src-card__title" },
                    `[${refNum}] ${chunk.filename || "Noticia"}`);
                const medio = chunk.institution || "Prensa";
                const fecha = chunk.date ? chunk.date.slice(0, 10) : "";
                metaLine = h("div", { "class": "src-card__meta" },
                    [medio, fecha].filter(Boolean).join("  ·  "));
                scoreLine = h("div", { "class": "src-card__score mono" },
                    `importancia ${(chunk.importance || 0).toFixed(2)}`);
            } else {
                titleLine = h("div", { "class": "src-card__title" },
                    `[${refNum}] ${chunk.doc_type || "—"}${chunk.page_start ? " · pág. " + chunk.page_start : ""}`);
                metaLine = h("div", { "class": "src-card__meta" },
                    `${chunk.filename || ""} · ${chunk.section || ""}`);
                scoreLine = h("div", { "class": "src-card__score mono" },
                    `importancia ${(chunk.importance || 0).toFixed(2)}`);
            }

            items.push(h("div", { "class": `src-card src-card--${cls}` }, [
                titleLine, metaLine, scoreLine,
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

        // Botón "Nuevo chat" del panel (si existe en el header).
        this.newBtn = $("#chat-new");
        if (this.newBtn) {
            this.newBtn.addEventListener("click", () => this.controller.newConversation());
        }

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
        if (value) {
            // Re-sincroniza con la conversación activa (pudo cambiar en la vista
            // inline mientras el panel estaba cerrado).
            this.controller.reloadActive();
            setTimeout(() => this.controller.input.focus(), 250);
        }
    }
}

BCCh.mountInlineChat = (container) => {
    container.innerHTML = "";

    // ── Barra de historial (izquierda, estilo Claude) ──────────────────────
    const newBtn = h("button", { "class": "chat-rail__new", type: "button" }, "+ Nuevo chat");
    const listEl = h("div", { "class": "chat-rail__list" });
    const rail = h("aside", { "class": "chat-rail" }, [newBtn, listEl]);

    // ── Área de chat (derecha) ──────────────────────────────────────────────
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

    // Barra superior del chat: toggle de la barra de historial (izq) + nuevo chat (der).
    const railToggle = h("button", {
        "class": "chat-main__toggle", type: "button",
        title: "Mostrar/ocultar conversaciones", "aria-label": "Mostrar/ocultar conversaciones",
        html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="9" y1="4" x2="9" y2="20"/></svg>',
    });
    const newBtnTop = h("button", {
        "class": "chat-main__new", type: "button",
        title: "Nuevo chat", "aria-label": "Nuevo chat",
    }, "+");
    const bar = h("div", { "class": "chat-main__bar" }, [railToggle, newBtnTop]);
    const main = h("div", { "class": "chat-main" }, [bar, messages, form]);

    const wrapper = h("div", { "class": "chat-inline chat-inline--with-rail" }, [rail, main]);
    container.appendChild(wrapper);

    // Colapsar/expandir la barra de historial (estado recordado entre sesiones).
    const RAIL_KEY = "bcch_chat_rail_collapsed";
    let railCollapsed = localStorage.getItem(RAIL_KEY) === "true";
    const applyRail = () => {
        wrapper.classList.toggle("chat-inline--rail-collapsed", railCollapsed);
        railToggle.setAttribute("aria-expanded", String(!railCollapsed));
    };
    railToggle.addEventListener("click", () => {
        railCollapsed = !railCollapsed;
        localStorage.setItem(RAIL_KEY, String(railCollapsed));
        applyRail();
    });
    applyRail();

    const controller = new ChatController({ messages, form, input, sendBtn, suggestions });

    // Re-renderiza la lista de conversaciones desde el store.
    const renderRail = () => {
        const activeId = BCCh.ChatStore.peekActiveId();
        listEl.innerHTML = "";
        BCCh.ChatStore.list().forEach((c) => {
            const title = h("span", { "class": "chat-rail__item-title" }, c.title || "Nueva conversación");
            const renameBtn = h("button", {
                "class": "chat-rail__act", type: "button",
                title: "Renombrar", "aria-label": "Renombrar",
                onClick: (e) => {
                    e.stopPropagation();
                    const nv = window.prompt("Nuevo título de la conversación:", c.title || "");
                    if (nv != null && nv.trim()) BCCh.ChatStore.rename(c.id, nv);
                },
            }, "✎");
            const delBtn = h("button", {
                "class": "chat-rail__act chat-rail__act--del", type: "button",
                title: "Borrar", "aria-label": "Borrar",
                onClick: (e) => {
                    e.stopPropagation();
                    if (window.confirm("¿Borrar esta conversación?")) {
                        const wasActive = controller.conversationId === c.id;
                        BCCh.ChatStore.remove(c.id);
                        if (wasActive) controller.reloadActive();
                    }
                },
            }, "🗑");
            const acts = h("div", { "class": "chat-rail__acts" }, [renameBtn, delBtn]);
            const item = h("div", {
                "class": "chat-rail__item",
                "data-active": String(c.id === activeId),
                title: c.title || "",
                onClick: () => controller.loadConversation(c.id),
            }, [title, acts]);
            listEl.appendChild(item);
        });
    };

    newBtn.addEventListener("click", () => controller.newConversation());
    newBtnTop.addEventListener("click", () => controller.newConversation());

    // Una sola suscripción de la barra a la vez: al re-montar la sección,
    // desuscribe la anterior (evita fugas de closures sobre rails detached).
    if (BCCh._inlineRailUnsub) BCCh._inlineRailUnsub();
    BCCh._inlineRailUnsub = BCCh.ChatStore.subscribe(renderRail);
    renderRail();

    return controller;
};

BCCh.ChatPanel = ChatPanel;
BCCh.ChatController = ChatController;
}());
