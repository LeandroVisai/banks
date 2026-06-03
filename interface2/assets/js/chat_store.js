/* ─────────────────────────────────────────────────────────────────────────
   CHAT STORE — conversaciones persistidas en localStorage (estilo Claude).

   Fuente única de verdad para el chat del Agente: la sección inline (Agente
   GOEM) y el panel lateral comparten la conversación ACTIVA y persisten entre
   navegaciones y recargas. Solo se pierde al iniciar un chat nuevo.

   Shape en localStorage["bcch_chats"]:
     { activeId, conversations: [ { id, title, createdAt, updatedAt,
         turns: [ {role:"user", content, attachmentNames}
                | {role:"assistant", data, elapsed} ] } ] }

   `data` del turno assistant es el ChatResponse completo (response, series_used,
   chunks_seen, attachment_visuals, …): suficiente para re-renderizar la
   conversación idéntica al recargarla.
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

(function () {
const STORAGE_KEY = "bcch_chats";
const MAX_CONVERSATIONS = 40;          // cap defensivo (descarta las más viejas)
const DEFAULT_TITLE = "Nueva conversación";

// pub/sub para que la barra de historial se refresque ante cualquier cambio.
const _listeners = new Set();
const _emit = () => { _listeners.forEach((cb) => { try { cb(); } catch (_) {} }); };

const _newId = () => `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
const _empty = () => ({ activeId: null, conversations: [] });

const _load = () => {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return _empty();
        const parsed = JSON.parse(raw);
        if (!parsed || !Array.isArray(parsed.conversations)) return _empty();
        return parsed;
    } catch (_) {
        return _empty();
    }
};

const _byRecent = (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0);

const _save = (state) => {
    if (state.conversations.length > MAX_CONVERSATIONS) {
        state.conversations.sort(_byRecent);
        state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS);
    }
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_) {
        // QuotaExceeded u otro: descarta la conversación más antigua y reintenta.
        if (state.conversations.length > 1) {
            state.conversations.sort(_byRecent);
            state.conversations.pop();
            _save(state);
        }
    }
};

const _active = (s) => s.conversations.find((c) => c.id === s.activeId) || null;

const _autoTitle = (text) => {
    const t = (text || "").trim().replace(/\s+/g, " ");
    if (!t) return DEFAULT_TITLE;
    return t.length > 42 ? t.slice(0, 42) + "…" : t;
};

const ChatStore = {
    /** Suscribe un callback a cambios del store. Devuelve la función para desuscribir. */
    subscribe(cb) { _listeners.add(cb); return () => _listeners.delete(cb); },

    /** Conversaciones ordenadas por actividad reciente (desc). */
    list() {
        return [..._load().conversations].sort(_byRecent);
    },

    get(id) { return _load().conversations.find((c) => c.id === id) || null; },

    /** Id de la conversación activa; crea una vacía si no hay ninguna válida. */
    activeId() {
        const s = _load();
        if (_active(s)) return s.activeId;
        return ChatStore.create().id;
    },

    /** Id activo SIN crear (puede ser null). Para resaltar en la barra. */
    peekActiveId() { return _load().activeId; },

    setActive(id) {
        const s = _load();
        if (s.conversations.some((c) => c.id === id)) {
            s.activeId = id;
            _save(s);
            _emit();
        }
    },

    /** Crea una conversación vacía y la deja como activa. */
    create() {
        const s = _load();
        const now = Date.now();
        const conv = { id: _newId(), title: DEFAULT_TITLE, createdAt: now, updatedAt: now, turns: [] };
        s.conversations.push(conv);
        s.activeId = conv.id;
        _save(s);
        _emit();
        return conv;
    },

    appendUserTurn(id, content, attachmentNames) {
        const s = _load();
        const conv = s.conversations.find((c) => c.id === id);
        if (!conv) return;
        conv.turns.push({ role: "user", content, attachmentNames: attachmentNames || [] });
        conv.updatedAt = Date.now();
        if (conv.title === DEFAULT_TITLE) conv.title = _autoTitle(content);
        _save(s);
        _emit();
    },

    appendAssistantTurn(id, data, elapsed) {
        const s = _load();
        const conv = s.conversations.find((c) => c.id === id);
        if (!conv) return;
        conv.turns.push({ role: "assistant", data, elapsed });
        conv.updatedAt = Date.now();
        _save(s);
        _emit();
    },

    rename(id, title) {
        const s = _load();
        const conv = s.conversations.find((c) => c.id === id);
        if (!conv) return;
        const clean = (title || "").trim().replace(/\s+/g, " ").slice(0, 60);
        conv.title = clean || DEFAULT_TITLE;
        _save(s);
        _emit();
    },

    /** Borra una conversación. Si era la activa, salta a la más reciente (o crea una). */
    remove(id) {
        const s = _load();
        const idx = s.conversations.findIndex((c) => c.id === id);
        if (idx === -1) return;
        s.conversations.splice(idx, 1);
        if (s.activeId === id) {
            const sorted = [...s.conversations].sort(_byRecent);
            s.activeId = sorted.length ? sorted[0].id : null;
        }
        _save(s);
        if (!s.conversations.length) { ChatStore.create(); return; }
        _emit();
    },
};

BCCh.ChatStore = ChatStore;
}());
