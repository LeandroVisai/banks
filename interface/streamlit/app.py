"""Banks RAG — Terminal Financiero BCCh."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="BCCh · Banks RAG",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system — IBM Plex (fuente técnica, no genérica) ────────────────────
# IBM Plex es la fuente de IBM Research/finanzas — distinctive vs Inter/Roboto

st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset tipográfico ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
}

/* ── Fondo con textura sutil (profundidad terminal) ─────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 0%, rgba(191,156,105,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(68,114,196,0.05) 0%, transparent 50%),
        repeating-linear-gradient(
            0deg, transparent, transparent 39px, rgba(255,255,255,0.015) 39px, rgba(255,255,255,0.015) 40px
        ),
        repeating-linear-gradient(
            90deg, transparent, transparent 39px, rgba(255,255,255,0.015) 39px, rgba(255,255,255,0.015) 40px
        ),
        #001730 !important;
}

/* ── Layout ────────────────────────────────────────────────────────────────── */
.block-container {
    padding: 0.75rem 1.5rem 0.5rem 1.5rem !important;
    max-width: 100% !important;
}

/* ── Monospace para datos numéricos ────────────────────────────────────────── */
.mono { font-family: 'IBM Plex Mono', monospace !important; }
div[data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Divider horizontal ─────────────────────────────────────────────────── */
.rule-gold  { height:1px; background:#BF9C69; opacity:0.6; margin:0.3rem 0 0.5rem; }
.rule-dim   { height:1px; background:#1a3a5c; margin:0.4rem 0; }

/* ── Header strip ─────────────────────────────────────────────────────────── */
.hdr {
    display:flex; align-items:baseline; justify-content:space-between;
    border-bottom: 1px solid rgba(191,156,105,0.5);
    padding-bottom: 0.35rem;
    margin-bottom: 0.5rem;
}
.hdr-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #BF9C69;
    text-transform: uppercase;
}
.hdr-sub {
    font-size: 0.72rem;
    color: #44546A;
    margin-left: 1rem;
    letter-spacing: 0.05em;
}
.hdr-clock {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #44546A;
    letter-spacing: 0.06em;
}

/* ── KPI ticker strip ──────────────────────────────────────────────────────── */
.kpi-strip {
    display: flex;
    gap: 0;
    border: 1px solid #1a3a5c;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.kpi-cell {
    flex: 1;
    padding: 0.45rem 0.8rem;
    border-right: 1px solid #1a3a5c;
    background: rgba(0,35,71,0.6);
    transition: background 0.15s ease;
}
.kpi-cell:last-child { border-right: none; }
.kpi-cell:hover { background: rgba(191,156,105,0.08); }
.kpi-label {
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #44546A;
    margin-bottom: 0.15rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #D4CFBE;
    line-height: 1.2;
}
.kpi-delta-up   { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#70AD47; }
.kpi-delta-down { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#C00000; }
.kpi-delta-neu  { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#7F7F7F; }

/* ── Panel section headers ─────────────────────────────────────────────────── */
.panel-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #44546A;
    margin-bottom: 0.4rem;
}

/* ── Source cards ──────────────────────────────────────────────────────────── */
.src-card {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.3rem;
    border-radius: 3px;
    border-left: 3px solid #BF9C69;
    background: rgba(0,35,71,0.5);
    transition: border-color 0.15s ease, background 0.15s ease;
}
.src-card:hover { background: rgba(191,156,105,0.07); }
.src-card.minutas    { border-left-color: #4472C4; }
.src-card.comunicado { border-left-color: #C00000; }
.src-card.ipom       { border-left-color: #70AD47; }
.src-card.ief        { border-left-color: #ED7D31; }
.src-card.research   { border-left-color: #57257D; }
.src-card.monitor_pm { border-left-color: #BF9C69; }

.src-name    { font-size:0.78rem; font-weight:500; color:#D4CFBE; }
.src-meta    { font-size:0.68rem; color:#44546A; }
.src-score   { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#BF9C69; margin-top:0.05rem; }

/* ── Chat messages tweaks ──────────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: rgba(0,35,71,0.35) !important;
    border-radius: 4px !important;
    border: 1px solid rgba(26,58,92,0.6) !important;
    margin-bottom: 0.4rem !important;
    padding: 0.6rem 0.8rem !important;
}

/* ── Suggestion pills ─────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.empty-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.4;
}
.empty-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #44546A;
    margin-bottom: 0.8rem;
}

/* ── Loading multi-step ────────────────────────────────────────────────────── */
.step-loader {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.8rem 1rem;
    background: rgba(0,35,71,0.5);
    border-radius: 4px;
    border-left: 3px solid #BF9C69;
    font-size: 0.78rem;
    color: #7F7F7F;
}
.step-active { color: #BF9C69; font-weight: 500; }

/* ── Route badge ───────────────────────────────────────────────────────────── */
.route-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    margin-right: 0.4rem;
}
.route-rag    { background: rgba(68,114,196,0.2); color:#4472C4; border:1px solid rgba(68,114,196,0.4); }
.route-sql    { background: rgba(112,173,71,0.2); color:#70AD47; border:1px solid rgba(112,173,71,0.4); }
.route-visual { background: rgba(237,125,49,0.2); color:#ED7D31; border:1px solid rgba(237,125,49,0.4); }
.route-time   { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:#44546A; }

/* ── Search result cards ───────────────────────────────────────────────────── */
.hit-card {
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-radius: 3px;
    border: 1px solid #1a3a5c;
    border-left: 3px solid #BF9C69;
    background: rgba(0,35,71,0.4);
}
.hit-header { font-size:0.75rem; color:#44546A; margin-bottom:0.3rem; }
.hit-text   { font-size:0.8rem; color:#D4CFBE; line-height:1.55; }
.hit-score  { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#BF9C69; }

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#1a3a5c; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#BF9C69; }

/* ── Reduce motion ─────────────────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    .src-card, .kpi-cell, div[data-testid="stChatMessage"] { transition: none !important; }
}
</style>
""")

# ── Estado ────────────────────────────────────────────────────────────────────

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("last_route", None)
st.session_state.setdefault("api_key", "")
st.session_state.setdefault("api_url", "http://localhost:8080")
st.session_state.setdefault("api_status", None)
st.session_state.setdefault("doc_filter", [])

# ── KPI data — leídos desde parquets reales ──────────────────────────────────

_PARQUET_BASE = Path(__file__).parent.parent.parent / "data_pipeline" / "parquet"


@st.cache_data(ttl=300)
def _load_kpis() -> list[tuple[str, str, str, str]]:
    import duckdb

    def _query(sql: str) -> list[dict]:
        with duckdb.connect(":memory:") as con:
            rel = con.sql(sql)
            cols = [d[0] for d in rel.description]
            return [{c: v for c, v in zip(cols, row)} for row in rel.fetchall()]

    kpis: list[tuple[str, str, str, str]] = []

    # TPM — sin parquet disponible
    kpis.append(("TPM", "—", "—", "neu"))

    # USD/CLP
    try:
        p = _PARQUET_BASE / "clp_monto.parquet"
        rows = _query(f"SELECT Fecha, CLP FROM read_parquet('{p}') ORDER BY Fecha DESC LIMIT 2")
        if rows:
            last, prev = rows[0]["CLP"], (rows[1]["CLP"] if len(rows) > 1 else None)
            val_str = f"{last:,.1f}"
            if prev:
                pct = (last - prev) / prev * 100
                delta_str = f"{pct:+.1f}%"
                direction = "up" if pct >= 0 else "down"
            else:
                delta_str, direction = "—", "neu"
            kpis.append(("USD/CLP", val_str, delta_str, direction))
        else:
            kpis.append(("USD/CLP", "—", "—", "neu"))
    except Exception:
        kpis.append(("USD/CLP", "—", "—", "neu"))

    # IPC 12m — expectativas de inflación
    try:
        p = _PARQUET_BASE / "expectativas_inflacion.parquet"
        rows = _query(
            f"SELECT Fecha, Valor FROM read_parquet('{p}') "
            f"WHERE Serie = '12M' ORDER BY Fecha DESC LIMIT 2"
        )
        if rows:
            last, prev = rows[0]["Valor"], (rows[1]["Valor"] if len(rows) > 1 else None)
            val_str = f"{last:.1f}%"
            if prev is not None:
                pp = last - prev
                delta_str = f"{pp:+.1f}pp"
                direction = "up" if pp >= 0 else "down"
            else:
                delta_str, direction = "—", "neu"
            kpis.append(("IPC 12m", val_str, delta_str, direction))
        else:
            kpis.append(("IPC 12m", "—", "—", "neu"))
    except Exception:
        kpis.append(("IPC 12m", "—", "—", "neu"))

    # Cobre (USD/lb — parquet almacena USc/lb, dividir por 100)
    try:
        p = _PARQUET_BASE / "cobre_dxy.parquet"
        rows = _query(f"SELECT Fecha, Cobre / 100.0 AS Cobre FROM read_parquet('{p}') ORDER BY Fecha DESC LIMIT 2")
        if rows:
            last, prev = rows[0]["Cobre"], (rows[1]["Cobre"] if len(rows) > 1 else None)
            val_str = f"{last:.3f}"
            if prev:
                pct = (last - prev) / prev * 100
                delta_str = f"{pct:+.1f}%"
                direction = "up" if pct >= 0 else "down"
            else:
                delta_str, direction = "—", "neu"
            kpis.append(("Cobre", val_str, delta_str, direction))
        else:
            kpis.append(("Cobre", "—", "—", "neu"))
    except Exception:
        kpis.append(("Cobre", "—", "—", "neu"))

    # IPSA — sin parquet disponible
    kpis.append(("IPSA", "—", "—", "neu"))

    return kpis


KPIS = _load_kpis()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.html('<div class="panel-label">Configuración</div>')
    st.session_state.api_url = st.text_input("Servidor", value=st.session_state.api_url, label_visibility="collapsed", placeholder="http://localhost:8080")
    st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed", placeholder="banks-key-…")

    if st.button("Verificar conexión", use_container_width=True):
        try:
            r = requests.get(f"{st.session_state.api_url}/readyz", timeout=5)
            st.session_state.api_status = r.json().get("status", "error")
        except Exception:
            st.session_state.api_status = "error"

    s = st.session_state.api_status
    if s == "ok":
        st.badge("API lista", icon=":material/check_circle:", color="green")
    elif s == "loading":
        st.badge("LLM cargando", icon=":material/hourglass_top:", color="orange")
    elif s == "error":
        st.badge("Sin conexión", icon=":material/error:", color="red")

    st.html('<div class="rule-dim"></div>')
    st.html('<div class="panel-label">Filtros de corpus</div>')
    st.session_state.doc_filter = st.multiselect(
        "docs", ["MINUTAS", "COMUNICADO", "REPORTE_RESEARCH", "MONITOR_PM", "IPOM", "IEF"],
        default=st.session_state.doc_filter, label_visibility="collapsed", placeholder="Todos los documentos",
    )
    st.html('<div class="rule-dim"></div>')
    st.caption(f"Banks RAG v1.0 · {datetime.now().strftime('%Y-%m-%d')}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if st.session_state.api_key:
        h["X-API-Key"] = st.session_state.api_key
    return h

def call_chat(message: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message}
    if st.session_state.doc_filter:
        payload["filters"] = {"doc_types": st.session_state.doc_filter}
    r = requests.post(f"{st.session_state.api_url}/v1/chat", json=payload, headers=_headers(), timeout=120)
    r.raise_for_status()
    return r.json()

def call_search(query: str, k: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "k": k}
    if st.session_state.doc_filter:
        payload["filters"] = {"doc_types": st.session_state.doc_filter}
    r = requests.post(f"{st.session_state.api_url}/v1/search", json=payload, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

DOC_CSS_CLASS = {
    "MINUTAS": "minutas", "COMUNICADO": "comunicado", "IPOM": "ipom",
    "IEF": "ief", "REPORTE_RESEARCH": "research", "MONITOR_PM": "monitor_pm",
}

def _src_card(s: dict) -> str:
    doc_type = s.get("doc_type", "—")
    doc_date = s.get("document_date", "")
    section  = s.get("section_type", "")
    page     = s.get("page")
    score    = s.get("score", 0.0)
    css_cls  = DOC_CSS_CLASS.get(doc_type, "")
    page_str = f" · pág. {page}" if page is not None else ""
    meta     = " · ".join(filter(None, [doc_date, section]))
    return f"""
    <div class="src-card {css_cls}">
        <div class="src-name">{doc_type}{page_str}</div>
        <div class="src-meta">{meta}</div>
        <div class="src-score">score {score:.4f}</div>
    </div>"""

def _route_html(route: str, elapsed: float | None) -> str:
    cls = f"route-{route}" if route in ("rag", "sql", "visual") else "route-rag"
    elapsed_str = f'<span class="route-time">{elapsed:.1f}s</span>' if elapsed else ""
    return f'<span class="route-pill {cls}">{route.upper()}</span>{elapsed_str}'

# ── HEADER ────────────────────────────────────────────────────────────────────

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
st.html(f"""
<div class="hdr">
    <div>
        <span class="hdr-title">BANCO CENTRAL DE CHILE</span>
        <span class="hdr-sub">Banks RAG · Terminal de Análisis</span>
    </div>
    <span class="hdr-clock">{ts}</span>
</div>
""")

# ── KPI TICKER ────────────────────────────────────────────────────────────────

cells = ""
for label, value, delta, direction in KPIS:
    delta_cls = {"up": "kpi-delta-up", "down": "kpi-delta-down"}.get(direction, "kpi-delta-neu")
    cells += f"""
    <div class="kpi-cell">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="{delta_cls}">{delta}</div>
    </div>"""

st.html(f'<div class="kpi-strip">{cells}</div>')

# ── TABS ──────────────────────────────────────────────────────────────────────

tab_chat, tab_search = st.tabs([
    ":material/chat: Agente",
    ":material/manage_search: Búsqueda",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    col_chat, col_src = st.columns([3, 2], gap="large")

    # ── Conversación ──────────────────────────────────────────────────────────
    with col_chat:
        st.html('<div class="panel-label">:material/smart_toy: Agente BCCh</div>')

        SUGGESTIONS = {
            "Decisión del Consejo": "¿Qué decidió el Consejo del Banco Central en su última reunión?",
            "Proyecciones inflación": "¿Cuáles son las proyecciones de inflación del BCCh?",
            "Riesgos externos": "¿Qué riesgos externos menciona el Consejo en las minutas?",
            "USD/CLP actual": "¿Cuánto vale el USD/CLP actualmente?",
            "Precio cobre": "Dame el precio histórico del cobre",
        }

        if not st.session_state.messages:
            st.html("""
            <div class="empty-state">
                <div class="empty-icon">⬡</div>
                <div class="empty-label">Consultas sugeridas</div>
            </div>
            """)
            selected = st.pills("s", list(SUGGESTIONS.keys()), label_visibility="collapsed")
            if selected:
                st.session_state.messages.append({
                    "role": "user", "content": SUGGESTIONS[selected],
                    "route": None, "elapsed": None,
                })
                st.rerun()

        for msg in st.session_state.messages:
            role   = msg["role"]
            avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("route"):
                    st.html(_route_html(msg["route"], msg.get("elapsed")))

        # Input + multi-step loading (mejor UX que spinner estático)
        if prompt := st.chat_input("Consulta al agente…"):
            st.session_state.messages.append({
                "role": "user", "content": prompt, "route": None, "elapsed": None,
            })
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                with st.status("Procesando consulta…", expanded=True) as status_box:
                    st.write(":material/manage_search: Analizando y enrutando consulta…")
                    try:
                        t0 = time.monotonic()
                        # Simular progreso visible para el usuario
                        time.sleep(0.3)
                        st.write(":material/database: Buscando en corpus y catálogo SQL…")
                        data    = call_chat(prompt)
                        elapsed = time.monotonic() - t0
                        st.write(":material/auto_awesome: Generando respuesta…")
                        status_box.update(label="Respuesta generada", state="complete", expanded=False)

                        response = data.get("response", "Sin respuesta.")
                        sources  = data.get("sources", [])
                        route    = data.get("route", "rag")

                        st.markdown(response)
                        st.html(_route_html(route, elapsed))

                        st.session_state.messages.append({
                            "role": "assistant", "content": response,
                            "route": route, "elapsed": elapsed,
                        })
                        st.session_state.last_sources = sources
                        st.session_state.last_route   = route

                    except requests.exceptions.ConnectionError:
                        status_box.update(label="Error de conexión", state="error", expanded=False)
                        st.error("Sin conexión con la API — verifica la URL en el panel lateral.", icon=":material/error:")
                    except requests.exceptions.Timeout:
                        status_box.update(label="Timeout", state="error", expanded=False)
                        st.error("Timeout (>120s). El modelo puede estar cargando.", icon=":material/hourglass_empty:")
                    except requests.exceptions.HTTPError as e:
                        status_box.update(label=f"Error {e.response.status_code}", state="error", expanded=False)
                        code = e.response.status_code
                        if code == 401:
                            st.error("API Key incorrecta.", icon=":material/lock:")
                        elif code == 429:
                            st.warning("Rate limit alcanzado.", icon=":material/speed:")
                        else:
                            st.error(f"Error {code}: {e.response.text[:150]}", icon=":material/error:")

        if st.session_state.messages:
            st.html('<div class="rule-dim"></div>')
            if st.button(":material/delete_sweep: Nueva conversación", use_container_width=False):
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.rerun()

    # ── Fuentes ───────────────────────────────────────────────────────────────
    with col_src:
        st.html('<div class="panel-label">:material/library_books: Fuentes citadas</div>')

        if not st.session_state.last_sources:
            st.html("""
            <div style="padding:1.5rem 0.5rem; text-align:center;">
                <div style="font-size:1.5rem; opacity:0.2; margin-bottom:0.5rem;">◈</div>
                <div style="font-size:0.7rem; letter-spacing:0.08em; text-transform:uppercase; color:#44546A;">
                    Las fuentes aparecerán tras cada consulta
                </div>
            </div>
            """)
        else:
            route = st.session_state.last_route
            if route:
                st.html(_route_html(route, None))
            st.html('<div class="rule-dim" style="margin:0.4rem 0;"></div>')

            cards_html = "".join(_src_card(s) for s in st.session_state.last_sources)
            st.html(cards_html)


# ══════════════════════════════════════════════════════════════════════════════
# TAB BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.html('<div class="panel-label">Búsqueda híbrida · HNSW + BM25 → RRF → Reranker</div>')

    s1, s2, s3 = st.columns([5, 1, 1])
    with s1:
        query = st.text_input("q", placeholder="tasa política monetaria inflación…", label_visibility="collapsed")
    with s2:
        k = st.number_input("k", 1, 20, 5, label_visibility="collapsed")
    with s3:
        run = st.button(":material/search: Buscar", type="primary", use_container_width=True)

    if run and query:
        with st.spinner(""):
            try:
                data = call_search(query, int(k))
                hits = data.get("hits", [])
                if not hits:
                    st.html('<div style="padding:1rem; font-size:0.8rem; color:#44546A;">Sin resultados para esta consulta.</div>')
                else:
                    st.html(f'<div class="panel-label">{len(hits)} resultado(s)</div>')
                    for i, hit in enumerate(hits, 1):
                        doc_type = hit.get("doc_type", "—")
                        doc_date = hit.get("document_date", "")
                        section  = hit.get("section_type", "")
                        score    = hit.get("score", 0.0)
                        text     = hit.get("text", "")
                        css_cls  = DOC_CSS_CLASS.get(doc_type, "")
                        meta     = " · ".join(filter(None, [doc_date, section]))
                        snippet  = text[:450] + ("…" if len(text) > 450 else "")
                        st.html(f"""
                        <div class="hit-card {css_cls}" style="border-left-color:{
                            {'minutas':'#4472C4','comunicado':'#C00000','ipom':'#70AD47',
                             'ief':'#ED7D31','research':'#57257D','monitor_pm':'#BF9C69'}.get(css_cls,'#BF9C69')
                        }">
                            <div class="hit-header">#{i} · {doc_type} · {meta}
                                <span class="hit-score" style="float:right">{score:.4f}</span>
                            </div>
                            <div class="hit-text">{snippet}</div>
                        </div>
                        """)
            except requests.exceptions.ConnectionError:
                st.error("Sin conexión.", icon=":material/error:")
            except requests.exceptions.HTTPError as e:
                st.error(f"Error {e.response.status_code}", icon=":material/error:")
    elif run:
        st.warning("Escribe una consulta.", icon=":material/warning:")
