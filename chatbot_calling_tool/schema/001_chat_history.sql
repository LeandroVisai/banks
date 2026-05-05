-- =============================================================================
-- 001_chat_history.sql — historial conversacional para el chatbot agentic.
-- Tablas distintas a las del chatbot/ clásico para no mezclar trazas.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata     JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        UUID        NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
    role              TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content           TEXT        NOT NULL,
    -- Diferencias respecto al chatbot/ clásico:
    -- En vez de rag_sources (única recuperación), guardamos el TRACE completo
    -- de qué herramientas llamó el agente, con qué args, qué devolvieron, y cuánto tardaron.
    tool_trace        JSONB,                       -- [{tool, args, result_summary, duration_ms}]
    cited_chunks      JSONB,                       -- chunks finales referenciados en la respuesta
    historical_series JSONB,                       -- series consultadas (si las hubo)
    iterations        INT,                         -- nº de ciclos LLM↔tools que se ejecutaron
    token_count       INT,
    latency_ms        INT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_session_created
    ON agent_messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated
    ON agent_sessions (updated_at DESC);

CREATE OR REPLACE FUNCTION _touch_agent_session_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE agent_sessions SET updated_at = NOW() WHERE session_id = NEW.session_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_agent_session_updated_at ON agent_messages;
CREATE TRIGGER trg_touch_agent_session_updated_at
    AFTER INSERT ON agent_messages
    FOR EACH ROW EXECUTE FUNCTION _touch_agent_session_updated_at();
