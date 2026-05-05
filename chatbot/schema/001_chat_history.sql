-- =============================================================================
-- 001_chat_history.sql
-- Sesiones y mensajes de conversación. Aplicado automáticamente al startup
-- por app/db.py.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata     JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        UUID        NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role              TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content           TEXT        NOT NULL,
    rag_sources       JSONB,                       -- [{ref, filename, page, ...}]
    historical_series JSONB,                       -- [{series_id, name, n_obs, ...}]
    token_count       INT,                         -- tokens generados (assistant)
    latency_ms        INT,                         -- tiempo de generación (assistant)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated
    ON chat_sessions (updated_at DESC);

-- Trigger: toca updated_at en sessions cuando llega un mensaje
CREATE OR REPLACE FUNCTION _touch_session_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE chat_sessions SET updated_at = NOW()
    WHERE  session_id = NEW.session_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_touch_session_updated_at ON chat_messages;
CREATE TRIGGER trg_touch_session_updated_at
    AFTER INSERT ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION _touch_session_updated_at();
