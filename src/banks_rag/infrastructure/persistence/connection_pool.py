"""Pool de conexiones PostgreSQL compartido del proceso (psycopg2 ThreadedConnectionPool).

Singleton lazy inicializado en el lifespan de FastAPI. Thread-safe: psycopg2's
ThreadedConnectionPool serializa internamente el acceso al pool con un lock.

Uso:
    # En lifespan:
    init_pool(min_conn=2, max_conn=10, host=..., port=..., ...)

    # En código de acceso a BD:
    with pooled_conn() as conn:
        cur = conn.cursor()
        ...
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psycopg2.extensions
    import psycopg2.pool

log = logging.getLogger(__name__)

_pool: "psycopg2.pool.ThreadedConnectionPool | None" = None
_pool_lock = threading.Lock()


def init_pool(
    *,
    min_conn: int = 2,
    max_conn: int = 10,
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "postgres",
    database: str = "rag_banco",
) -> None:
    """Inicializa el pool de conexiones. Idempotente."""
    global _pool
    if _pool is not None:
        return
    with _pool_lock:
        if _pool is not None:
            return
        import psycopg2.pool

        _pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        log.info("PostgreSQL pool inicializado (min=%d, max=%d, db=%s)", min_conn, max_conn, database)


def get_pool() -> "psycopg2.pool.ThreadedConnectionPool | None":
    """Devuelve el pool si está inicializado, None si no."""
    return _pool


def close_pool() -> None:
    """Cierra todas las conexiones del pool. Llamar en shutdown del API."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None
            log.info("PostgreSQL pool cerrado")


@contextmanager
def pooled_conn(*, autocommit: bool = True):
    """Context manager que obtiene una conexión del pool y la devuelve al terminar.

    Nunca llama a ``conn.close()`` — devuelve la conexión al pool en el
    ``finally``, lo que la deja disponible para el siguiente request.

    Si el pool no está inicializado, abre y cierra una conexión directa
    (fallback transparente para tests y CLI).
    """
    pool = get_pool()
    if pool is None:
        import psycopg2

        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            user=os.getenv("PGUSER", os.getenv("USER", "postgres")),
            password=os.getenv("PGPASSWORD", "postgres"),
            database=os.getenv("PGDATABASE", "rag_banco"),
        )
        conn.autocommit = autocommit
        try:
            yield conn
            if not autocommit:
                conn.commit()
        except Exception:
            if not autocommit:
                conn.rollback()
            raise
        finally:
            conn.close()
        return

    conn = pool.getconn()
    conn.autocommit = autocommit
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        pool.putconn(conn)
