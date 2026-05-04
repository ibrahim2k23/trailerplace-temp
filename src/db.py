"""
Postgres connection helpers (Azure / generic).

**Single URL (recommended)**  
Set ``DATABASE_URL`` or ``TRAILERPLACE_DATABASE_URL`` to
``postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require``.

**Separate variables** (any of these names; avoid bare ``USER`` on Windows — it is
often already set to your Windows account and ``.env`` will not override it):

- Host: ``PGHOST``, ``DB_HOST``, or ``HOST``
- Login: ``PGUSER`` or ``DB_USER`` (not ``USER``)
- Password: ``PGPASSWORD``, ``PASSWORD``, or ``DB_PASSWORD``
- Port: ``PGPORT``, ``DB_PORT``, or ``PORT`` (prefer ``PGPORT`` if you also set ``PORT`` for HTTP)
- Database name: ``PGDATABASE``, ``DATABASE``, or ``DB_NAME``
"""
from __future__ import annotations

import os
import threading
from typing import Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None
_lock = threading.Lock()

_URL_ENV_NAMES = (
    "DATABASE_URL",
    "TRAILERPLACE_DATABASE_URL",
)


def get_database_url() -> Optional[str]:
    for n in _URL_ENV_NAMES:
        u = (os.getenv(n) or "").strip()
        if u:
            if u.startswith("postgres://"):
                u = u.replace("postgres://", "postgresql+psycopg://", 1)
            elif u.startswith("postgresql://") and "+" not in u.split("://", 1)[0]:
                u = u.replace("postgresql://", "postgresql+psycopg://", 1)
            return u

    host = (
        (os.getenv("PGHOST") or "").strip()
        or (os.getenv("DB_HOST") or "").strip()
        or (os.getenv("HOST") or "").strip()
    )
    # Never use bare os.environ["USER"] — on Windows it is the OS login and .env cannot override it.
    user = (
        (os.getenv("PGUSER") or "").strip()
        or (os.getenv("DB_USER") or "").strip()
        or (os.getenv("DATABASE_USER") or "").strip()
    )
    password = os.getenv("PGPASSWORD") or os.getenv("PASSWORD") or os.getenv("DB_PASSWORD")
    port = (
        (os.getenv("PGPORT") or "").strip()
        or (os.getenv("DB_PORT") or "").strip()
        or (os.getenv("PORT") or "").strip()
        or "5432"
    )
    database = (
        (os.getenv("PGDATABASE") or "").strip()
        or (os.getenv("DATABASE") or "").strip()
        or (os.getenv("DB_NAME") or "").strip()
        or "postgres"
    )
    if host and user and password is not None:
        return (
            f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(str(password))}"
            f"@{host}:{port}/{database}?sslmode=require"
        )
    return None


def get_engine() -> Optional[Engine]:
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            return _engine
        url = get_database_url()
        if not url:
            return None
        _engine = create_engine(url, pool_pre_ping=True)
        _session_factory = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
        )
        return _engine


def get_session_factory() -> Optional[sessionmaker[Session]]:
    get_engine()
    return _session_factory
