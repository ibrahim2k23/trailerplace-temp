from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)


def database_enabled() -> bool:
    required = ["HOST", "PGUSER", "PASSWORD", "DATABASE", "PORT"]
    return all((os.getenv(k) or "").strip() for k in required)


def _postgres_driver_name() -> str:
    if find_spec("psycopg2"):
        return "postgresql+psycopg2"
    if find_spec("psycopg"):
        return "postgresql+psycopg"
    raise RuntimeError("Install a Postgres driver with `uv add psycopg` or `uv add psycopg2-binary`")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    if not database_enabled():
        raise RuntimeError("Azure Postgres env vars are not fully configured")
    url = URL.create(
        _postgres_driver_name(),
        username=(os.getenv("PGUSER") or "").strip(),
        password=os.getenv("PASSWORD") or "",
        host=(os.getenv("HOST") or "").strip(),
        port=int((os.getenv("PORT") or "5432").strip()),
        database=(os.getenv("DATABASE") or "").strip(),
        query={"sslmode": "require"},
    )
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def ensure_schema() -> None:
    """Create missing chatbot tables without touching existing data."""
    if not database_enabled():
        return
    from src.db_models import Base

    Base.metadata.create_all(get_engine())
    logger.info("Chatbot persistence schema is ready")
