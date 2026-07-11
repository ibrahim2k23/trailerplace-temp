from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import conversation_store, db, tracing
from src.api import readiness
from src.api.routes import ensure_graph, router
from src.config import settings
from src.log_setup import configure_trailerplace_logging

logger = logging.getLogger(__name__)


def run_startup() -> None:
    """Migrate, compile the graph, probe the database (M8 §6).

    Every failure is recorded on the readiness state rather than raised: the process
    still serves so /health can report *why* it is not ok.
    """
    try:
        # Before the graph compiles, so the first turn's spans are exported (M9 §2).
        tracing.configure_langsmith()

        if db.database_enabled():
            if settings.db_auto_create:
                db.run_migrations()
            else:
                logger.info("DB_AUTO_CREATE is off — run `alembic upgrade head` before first boot")

        # Compile once here so the first /chat doesn't pay the build cost inside its
        # own timeout budget.
        ensure_graph()
        readiness.mark_graph_ready()

        # Persistence-off is a valid deployment: there is nothing to probe.
        if conversation_store.persistence_enabled() and not db.ping():
            readiness.mark_failed("database unreachable")
            return
        readiness.mark_db_ready()
    except Exception as exc:  # noqa: BLE001 - surface the reason through /health
        logger.exception("Startup failed")
        readiness.mark_failed(f"{type(exc).__name__}: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    readiness.reset()
    run_startup()
    yield


def create_app() -> FastAPI:
    # Bug fix: this was defined in src/log_setup.py since M0 but never actually
    # called anywhere, so the root logger had no handlers and every logger.info()
    # in the codebase (the per-turn JSON record, the human-readable conversation
    # reasoning block, every TOOL search/inventory_lookup/email/outbox line) was
    # silently dropped -- console AND the daily log file. Called here, before the
    # FastAPI app is built and before uvicorn.run() performs its own logging setup
    # (uvicorn's default config has no "root" entry, so it does not touch these
    # handlers once they exist).
    configure_trailerplace_logging()
    app = FastAPI(title="TrailerPlace Chatbot API", lifespan=lifespan)
    app.include_router(router)
    # Wire the transactional-outbox drain to the email sender (M7).
    conversation_store.register_default_outbox_handlers()
    return app
