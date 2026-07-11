from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src import conversation_store as store
from src import db
from src.api import routes
from src.api.app import create_app
from src.db_models import Base, ChatbotLead, ChatbotOutbox
from src.graph.state import _sessions
from tests.conftest import FakeEmailSender, FakeLLM
from tests.unit.llm_helpers import sample_analysis, sample_reply


# --------------------------------------------------------------------------- #
# Persistence-off: the email path fires directly through email_sender.
# --------------------------------------------------------------------------- #
def test_escalation_with_contact_sends_directly_persistence_off(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    fake = FakeEmailSender()
    monkeypatch.setattr("src.graph.nodes.email_actions.email_sender.send_email", fake.send_email)
    _sessions.clear()
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(
                    intent="team_request_escalation",
                    category_mentioned=None,
                    email_triggers=[{"kind": "escalation", "faq_key": None, "listing_reference": None, "description": "wants a call"}],
                    contact={"name": "Jane", "email": None, "phone": "555-1234"},
                ),
                sample_reply("I've passed your query to our team."),
            ]
        )
    )
    client = TestClient(create_app())
    resp = client.post("/chat", json={"session_id": str(uuid.uuid4()), "turn_id": str(uuid.uuid4()), "message": "have someone call me", "customer_full_name": "Jane", "customer_phone": "555-1234"})
    assert resp.status_code == 200
    assert len(fake.sent) == 1
    assert "[Escalation]" in fake.sent[0]["body"]


# --------------------------------------------------------------------------- #
# Persistence-on outbox mechanics (requires a disposable Postgres).
# --------------------------------------------------------------------------- #
pytest.importorskip("psycopg")


@pytest.fixture()
def engine(monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    db_name = urlparse(url).path.lstrip("/")
    if "test" not in db_name.lower() and os.getenv("ALLOW_DESTRUCTIVE_DB_TESTS") != "1":
        pytest.skip("DB tests require a test database")
    engine = create_engine(url, pool_pre_ping=True)
    with engine.begin() as conn:
        Base.metadata.drop_all(conn)
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db, "database_enabled", lambda: True)
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    monkeypatch.setattr(db, "get_session_factory", lambda: factory)
    monkeypatch.setattr(store, "persistence_enabled", lambda: True)
    yield engine
    with engine.begin() as conn:
        Base.metadata.drop_all(conn)
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    engine.dispose()


pytestmark = pytest.mark.db


def _drain_outbox_pool() -> None:
    """deliver_pending_outbox_async fires on the background persistence pool now
    (M9 fix — a slow email send must not block the /chat response). Wait it out
    before asserting on outbox/lead state, the same pattern test_db.py uses for
    enqueue_save_user_feedback."""
    for future in [store._pool.submit(lambda: None) for _ in range(2)]:
        future.result(timeout=10)


def _lead_type(engine, sid) -> str:
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        return s.execute(select(ChatbotLead.lead_type).where(ChatbotLead.psid == sid)).scalar_one()


def _outbox(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as s:
        return list(s.execute(select(ChatbotOutbox)).scalars())


def _run_escalation_chat(engine, fake, session_id, turn_id):
    _sessions.pop(session_id, None)
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(
                    intent="team_request_escalation",
                    category_mentioned=None,
                    email_triggers=[{"kind": "escalation", "faq_key": None, "listing_reference": None, "description": "wants a call"}],
                    contact={"name": "Jane", "email": None, "phone": "555-1234"},
                ),
                sample_reply("Passed to the team."),
            ]
        )
    )
    # create_app() wires the default (real) handlers, so override AFTER it is built
    # but before the POST triggers the post-commit drain.
    client = TestClient(create_app())
    store.register_outbox_handler("escalation_alert", fake.handler)
    return client.post("/chat", json={"session_id": session_id, "turn_id": turn_id, "message": "call me", "customer_full_name": "Jane", "customer_phone": "555-1234"})


def test_outbox_created_in_transaction_drained_and_lead_hard(engine):
    fake = FakeEmailSender()
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    resp = _run_escalation_chat(engine, fake, sid, tid)
    assert resp.status_code == 200
    _drain_outbox_pool()  # M9: the drain is now async, wait it out before asserting
    rows = _outbox(engine)
    assert len(rows) == 1 and rows[0].status == "sent"
    assert len(fake.sent) == 1
    assert _lead_type(engine, sid) == "hard"


def test_failed_handler_leaves_row_retryable(engine):
    fake = FakeEmailSender(succeed=False)
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    _run_escalation_chat(engine, fake, sid, tid)
    _drain_outbox_pool()
    rows = _outbox(engine)
    assert len(rows) == 1 and rows[0].status == "failed"
    # Next drain retries the failed row.
    good = FakeEmailSender()
    store.register_outbox_handler("escalation_alert", good.handler)
    store.deliver_pending_outbox()
    assert _outbox(engine)[0].status == "sent"


def test_duplicate_turn_replay_does_not_enqueue_second_event(engine):
    fake = FakeEmailSender()
    sid, tid = str(uuid.uuid4()), str(uuid.uuid4())
    _run_escalation_chat(engine, fake, sid, tid)
    # Replay the exact same turn_id + message -> receipt short-circuits, no new event.
    _sessions.pop(sid, None)
    client = TestClient(create_app())
    replay = client.post("/chat", json={"session_id": sid, "turn_id": tid, "message": "call me", "customer_full_name": "Jane", "customer_phone": "555-1234"})
    assert replay.status_code == 200
    assert len(_outbox(engine)) == 1
