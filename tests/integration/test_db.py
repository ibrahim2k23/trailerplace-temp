from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src import db
from src import conversation_store as store
from src.db_models import Base, ChatbotConversation, ChatbotOutbox, ChatbotTurn

pytestmark = pytest.mark.db


@pytest.fixture()
def db_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    db_name = urlparse(url).path.lstrip("/")
    if "test" not in db_name.lower() and os.getenv("ALLOW_DESTRUCTIVE_DB_TESTS") != "1":
        pytest.skip("DB tests require a test database; set ALLOW_DESTRUCTIVE_DB_TESTS=1 only for disposable DBs")
    return url


@pytest.fixture()
def engine(db_url, monkeypatch):
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        Base.metadata.drop_all(conn)
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
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


def test_alembic_upgrade_head_idempotent(engine):
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    command.upgrade(cfg, "head")
    tables = set(inspect(engine).get_table_names())
    assert {"chatbot_leads", "chatbot_conversations", "chatbot_turns", "chatbot_outbox"} <= tables


def test_durable_turn_receipts_and_mismatch(engine):
    Base.metadata.create_all(engine)
    sid, tid = str(uuid.uuid4()), uuid.uuid4()
    lead_id = store.create_or_get_soft_lead(session_id=sid)
    with store.durable_turn(sid, tid, "hello") as (session, row, receipt):
        assert row is None
        assert receipt is None
        session.add(ChatbotConversation(session_id=uuid.UUID(sid), lead_id=uuid.UUID(lead_id), conversation=[]))
        session.add(ChatbotTurn(session_id=uuid.UUID(sid), turn_id=tid, request_message="hello", response={"ok": True}))
    with store.durable_turn(sid, tid, "hello") as (_, row, receipt):
        assert row is not None
        assert receipt.response == {"ok": True}
    with pytest.raises(ValueError):
        with store.durable_turn(sid, tid, "different"):
            pass


def test_leads_conversation_restore_close_and_feedback(engine):
    Base.metadata.create_all(engine)
    sid = str(uuid.uuid4())
    lead_id = store.create_or_get_soft_lead(session_id=sid)
    store.update_lead_contact(session_id=sid, full_name="A", email="a@test.com")
    store.update_lead_item_of_interest(sid, "Dump")
    store.promote_lead_to_hard(sid)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "user_feedback": "good"},
        {"role": "user", "content": "more"},
    ]
    store.upsert_conversation(session_id=sid, lead_id=lead_id, conversation=store._messages_to_conversation(messages))
    restored = store.restore_session(sid)
    assert restored["exists"] is True
    assert restored["messages"][0]["content"] == "hi"
    assert store.get_conversation(sid)[0]["feedback"] == "good"
    store.close_session(sid)
    assert store.restore_session(sid)["closed"] is True


def test_outbox_delivery_retry_and_unique_event(engine):
    Base.metadata.create_all(engine)
    sid, tid = uuid.uuid4(), uuid.uuid4()
    calls = []
    store.register_outbox_handler("team_request", lambda **payload: calls.append(payload))
    with db.get_session_factory()() as session:
        store.enqueue_outbox_event(session, session_id=sid, turn_id=tid, event_key="team:1", event_type="team_request", payload={"subject": "s"})
        session.commit()
    store.deliver_pending_outbox()
    assert calls == [{"subject": "s"}]
    with db.get_session_factory()() as session:
        row = session.query(ChatbotOutbox).one()
        assert row.status == "sent"


# --- Milestone 8: restart restore + feedback round trip -------------------------


def _drain_feedback_pool() -> None:
    """enqueue_save_user_feedback writes on a background thread; wait it out.

    The pool has two workers, so queue one no-op per worker to be sure every earlier
    task has been picked up and finished.
    """
    for future in [store._pool.submit(lambda: None) for _ in range(2)]:
        future.result(timeout=10)


def test_restore_after_backend_restart(engine):
    """A cold process reloads the conversation from state_snapshot, text only."""
    Base.metadata.create_all(engine)
    sid = str(uuid.uuid4())
    lead_id = store.create_or_get_soft_lead(session_id=sid)
    messages = [
        {"role": "user", "content": "dump for dirt", "listings": None},
        {"role": "assistant", "content": "How heavy?", "listings": [{"url": "https://x/1"}]},
    ]
    with store.durable_turn(sid, uuid.uuid4(), "dump for dirt") as (session, row, _):
        session.add(
            ChatbotConversation(
                session_id=uuid.UUID(sid),
                lead_id=uuid.UUID(lead_id),
                conversation=store._messages_to_conversation(messages),
                state_snapshot={"session_id": sid, "messages": messages},
            )
        )

    # Simulate a restart: the in-memory working store is gone, the DB is not.
    from src.graph.state import _sessions

    _sessions.clear()

    restored = store.restore_session(sid)
    assert restored["exists"] is True and restored["closed"] is False
    assert [m["content"] for m in restored["messages"]] == ["dump for dirt", "How heavy?"]
    # Listing dicts stay in the snapshot; the route is what nulls them for the frontend.
    assert restored["messages"][1]["listings"] == [{"url": "https://x/1"}]


def test_feedback_lands_on_turn_0_and_turn_3_and_survives_later_turns(engine):
    Base.metadata.create_all(engine)
    sid = str(uuid.uuid4())
    lead_id = store.create_or_get_soft_lead(session_id=sid)
    messages = []
    for i in range(4):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    store.upsert_conversation(
        session_id=sid, lead_id=lead_id, conversation=store._messages_to_conversation(messages)
    )

    store.enqueue_save_user_feedback(sid, 0, "great", "2026-07-09T00:00:00+00:00")
    store.enqueue_save_user_feedback(sid, 3, "wrong trailer", "2026-07-09T00:00:00+00:00")
    _drain_feedback_pool()

    conversation = store.get_conversation(sid)
    assert conversation[0]["feedback"] == "great"
    assert conversation[3]["feedback"] == "wrong trailer"

    # A later turn rewrites the conversation list; _merge_existing_feedback keeps it.
    messages.extend([{"role": "user", "content": "q4"}, {"role": "assistant", "content": "a4"}])
    store.upsert_conversation(
        session_id=sid, lead_id=lead_id, conversation=store._messages_to_conversation(messages)
    )
    conversation = store.get_conversation(sid)
    assert conversation[0]["feedback"] == "great"
    assert conversation[3]["feedback"] == "wrong trailer"

    # And it comes back on the matching assistant message (M8 §3).
    with db.get_session_factory()() as session:
        row = session.get(ChatbotConversation, uuid.UUID(sid))
        row.state_snapshot = {"session_id": sid, "messages": messages}
        session.commit()
    restored = store.restore_session(sid)
    assert restored["messages"][1]["user_feedback"] == "great"
    assert restored["messages"][7]["user_feedback"] == "wrong trailer"


def test_closed_session_reports_closed(engine):
    Base.metadata.create_all(engine)
    sid = str(uuid.uuid4())
    lead_id = store.create_or_get_soft_lead(session_id=sid)
    store.upsert_conversation(session_id=sid, lead_id=lead_id, conversation=[])
    store.close_session(sid)
    restored = store.restore_session(sid)
    assert restored["exists"] is True and restored["closed"] is True
