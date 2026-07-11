"""Milestone 8 — full API contract + persistence hardening.

Everything here is network-free: the graph is faked, persistence is off.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import readiness, routes
from src.api.app import create_app
from src.graph.state import _sessions
from src.models import TrailerListing
from tests.conftest import FakeLLM
from tests.unit.llm_helpers import sample_reply

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "chat_response_contract.json"
SESSION = "44444444-4444-4444-4444-444444444444"
TURN = "55555555-5555-5555-5555-555555555555"


class _FakeGraph:
    """Stands in for the compiled LangGraph. Records the state it was invoked with."""

    def __init__(self, *, reply=None, raises=None, sleep=0.0, listings=None):
        self._reply = reply or sample_reply("Sure, I can help.")
        self._raises = raises
        self._sleep = sleep
        self._listings = listings or []
        self.seen_state: dict | None = None

    def invoke(self, state):
        self.seen_state = state
        if self._sleep:
            time.sleep(self._sleep)
        if self._raises:
            raise self._raises
        return {
            **state,
            "customer_name": "John",
            "customer_phone": "555-1234",
            "turn_outcome": {"reply": self._reply, "listings": self._listings},
        }


@pytest.fixture
def offline(monkeypatch):
    """Persistence off, graph faked, in-memory state clean."""
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    _sessions.clear()

    def _install(graph):
        monkeypatch.setattr(routes, "_get_graph", lambda: graph)
        return TestClient(create_app())

    return _install


def _post(client, **overrides):
    body = {"session_id": SESSION, "turn_id": TURN, "message": "dump for dirt"}
    body.update(overrides)
    return client.post("/chat", json=body)


# --- §1 response contract ------------------------------------------------------


def test_chat_response_matches_golden_contract(offline):
    client = offline(_FakeGraph())
    response = _post(client)
    assert response.status_code == 200
    body = response.json()

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    expected.pop("_comment")
    assert set(body) == set(expected)
    # app.py probes this one with `"customer_email" in data`, so it must always be a key.
    assert "customer_email" in body
    assert body["sales_phase"] == "main"
    assert body["onboarding_api_messages"] == []
    assert body["main_prior_messages"] is None
    assert body["thinking_context"] is None


def test_listings_survive_app_py_parser(offline):
    """app.py wraps TrailerListing(...) in a bare try/except and silently drops failures.

    This copies its construction block verbatim so a shape regression fails loudly here
    instead of making cards vanish in the UI.
    """
    listing = {
        "title": "2026 Iron Bull FHG24K Dump Trailer",
        "condition": "New",
        "price": "$18,500",
        "category": "Dump > Standard",
        "make": "Iron Bull Trailers",
        "color": "Black",
        "hitch_type": "Gooseneck",
        "year": 2026,
        "length": "24",
        "width": "8.5",
        "axles": 2,
        "gvwr": "24000",
        "payload_capacity": "18000",
        "material": "Steel",
        "floor": "Steel",
        "url": "https://trailerplace.com/l/12914",
        "relevance_score": 0.91,
    }
    reply = sample_reply("Here it is.", urls=[listing["url"]])
    client = offline(_FakeGraph(reply=reply, listings=[listing]))

    returned = _post(client).json()["listings"]
    assert len(returned) == 1

    parsed = []
    for d in returned:
        if not isinstance(d, dict):
            continue
        try:
            parsed.append(
                TrailerListing(
                    listing_id=str(d.get("url") or d.get("title") or ""),
                    title=str(d.get("title") or ""),
                    condition=str(d.get("condition") or "New"),
                    price=float(d["price"].replace("$", "").replace(",", ""))
                    if isinstance(d.get("price"), str) and d.get("price") not in ("Call for price", None, "")
                    else d.get("price"),
                    price_display=str(d.get("price") or "") or None,
                    payments_from=None,
                    category_subcategory=str(d.get("category") or ""),
                    make=str(d.get("make") or ""),
                    color=str(d.get("color") or ""),
                    hitch_type=d.get("hitch_type"),
                    year=d.get("year"),
                    width=d.get("width"),
                    length=d.get("length"),
                    axles=d.get("axles"),
                    gvwr=d.get("gvwr"),
                    payload_capacity=d.get("payload_capacity"),
                    trailer_material=d.get("material"),
                    floor=d.get("floor"),
                    url=str(d.get("url") or ""),
                    score=d.get("relevance_score"),
                )
            )
        except Exception:
            pass

    assert len(parsed) == 1, "app.py's parser silently dropped the listing"
    assert parsed[0].price == 18500.0


# --- §4 robustness -------------------------------------------------------------


def test_graph_error_returns_200_apology(offline, caplog):
    client = offline(_FakeGraph(raises=RuntimeError("pinecone exploded")))
    response = _post(client)
    assert response.status_code == 200
    body = response.json()
    assert body["assistant_text"] == routes.ERROR_ASSISTANT_TEXT
    assert body["listings"] == []
    assert "customer_email" in body
    assert "Chat turn failed" in caplog.text


def test_chat_timeout_returns_200_apology(offline, monkeypatch):
    # Settings is a frozen dataclass: swap the whole object, not one field.
    monkeypatch.setattr(routes, "settings", replace(routes.settings, chat_timeout_seconds=0.05))
    client = offline(_FakeGraph(sleep=0.5))
    response = _post(client)
    assert response.status_code == 200
    assert response.json()["assistant_text"] == routes.ERROR_ASSISTANT_TEXT


def test_oversized_message_is_truncated(offline, monkeypatch):
    monkeypatch.setattr(routes, "settings", replace(routes.settings, chat_max_message_chars=32))
    graph = _FakeGraph()
    client = offline(graph)
    _post(client, message="x" * 500)
    assert graph.seen_state["messages"][-1]["content"] == "x" * 32


@pytest.mark.parametrize("field", ["session_id", "turn_id"])
def test_malformed_uuid_returns_422(offline, field):
    client = offline(_FakeGraph())
    assert _post(client, **{field: "not-a-uuid"}).status_code == 422


def test_reset_unknown_session_is_ok(offline):
    client = offline(_FakeGraph())
    assert client.post("/session/reset", json={"session_id": "never-seen"}).status_code == 200


# --- §5 closed session / §6 health ---------------------------------------------


def test_closed_session_shape(offline, monkeypatch):
    monkeypatch.setattr(
        "src.conversation_store.restore_session",
        lambda sid: {"exists": True, "closed": True, "messages": []},
    )
    client = offline(_FakeGraph())
    body = client.get(f"/session/{SESSION}").json()
    assert body["exists"] is True and body["closed"] is True


def test_health_is_503_until_startup_then_ok(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    monkeypatch.setattr("src.db.database_enabled", lambda: False)
    routes.set_graph_client(FakeLLM([]))

    app = create_app()
    readiness.reset()

    # Lifespan has not run: not ready.
    early = TestClient(app).get("/health")
    assert early.status_code == 503
    assert early.json()["status"] == "starting"

    # Entering the context manager runs the lifespan.
    with TestClient(app) as client:
        ready = client.get("/health")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ok"}


def test_health_reports_error_when_db_unreachable(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: True)
    monkeypatch.setattr("src.db.database_enabled", lambda: False)
    monkeypatch.setattr("src.db.ping", lambda: False)
    routes.set_graph_client(FakeLLM([]))

    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "error"
