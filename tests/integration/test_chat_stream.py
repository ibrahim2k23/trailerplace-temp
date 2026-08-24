"""POST /chat/stream — the streamed turn the UI types out, one bubble per trailer.

Network-free: the graph is faked and persistence is off, exactly as test_api_contract does.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from src.api import routes
from src.api.app import create_app
from src.config import settings
from src.graph.state import _sessions
from tests.integration.test_api_contract import SESSION, TURN, _FakeGraph
from tests.unit.llm_helpers import sample_reply

LISTING_REPLY = (
    "Here are some options for livestock trailers:\n\n"
    "1. [2025 Galyean Cattle Trailer - 15221](https://trailerplace.com/a)\n"
    "   - Price: $24,250\n"
    "2. [2026 Galyean 32' Cattle Trailer - 15079](https://trailerplace.com/b)\n"
    "   - Price: $32,250\n\n"
    "Do any of these look like a fit?"
)


@pytest.fixture
def stream_client(monkeypatch):
    """A client whose turns are faked and whose typing delays are removed."""
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    _sessions.clear()
    # The pacing is real time on a real turn; a test must not sit through it.
    monkeypatch.setattr(
        routes, "settings", replace(settings, chat_stream_delta_seconds=0.0, chat_stream_chunk_pause_seconds=0.0)
    )

    def _install(graph):
        monkeypatch.setattr(routes, "_get_graph", lambda: graph)
        return TestClient(create_app())

    return _install


def _events(client, **overrides) -> list[tuple[str, dict]]:
    body = {"session_id": SESSION, "turn_id": TURN, "message": "livestock trailer"}
    body.update(overrides)
    out: list[tuple[str, dict]] = []
    with client.stream("POST", "/chat/stream", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        event = "message"
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                out.append((event, json.loads(line.split(":", 1)[1].strip())))
    return out


def test_stream_sends_one_chunk_per_trailer(stream_client):
    client = stream_client(_FakeGraph(reply=sample_reply(LISTING_REPLY)))
    events = _events(client)

    starts = [body for name, body in events if name == "chunk_start"]
    ends = [body["text"] for name, body in events if name == "chunk_end"]
    assert len(starts) == len(ends) == 4
    assert starts[0]["total"] == 4
    assert ends[0].startswith("Here are some options")
    assert ends[1].startswith("1. [2025 Galyean")
    assert ends[2].startswith("2. [2026 Galyean")
    assert ends[3].startswith("Do any of these")


def test_deltas_rebuild_each_chunk_exactly(stream_client):
    client = stream_client(_FakeGraph(reply=sample_reply(LISTING_REPLY)))
    events = _events(client)

    typed: dict[int, str] = {}
    for name, body in events:
        if name == "delta":
            typed[body["index"]] = typed.get(body["index"], "") + body["text"]
    for name, body in events:
        if name == "chunk_end":
            assert typed[body["index"]] == body["text"]


def test_done_carries_the_same_body_as_plain_chat(stream_client):
    """A client that ignores the typing entirely still gets the full /chat contract."""
    client = stream_client(_FakeGraph(reply=sample_reply(LISTING_REPLY)))
    events = _events(client)
    done = [body for name, body in events if name == "done"]
    assert len(done) == 1
    body = done[0]
    assert body["assistant_text"] == LISTING_REPLY
    # The chunk list on `done` is exactly what was typed out, so a non-typing client can
    # render the same bubbles from it in one pass.
    assert body["chunks"] == [b["text"] for name, b in events if name == "chunk_end"]
    for key in ("listings", "sales_phase", "customer_email", "main_prior_messages", "thinking_context"):
        assert key in body


def test_a_failed_turn_ends_the_stream_with_an_error_event(stream_client):
    """The status code is already spent once bytes are flowing, so failure is an event."""
    client = stream_client(_FakeGraph(raises=RuntimeError("boom")))
    events = _events(client)
    # A contained graph failure is still a served turn: /chat answers 200 with the apology,
    # so the stream types that apology out rather than erroring.
    done = [body for name, body in events if name == "done"]
    assert len(done) == 1
    assert "something went wrong" in done[0]["assistant_text"].lower()


def test_malformed_session_id_is_rejected_before_the_stream_opens(stream_client):
    client = stream_client(_FakeGraph())
    response = client.post("/chat/stream", json={"session_id": "not-a-uuid", "message": "hi"})
    assert response.status_code == 422


def test_streaming_can_be_switched_off(stream_client, monkeypatch):
    client = stream_client(_FakeGraph())
    monkeypatch.setattr(routes, "settings", replace(settings, chat_stream_enabled=False))
    response = client.post("/chat/stream", json={"session_id": SESSION, "turn_id": TURN, "message": "hi"})
    assert response.status_code == 404
