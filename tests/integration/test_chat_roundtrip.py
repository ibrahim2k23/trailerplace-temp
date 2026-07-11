from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api import routes
from src.graph.nodes import inventory_lookup as inventory_lookup_module
from src.graph.state import _sessions
from tests.conftest import FakeLLM
from tests.unit.llm_helpers import sample_analysis, sample_reply


def _lookup_match(url: str) -> dict:
    return {
        "title": "2026 Iron Bull FHG24K Dump Trailer",
        "url": url,
        "stock_number": "12914",
        "make": "Iron Bull Trailers",
        "price": "$18,500",
        "relevance_score": 100.0,
    }


def test_chat_two_turn_roundtrip_persistence_off(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    _sessions.clear()
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(category_mentioned="Dump", slot_answers=[{"slot_name": "haul_material", "raw_answer": "dirt"}]),
                sample_reply("What rough haul weight per load?"),
                sample_analysis(intent="qualification_answer", category_mentioned=None, slot_answers=[{"slot_name": "haul_weight_lbs", "raw_answer": "2 tons"}]),
                sample_reply("Thanks, I can use that."),
            ]
        )
    )
    client = TestClient(create_app())
    first = client.post("/chat", json={"session_id": "11111111-1111-1111-1111-111111111111", "turn_id": "22222222-2222-2222-2222-222222222222", "message": "dump for dirt"})
    assert first.status_code == 200
    assert first.json()["sales_phase"] == "main"
    second = client.post("/chat", json={"session_id": "11111111-1111-1111-1111-111111111111", "turn_id": "33333333-3333-3333-3333-333333333333", "message": "2 tons"})
    assert second.status_code == 200
    assert _sessions["11111111-1111-1111-1111-111111111111"]["slots"]["haul_material"] == "dirt"


def test_session_restore_text_only_and_reset(monkeypatch):
    monkeypatch.setattr("src.conversation_store.restore_session", lambda sid: {"exists": True, "messages": [{"role": "assistant", "content": "hi", "listings": [{"url": "x"}]}], "sales_phase": "main"})
    closed = []
    monkeypatch.setattr("src.conversation_store.close_session", lambda sid: closed.append(sid))
    _sessions["s1"] = {"session_id": "s1"}
    client = TestClient(create_app())
    restored = client.get("/session/s1").json()
    assert restored["messages"][0]["listings"] is None
    reset = client.post("/session/reset", json={"session_id": "s1"})
    assert reset.status_code == 200
    assert "s1" not in _sessions
    assert closed == ["s1"]


def test_width_question_roundtrip(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    _sessions.clear()
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(
                    category_mentioned="Equipment",
                    extracted={**sample_analysis().extracted.model_dump(), "haul_item": "tractor", "payload_lbs": 7000},
                    haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
                    slot_answers=[{"slot_name": "haul_length_ft", "raw_answer": "14 ft"}, {"slot_name": "hitch_type", "raw_answer": "bumper pull"}],
                ),
                sample_reply("About how wide is that tractor?"),
                sample_analysis(intent="qualification_answer", category_mentioned=None, slot_answers=[{"slot_name": "item_or_trailer_width_ft", "raw_answer": "6 ft"}]),
                sample_reply("Great, I have what I need."),
            ]
        )
    )
    client = TestClient(create_app())
    sid = "44444444-4444-4444-4444-444444444444"
    first = client.post("/chat", json={"session_id": sid, "turn_id": "55555555-5555-5555-5555-555555555555", "message": "equipment for tractor"})
    assert first.status_code == 200
    assert _sessions[sid]["pending_question_slot"] == "item_or_trailer_width_ft"
    second = client.post("/chat", json={"session_id": sid, "turn_id": "66666666-6666-6666-6666-666666666666", "message": "6 ft"})
    assert second.status_code == 200
    assert _sessions[sid]["slots"]["item_or_trailer_width_ft"] == "6 ft"  # category slot: raw answer
    assert _sessions[sid]["slots"]["width_ft"] == 6.0  # metadata target: parsed number


def test_first_turn_inventory_lookup_defers_contact_invite(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {
            "match_status": "exact",
            "matches": [_lookup_match("https://example.com/1")],
            "requested_label": "Iron Bull FHG24K",
        },
    )
    _sessions.clear()
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(
                    intent="inventory_lookup",
                    category_mentioned=None,
                    inventory_lookup={
                        "is_lookup": True,
                        "year": None,
                        "make": "Iron Bull Trailers",
                        "model_text": "FHG24K",
                        "stock_number": None,
                        "wants": "price",
                        "confidence": "high",
                    },
                ),
                sample_reply("Yes, here's the FHG24K.", urls=["https://example.com/1"]),
                sample_analysis(intent="smalltalk_other", category_mentioned=None),
                sample_reply("Happy to help further."),
            ]
        )
    )
    client = TestClient(create_app())
    sid = "77777777-7777-7777-7777-777777777777"
    first = client.post("/chat", json={"session_id": sid, "turn_id": "88888888-8888-8888-8888-888888888888", "message": "how much is the iron bull fhg24k"})
    assert first.status_code == 200
    assert first.json()["listings"] == [_lookup_match("https://example.com/1")]
    # First-turn lookup defers the contact invite (Locked Decision: inventory lookup exception).
    assert _sessions[sid]["contact_prompted_initial"] is False

    second = client.post("/chat", json={"session_id": sid, "turn_id": "99999999-9999-9999-9999-999999999999", "message": "thanks"})
    assert second.status_code == 200
    assert _sessions[sid]["contact_prompted_initial"] is True


def test_show_only_llm_mentioned_cards_filters_uncited_listings(monkeypatch):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {
            "match_status": "ambiguous",
            "matches": [_lookup_match("https://example.com/1"), _lookup_match("https://example.com/2")],
            "requested_label": "Iron Bull FHG",
        },
    )
    _sessions.clear()
    routes.set_graph_client(
        FakeLLM(
            [
                sample_analysis(
                    intent="inventory_lookup",
                    category_mentioned=None,
                    inventory_lookup={
                        "is_lookup": True,
                        "year": None,
                        "make": "Iron Bull Trailers",
                        "model_text": "FHG",
                        "stock_number": None,
                        "wants": "price",
                        "confidence": "high",
                    },
                ),
                # Only cites one of the two candidates shown.
                sample_reply("Did you mean the FHG24K?", urls=["https://example.com/1"]),
            ]
        )
    )
    client = TestClient(create_app())
    sid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    response = client.post("/chat", json={"session_id": sid, "turn_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "message": "do you have the FHG"})
    assert response.status_code == 200
    listings = response.json()["listings"]
    assert [item["url"] for item in listings] == ["https://example.com/1"]
