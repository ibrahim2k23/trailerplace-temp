from __future__ import annotations

import pytest

from src.graph.nodes import search as search_module
from src.graph.state import new_session_state


def _fake_search(calls, results_by_call):
    """FakePinecone: records each search_pinecone_listings call and returns queued results."""

    def _search(**kwargs):
        calls.append(kwargs)
        return results_by_call.pop(0) if results_by_call else []

    return _search


def _listing(url: str, **overrides) -> dict:
    base = {
        "title": "2024 Diamond C Dump",
        "condition": "New",
        "price": 12000.0,
        "category": "Dump",
        "make": "Diamond C",
        "color": "Black",
        "hitch_type": "Bumper Pull",
        "year": 2024,
        "length": "14",
        "width": "7",
        "axles": "2",
        "gvwr": "14000",
        "payload_capacity": "10000",
        "material": "Steel",
        "floor": "Steel",
        "url": url,
        "relevance_score": 0.9,
    }
    base.update(overrides)
    return base


def test_gate_refuses_when_qualification_incomplete():
    state = new_session_state("s1")
    state["qualification_complete"] = False
    with pytest.raises(AssertionError):
        search_module.search_node(state)


def test_metadata_filters_built_from_slots(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search(calls, [[_listing("u1")]]))
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["qualification_complete"] = True
    state["slots"] = {
        "haul_length_ft": 14.0,
        "item_or_trailer_width_ft": 7.0,
        "haul_weight_lbs": 7000.0,
        "hitch_type": ["Bumper Pull"],
    }
    search_module.search_node(state)
    assert len(calls) == 1
    filters = calls[0]["metadata_filters"]
    assert filters["length_ft"] == 14.0
    assert filters["width_ft"] == 7.0
    assert filters["payload_lbs"] == 7000.0
    assert filters["hitch_type"] == "Bumper Pull"


def test_roll_off_bin_size_maps_to_length_ft(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search(calls, [[]]))
    state = new_session_state("s1")
    state["category"] = "Roll Off"
    state["qualification_complete"] = True
    state["slots"] = {"bin_size": "15 yd"}
    search_module.search_node(state)
    assert calls[0]["metadata_filters"]["length_ft"] == 15.0


def test_brand_preference_becomes_make_filter(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search(calls, [[_listing("u1")]]))
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    state["brand_preference"] = "Diamond C"
    search_module.search_node(state)
    assert calls[0]["metadata_filters"]["make"] == "Diamond C"


def test_zero_result_brand_fallback_reruns_without_make(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        search_module,
        "search_pinecone_listings",
        _fake_search(calls, [[], [_listing("u1"), _listing("u2")]]),
    )
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    state["brand_preference"] = "Obscure Brand"
    search_module.search_node(state)
    assert len(calls) == 2
    assert "make" in calls[0]["metadata_filters"]
    assert "make" not in calls[1]["metadata_filters"]
    assert state["turn_outcome"]["brand_relaxed"] is True
    assert state["turn_outcome"]["result_count"] == 2


def test_already_shown_urls_passed_through_for_dedupe(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search(calls, [[_listing("u2")]]))
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    state["shown_urls"] = ["u1"]
    search_module.search_node(state)
    assert calls[0]["already_shown_urls"] == ["u1"]
    # Finding a listing is not showing it: respond records what actually reached the
    # customer (nodes/respond.py::_record_shown_listings), so search leaves this alone.
    assert set(state["shown_urls"]) == {"u1"}
    assert [item["url"] for item in state["turn_outcome"]["listings"]] == ["u2"]


def test_result_shape_matches_app_listing_parser(monkeypatch):
    """Mirrors app.py's TrailerListing construction (app.py:1481-1509) so a real
    search result never silently drops a card."""
    from src.models import TrailerListing

    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search([], [[_listing("u1")]]))
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    search_module.search_node(state)
    d = state["turn_outcome"]["listings"][0]

    listing = TrailerListing(
        listing_id=str(d.get("url") or d.get("title") or ""),
        title=str(d.get("title") or ""),
        condition=str(d.get("condition") or "New"),
        price=d.get("price"),
        price_display=str(d.get("price") or "") or None,
        payments_from=None,
        category_subcategory=str(d.get("category") or ""),
        make=str(d.get("make") or ""),
        color=str(d.get("color") or ""),
        hitch_type=d.get("hitch_type"),
        year=d.get("year"),
        length=d.get("length"),
        width=d.get("width"),
        axles=d.get("axles"),
        gvwr=d.get("gvwr"),
        payload_capacity=d.get("payload_capacity"),
        trailer_material=d.get("material"),
        floor=d.get("floor"),
        url=str(d.get("url") or ""),
        score=d.get("relevance_score"),
    )
    assert listing.url == "u1"
    assert listing.trailer_material == "Steel"


def test_search_produces_results_shown_system_trigger(monkeypatch):
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search([], [[_listing("u1")]]))
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    search_module.search_node(state)
    triggers = state["turn_outcome"]["system_email_triggers"]
    assert len(triggers) == 1
    assert triggers[0]["kind"] == "results_shown"
    assert "Dump" in triggers[0]["description"]


def test_no_results_no_system_trigger(monkeypatch):
    monkeypatch.setattr(search_module, "search_pinecone_listings", _fake_search([], [[]]))
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["qualification_complete"] = True
    search_module.search_node(state)
    assert state["turn_outcome"].get("system_email_triggers", []) == []
