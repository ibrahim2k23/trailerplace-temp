from __future__ import annotations

import pytest

from src.graph.nodes import inventory_lookup as inventory_lookup_module
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis


def _lookup_analysis(**inventory_overrides):
    inventory = {
        "is_lookup": True,
        "year": None,
        "make": "Iron Bull Trailers",
        "model_text": "FHG24K",
        "stock_number": None,
        "wants": "price",
        "confidence": "high",
    }
    inventory.update(inventory_overrides)
    return sample_analysis(intent="inventory_lookup", category_mentioned=None, inventory_lookup=inventory)


def _match(url: str, **overrides) -> dict:
    base = {
        "title": "2026 Iron Bull FHG24K Dump Trailer",
        "url": url,
        "stock_number": "12914",
        "make": "Iron Bull Trailers",
        "relevance_score": 100.0,
    }
    base.update(overrides)
    return base


def test_gate_refuses_low_confidence():
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis(confidence="low")
    with pytest.raises(AssertionError):
        inventory_lookup_module.inventory_lookup_node(state)


def test_gate_refuses_not_a_lookup():
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis(is_lookup=False)
    with pytest.raises(AssertionError):
        inventory_lookup_module.inventory_lookup_node(state)


def test_matches_appended_and_deduped(monkeypatch):
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {
            "match_status": "exact",
            "matches": [_match("https://example.com/1"), _match("https://example.com/already-shown")],
            "requested_label": "Iron Bull FHG24K",
        },
    )
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis()
    state["shown_urls"] = ["https://example.com/already-shown"]
    inventory_lookup_module.inventory_lookup_node(state)
    assert state["turn_outcome"]["inventory_match_status"] == "exact"
    assert state["turn_outcome"]["listings"] == [
        _match("https://example.com/1"),
        _match("https://example.com/already-shown"),
    ]
    # Finding a match is not showing it. Respond records what actually reached the customer
    # (nodes/respond.py::_record_shown_listings), so the lookup leaves shown_* alone.
    assert state["shown_listings"] == []
    assert set(state["shown_urls"]) == {"https://example.com/already-shown"}


def test_qualification_state_untouched_by_lookup(monkeypatch):
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {"match_status": "exact", "matches": [_match("u1")], "requested_label": "x"},
    )
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis()
    state["category"] = "Dump"
    state["slots"] = {"haul_material": "dirt"}
    state["brand_preference"] = "Diamond C"
    state["skipped_slots"] = ["haul_weight_lbs"]
    state["qualification_complete"] = False
    state["pending_question_slot"] = "haul_length_ft"
    inventory_lookup_module.inventory_lookup_node(state)
    assert state["category"] == "Dump"
    assert state["slots"] == {"haul_material": "dirt"}
    assert state["brand_preference"] == "Diamond C"
    assert state["skipped_slots"] == ["haul_weight_lbs"]
    assert state["qualification_complete"] is False
    assert state["pending_question_slot"] == "haul_length_ft"


def test_contract_flags_preserved_for_respond_node(monkeypatch):
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {"match_status": "no_exact", "matches": [], "requested_label": "x"},
    )
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis()
    inventory_lookup_module.inventory_lookup_node(state)
    assert state["turn_outcome"]["inventory_lookup_ran"] is True
    assert state["turn_outcome"]["contact_invite_suppressed"] is True


def test_no_matches_no_system_trigger(monkeypatch):
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {"match_status": "no_exact", "matches": [], "requested_label": "x"},
    )
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis()
    inventory_lookup_module.inventory_lookup_node(state)
    assert state["turn_outcome"].get("system_email_triggers", []) == []


def test_matches_produce_results_shown_system_trigger(monkeypatch):
    monkeypatch.setattr(
        inventory_lookup_module,
        "lookup_inventory",
        lambda **kwargs: {
            "match_status": "exact",
            "matches": [_match("u1")],
            "requested_label": "Iron Bull FHG24K",
        },
    )
    state = new_session_state("s1")
    state["turn"] = _lookup_analysis()
    inventory_lookup_module.inventory_lookup_node(state)
    triggers = state["turn_outcome"]["system_email_triggers"]
    assert len(triggers) == 1
    assert triggers[0]["kind"] == "results_shown"
    assert "Iron Bull FHG24K" in triggers[0]["description"]
