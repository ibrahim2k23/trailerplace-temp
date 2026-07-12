from __future__ import annotations

import pytest

from src.graph.nodes import email_actions as email_actions_module
from src.graph.nodes.email_actions import email_actions_node
from tests.conftest import FakeEmailSender
from tests.unit.llm_helpers import sample_analysis


@pytest.fixture(autouse=True)
def _persistence_off_with_fake_sender(monkeypatch):
    """Run the node in persistence-off mode so it sends directly via a fake sender."""
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    fake = FakeEmailSender()
    monkeypatch.setattr(email_actions_module.email_sender, "send_email", fake.send_email)
    return fake


def _state(**over):
    state = {
        "session_id": "sess-1",
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_declined": False,
        "contact_followup_pending": None,
        "pending_email_actions": [],
        "shown_listings": [],
        "turn": None,
        "turn_outcome": {"canned_keys": [], "emails_sent": [], "system_email_triggers": []},
    }
    state.update(over)
    return state


def _turn(email_triggers):
    return sample_analysis(email_triggers=email_triggers)


def _trigger(kind, **over):
    base = {"kind": kind, "faq_key": None, "listing_reference": None, "description": f"{kind} request"}
    base.update(over)
    return base


def test_complete_contact_sends_immediately(_persistence_off_with_fake_sender):
    state = _state(customer_name="Jane", customer_phone="555-1", turn=_turn([_trigger("escalation")]))
    result = email_actions_node(state)
    outcome = result["turn_outcome"]
    assert outcome["emails_sent"] == ["Escalation"]
    assert "escalation" in outcome["canned_keys"]
    assert len(outcome["outbox_events"]) == 1
    assert result["pending_email_actions"] == []
    assert len(_persistence_off_with_fake_sender.sent) == 1


def test_partial_contact_stashes_and_asks(_persistence_off_with_fake_sender):
    state = _state(customer_name="Jane", turn=_turn([_trigger("faq", faq_key="financing")]))
    result = email_actions_node(state)
    outcome = result["turn_outcome"]
    # FAQ answer still delivered even though the email waits.
    assert "financing" in outcome["canned_keys"]
    assert outcome["emails_sent"] == []
    # We name every piece still missing, so one ask covers them all.
    assert result["contact_followup_pending"] == "email or phone"
    assert len(result["pending_email_actions"]) == 1
    assert "deferred" in outcome["email_status"]
    assert _persistence_off_with_fake_sender.sent == []


def test_stashed_then_provided_sends(_persistence_off_with_fake_sender):
    # Turn 1: partial contact stashes the FAQ email.
    state = _state(customer_name="Jane", turn=_turn([_trigger("faq", faq_key="financing")]))
    email_actions_node(state)
    assert len(state["pending_email_actions"]) == 1
    # Turn 2: contact method arrives, no new trigger.
    state["customer_phone"] = "555-1"
    state["turn"] = _turn([])
    state["turn_outcome"] = {"canned_keys": [], "emails_sent": [], "system_email_triggers": []}
    result = email_actions_node(state)
    assert result["turn_outcome"]["emails_sent"] == ["FAQ – financing"]
    assert result["pending_email_actions"] == []


def test_declined_drops_stashed_and_never_asks(_persistence_off_with_fake_sender):
    state = _state(
        customer_name="Jane",
        contact_declined=True,
        pending_email_actions=[{"kind": "escalation", "reason": "Escalation", "event_type": "escalation_alert", "canned_key": "escalation", "description": "x", "is_system": False}],
        turn=_turn([]),
    )
    result = email_actions_node(state)
    assert result["pending_email_actions"] == []
    assert result["contact_followup_pending"] is None
    assert result["turn_outcome"]["email_status"] == "skipped (user declined)"
    assert _persistence_off_with_fake_sender.sent == []


def test_listing_interest_selected_variant():
    state = _state(
        customer_name="Jane",
        customer_email="j@x.com",
        shown_listings=[{"title": "2026 Dump", "url": "https://x/1"}, {"title": "Iron Bull", "url": "https://x/2"}],
        turn=_turn([_trigger("listing_interest", listing_reference=2)]),
    )
    result = email_actions_node(state)
    outcome = result["turn_outcome"]
    assert "listing_interest_selected" in outcome["canned_keys"]
    body = outcome["outbox_events"][0]["payload"]["body"]
    assert "Iron Bull" in body and "https://x/2" in body


def test_listing_interest_unselected_and_fallback():
    # Interest but no valid reference, listings exist -> unselected.
    state = _state(customer_name="Jane", customer_email="j@x.com", shown_listings=[{"title": "A", "url": "u"}], turn=_turn([_trigger("listing_interest")]))
    assert "listing_interest_unselected" in email_actions_node(state)["turn_outcome"]["canned_keys"]
    # No listings at all -> fallback.
    state2 = _state(customer_name="Jane", customer_email="j@x.com", shown_listings=[], turn=_turn([_trigger("listing_interest")]))
    assert "listing_interest_fallback" in email_actions_node(state2)["turn_outcome"]["canned_keys"]


def test_multi_trigger_fires_all_in_order():
    state = _state(
        customer_name="Jane",
        customer_email="j@x.com",
        turn=_turn([_trigger("faq", faq_key="financing"), _trigger("escalation")]),
    )
    outcome = email_actions_node(state)["turn_outcome"]
    assert outcome["emails_sent"] == ["FAQ – financing", "Escalation"]
    assert "financing" in outcome["canned_keys"] and "escalation" in outcome["canned_keys"]
    assert len(outcome["outbox_events"]) == 2


def test_multi_trigger_gate_blocked_then_sent():
    state = _state(turn=_turn([_trigger("faq", faq_key="financing"), _trigger("escalation")]))
    email_actions_node(state)
    assert len(state["pending_email_actions"]) == 2
    state["customer_name"] = "Jane"
    state["customer_phone"] = "555-1"
    state["turn"] = _turn([])
    state["turn_outcome"] = {"canned_keys": [], "emails_sent": [], "system_email_triggers": []}
    outcome = email_actions_node(state)["turn_outcome"]
    assert outcome["emails_sent"] == ["FAQ – financing", "Escalation"]


def test_system_results_shown_sent_but_not_in_canned():
    state = _state(customer_name="Jane", customer_phone="555-1")
    state["turn"] = _turn([])
    state["turn_outcome"]["system_email_triggers"] = [{"kind": "results_shown", "description": "Pinecone search — 3 results — Dump"}]
    outcome = email_actions_node(state)["turn_outcome"]
    assert outcome["emails_sent"] == ["Results Shown to User"]
    assert outcome["canned_keys"] == []


def test_system_alert_stashed_silently_without_followup():
    state = _state()  # no contact
    state["turn"] = _turn([])
    state["turn_outcome"]["system_email_triggers"] = [{"kind": "results_shown", "description": "d"}]
    result = email_actions_node(state)
    assert len(result["pending_email_actions"]) == 1
    # System alerts never trigger a contact ask.
    assert result["contact_followup_pending"] is None
    assert result["turn_outcome"].get("email_status") is None


def test_unanswered_question_alert_gated_like_others():
    state = _state(customer_name="Jane", customer_email="j@x.com")
    state["turn"] = _turn([])
    state["turn_outcome"]["system_email_triggers"] = [{"kind": "unanswered_question", "description": "Skipped: haul_material"}]
    outcome = email_actions_node(state)["turn_outcome"]
    assert outcome["emails_sent"] == ["Unanswered Question"]


def test_double_pass_idempotent():
    state = _state(customer_name="Jane", customer_phone="555-1", turn=_turn([_trigger("escalation")]))
    # Pass 1: escalation processed and consumed.
    email_actions_node(state)
    assert state["turn"].email_triggers == []
    assert state["turn_outcome"]["emails_sent"] == ["Escalation"]
    # Search adds a results alert after pass 1.
    state["turn_outcome"]["system_email_triggers"] = [{"kind": "results_shown", "description": "d"}]
    # Pass 2: only the new results alert is sent, escalation not re-sent.
    email_actions_node(state)
    assert state["turn_outcome"]["emails_sent"] == ["Escalation", "Results Shown to User"]
    assert len(state["turn_outcome"]["outbox_events"]) == 2
