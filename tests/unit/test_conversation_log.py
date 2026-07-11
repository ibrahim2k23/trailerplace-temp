"""Human-readable conversation reasoning + state log (src/conversation_log.py)."""
from __future__ import annotations

import logging

from src import conversation_log
from tests.unit.llm_helpers import sample_analysis


def _sample_state(**overrides):
    state = {
        "category": "Dump",
        "clarification_key": None,
        "slots": {"haul_material": "gravel", "haul_weight_lbs": 6000.0},
        "skipped_slots": [],
        "qualification_complete": False,
        "pending_question_slot": "dump_mechanism",
        "pending_question_repeats": 0,
        "brand_preference": None,
        "non_metadata_features": [],
        "customer_name": "John",
        "customer_email": None,
        "customer_phone": "555-1234",
        "contact_declined": False,
        "pending_category_change": None,
        "pending_email_actions": [],
        "contact_followup_pending": None,
        "shown_listings": [],
        "shown_urls": [],
    }
    state.update(overrides)
    return state


def test_full_turn_block_contains_every_required_section(caplog):
    analysis = sample_analysis(intent="qualification_answer", category_mentioned="Dump")
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="sess-123",
            turn_id="turn-456",
            user_message="7x14 dump trailer for dirt",
            analysis=analysis,
            assistant_text="Got it, a 7x14 Dump trailer for dirt.",
            cited_listing_urls=["https://example.com/1"],
            listings_returned=1,
            turn_outcome={"search_ran": True, "result_count": 1, "emails_sent": []},
            state=_sample_state(),
        )
    block = caplog.records[0].message
    # Session/turn identity — the user's #1 ask.
    assert "session=sess-123" in block
    assert "turn=turn-456" in block
    # Raw user message and assistant reply.
    assert "USER: 7x14 dump trailer for dirt" in block
    assert "ASSISTANT: Got it, a 7x14 Dump trailer for dirt." in block
    # Reasoning (Analyze LLM output).
    assert "intent: qualification_answer" in block
    assert "category_mentioned: Dump" in block
    # State after the turn.
    assert "STATE AFTER TURN:" in block
    assert "category: Dump" in block
    assert "haul_material" in block
    assert "pending_question: dump_mechanism" in block
    # Tools.
    assert "TOOLS FIRED: search(results=1" in block


def test_block_shows_no_analysis_placeholder_for_a_replay(caplog):
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="hi", analysis=None,
            assistant_text="hi there", cited_listing_urls=None, listings_returned=0,
            turn_outcome={}, state=_sample_state(),
        )
    block = caplog.records[0].message
    assert "no analysis available" in block
    assert "TOOLS FIRED: -" in block


def test_block_shows_error_instead_of_assistant_text(caplog):
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="hi", analysis=None,
            assistant_text=None, cited_listing_urls=None, listings_returned=0,
            turn_outcome={}, state=_sample_state(), error="graph run exceeded 150.0s",
        )
    block = caplog.records[0].message
    assert "ERROR: graph run exceeded 150.0s" in block
    assert "ASSISTANT:" not in block


def test_haul_classification_and_inventory_lookup_lines_appear_when_relevant(caplog):
    analysis = sample_analysis(
        haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
        inventory_lookup={"is_lookup": True, "year": 2026, "make": "Iron Bull Trailers", "model_text": "fhg 24k", "stock_number": None, "wants": "price", "confidence": "high"},
    )
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="msg", analysis=analysis,
            assistant_text="reply", cited_listing_urls=[], listings_returned=0,
            turn_outcome={}, state=_sample_state(),
        )
    block = caplog.records[0].message
    assert "needs_width=True" in block
    assert "matched=tractor" in block
    assert "inventory_lookup: year=2026 make=Iron Bull Trailers model=fhg 24k" in block


def test_email_triggers_and_interruption_lines(caplog):
    analysis = sample_analysis(
        email_triggers=[{"kind": "faq", "faq_key": "financing", "listing_reference": None, "description": "asked about financing"}],
        answered_current_question=False,
        user_question_to_answer="what's the difference between tandem and single axles?",
    )
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="msg", analysis=analysis,
            assistant_text="reply", cited_listing_urls=[], listings_returned=0,
            turn_outcome={}, state=_sample_state(),
        )
    block = caplog.records[0].message
    assert "email_triggers: faq[financing]" in block
    assert "interruption:" in block
    assert "tandem and single axles" in block


def test_tools_fired_reflects_search_lookup_and_email(caplog):
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="msg", analysis=None,
            assistant_text="reply", cited_listing_urls=[], listings_returned=0,
            turn_outcome={
                "inventory_lookup_ran": True,
                "inventory_match_status": "exact",
                "emails_sent": ["Escalation", "FAQ – financing"],
                "email_status": "sent: ['Escalation']",
            },
            state=_sample_state(),
        )
    block = caplog.records[0].message
    assert "inventory_lookup(status=exact)" in block
    assert "email(Escalation, FAQ – financing)" in block
    assert "EMAIL STATUS: sent: ['Escalation']" in block


def test_never_raises_on_a_malformed_analysis_object(caplog):
    class _Broken:
        # Accessing .extracted raises -- the logger must swallow it, not crash the turn.
        @property
        def extracted(self):
            raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="msg", analysis=_Broken(),
            assistant_text="reply", cited_listing_urls=[], listings_returned=0,
            turn_outcome={}, state=_sample_state(),
        )  # must not raise
    # The exception is itself logged (at ERROR, via logger.exception).
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_empty_and_none_fields_render_as_dash(caplog):
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        conversation_log.log_conversation_turn(
            session_id="s", turn_id="t", user_message="msg", analysis=None,
            assistant_text="reply", cited_listing_urls=[], listings_returned=0,
            turn_outcome={}, state=_sample_state(category=None, brand_preference=None, pending_question_slot=None),
        )
    block = caplog.records[0].message
    assert "category: -" in block
    assert "pending_question: -" in block
