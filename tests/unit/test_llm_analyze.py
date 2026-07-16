from __future__ import annotations

from tests.conftest import FakeLLM

from src.llm.analyze import analyze_turn, build_analyze_prompt, normalize_analysis_values
from src.llm.schemas import TurnAnalysis
from tests.unit.llm_helpers import sample_analysis


def test_analyze_prompt_contains_required_blocks():
    state = {
        "category": "Dump",
        "messages": [{"role": "user", "content": "what about 83 inches wide?"}],
        "pending_question_slot": "haul_weight_lbs",
        "pending_question_repeats": 1,
        "slots": {"trailer_length_ft": 14, "hitch_type": None},
        "slot_sources": {"trailer_length_ft": "user"},
        "skipped_slots": ["payload_lbs"],
        "customer_name": "John",
        "customer_email": None,
        "customer_phone": "555",
        "shown_listings": [{"title": "Iron Bull Dump 14 ft"}],
    }
    system, messages = build_analyze_prompt(state)
    assert "=== TRAILER CATEGORIES ===" in system
    assert "=== KNOWN MAKES/BRANDS ===" in system
    assert 'Pending question: "What\'s the rough haul weight per load?"' in system
    assert "already re-asked 1 time(s)" in system
    assert "Already collected" in system and "trailer_length_ft" in system
    assert "No-preference slots (stored null): ['hitch_type']" in system
    assert "Category notes" in system
    assert "Iron Bull Dump 14 ft" in system
    assert "Contact: name=John" in system
    assert "HAUL CLASSIFICATION" in system
    assert "AxB = width x length" in system
    assert "83 inches" in system and "6.92" in system
    assert '"3 ft sides" -> trailer_height_ft=3' in system
    assert '"3 inch walls" ->' in system and "trailer_height_ft=0.25" in system
    assert '"20 footer" or "20-footer" ->' in system and "trailer_length_ft=20" in system
    assert messages == state["messages"]


def test_analyze_prompt_carries_the_critical_rules_section():
    system, _ = build_analyze_prompt({"messages": [{"role": "user", "content": "hi"}]})
    assert "CRITICAL RULES - THESE OUTRANK EVERYTHING ELSE" in system
    assert "CATEGORY CHANGES ARE NEVER MISSED" in system
    assert "A CATEGORY CHANGE RESTARTS QUALIFICATION FROM ZERO" in system
    assert "YOUR LABELS ARE THE SEARCH GATE" in system
    assert "VAGUE MESSAGES MEAN WHAT OUR LAST MESSAGE MAKES THEM MEAN" in system


def test_analyze_prompt_contains_inventory_lookup_guards():
    system, _ = build_analyze_prompt({"messages": [{"role": "user", "content": "stock 12345"}]})
    assert "NEVER a stock number: weights" in system
    assert "FILL THE BLOCK REGARDLESS OF INTENT" in system
    assert "PHRASING NEVER MATTERS" in system
    assert "Coexistence rules" in system
    assert "A lookup NEVER changes the selected category" in system


def test_analyze_turn_uses_fake_llm_and_normalizes_values():
    base = sample_analysis()
    queued = base.model_copy(update={"extracted": base.extracted.model_copy(update={"trailer_width_ft": "83 inches"})})
    llm = FakeLLM([queued])
    result = analyze_turn(llm, {"category": "Dump", "messages": [{"role": "user", "content": "83 inches"}]})
    assert isinstance(result, TurnAnalysis)
    assert round(float(result.extracted.trailer_width_ft), 2) == 6.92
    assert llm.calls[0]["schema"] is TurnAnalysis


def test_safety_net_numeric_passthrough():
    analysis = sample_analysis(extracted={**sample_analysis().extracted.model_dump(), "trailer_width_ft": 6.92})
    normalized = normalize_analysis_values(analysis, "Dump")
    assert normalized.extracted.trailer_width_ft == 6.92


def test_normalize_leaves_slot_answers_untouched():
    # Slot-answer -> metadata-target mapping now lives in apply_analysis; the safety
    # net must pass slot answers through verbatim (no string re-encoding here).
    analysis = sample_analysis(
        slot_answers=[
            {"slot_name": "bin_size", "raw_answer": "15 yd"},
            {"slot_name": "cargo_size", "raw_answer": "83 inches"},
        ]
    )
    normalized = normalize_analysis_values(analysis, "Roll Off")
    assert normalized.slot_answers[0].raw_answer == "15 yd"
    assert normalized.slot_answers[1].raw_answer == "83 inches"
