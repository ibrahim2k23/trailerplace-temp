"""A negative measurement is refused and re-asked. Zero and vague answers are NOT touched.

The three cases are deliberately different:
  negative -> refused, nothing stored, question asked again with a reason
  zero     -> no preference (stored null, never re-asked) - same as a vague answer
  vague    -> unchanged, exactly as before
"""
from __future__ import annotations

import pytest

from src.domain.slot_map import is_impossible_measurement, normalize_answer_for_slot
from src.graph.state import new_session_state
from src.llm.respond import _decision_lines
from tests.unit.llm_helpers import sample_analysis

NEGATIVES = [
    ("Dump", "haul_weight_lbs", "-500 lbs"),
    ("Equipment", "haul_length_ft", "-3 ft"),
    ("Utility", "axle_capacity_lbs", "-7000"),
    ("Car Hauler", "vehicle_length_ft", "-12"),
]

ZEROS = [
    ("Dump", "haul_weight_lbs", "0 lbs"),
    ("Equipment", "haul_length_ft", "0"),
    ("Equipment", "haul_length_ft", "0 ft"),
]

VAGUE = [
    ("Dump", "haul_weight_lbs", "no preference"),
    ("Dump", "haul_weight_lbs", "whatever you'd recommend is fine"),
    ("Utility", "trailer_size", "as big as you have"),
]

# Ranges are the reason the sign check cannot be a plain "-" search.
VALID = [
    ("Dump", "haul_weight_lbs", "5,000-7,000 lbs", 5000.0),
    ("Equipment", "haul_length_ft", "15-18 ft", 15.0),
    ("Dump", "haul_weight_lbs", "2000 lbs", 2000.0),
]


@pytest.mark.parametrize("category,slot,answer", NEGATIVES)
def test_a_negative_is_refused(category, slot, answer):
    assert is_impossible_measurement(category, slot, answer) is True


@pytest.mark.parametrize("category,slot,answer", ZEROS)
def test_a_zero_is_no_preference_not_a_refusal(category, slot, answer):
    """Zero must behave exactly like a vague answer: stored null, question closed."""
    assert is_impossible_measurement(category, slot, answer) is False
    assert normalize_answer_for_slot(category, slot, answer) is None


@pytest.mark.parametrize("category,slot,answer", VAGUE)
def test_vague_answers_are_untouched(category, slot, answer):
    assert is_impossible_measurement(category, slot, answer) is False
    assert normalize_answer_for_slot(category, slot, answer) is None


@pytest.mark.parametrize("category,slot,answer,expected", VALID)
def test_valid_answers_and_ranges_are_untouched(category, slot, answer, expected):
    """A range dash is not a minus sign - "5000-7000" must not read as -7000."""
    assert is_impossible_measurement(category, slot, answer) is False
    assert normalize_answer_for_slot(category, slot, answer) == expected


def test_free_text_slots_are_never_refused():
    """Only measurements have an ordering; "a -5 ton digger" is a description."""
    assert is_impossible_measurement("Equipment", "haul_item", "a -5 ton digger") is False


def test_a_refusal_reaches_the_reply_as_an_order():
    """Refusing silently would re-ask the same question with no explanation, which reads as
    not having listened."""
    state = new_session_state("s1")
    state["category"] = "Dump"
    outcome = {
        "rejected_answers": [{"slot": "haul_weight_lbs", "raw_answer": "-500 lbs"}],
        "next_question": "What's the rough haul weight per load?",
    }
    lines = "\n".join(_decision_lines(state, sample_analysis(), outcome))
    assert '"-500 lbs"' in lines, "their own words must be quoted back so a typo is obvious"
    assert "NEGATIVE MEASUREMENT" in lines
    assert "NOT recorded" in lines
    # The way out has to be offered, or a customer with no real figure is stuck.
    assert "skip" in lines.lower()
    # And it must not turn into a telling-off.
    assert "do not blame them" in lines.lower()
