from __future__ import annotations

from src.domain.slot_map import can_autofill_slot, equivalent_slots
from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted, say


def apply_with(state, analysis):
    state["turn"] = analysis
    return apply_analysis_to_state(state)


def next_question(state) -> str | None:
    qualification_node(state)
    return state["turn_outcome"].get("next_question")


def test_a_length_is_a_length_whatever_the_category_calls_it():
    assert "haul_length_ft" in equivalent_slots("trailer_length_ft")
    assert "vehicle_length_ft" in equivalent_slots("trailer_length_ft")
    assert "trailer_length_ft" in equivalent_slots("haul_length_ft")
    assert "payload_need" in equivalent_slots("haul_weight_lbs")
    # A combined size question needs more than one number, so a lone length cannot answer it.
    assert not can_autofill_slot("trailer_size")
    assert not can_autofill_slot("cargo_size")
    # Bin size is a yardage — a length carried in from another category is not one.
    assert not can_autofill_slot("bin_size")


def test_category_named_in_the_same_breath_as_its_measurements():
    # "an equipment trailer, 20ft long" put the 20 in trailer_length_ft, leaving Equipment's
    # haul_length_ft empty — so we asked for a length we had just been given.
    state = new_session_state("s1")
    say(state, "I want an equipment trailer, 20ft long, 7000 lbs")
    apply_with(
        state,
        sample_analysis(
            intent="category_selection",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "trailer_length_ft": 20.0, "payload_lbs": 7000.0},
            slot_answers=[],
        ),
    )
    assert state["slots"]["haul_length_ft"] == 20.0
    assert state["slots"]["haul_weight_lbs"] == 7000.0
    assert next_question(state) != "How long is the equipment you need to haul (in feet)?"


def test_measurements_carry_across_a_category_change_under_the_new_name():
    state = new_session_state("s1")
    say(state, "I need a livestock trailer, 24ft")
    apply_with(
        state,
        sample_analysis(intent="category_selection", category_mentioned="Livestock",
                        extracted={**_empty_extracted(), "trailer_length_ft": 24.0}, slot_answers=[]),
    )
    say(state, "actually make it an equipment trailer")
    apply_with(state, sample_analysis(intent="category_change", category_mentioned="Equipment",
                                      extracted=_empty_extracted(), slot_answers=[]))
    say(state, "keep the length")
    apply_with(state, sample_analysis(keep_fields_answer="all", category_mentioned=None,
                                      extracted=_empty_extracted(), slot_answers=[]))
    assert state["category"] == "Equipment"
    assert state["slots"]["haul_length_ft"] == 24.0
    # The only thing still missing is what Equipment actually asks that Livestock never did.
    assert next_question(state) == "What equipment will you be hauling (e.g. skid steer, mini excavator, tractor)?"


def test_features_given_before_a_category_land_in_it_afterwards():
    state = new_session_state("s1")
    say(state, "I need something 22ft long that can carry 9000 lbs")
    apply_with(
        state,
        sample_analysis(intent="feature_request_no_category", category_mentioned=None,
                        extracted={**_empty_extracted(), "trailer_length_ft": 22.0, "payload_lbs": 9000.0},
                        slot_answers=[]),
    )
    assert state["category"] is None
    say(state, "let's go with a car hauler")
    apply_with(state, sample_analysis(intent="category_selection", category_mentioned="Car Hauler",
                                      extracted=_empty_extracted(), slot_answers=[]))
    # Car Hauler calls a length `vehicle_length_ft` — the 22 they already gave answers it.
    assert state["slots"]["vehicle_length_ft"] == 22.0
    assert state["slots"]["haul_weight_lbs"] == 9000.0
    assert next_question(state) == "What type of vehicle will you be hauling (make/model or class)?"


def test_a_skipped_question_stays_skipped():
    # They declined to give a length. A number arriving under another name is not consent to
    # fill it in for them.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["skipped_slots"] = ["haul_length_ft"]
    say(state, "24ft trailer")
    apply_with(state, sample_analysis(category_mentioned="Equipment",
                                      extracted={**_empty_extracted(), "trailer_length_ft": 24.0}, slot_answers=[]))
    assert "haul_length_ft" not in state["slots"]
    assert "haul_length_ft" in state["skipped_slots"]
