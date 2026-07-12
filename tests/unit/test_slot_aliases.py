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


def test_the_cargo_question_is_one_question_under_many_names():
    assert "haul_material" in equivalent_slots("haul_item")
    assert "vehicle_type" in equivalent_slots("haul_item")
    assert "use_case" in equivalent_slots("haul_item")
    assert "haul_item" in equivalent_slots("haul_material")
    # It never leaks across groups: cargo is not a measurement.
    assert "trailer_length_ft" not in equivalent_slots("haul_item")


def test_cargo_given_before_a_category_answers_that_categorys_cargo_question():
    # Seen live: "a trailer to haul random things, ~6000 lbs" -> haul_item. Then they pick
    # Dump, whose cargo question is `haul_material`, and we asked "what material will you be
    # hauling?" — a question they had just answered.
    state = new_session_state("s1")
    say(state, "a trailer to haul random things, around 6000 lbs")
    apply_with(
        state,
        sample_analysis(intent="feature_request_no_category", category_mentioned=None,
                        extracted={**_empty_extracted(), "haul_item": "random things", "payload_lbs": 6000.0},
                        slot_answers=[]),
    )
    say(state, "I think I'd like a dump trailer")
    apply_with(state, sample_analysis(intent="category_selection", category_mentioned="Dump",
                                      extracted=_empty_extracted(), slot_answers=[]))
    assert state["slots"]["haul_material"] == "random things"
    assert next_question(state) is None  # nothing left to ask — Dump is fully qualified
    assert state["qualification_complete"] is True


def test_cargo_maps_to_vehicle_type_and_use_case_too():
    car = new_session_state("s1")
    say(car, "I need to haul my Mustang, 3500 lbs, 16ft")
    apply_with(car, sample_analysis(intent="feature_request_no_category", category_mentioned=None,
                                    extracted={**_empty_extracted(), "haul_item": "my Mustang",
                                               "payload_lbs": 3500.0, "trailer_length_ft": 16.0},
                                    slot_answers=[]))
    say(car, "a car hauler then")
    apply_with(car, sample_analysis(intent="category_selection", category_mentioned="Car Hauler",
                                    extracted=_empty_extracted(), slot_answers=[]))
    assert car["slots"]["vehicle_type"] == "my Mustang"
    assert car["slots"]["vehicle_length_ft"] == 16.0
    assert next_question(car) is None

    enclosed = new_session_state("s2")
    say(enclosed, "something to carry my woodworking tools")
    apply_with(enclosed, sample_analysis(intent="feature_request_no_category", category_mentioned=None,
                                         extracted={**_empty_extracted(), "haul_item": "woodworking tools"},
                                         slot_answers=[]))
    say(enclosed, "an enclosed trailer")
    apply_with(enclosed, sample_analysis(intent="category_selection", category_mentioned="Enclosed",
                                         extracted=_empty_extracted(), slot_answers=[]))
    assert enclosed["slots"]["use_case"] == "woodworking tools"
    # A cargo description does not answer a SIZE question — that one still gets asked.
    assert next_question(enclosed) == "What's the rough size of the cargo you need to fit (length × width × height)?"


def test_a_hitch_named_before_the_category_answers_equipments_hitch_question():
    state = new_session_state("s1")
    say(state, "something gooseneck, 20ft, 9000 lbs")
    apply_with(state, sample_analysis(intent="feature_request_no_category", category_mentioned=None,
                                      extracted={**_empty_extracted(), "hitch_type": ["Gooseneck"],
                                                 "trailer_length_ft": 20.0, "payload_lbs": 9000.0},
                                      slot_answers=[]))
    say(state, "make it an equipment trailer for my skid steer")
    apply_with(state, sample_analysis(intent="category_selection", category_mentioned="Equipment",
                                      extracted={**_empty_extracted(), "haul_item": "skid steer"}, slot_answers=[]))
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["slots"]["haul_length_ft"] == 20.0
    # Equipment asks for hitch — we already have it, so it is not asked again.
    assert next_question(state) is None


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
