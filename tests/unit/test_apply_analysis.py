from __future__ import annotations

from src.graph.apply_analysis import INJECTED_WIDTH_SLOT, apply_analysis_to_state
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis


def apply_with(state, analysis):
    state["turn"] = analysis
    return apply_analysis_to_state(state)


def _empty_extracted() -> dict:
    """A sample_analysis extracted block with no values, so it adds nothing on apply."""
    return {
        **sample_analysis().extracted.model_dump(),
        "trailer_length_ft": None,
        "trailer_width_ft": None,
        "trailer_height_ft": None,
        "payload_lbs": None,
        "hitch_type": None,
        "haul_item": None,
    }


def test_size_pair_answer_is_normalized_into_both_dimensions():
    # "8x25" answering the length question used to be stored verbatim, which clobbered the
    # 25.0 the extractor had just parsed. The slot then held text, so search built no
    # length filter and the fit rerank saw required_length_ft=None — the size was lost.
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "I think something like 8x25 would be good")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Livestock",
            extracted={**_empty_extracted(), "trailer_length_ft": 25.0, "trailer_width_ft": 8.0},
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "8x25"}],
        ),
    )
    assert state["slots"]["trailer_length_ft"] == 25.0
    assert state["slots"]["trailer_width_ft"] == 8.0
    assert state["slots"]["length_ft"] == 25.0


def test_size_pair_survives_even_when_the_extractor_missed_it():
    # The pair is parsed from the answer itself, so both dimensions land even if the LLM
    # extracted no numbers at all.
    state = new_session_state("s1")
    state["category"] = "Utility"
    say(state, "7x16")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Utility",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "trailer_size", "raw_answer": "7x16"}],
        ),
    )
    assert state["slots"]["trailer_length_ft"] == 16.0
    assert state["slots"]["trailer_width_ft"] == 7.0
    assert state["slots"]["length_ft"] == 16.0
    assert state["slots"]["width_ft"] == 7.0


def test_hitch_slot_answer_canonicalizes_typos_and_spacing():
    state = new_session_state("s1")
    state["category"] = "Equipment"
    say(state, "goose neck please")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "hitch_type": None},
            slot_answers=[{"slot_name": "hitch_type", "raw_answer": "goose neck please"}],
        ),
    )
    assert state["slots"]["hitch_type"] == ["Gooseneck"]


def test_hitch_either_answer_means_no_preference():
    # "Either is fine" is not a specific preference, so it's null (not a 2-item "both"
    # list) - a null hitch_type and an unset one behave identically downstream.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    say(state, "either is fine")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "hitch_type": None},
            slot_answers=[{"slot_name": "hitch_type", "raw_answer": "either is fine"}],
        ),
    )
    assert state["slots"]["hitch_type"] is None


def test_hitch_extraction_collapses_both_option_list_to_no_preference():
    # If the LLM's OWN extraction still emits the old "both options" convention despite
    # the updated prompt guidance, that must also collapse to null, not a 2-item list.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "hitch_type": ["Bumper Pull", "Gooseneck"]},
        ),
    )
    assert state["slots"]["hitch_type"] is None


def test_hitch_slot_answer_does_not_clobber_a_clean_extraction_in_the_same_turn():
    # A vague/unrecognized slot_answers entry for hitch_type must never downgrade the
    # clean list the LLM's constrained extraction already produced this same turn -
    # search only builds a hitch filter from that list form.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    say(state, "the gooseneck one, whichever fits our truck")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "hitch_type": ["Gooseneck"]},
            slot_answers=[{"slot_name": "hitch_type", "raw_answer": "whichever fits our truck"}],
        ),
    )
    assert state["slots"]["hitch_type"] == ["Gooseneck"]


def test_unrecognized_hitch_text_is_no_preference():
    # A genuine typo/unrecognized spelling the term list doesn't catch is also null -
    # never invented, and never kept as raw unusable text either.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    say(state, "whichever fits our truck")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Equipment",
            extracted={**_empty_extracted(), "hitch_type": None},
            slot_answers=[{"slot_name": "hitch_type", "raw_answer": "whichever fits our truck"}],
        ),
    )
    assert state["slots"]["hitch_type"] is None


def test_vague_size_answer_is_no_preference_not_raw_text():
    # An unparseable measurement answer is null (no preference) so search never treats
    # raw prose as a filter value; the question still advances since the slot is set.
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "as long as you can get")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Livestock",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "as long as you can get"}],
        ),
    )
    assert state["slots"]["trailer_length_ft"] is None


def test_size_range_answer_resolves_to_the_smallest_side():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "somewhere between 15 and 18 ft")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Livestock",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "15-18 ft"}],
        ),
    )
    assert state["slots"]["trailer_length_ft"] == 15.0


def test_dimension_triple_answer_populates_width_length_and_height():
    state = new_session_state("s1")
    state["category"] = "Enclosed"
    say(state, "8x20x7")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Enclosed",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "cargo_size", "raw_answer": "8x20x7"}],
        ),
    )
    assert state["slots"]["trailer_width_ft"] == 8.0
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert state["slots"]["trailer_height_ft"] == 7.0
    assert state["slots"]["length_ft"] == 20.0
    assert state["slots"]["width_ft"] == 8.0
    assert state["slots"]["height_ft"] == 7.0


def test_bin_size_maps_to_trailer_length_and_follows_measurement_rules():
    state = new_session_state("s1")
    state["category"] = "Roll Off"
    say(state, "15-20 yard bin")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Roll Off",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "bin_size", "raw_answer": "15-20 yard bin"}],
        ),
    )
    # Smallest of the range, mirrored into the canonical length slot everything else reads.
    assert state["slots"]["bin_size"] == 15.0
    assert state["slots"]["trailer_length_ft"] == 15.0
    assert state["slots"]["length_ft"] == 15.0


def test_bin_size_vague_answer_is_no_preference():
    state = new_session_state("s1")
    state["category"] = "Roll Off"
    say(state, "whatever you have the most of")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Roll Off",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "bin_size", "raw_answer": "whatever you have the most of"}],
        ),
    )
    assert state["slots"]["bin_size"] is None
    assert "trailer_length_ft" not in state["slots"]


def test_aluminum_subcategory_vague_answer_is_no_preference():
    state = new_session_state("s1")
    state["category"] = "Aluminum"
    say(state, "not sure, whatever's popular")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Aluminum",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "base_category", "raw_answer": "not sure, whatever's popular"}],
        ),
    )
    assert state["slots"]["base_category"] is None


def test_aluminum_subcategory_recognized_answer_is_canonicalized():
    state = new_session_state("s1")
    state["category"] = "Aluminum"
    say(state, "an equipment trailer please")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Aluminum",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "base_category", "raw_answer": "an equipment trailer please"}],
        ),
    )
    assert state["slots"]["base_category"] == "Equipment"


def test_contact_merge_and_decline():
    state = new_session_state("s1")
    apply_with(state, sample_analysis(contact={"name": "Jane", "email": "j@test.com", "phone": None}))
    assert state["customer_name"] == "Jane"
    assert state["customer_email"] == "j@test.com"
    apply_with(state, sample_analysis(intent="contact_declined"))
    assert state["contact_declined"] is True


def test_defaults_then_user_override(monkeypatch):
    import src.graph.apply_analysis as mod

    monkeypatch.setattr(mod, "defaults_for", lambda category: {"hitch_type": "Bumper Pull"})
    state = new_session_state("s1")
    apply_with(state, sample_analysis(category_mentioned="Utility"))
    assert state["slots"]["hitch_type"] == "Bumper Pull"
    assert state["slot_sources"]["hitch_type"] == "default"
    apply_with(state, sample_analysis(slot_answers=[{"slot_name": "hitch_type", "raw_answer": "Gooseneck"}]))
    # Canonicalized to the same ["Bumper Pull"]/["Gooseneck"] list form the LLM's
    # constrained extraction uses, so search's filter builder can read it either way.
    assert state["slots"]["hitch_type"] == ["Gooseneck"]
    assert state["slot_sources"]["hitch_type"] == "user"


def test_flatbed_defaults_to_eight_foot_width_then_user_can_override():
    state = new_session_state("flatbed-default-width")

    apply_with(
        state,
        sample_analysis(
            category_mentioned="Flatbed",
            extracted=_empty_extracted(),
            slot_answers=[],
        ),
    )

    assert state["slots"]["trailer_width_ft"] == 8.0
    assert state["slot_sources"]["trailer_width_ft"] == "default"

    apply_with(
        state,
        sample_analysis(
            category_mentioned=None,
            extracted={**_empty_extracted(), "trailer_width_ft": 8.5},
            slot_answers=[],
        ),
    )

    assert state["slots"]["trailer_width_ft"] == 8.5
    assert state["slot_sources"]["trailer_width_ft"] == "user"


def say(state, text):
    """Set the latest user message — the category rules read the raw text for term tiers."""
    state.setdefault("messages", []).append({"role": "user", "content": text})


def test_category_change_keep_all_carries_measurements():
    state = new_session_state("s1")
    state["category"] = "Tilt"
    say(state, "make it 18 ft long")
    apply_with(state, sample_analysis(category_mentioned=None, slot_answers=[], extracted={**_empty_extracted(), "trailer_length_ft": 18.0}))
    assert state["slots"]["trailer_length_ft"] == 18.0
    # The switch takes effect at once (so anything said in the same breath lands in the new
    # category), and we still pause to ask which measurements to carry over.
    say(state, "switch me to a dump trailer instead")
    apply_with(state, sample_analysis(intent="category_change", category_mentioned="Dump", slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Dump"
    assert state["qualification_complete"] is False
    assert state["pending_category_change"]["new_category"] == "Dump"
    assert state["pending_category_change"]["dimensions"] == {"length": 18.0}
    # Keeping all keeps the carried measurements.
    say(state, "keep all of them")
    apply_with(state, sample_analysis(keep_fields_answer="all", category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Dump"
    assert state["pending_category_change"] is None
    assert state["slots"]["trailer_length_ft"] == 18.0


def test_category_change_keeps_measurements_given_in_the_same_message():
    # "switch to a dump trailer, 20 ft" — the 20 is a statement about the DUMP trailer, so it
    # must survive the switch and be the value we offer to carry over (not the stale 26).
    state = new_session_state("s1")
    state["category"] = "Livestock"
    state["slots"] = {"trailer_length_ft": 26.0, "trailer_width_ft": 8.0}
    state["slot_sources"] = {"trailer_length_ft": "user", "trailer_width_ft": "user"}
    say(state, "let's go with a dump trailer, it should be 20 ft long")
    apply_with(
        state,
        sample_analysis(
            intent="category_change",
            category_mentioned="Dump",
            extracted={**_empty_extracted(), "trailer_length_ft": 20.0},
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "20ft"}],
        ),
    )
    assert state["category"] == "Dump"
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert state["pending_category_change"]["dimensions"] == {"length": 20.0, "width": 8.0}
    # "start fresh" drops the leftover width but NOT the length they just gave us.
    say(state, "no, start fresh")
    apply_with(state, sample_analysis(keep_fields_answer="none", category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert "trailer_width_ft" not in state["slots"]


# --- Rule 1: no category chosen yet ------------------------------------------------


def test_rule1_cargo_term_alone_picks_the_category():
    state = new_session_state("s1")
    say(state, "I need something to haul a tractor")
    apply_with(state, sample_analysis(intent="feature_request_no_category", category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Equipment"  # "tractor" is an Equipment cargo term
    assert state["pending_category_suggestion"] is None


def test_rule1_named_type_plus_conflicting_cargo_asks_to_confirm():
    # "a tilt trailer to haul a tractor": take the named type, then offer Equipment.
    state = new_session_state("s1")
    say(state, "I need a tilt trailer to haul a tractor")
    apply_with(state, sample_analysis(intent="category_selection", category_mentioned="Tilt", slot_answers=[], extracted={**_empty_extracted(), "haul_item": "tractor"}))
    assert state["category"] == "Tilt"  # honour what they actually named
    suggestion = state["pending_category_suggestion"]
    assert suggestion["suggested_category"] == "Equipment"
    assert suggestion["from_category"] == "Tilt"


# --- Rule 2: mid-qualification, cargo term implies a different category ------------


def test_rule2_cargo_term_suggests_switch_and_yes_applies_it():
    state = new_session_state("s1")
    state["category"] = "Tilt"
    state["slots"] = {"trailer_length_ft": 20.0}
    say(state, "it's for hauling a tractor")
    apply_with(state, sample_analysis(intent="qualification_answer", category_mentioned=None, slot_answers=[], extracted={**_empty_extracted(), "haul_item": "tractor"}))
    assert state["category"] == "Tilt"  # not switched yet — we ask first
    assert state["pending_category_suggestion"]["suggested_category"] == "Equipment"
    # "yes" switches and carries the measurements over without a second question.
    say(state, "yes, switch me")
    apply_with(state, sample_analysis(category_confirm_answer="yes", category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Equipment"
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert state["pending_category_suggestion"] is None


def test_rule2_declining_the_suggestion_stays_put_and_never_reasks():
    state = new_session_state("s1")
    state["category"] = "Tilt"
    say(state, "it's for hauling a tractor")
    apply_with(state, sample_analysis(intent="qualification_answer", category_mentioned=None, slot_answers=[], extracted={**_empty_extracted(), "haul_item": "tractor"}))
    assert state["pending_category_suggestion"]["suggested_category"] == "Equipment"
    say(state, "no, tilt is fine")
    apply_with(state, sample_analysis(category_confirm_answer="no", category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Tilt"
    assert state["pending_category_suggestion"] is None


# --- Rule 3: results already shown -> switch outright, ask keep/drop ---------------


def test_rule3_after_results_a_cargo_term_switches_and_asks_keep_drop():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    state["slots"] = {"trailer_length_ft": 20.0}
    state["shown_urls"] = ["https://x/1"]  # results already shown
    say(state, "actually I need to haul a tractor")
    apply_with(state, sample_analysis(intent="requirement_change", category_mentioned=None, slot_answers=[], extracted={**_empty_extracted(), "haul_item": "tractor"}))
    # No confirmation step once results are on the table — go straight to keep/drop.
    assert state["pending_category_suggestion"] is None
    assert state["pending_category_change"]["new_category"] == "Equipment"
    assert state["pending_category_change"]["dimensions"] == {"length": 20.0}


# --- The critical guard: informational questions never move the category -----------


def test_general_question_about_a_category_never_changes_it():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "which trailer type is best for hauling a tractor?")
    apply_with(
        state,
        sample_analysis(
            intent="category_exploration",
            category_mentioned="Equipment",
            is_category_info_only=True,
            slot_answers=[],
            extracted=_empty_extracted(),
        ),
    )
    assert state["category"] == "Livestock"
    assert state["pending_category_suggestion"] is None
    assert state["pending_category_change"] is None


def test_info_only_question_with_no_category_does_not_select_one():
    state = new_session_state("s1")
    say(state, "what is a dump trailer used for?")
    apply_with(
        state,
        sample_analysis(intent="general_question", category_mentioned="Dump", is_category_info_only=True, slot_answers=[], extracted=_empty_extracted()),
    )
    assert state["category"] is None


def test_category_change_no_measurements_switches_immediately():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    state["slots"] = {"haul_item": "cattle"}
    # No length/width/payload collected -> nothing to keep -> switch without a keep/drop prompt.
    apply_with(state, sample_analysis(intent="category_change", category_mentioned="Dump", slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Dump"
    assert state["pending_category_change"] is None
    assert "haul_item" not in state["slots"]


def test_category_change_keep_some_drops_non_dimension_features():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    state["slots"] = {"trailer_length_ft": 20.0, "trailer_width_ft": 8.0, "haul_item": "cattle"}
    state["brand_preference"] = "Galyean"
    state["non_metadata_features"] = ["swing gate"]
    apply_with(state, sample_analysis(intent="category_change", category_mentioned="Dump", slot_answers=[], extracted=_empty_extracted()))
    assert state["pending_category_change"]["dimensions"] == {"length": 20.0, "width": 8.0}
    apply_with(state, sample_analysis(keep_fields_answer="some", kept_fields=["trailer_length_ft"], category_mentioned=None, slot_answers=[], extracted=_empty_extracted()))
    assert state["category"] == "Dump"
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert "trailer_width_ft" not in state["slots"]
    assert "haul_item" not in state["slots"]
    assert state["brand_preference"] is None
    assert state["non_metadata_features"] == []


def test_skip_current_skip_all_repeat_and_drop():
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["pending_question_slot"] = "haul_weight_lbs"
    apply_with(state, sample_analysis(intent="skip_current"))
    assert "haul_weight_lbs" in state["skipped_slots"]
    apply_with(state, sample_analysis(intent="drop_requirements", dropped_fields=["haul_item"]))
    assert "haul_item" not in state["slots"]
    state["pending_question_slot"] = "hitch_type"
    apply_with(state, sample_analysis(intent="general_question", answered_current_question=False))
    apply_with(state, sample_analysis(intent="general_question", answered_current_question=False))
    assert "hitch_type" in state["skipped_slots"]
    assert state["turn_outcome"]["system_email_triggers"]
    apply_with(state, sample_analysis(intent="skip_all_show_results"))
    assert state["qualification_complete"] is True


def test_slot_answers_map_to_numeric_target_slots():
    state = new_session_state("s1")
    state["category"] = "Equipment"
    apply_with(
        state,
        sample_analysis(
            category_mentioned=None,
            slot_answers=[
                {"slot_name": "haul_weight_lbs", "raw_answer": "2 tons"},
                {"slot_name": "haul_length_ft", "raw_answer": "83 inches"},
            ],
        ),
    )
    # Numbers stored under the metadata target keys, not a re-encoded string.
    assert state["slots"]["payload_lbs"] == 4000.0
    assert round(state["slots"]["length_ft"], 3) == 6.917
    # Category slots still marked answered so qualification advances.
    assert "haul_weight_lbs" in state["slots"] and "haul_length_ft" in state["slots"]
    assert all("=" not in str(value) for value in state["slots"].values() if value is not None)


def test_bin_size_maps_to_length_for_rolloff():
    state = new_session_state("s1")
    state["category"] = "Roll Off"
    apply_with(state, sample_analysis(category_mentioned=None, slot_answers=[{"slot_name": "bin_size", "raw_answer": "15 yd"}]))
    assert state["slots"]["length_ft"] == 15.0
    assert "bin_size" in state["slots"]


def test_cargo_size_vague_answer_still_marked_answered():
    state = new_session_state("s1")
    state["category"] = "Enclosed"
    state["pending_question_slot"] = "cargo_size"
    apply_with(
        state,
        sample_analysis(intent="qualification_answer", category_mentioned=None, answered_current_question=True, slot_answers=[]),
    )
    assert "cargo_size" in state["slots"]  # answered even with no parseable value -> next question


def test_clarification_triggered_then_resolved():
    state = new_session_state("s1")
    state["messages"] = [{"role": "user", "content": "I need an office trailer"}]
    apply_with(state, sample_analysis(category_mentioned="Fiber", is_category_info_only=False))
    assert state["clarification_key"] == "office_trailer_use"
    assert state["turn_outcome"]["clarification_question"]
    assert state["category"] is None
    state["messages"].append({"role": "user", "content": "it's for fiber splicing work"})
    apply_with(state, sample_analysis(category_mentioned=None, is_category_info_only=False))
    assert state["category"] == "Fiber"
    assert state["clarification_key"] is None


def test_no_preference_and_width_injection_cases():
    # needs_width_question (a SIZE judgment) -> the width question is injected.
    # category_mentioned stays None throughout: this test is about width, not category moves.
    state = new_session_state("s1")
    state["category"] = "Equipment"
    analysis = sample_analysis(
        category_mentioned=None,
        # No width given anywhere (the default sample carries a 7x14 size answer) —
        # otherwise there is nothing left to ask.
        extracted={**_empty_extracted(), "numeric_no_preference": ["haul_weight_lbs"]},
        slot_answers=[],
        haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
    )
    apply_with(state, analysis)
    assert state["slots"]["haul_weight_lbs"] is None
    assert INJECTED_WIDTH_SLOT in state["injected_required_slots"]
    # Width is a size question, NOT a weight question: a lightweight utility load does
    # not inject width (and must not skip width either — the two are independent).
    utility = new_session_state("s2")
    utility["category"] = "Utility"
    apply_with(utility, sample_analysis(category_mentioned=None, haul_classification={"is_lightweight_utility_load": True, "needs_width_question": False, "haul_item_matched": "golf cart"}))
    assert INJECTED_WIDTH_SLOT not in utility["injected_required_slots"]
    # Width-excluded category never gets the width question, even for a big load.
    rolloff = new_session_state("s3")
    rolloff["category"] = "Roll Off"
    apply_with(rolloff, analysis)
    assert INJECTED_WIDTH_SLOT not in rolloff["injected_required_slots"]
    # Invariant: a width flag with no matched cargo is cleared before injection.
    invalid = new_session_state("s4")
    invalid["category"] = "Equipment"
    apply_with(invalid, sample_analysis(category_mentioned=None, haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": None}))
    assert INJECTED_WIDTH_SLOT not in invalid["injected_required_slots"]


def test_lightweight_utility_load_skips_the_weight_question():
    # is_lightweight_utility_load (a WEIGHT judgment) -> the weight question is skipped:
    # we already know a golf cart is light, so we don't ask "rough total weight?".
    state = new_session_state("s1")
    state["category"] = "Utility"
    apply_with(
        state,
        sample_analysis(
            category_mentioned="Utility",
            slot_answers=[{"slot_name": "haul_item", "raw_answer": "golf cart"}],
            haul_classification={"is_lightweight_utility_load": True, "needs_width_question": False, "haul_item_matched": "golf cart"},
        ),
    )
    assert "haul_weight_lbs" in state["skipped_slots"]
    assert "haul_weight_lbs" not in state["slots"]
    # Width is unaffected — a light load is not a wide load.
    assert INJECTED_WIDTH_SLOT not in state["injected_required_slots"]


def test_lightweight_does_not_skip_weight_the_user_actually_gave():
    # If the user volunteers a weight in the same message, keep it — "don't ask" is not
    # "discard what they told us".
    state = new_session_state("s1")
    state["category"] = "Utility"
    apply_with(
        state,
        sample_analysis(
            category_mentioned="Utility",
            slot_answers=[
                {"slot_name": "haul_item", "raw_answer": "golf cart"},
                {"slot_name": "haul_weight_lbs", "raw_answer": "900 lbs"},
            ],
            haul_classification={"is_lightweight_utility_load": True, "needs_width_question": False, "haul_item_matched": "golf cart"},
        ),
    )
    # A numeric slot stores the parsed number, not the raw text — search's metadata filter
    # and the fit rerank both read this slot and can only use a number.
    assert state["slots"]["haul_weight_lbs"] == 900.0
    assert state["slots"]["payload_lbs"] == 900.0
    assert "haul_weight_lbs" not in state["skipped_slots"]


def test_lightweight_flag_only_skips_weight_for_utility():
    # is_lightweight_utility_load is Utility-only; the invariant would clear it elsewhere,
    # but guard belt-and-suspenders: a non-Utility category never has its weight skipped here.
    state = new_session_state("s1")
    state["category"] = "Dump"
    apply_with(
        state,
        sample_analysis(
            category_mentioned=None,
            slot_answers=[],
            haul_classification={"is_lightweight_utility_load": True, "needs_width_question": False, "haul_item_matched": "gravel"},
        ),
    )
    assert "haul_weight_lbs" not in state["skipped_slots"]


def test_aluminum_named_with_another_type_keeps_aluminum_as_the_category():
    # "a utility trailer in aluminum" names two types, and only Aluminum is a category we stock.
    # The other one is the base_category it should be built as. Taking Utility (it came first in
    # the sentence) dropped Aluminum entirely and qualified them for a steel utility trailer.
    state = new_session_state("s1")
    say(state, "I want a utility trailer in aluminum")
    apply_with(
        state,
        sample_analysis(category_mentioned="Aluminum", slot_answers=[], extracted=_empty_extracted()),
    )
    assert state["category"] == "Aluminum"
    assert state["slots"]["base_category"] == "Utility"


def test_answering_the_base_category_question_does_not_change_the_category():
    # Seen live: mid-Aluminum, we ask "what type do you want it in - utility, equipment,
    # enclosed?", they say "utility", and we read their ANSWER as a request to leave Aluminum:
    # category switched, Aluminum slots wiped, qualification restarted from the top.
    state = new_session_state("s1")
    state["category"] = "Aluminum"
    state["slots"] = {"payload_need": 4000.0}
    state["slot_sources"] = {"payload_need": "user"}
    state["pending_question_slot"] = "base_category"
    say(state, "utility")
    apply_with(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned=None,
            slot_answers=[{"slot_name": "base_category", "raw_answer": "utility"}],
            extracted=_empty_extracted(),
        ),
    )
    assert state["category"] == "Aluminum"
    assert state["pending_category_change"] is None
    assert state["slots"]["base_category"] == "Utility"
    assert state["slots"]["payload_need"] == 4000.0  # nothing was wiped


def test_base_category_survives_an_extractor_that_named_the_category_instead():
    # The extractor keeps reading the type word as a category and emits no slot answer at all,
    # which left base_category empty (or nulled by the answered-but-unparsed fallback) and the
    # subcategory filter unset. The answer is recovered from what they actually said.
    state = new_session_state("s1")
    state["category"] = "Aluminum"
    state["pending_question_slot"] = "base_category"
    say(state, "enclosed would work")
    apply_with(
        state,
        sample_analysis(
            intent="category_change",
            category_mentioned="Enclosed",
            slot_answers=[],
            extracted=_empty_extracted(),
        ),
    )
    assert state["category"] == "Aluminum"
    assert state["slots"]["base_category"] == "Enclosed"


def test_a_type_word_outside_the_aluminum_base_question_is_still_a_category_change():
    # The guard is scoped to the question we asked. Once base_category is settled, "show me dump
    # trailers instead" must still move them off Aluminum like any other category change.
    state = new_session_state("s1")
    state["category"] = "Aluminum"
    state["slots"] = {"base_category": "Utility"}
    state["slot_sources"] = {"base_category": "user"}
    say(state, "actually show me dump trailers instead")
    apply_with(
        state,
        sample_analysis(intent="category_change", category_mentioned="Dump", slot_answers=[], extracted=_empty_extracted()),
    )
    assert state["category"] == "Dump"
    assert "base_category" not in state["slots"]


def test_contact_details_and_a_category_in_one_message_still_set_the_category():
    # Seen live: "My name is Ibrahim, my email is ibrahim@x.ai. I need an Aluminum trailer, base
    # category equipment, payload 7-9k." The extractor called the whole thing contact_info_provided,
    # which was not a category-action intent — so Aluminum was dropped, the slots were stored under
    # no category at all, and we replied by asking him to pick a trailer type he had just named.
    state = new_session_state("s1")
    say(state, "My name is Ibrahim and my email is ibrahim@esided.ai. I need an Aluminum trailer, an equipment one.")
    apply_with(
        state,
        sample_analysis(
            intent="contact_info_provided",
            category_mentioned="Aluminum",
            contact={"name": "Ibrahim", "email": "ibrahim@esided.ai", "phone": None},
            slot_answers=[{"slot_name": "base_category", "raw_answer": "equipment"}],
            extracted={**_empty_extracted(), "payload_lbs": 7000.0},
        ),
    )
    assert state["category"] == "Aluminum"
    assert state["slots"]["base_category"] == "Equipment"
    assert state["slots"]["payload_need"] == 7000.0


def test_contact_details_alone_never_start_a_category():
    state = new_session_state("s1")
    say(state, "It's Ibrahim, ibrahim@esided.ai")
    apply_with(
        state,
        sample_analysis(
            intent="contact_info_provided",
            category_mentioned=None,
            contact={"name": "Ibrahim", "email": "ibrahim@esided.ai", "phone": None},
            slot_answers=[],
            extracted=_empty_extracted(),
        ),
    )
    assert not state.get("category")


def test_contact_details_never_change_a_category_already_chosen():
    # They gave us their email while on Dump and mentioned a tilt trailer in passing. Handing over
    # contact details says nothing about changing their mind, so it can start a category but never
    # move one.
    state = new_session_state("s1")
    state["category"] = "Dump"
    say(state, "sure, ibrahim@esided.ai — a friend of mine has a tilt trailer")
    apply_with(
        state,
        sample_analysis(
            intent="contact_info_provided",
            category_mentioned="Tilt",
            contact={"name": None, "email": "ibrahim@esided.ai", "phone": None},
            slot_answers=[],
            extracted=_empty_extracted(),
        ),
    )
    assert state["category"] == "Dump"
