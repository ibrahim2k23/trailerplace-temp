from __future__ import annotations

from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.build import should_search
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted, say


def qualified_state(**overrides) -> dict:
    """A customer who has answered everything and already has results on screen."""
    state = new_session_state("s1")
    state["category"] = "Livestock"
    state["slots"] = {"trailer_length_ft": 26.0, "trailer_width_ft": 8.0, "length_ft": 26.0}
    state["slot_sources"] = {key: "user" for key in state["slots"]}
    state["qualification_complete"] = True
    state["search_pending"] = False
    state["shown_urls"] = ["https://x/1"]
    state.update(overrides)
    return state


def turn(state, analysis):
    state["turn"] = analysis
    apply_analysis_to_state(state)
    return should_search(state)


def test_listing_interest_does_not_re_search():
    # "yeah I like the 81382" — the LLM re-states the size it already knows; nothing moved,
    # so there is nothing new to look up.
    state = qualified_state()
    say(state, "yeah I like the 81382")
    assert not turn(
        state,
        sample_analysis(
            intent="listing_interest",
            category_mentioned="Livestock",
            listing_reference=1,
            extracted={**_empty_extracted(), "trailer_length_ft": 26.0, "trailer_width_ft": 8.0},
            slot_answers=[{"slot_name": "trailer_length_ft", "raw_answer": "8x26"}],
        ),
    )


def test_contact_info_does_not_re_search():
    state = qualified_state()
    say(state, "sure, my email is ibrahim@example.com")
    assert not turn(
        state,
        sample_analysis(
            intent="contact_info_provided",
            category_mentioned="Livestock",
            extracted=_empty_extracted(),
            slot_answers=[],
            contact={"name": None, "email": "ibrahim@example.com", "phone": None},
        ),
    )


def test_faq_does_not_re_search():
    state = qualified_state()
    say(state, "do you offer financing?")
    assert not turn(
        state,
        sample_analysis(intent="faq", category_mentioned=None, extracted=_empty_extracted(), slot_answers=[]),
    )


def test_changed_requirement_re_searches():
    state = qualified_state()
    say(state, "what about with a bumper pull?")
    assert turn(
        state,
        sample_analysis(
            intent="requirement_change",
            category_mentioned="Livestock",
            extracted={**_empty_extracted(), "hitch_type": ["Bumper Pull"]},
            slot_answers=[],
        ),
    )


def test_asking_for_more_results_re_searches():
    state = qualified_state()
    say(state, "show me more options")
    assert turn(
        state,
        sample_analysis(intent="show_more_results", category_mentioned="Livestock", extracted=_empty_extracted(), slot_answers=[]),
    )


def test_finishing_qualification_searches_once_then_stops():
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "26 ft, gooseneck")
    assert turn(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Livestock",
            extracted={**_empty_extracted(), "trailer_length_ft": 26.0, "hitch_type": ["Gooseneck"]},
            slot_answers=[],
            # everything else waved off
        ),
    ) is False  # still questions left -> no search yet
    state["qualification_complete"] = True
    assert should_search(state) is True  # the answer above left the search dirty
    state["search_pending"] = False  # ... and search_node clears it
    assert should_search(state) is False


def test_category_change_blocks_search_until_its_questions_are_asked():
    state = qualified_state()
    say(state, "actually I need an equipment trailer")
    # The change itself pauses on the keep/drop question.
    assert not turn(
        state,
        sample_analysis(
            intent="category_change",
            category_mentioned="Equipment",
            extracted=_empty_extracted(),
            slot_answers=[],
        ),
    )
    assert state["category"] == "Equipment"
    assert state["qualification_complete"] is False
    # Even after the keep/drop answer, Equipment's own questions are unanswered -> no search.
    say(state, "keep them all")
    assert not turn(
        state,
        sample_analysis(keep_fields_answer="all", category_mentioned=None, extracted=_empty_extracted(), slot_answers=[]),
    )
    assert state["pending_category_change"] is None


def test_unanswered_keep_drop_question_never_blocks_qualification():
    # The customer replies to "which measurements should we carry over?" with something else
    # entirely. The pending change must resolve anyway — leaving it set froze the router on
    # `respond` forever, so the new category's questions were never asked.
    state = qualified_state()
    say(state, "I am also looking for a utility trailer")
    turn(state, sample_analysis(intent="category_change", category_mentioned="Utility", extracted=_empty_extracted(), slot_answers=[]))
    assert state["pending_category_change"]["dimensions"] == {"length": 26.0, "width": 8.0}

    say(state, "I think 18ft would be good")
    assert not turn(
        state,
        sample_analysis(
            intent="requirement_change",
            category_mentioned="Utility",
            keep_fields_answer=None,
            extracted={**_empty_extracted(), "trailer_length_ft": 18.0},
            slot_answers=[{"slot_name": "trailer_size", "raw_answer": "I think 18ft would be good"}],
        ),
    )
    assert state["pending_category_change"] is None
    assert state["slots"]["trailer_length_ft"] == 18.0
    # A lone number answering a combined size question is a LENGTH — never also a width.
    assert state["slots"]["length_ft"] == 18.0
    assert "width_ft" not in state["slots"]
    # ... and the Utility questions now actually get asked.
    qualification_node(state)
    assert state["turn_outcome"]["next_question"] == "What will you be hauling on the utility trailer?"


def test_blank_and_foreign_slot_answers_are_not_answers():
    # Seen live: answering "20ft would be good" for Livestock, the extractor also emitted
    # haul_item='', haul_weight_lbs='' and hitch_type='' — blank answers to slots Livestock
    # does not even have. Stored, they mark questions answered and put haul_item="" into the
    # search query.
    state = new_session_state("s1")
    state["category"] = "Livestock"
    say(state, "I am looking for a livestock trailer. 20ft would be good")
    turn(
        state,
        sample_analysis(
            intent="category_selection",
            category_mentioned="Livestock",
            extracted={**_empty_extracted(), "trailer_length_ft": 20.0},
            slot_answers=[
                {"slot_name": "haul_item", "raw_answer": ""},
                {"slot_name": "haul_weight_lbs", "raw_answer": ""},
                {"slot_name": "hitch_type", "raw_answer": ""},
            ],
        ),
    )
    assert state["slots"] == {"trailer_length_ft": 20.0}


def test_blank_answer_leaves_its_own_question_unasked():
    state = new_session_state("s1")
    state["category"] = "Utility"
    say(state, "I need a utility trailer")
    turn(
        state,
        sample_analysis(
            intent="category_selection",
            category_mentioned="Utility",
            extracted=_empty_extracted(),
            slot_answers=[{"slot_name": "haul_item", "raw_answer": ""}],
        ),
    )
    assert "haul_item" not in state["slots"]
    qualification_node(state)
    assert state["qualification_complete"] is False
    assert state["turn_outcome"]["next_question"] == "What will you be hauling on the utility trailer?"


def test_a_width_they_already_gave_is_not_asked_for_again():
    # Seen live: "around 20ft long, 6ft wide" still injected "how wide is that item or trailer you
    # need to haul?" — so qualification stayed open and the search never ran. Uses Tilt, a
    # width-ELIGIBLE category, so the guard under test is the one that fires.
    state = new_session_state("s1")
    state["category"] = "Tilt"
    # NB: cargo with no category of its own — "skid steer" would imply Equipment and raise a
    # category-switch suggestion, which blocks the search for an unrelated reason.
    say(state, "a trailer to haul my forklift, around 20ft long, 6ft wide, about 7000 lbs")
    assert turn(
        state,
        sample_analysis(
            intent="qualification_answer",
            category_mentioned="Tilt",
            extracted={
                **_empty_extracted(),
                "trailer_length_ft": 20.0,
                "trailer_width_ft": 6.0,
                "payload_lbs": 7000.0,
                "haul_item": "forklift",
            },
            slot_answers=[{"slot_name": "haul_weight_lbs", "raw_answer": "7000 lbs"}],
            haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "forklift"},
        ),
    ) is False  # qualification_node hasn't run yet in this helper
    assert "item_or_trailer_width_ft" not in state["injected_required_slots"]
    qualification_node(state)
    assert state["qualification_complete"] is True
    assert should_search(state)


def test_width_exempt_categories_never_get_the_injected_width_question():
    # Livestock/Dump/Utility/Flatbed/Enclosed/Aluminum bodies come as they come — their width is
    # not a choice the customer makes, so we never ask it however wide the cargo is.
    for category in ("Livestock", "Dump", "Utility", "Flatbed", "Enclosed", "Aluminum"):
        state = new_session_state("s1")
        state["category"] = category
        say(state, "I need to haul a full-size tractor")
        turn(
            state,
            sample_analysis(
                intent="qualification_answer",
                category_mentioned=category,
                extracted={**_empty_extracted(), "haul_item": "tractor"},
                slot_answers=[],
                haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
            ),
        )
        assert "item_or_trailer_width_ft" not in state["injected_required_slots"], category


def test_width_eligible_categories_still_get_the_injected_width_question():
    # The exemption must not disarm the width question everywhere — Car Hauler, Equipment and
    # Tilt are exactly where a wide load decides whether the trailer works.
    for category in ("Car Hauler", "Equipment", "Tilt"):
        state = new_session_state("s1")
        state["category"] = category
        say(state, "I need to haul a full-size tractor")
        turn(
            state,
            sample_analysis(
                intent="qualification_answer",
                category_mentioned=category,
                extracted={**_empty_extracted(), "haul_item": "tractor"},
                slot_answers=[],
                haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
            ),
        )
        assert "item_or_trailer_width_ft" in state["injected_required_slots"], category


def test_looking_for_a_trailer_to_haul_a_tractor_moves_the_category():
    # A statement of need is a WANT, not an information question: the cargo picks the category.
    state = qualified_state()
    say(state, "I am looking for a trailer to haul a tractor")
    turn(
        state,
        sample_analysis(
            intent="category_exploration",
            category_mentioned=None,
            is_category_info_only=False,
            extracted={**_empty_extracted(), "haul_item": "tractor"},
            slot_answers=[],
            haul_classification={"is_lightweight_utility_load": False, "needs_width_question": True, "haul_item_matched": "tractor"},
        ),
    )
    assert state["category"] == "Equipment"


def test_asking_which_trailer_suits_a_tractor_does_not_move_the_category():
    state = qualified_state()
    say(state, "which trailer is best for hauling a tractor?")
    assert not turn(
        state,
        sample_analysis(
            intent="category_exploration",
            category_mentioned=None,
            is_category_info_only=True,
            extracted={**_empty_extracted(), "haul_item": "tractor"},
            slot_answers=[],
        ),
    )
    assert state["category"] == "Livestock"
