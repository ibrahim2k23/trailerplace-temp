from __future__ import annotations

from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.build import _route
from src.graph.contact_gate import contact_gate_pending
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted, say


def contact(name=None, email=None, phone=None) -> dict:
    return {"name": name, "email": email, "phone": phone}


def turn(state, text, analysis):
    """One turn through analyze->apply->route, including respond's contact ask."""
    say(state, text)
    state["turn"] = analysis
    apply_analysis_to_state(state)
    route = _route(state)
    if route == "respond" and contact_gate_pending(state):
        # what respond_node does when the gate owns the turn
        state["contact_asks"] = int(state.get("contact_asks", 0) or 0) + 1
    return route


def test_gate_asks_for_contact_before_anything_else():
    state = new_session_state("s1")
    route = turn(
        state,
        "I need a dump trailer, 20ft",
        sample_analysis(intent="category_selection", category_mentioned="Dump", contact=contact(),
                        extracted={**_empty_extracted(), "trailer_length_ft": 20.0}, slot_answers=[]),
    )
    # We never reach qualification or search — but their request is safely recorded.
    assert route == "respond"
    assert state["category"] == "Dump"
    assert state["slots"]["trailer_length_ft"] == 20.0
    assert state["contact_asks"] == 1


def test_partial_contact_gets_one_more_ask_then_the_gate_closes():
    state = new_session_state("s1")
    turn(state, "I need a dump trailer", sample_analysis(intent="category_selection", category_mentioned="Dump", contact=contact(), extracted=_empty_extracted(), slot_answers=[]))
    # Name only -> we still lack a way to reach them, so ask once more.
    route = turn(state, "I'm Ibrahim", sample_analysis(intent="contact_info_provided", contact=contact(name="Ibrahim"), extracted=_empty_extracted(), slot_answers=[]))
    assert route == "respond"
    assert contact_gate_pending(state)
    assert state["contact_asks"] == 2
    # They give the rest -> gate closes and qualification finally starts.
    route = turn(state, "03304388550", sample_analysis(intent="contact_info_provided", contact=contact(phone="03304388550"), extracted=_empty_extracted(), slot_answers=[]))
    assert not contact_gate_pending(state)
    assert route == "qualification"


def test_ignoring_the_ask_closes_the_gate_for_good():
    state = new_session_state("s1")
    turn(state, "I need a dump trailer", sample_analysis(intent="category_selection", category_mentioned="Dump", contact=contact(), extracted=_empty_extracted(), slot_answers=[]))
    route = turn(state, "let's just get on with it", sample_analysis(intent="skip_current", contact=contact(), extracted=_empty_extracted(), slot_answers=[]))
    assert not contact_gate_pending(state)
    assert route == "qualification"
    assert state["contact_asks"] == 1  # never asked a second time


def test_declining_closes_the_gate_for_good():
    state = new_session_state("s1")
    turn(state, "I need a dump trailer", sample_analysis(intent="category_selection", category_mentioned="Dump", contact=contact(), extracted=_empty_extracted(), slot_answers=[]))
    route = turn(state, "I'd rather not share that", sample_analysis(intent="contact_declined", contact=contact(), extracted=_empty_extracted(), slot_answers=[]))
    assert not contact_gate_pending(state)
    assert route == "qualification"


def test_full_contact_up_front_never_triggers_the_gate():
    state = new_session_state("s1")
    route = turn(
        state,
        "Hi, I'm Ibrahim, ibrahim@x.ai — I need a dump trailer",
        sample_analysis(intent="category_selection", category_mentioned="Dump",
                        contact=contact(name="Ibrahim", email="ibrahim@x.ai"), extracted=_empty_extracted(), slot_answers=[]),
    )
    assert route == "qualification"
    assert state["contact_asks"] == 0


def test_a_contact_detour_costs_the_pending_question_one_strike_not_two():
    # An email trigger needs contact mid-qualification. The customer keeps talking about other
    # things. Our own detour must not burn both strikes and skip their question out from under
    # them — it is charged once.
    state = new_session_state("s1")
    state["category"] = "Dump"
    state["contact_gate_closed"] = True  # opening gate already settled
    state["pending_question_slot"] = "haul_material"
    state["contact_followup_pending"] = "contact_method"
    state["customer_name"] = "Ibrahim"

    for _ in range(3):
        turn(state, "what are your hours?", sample_analysis(intent="general_question", contact=contact(), extracted=_empty_extracted(), slot_answers=[], answered_current_question=False))
    assert state["pending_question_repeats"] == 1
    assert "haul_material" not in state["skipped_slots"]

    # Once the detour ends, the normal two-strike rule resumes.
    state["contact_followup_pending"] = None
    turn(state, "what are your hours?", sample_analysis(intent="general_question", contact=contact(), extracted=_empty_extracted(), slot_answers=[], answered_current_question=False))
    assert "haul_material" in state["skipped_slots"]
