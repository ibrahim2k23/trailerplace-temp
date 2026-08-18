from __future__ import annotations

from src.graph.apply_analysis import INJECTED_WIDTH_SLOT
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state


def test_question_order_skips_answered_and_skipped():
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["slots"] = {"haul_item": "tractor"}
    state["skipped_slots"] = ["haul_weight_lbs"]
    qualification_node(state)
    assert state["pending_question_slot"] == "haul_length_ft"
    assert "long is the load" in state["turn_outcome"]["next_question"]


def test_completion_flag_when_no_questions_left():
    state = new_session_state("s1")
    state["category"] = "Utility"
    state["slots"] = {"haul_item": "golf cart", "haul_weight_lbs": 1000, "axle_capacity_lbs": 3500}
    qualification_node(state)
    assert state["qualification_complete"] is True


def test_utility_asks_axle_capacity_after_the_weight_question():
    state = new_session_state("s1")
    state["category"] = "Utility"
    state["slots"] = {"haul_item": "golf cart", "haul_weight_lbs": 1000}
    qualification_node(state)
    assert state["pending_question_slot"] == "axle_capacity_lbs"
    assert "axle capacity" in state["turn_outcome"]["next_question"].lower()


def test_injected_width_question_sequences_like_required_slot():
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["slots"] = {"haul_item": "tractor", "haul_weight_lbs": 7000, "haul_length_ft": 14, "hitch_type": ["Bumper Pull"]}
    state["injected_required_slots"] = [INJECTED_WIDTH_SLOT]
    qualification_node(state)
    assert state["pending_question_slot"] == INJECTED_WIDTH_SLOT
    assert "wide" in state["turn_outcome"]["next_question"]
    state["skipped_slots"] = [INJECTED_WIDTH_SLOT]
    qualification_node(state)
    assert state["qualification_complete"] is True
