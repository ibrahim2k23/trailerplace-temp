from __future__ import annotations

from src.domain.trailer_fields import get_trailer_fields
from src.graph.apply_analysis import INJECTED_WIDTH_QUESTION, INJECTED_WIDTH_SLOT, required_slots_for_state


def qualification_node(state: dict) -> dict:
    outcome = state.setdefault("turn_outcome", {})
    if not state.get("category"):
        outcome["next_question"] = "What type of trailer are you looking for?"
        return state
    spec = get_trailer_fields(state["category"])
    questions = dict(spec.questions)
    questions[INJECTED_WIDTH_SLOT] = INJECTED_WIDTH_QUESTION
    for slot in required_slots_for_state(state):
        if slot not in state.get("slots", {}) and slot not in state.get("skipped_slots", []):
            previous = state.get("pending_question_slot")
            state["pending_question_slot"] = slot
            state["pending_question_repeats"] = state.get("pending_question_repeats", 0) if previous == slot else 0
            outcome["next_question"] = questions.get(slot, f"What is your preference for {slot}?")
            state["qualification_complete"] = False
            return state
    state["pending_question_slot"] = None
    state["pending_question_repeats"] = 0
    state["qualification_complete"] = True
    outcome["qualification_complete"] = True
    return state
