from __future__ import annotations

import logging

from src.domain.categories import unstocked_categories
from src.domain.slot_map import slot_value_kind
from src.domain.trailer_fields import get_trailer_fields
from src.graph.apply_analysis import INJECTED_WIDTH_QUESTION, INJECTED_WIDTH_SLOT, required_slots_for_state

logger = logging.getLogger(__name__)


def _axle_capacity_already_answers(state: dict, slot: str) -> bool:
    """True when the customer gave an axle rating and `slot` is this category's weight question.

    An axle rating is NOT a load weight - the two are different facts, and the rest of the
    system is careful to keep them apart (see slot_map's axle notes: filing axle under the
    payload kind would let a load weight silently auto-fill the axle rating). This is about
    the QUESTION, not the value.

    "How heavy is your load?" is asked for one reason: to size the trailer. Someone who has
    already said "I want 7,000 lb axles" has specified that themselves, from the capacity end.
    Asking them the weight question anyway reads as not having listened, and the answer would
    not change the search - the axle rating they gave is already a rerank signal in every
    category (see _ALWAYS_VALID_NUMERIC_SLOTS).

    Applies to every category, under whichever name it asks: haul_weight_lbs on
    Equipment/Utility/Dump/Tilt, payload_need on Aluminum, total_weight, payload_lbs.
    """
    if slot_value_kind(slot) != "payload_lbs":
        return False
    slots = state.get("slots", {})
    # Either rating counts. A total ("14,000 lbs across the axles") specifies the trailer from
    # the capacity end just as squarely as a per-axle figure does, and asking the load weight
    # after it reads the same way: as not having listened.
    return any(
        isinstance(rating, (int, float)) and not isinstance(rating, bool) and rating > 0
        # A positive number only. A 0 (or a non-numeric leftover) tells us nothing about
        # capacity, and skipping the weight question on it would lose BOTH facts.
        for rating in (slots.get("axle_capacity_lbs"), slots.get("total_axle_capacity_lbs"))
    )


def qualification_node(state: dict) -> dict:
    outcome = state.setdefault("turn_outcome", {})
    if not state.get("category"):
        # No category means nothing to search, whatever else happened this turn ("just show
        # me what you have" before they've told us what they want). An axle rating given
        # here is already stored and waits for us - the category still has to be settled.
        state["qualification_complete"] = False
        outcome["next_question"] = "What type of trailer are you looking for?"
        return state
    if state["category"] in unstocked_categories():
        # Nothing to qualify FOR. The respond prompt already forbids qualifying for a type we
        # do not stock, and the model ignored it - it had a concrete question in front of it,
        # and a concrete instruction beats a general rule every time. Live, a Diesel Tank
        # request was asked its fuel type and tank capacity before reaching a search that
        # could only ever return nothing. Withholding the question is what makes the rule
        # stick; qualification_complete stays False so no empty search runs either.
        logger.info(
            "qualification | %s is not stocked: no questions, no search",
            state["category"],
        )
        outcome["unstocked_category"] = state["category"]
        state["qualification_complete"] = False
        return state
    if outcome.get("clarification_question"):
        # One question per turn. A clarification is already outstanding - which axle capacity
        # they meant, or how many axles - and stacking a slot question on top of it asks two
        # things at once, which reliably gets one of them answered and the other lost.
        state["qualification_complete"] = False
        return state
    spec = get_trailer_fields(state["category"])
    questions = dict(spec.questions)
    questions[INJECTED_WIDTH_SLOT] = INJECTED_WIDTH_QUESTION
    for slot in required_slots_for_state(state):
        if slot in state.get("slots", {}) or slot in state.get("skipped_slots", []):
            continue
        if _axle_capacity_already_answers(state, slot):
            slots = state.get("slots", {})
            logger.info(
                "qualification | skipping %s for %s: axle capacity already given "
                "(per_axle=%s total=%s)",
                slot, state.get("category"),
                slots.get("axle_capacity_lbs"), slots.get("total_axle_capacity_lbs"),
            )
            continue
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
