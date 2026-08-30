"""An axle rating the customer volunteered stands in for the weight QUESTION.

The value is still not a load weight - nothing here files axle capacity under payload. What
changes is only whether we put the weight question to someone who has already told us the
capacity they want.
"""
from __future__ import annotations

import pytest

from src.domain.categories import CANONICAL_CATEGORIES
from src.domain.slot_map import slot_value_kind
from src.domain.trailer_fields import get_trailer_fields
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state


def _weight_slot(category: str) -> str | None:
    """This category's weight question, under whichever name it asks it."""
    return next(
        (slot for slot in get_trailer_fields(category).required if slot_value_kind(slot) == "payload_lbs"),
        None,
    )


WEIGHT_CATEGORIES = [c for c in CANONICAL_CATEGORIES if _weight_slot(c)]
NO_WEIGHT_CATEGORIES = [c for c in CANONICAL_CATEGORIES if not _weight_slot(c)]


def _state(category: str, *, axle: float | None, answered_others: bool = True) -> dict:
    state = new_session_state("s1")
    state["category"] = category
    slots = {}
    if answered_others:
        slots = {s: "x" for s in get_trailer_fields(category).required if slot_value_kind(s) != "payload_lbs"}
    if axle is not None:
        slots["axle_capacity_lbs"] = axle
    state["slots"] = slots
    return state


def test_there_are_categories_on_both_sides():
    """Guards the parametrised tests below from silently covering nothing."""
    assert WEIGHT_CATEGORIES and NO_WEIGHT_CATEGORIES


@pytest.mark.parametrize("category", WEIGHT_CATEGORIES)
def test_weight_question_is_asked_when_no_axle_rating_was_given(category):
    state = _state(category, axle=None)
    qualification_node(state)
    assert state["pending_question_slot"] == _weight_slot(category)
    assert state["qualification_complete"] is False


@pytest.mark.parametrize("category", WEIGHT_CATEGORIES)
def test_weight_question_is_skipped_when_an_axle_rating_was_given(category):
    state = _state(category, axle=7000.0)
    qualification_node(state)
    assert state["pending_question_slot"] is None
    assert state["qualification_complete"] is True


@pytest.mark.parametrize("category", NO_WEIGHT_CATEGORIES)
def test_categories_without_a_weight_question_are_unaffected(category):
    for axle in (None, 7000.0):
        state = _state(category, axle=axle)
        qualification_node(state)
        assert state["qualification_complete"] is True


@pytest.mark.parametrize("category", WEIGHT_CATEGORIES)
def test_the_axle_value_itself_is_never_filed_as_a_payload(category):
    """The skip must not become an auto-fill: a 7,000 lb axle is not a 7,000 lb load, and
    storing it as one would filter the search on a weight the customer never stated."""
    state = _state(category, axle=7000.0)
    qualification_node(state)
    assert _weight_slot(category) not in state["slots"]
    assert state["slots"]["axle_capacity_lbs"] == 7000.0


@pytest.mark.parametrize("category", WEIGHT_CATEGORIES)
def test_the_other_questions_are_still_asked(category):
    """Only the weight question goes. An axle rating answers nothing else."""
    state = new_session_state("s1")
    state["category"] = category
    state["slots"] = {"axle_capacity_lbs": 7000.0}
    qualification_node(state)
    required = [s for s in get_trailer_fields(category).required if slot_value_kind(s) != "payload_lbs"]
    if required:
        assert state["pending_question_slot"] == required[0]
        assert state["qualification_complete"] is False


def test_an_axle_rating_with_no_category_still_asks_for_the_category():
    """Nothing to search without a category, whatever else they told us."""
    state = new_session_state("s1")
    state["slots"] = {"axle_capacity_lbs": 7000.0}
    qualification_node(state)
    assert state["qualification_complete"] is False
    assert "type of trailer" in state["turn_outcome"]["next_question"]
    # And the rating survives for whichever category they pick next.
    assert state["slots"]["axle_capacity_lbs"] == 7000.0


def test_a_zero_axle_rating_is_not_treated_as_given():
    """`is not None`, not truthiness - but a 0 rating is meaningless, so it must not skip."""
    state = _state("Dump", axle=0.0)
    qualification_node(state)
    # 0.0 is not a usable rating; the weight question must still be asked.
    assert state["pending_question_slot"] == _weight_slot("Dump")
