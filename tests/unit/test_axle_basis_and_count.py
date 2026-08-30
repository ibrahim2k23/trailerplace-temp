"""Per-axle vs total, and how many axles.

A per-axle rating and a total are different trailers, and the old prompt resolved that
ambiguity silently by always reading a capacity as per-axle. These tests cover the three ways
it now goes: read confidently from the wording, asked about when it genuinely cannot be read,
and refused when the count is not one we stock.
"""
from __future__ import annotations

import pytest

from src.graph.apply_analysis import (
    AXLE_BASIS_QUESTION,
    AXLE_COUNT_QUESTION,
    apply_analysis_to_state,
)
from src.graph.nodes.qualification import qualification_node
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis


def _turn(category: str = "Equipment", *, text: str = "", **extracted) -> dict:
    state = new_session_state("s1")
    state["category"] = category
    state["messages"] = [{"role": "user", "content": text}]
    state["turn"] = sample_analysis(
        category_mentioned=category,
        intent="qualification_answer",
        extracted=extracted,
        slot_answers=[],
    )
    return apply_analysis_to_state(state)


def _reply(state: dict, text: str) -> dict:
    """Send a follow-up message into an existing conversation."""
    state["messages"] = [{"role": "user", "content": text}]
    state["turn"] = sample_analysis(intent="qualification_answer", extracted={}, slot_answers=[])
    return apply_analysis_to_state(state)


# --- read confidently, ask nothing -----------------------------------------------------


def test_a_per_axle_rating_is_stored_as_per_axle():
    state = _turn(text="I want 7,000 lb axles",
                  axle_capacity_lbs=7000.0, axle_capacity_basis="per_axle")
    assert state["slots"]["axle_capacity_lbs"] == 7000.0
    assert state["slots"].get("total_axle_capacity_lbs") is None
    assert state["pending_axle_basis"] is None


def test_a_total_is_stored_as_a_total_and_never_as_per_axle():
    """The regression this whole change exists for: 14,000 total is not 14,000 per axle."""
    state = _turn(text="it needs to carry 14,000 lbs across the axles",
                  total_axle_capacity_lbs=14000.0, axle_capacity_basis="total")
    assert state["slots"]["total_axle_capacity_lbs"] == 14000.0
    assert state["slots"].get("axle_capacity_lbs") is None


def test_a_count_given_with_the_rating_is_not_asked_about_again():
    """A phrase like 2-7,000# axles carries both facts, so the count question never comes up."""
    state = _turn(text="2-7,000# axles", axle_capacity_lbs=7000.0,
                  axle_count=2, axle_capacity_basis="per_axle")
    assert state["slots"]["axle_count"] == 2
    assert state["turn_outcome"].get("clarification_question") != AXLE_COUNT_QUESTION


# --- ask, when it genuinely cannot be read ---------------------------------------------


def test_an_unclear_capacity_is_held_and_questioned_not_guessed():
    state = _turn(text="14,000 lbs of axle capacity",
                  axle_capacity_lbs=14000.0, axle_capacity_basis="unclear")
    assert state["turn_outcome"]["clarification_question"] == AXLE_BASIS_QUESTION
    # Held, NOT filed as a per-axle rating - that was the old silent wrong answer.
    assert state["slots"].get("axle_capacity_lbs") is None
    assert state["pending_axle_basis"] == {"value": 14000.0}


@pytest.mark.parametrize(
    "reply,slot",
    [
        ("per axle", "axle_capacity_lbs"),
        ("each axle", "axle_capacity_lbs"),
        ("that is the total", "total_axle_capacity_lbs"),
        ("combined", "total_axle_capacity_lbs"),
    ],
)
def test_the_held_capacity_lands_where_their_answer_puts_it(reply, slot):
    state = _turn(text="14,000 lbs of axle capacity",
                  axle_capacity_lbs=14000.0, axle_capacity_basis="unclear")
    state = _reply(state, reply)

    assert state["slots"][slot] == 14000.0
    assert state["pending_axle_basis"] is None


def test_an_unreadable_reply_asks_once_more_rather_than_guessing():
    state = _turn(text="14,000 lbs of axle capacity",
                  axle_capacity_lbs=14000.0, axle_capacity_basis="unclear")
    state = _reply(state, "hmm not sure what you mean")

    assert state["turn_outcome"]["clarification_question"] == AXLE_BASIS_QUESTION
    assert state["pending_axle_basis"] == {"value": 14000.0}


def test_a_per_axle_rating_with_no_count_asks_how_many():
    state = _turn(text="7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    assert state["turn_outcome"]["clarification_question"] == AXLE_COUNT_QUESTION


def test_the_count_question_is_not_asked_after_they_declined_it():
    state = _turn(text="7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    state["skipped_slots"] = ["axle_count"]
    state = _reply(state, "skip that")

    assert state["turn_outcome"].get("clarification_question") != AXLE_COUNT_QUESTION


# --- refuse a count we do not stock ------------------------------------------------------


@pytest.mark.parametrize("count", [0, 5, 7])
def test_a_count_outside_one_to_four_is_refused_not_stored(count):
    state = _turn(text="{} axles".format(count), axle_count=count)

    rejected = state["turn_outcome"]["rejected_answers"]
    assert [r["reason"] for r in rejected] == ["axle_count_range"]
    # NOT stored, and NOT stored as None either: a null would read as "asked, no preference"
    # and close the question for good, which is the opposite of asking again.
    assert "axle_count" not in state["slots"]


def test_a_zero_count_is_refused_rather_than_swallowed_as_no_preference():
    """The ordering trap.

    The zero rule turns a stated 0 into "no preference" for every other measurement. A
    zero-axle trailer is not a preference, it is an answer that cannot be true, so the range
    check has to run before that rule sees it.
    """
    state = _turn(text="0 axles", axle_count=0)
    assert state["turn_outcome"]["rejected_answers"][0]["reason"] == "axle_count_range"
    assert "axle_count" not in state["slots"]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_a_count_inside_the_range_is_stored(count):
    state = _turn(text="{} axles".format(count), axle_count=count)
    assert state["slots"]["axle_count"] == count
    assert not state["turn_outcome"].get("rejected_answers")


# --- the weight question --------------------------------------------------------------


@pytest.mark.parametrize("slot", ["axle_capacity_lbs", "total_axle_capacity_lbs"])
def test_either_axle_rating_skips_the_weight_question(slot):
    """A total specifies the trailer from the capacity end just as a per-axle rating does."""
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["slots"] = {slot: 7000.0, "haul_item": "a skid steer", "hitch_type": ["Gooseneck"],
                      "haul_length_ft": 20.0, "item_or_trailer_width_ft": 8.0}
    state = qualification_node(state)

    assert state["turn_outcome"].get("next_question") is None
    assert state["qualification_complete"] is True


def test_only_one_question_is_asked_per_turn():
    """A clarification outstanding must suppress the slot question, or two are asked at once."""
    state = new_session_state("s1")
    state["category"] = "Equipment"
    state["turn_outcome"] = {"clarification_question": AXLE_BASIS_QUESTION}
    state = qualification_node(state)

    assert state["turn_outcome"].get("next_question") is None
    assert state["qualification_complete"] is False


# --- the deterministic backstop --------------------------------------------------------
#
# Live, the model read "14,000 lbs of axle capacity" as confidently per_axle and the
# question was never put. Its confidence is not evidence, so the wording gets a veto.


@pytest.mark.parametrize(
    "text,model_basis,expected",
    [
        # No marker either way: unclear, whatever the model claimed.
        ("Also it needs 14,000 lbs of axle capacity.", "per_axle", "unclear"),
        ("I need 14,000 lbs of axle capacity", "total", "unclear"),
        # The rating hangs off the axles themselves - conventional, not ambiguous.
        ("I want 7,000 lb axles", "per_axle", "per_axle"),
        ("2-7,000# axles", "per_axle", "per_axle"),
        ("tandem 5200# axles", "per_axle", "per_axle"),
        # They said which, in so many words.
        ("14,000 lbs total across the axles", "total", "total"),
        ("7000 per axle", "per_axle", "per_axle"),
        # Nothing stated at all stays nothing.
        ("just a dump trailer please", None, None),
    ],
)
def test_the_wording_can_veto_the_models_confidence(text, model_basis, expected):
    from src.graph.apply_analysis import infer_axle_basis_from_text

    assert infer_axle_basis_from_text(text, model_basis) == expected


def test_a_bare_capacity_is_questioned_even_when_the_model_was_sure():
    """The exact live failure: the model said per_axle, and we ask anyway."""
    state = _turn(text="Also it needs 14,000 lbs of axle capacity.",
                  axle_capacity_lbs=14000.0, axle_capacity_basis="per_axle")

    assert state["turn_outcome"]["clarification_question"] == AXLE_BASIS_QUESTION
    assert state["slots"].get("axle_capacity_lbs") is None
    assert state["pending_axle_basis"] == {"value": 14000.0}


def test_conventional_per_axle_wording_is_never_questioned():
    """The other half: asking about "7,000 lb axles" would be a needless extra turn."""
    state = _turn(text="I want 7,000 lb axles",
                  axle_capacity_lbs=7000.0, axle_capacity_basis="per_axle")

    assert state["turn_outcome"].get("clarification_question") != AXLE_BASIS_QUESTION
    assert state["slots"]["axle_capacity_lbs"] == 7000.0


def test_a_bare_capacity_is_held_whichever_slot_the_model_guessed():
    """The live shape that slipped through the first fix.

    Asked about "14,000 lbs of axle capacity", the model returned it as a confident TOTAL,
    not as a per-axle rating. A hold watching only axle_capacity_lbs never saw it, and the
    question went unasked - the guess was different from the old prompt's, but still a guess.
    """
    state = _turn(text="Also it needs 14,000 lbs of axle capacity.",
                  total_axle_capacity_lbs=14000.0, axle_capacity_basis="total")

    assert state["turn_outcome"]["clarification_question"] == AXLE_BASIS_QUESTION
    assert state["slots"].get("total_axle_capacity_lbs") is None
    assert state["slots"].get("axle_capacity_lbs") is None
    assert state["pending_axle_basis"] == {"value": 14000.0}


def test_the_parked_number_wins_over_the_models_re_reading():
    """Live, answering "per axle" made the model re-read 14,000 and hand back 7,000.

    It had quietly assumed two axles and halved the figure. The customer said 14,000 per
    axle, so the parked value is filed last and wins over anything re-extracted that turn.
    """
    state = _turn(text="Also it needs 14,000 lbs of axle capacity.",
                  total_axle_capacity_lbs=14000.0, axle_capacity_basis="total")
    assert state["pending_axle_basis"] == {"value": 14000.0}

    # The answering turn, where the model offers its halved re-reading.
    state["messages"] = [{"role": "user", "content": "Per axle, please."}]
    state["turn"] = sample_analysis(
        intent="qualification_answer",
        extracted={"axle_capacity_lbs": 7000.0, "axle_capacity_basis": "per_axle"},
        slot_answers=[],
    )
    state = apply_analysis_to_state(state)

    assert state["slots"]["axle_capacity_lbs"] == 14000.0, "their number, not the model's"
    assert state["slots"].get("total_axle_capacity_lbs") is None
    assert state["pending_axle_basis"] is None


# --- settling the count question ---------------------------------------------------------


def test_the_count_question_names_the_options():
    """Numbers alone make people guess at the vocabulary; the words are how they think."""
    assert "single" in AXLE_COUNT_QUESTION.lower()
    assert "tandem" in AXLE_COUNT_QUESTION.lower()
    assert "triple" in AXLE_COUNT_QUESTION.lower()
    # The stocked range is enforced when they answer, not advertised in the question.
    assert "four" not in AXLE_COUNT_QUESTION.lower()


@pytest.mark.parametrize(
    "shrug",
    ["I'm not sure", "no preference", "whatever you'd recommend", "I don't know"],
)
def test_not_knowing_is_recorded_as_no_preference_not_asked_again(shrug):
    """The loop bug: nothing was stored, so the question came back every turn forever."""
    state = _turn(text="7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    assert state["turn_outcome"]["clarification_question"] == AXLE_COUNT_QUESTION

    state = _reply(state, shrug)

    # Null is how every other unanswered question is recorded: asked, no preference.
    assert state["slots"]["axle_count"] is None
    assert state["turn_outcome"].get("clarification_question") != AXLE_COUNT_QUESTION

    # And a third turn must not raise it again either.
    state = _reply(state, "what colours do you have?")
    assert state["turn_outcome"].get("clarification_question") != AXLE_COUNT_QUESTION


def test_a_real_answer_settles_the_question():
    state = _turn(text="7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    state["messages"] = [{"role": "user", "content": "tandem"}]
    state["turn"] = sample_analysis(
        intent="qualification_answer", extracted={"axle_count": 2}, slot_answers=[]
    )
    state = apply_analysis_to_state(state)

    assert state["slots"]["axle_count"] == 2
    assert state["turn_outcome"].get("clarification_question") != AXLE_COUNT_QUESTION


def test_an_out_of_range_answer_keeps_the_question_open():
    """A refusal must NOT be settled as "no preference" - they are being asked again."""
    state = _turn(text="7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    state["messages"] = [{"role": "user", "content": "seven axles"}]
    state["turn"] = sample_analysis(
        intent="qualification_answer", extracted={"axle_count": 7}, slot_answers=[]
    )
    state = apply_analysis_to_state(state)

    assert state["turn_outcome"]["rejected_answers"][0]["reason"] == "axle_count_range"
    assert "axle_count" not in state["slots"], "not stored, and not nulled either"
    assert state["pending_axle_count_question"] is True, "still open, they get another go"


# --- their words beat the model's fields --------------------------------------------------


def test_the_parked_number_is_the_one_they_said_not_the_models_halved_copy():
    """Live, the model emitted BOTH a halved per-axle guess and the real total.

    Parking whichever field came first held 7,000 - so answering "that's the total" stored
    7,000 as the total, half what they asked for. The figure in their sentence is the truth.
    """
    state = _turn(text="Also it needs 14,000 lbs of axle capacity.",
                  axle_capacity_lbs=7000.0, total_axle_capacity_lbs=14000.0,
                  axle_capacity_basis="per_axle")
    assert state["pending_axle_basis"] == {"value": 14000.0}

    state = _reply(state, "That is the total across all of them.")
    assert state["slots"]["total_axle_capacity_lbs"] == 14000.0


@pytest.mark.parametrize(
    "reply,count",
    [("Single.", 1), ("Tandem.", 2), ("Triple.", 3), ("two", 2), ("4 axles", 4)],
)
def test_a_bare_count_word_is_read_from_their_reply(reply, count):
    """The model does not reliably turn a bare "Single." into a number.

    It was answering a question it had not really been asked, so nothing was extracted and
    the answer was written off as no preference. The words are read directly instead.
    """
    state = _turn(text="I want 7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    assert state["turn_outcome"]["clarification_question"] == AXLE_COUNT_QUESTION

    state = _reply(state, reply)
    assert state["slots"]["axle_count"] == count


def test_a_spoken_count_out_of_range_is_still_refused():
    """Reading their words must not become a way round the 1-4 rule."""
    state = _turn(text="I want 7,000 lb axles", axle_capacity_lbs=7000.0,
                  axle_capacity_basis="per_axle")
    state = _reply(state, "seven")

    assert state["turn_outcome"]["rejected_answers"][0]["reason"] == "axle_count_range"
    assert "axle_count" not in state["slots"]
    assert state["pending_axle_count_question"] is True
