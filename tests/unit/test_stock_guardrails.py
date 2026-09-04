"""The two prompt guardrails and the Concession category they were added alongside.

These are PROMPT guardrails by request: nothing in code forces the model to obey them, so what
is testable is that the prompt states them, states them unambiguously, and is fed the right
facts. The facts are the part that would rot silently - an unstocked category list built by
hand would drift from the catalogue the day stock changed.
"""
from __future__ import annotations

import pytest

from src.domain import brands
from src.domain.categories import (
    CANONICAL_CATEGORIES,
    advertised_categories_line,
    resolve_category_from_text,
    unstocked_categories,
    unstocked_categories_block,
)
from src.domain.trailer_fields import get_trailer_fields
from src.graph.state import new_session_state
from src.llm.respond import build_respond_prompt
from tests.unit.llm_helpers import sample_analysis

PHONE = "979-532-1486"


def _respond_prompt(category: str = "Dump") -> str:
    state = new_session_state("s1")
    state["category"] = category
    system, _ = build_respond_prompt(state, sample_analysis(), {})
    return system


# --- Concession ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["I need a concession trailer", "looking for a food trailer", "do you have food trailers"],
)
def test_concession_terms_resolve(text):
    """Recognising the type is the point: misheard, "food trailer" would route to Enclosed
    or nowhere, and the customer would never be told we do not stock it."""
    assert resolve_category_from_text(text).category == "Concession"


def test_concession_is_canonical_but_not_advertised():
    """It is a type we KNOW, not a type we SELL. Both halves matter."""
    assert "Concession" in CANONICAL_CATEGORIES
    assert "Concession" in unstocked_categories()
    assert "Concession" not in advertised_categories_line()


def test_concession_has_its_own_qualification_spec():
    """Without one it silently falls back to _DEFAULT_SPEC's "what will you be hauling?"."""
    spec = get_trailer_fields("Concession")
    assert spec.category == "Concession"
    # Length rather than cargo_size: concession trailers are bought by length, and length
    # is the ONLY required slot - use_case was dropped so the flow stays one question long.
    assert spec.required == ["trailer_length_ft"]
    assert "length" in spec.questions["trailer_length_ft"].lower()
    assert "serving" in spec.questions["ac_windows_cabinets"]


# --- Guardrail 1: not in stock -------------------------------------------------


def test_unstocked_block_is_derived_from_the_catalogue():
    """Hand-maintained, this list would advertise a sold-out category as available."""
    block = unstocked_categories_block()
    assert "Concession" in block
    # The customer's words, not ours - nobody asks for "a Concession".
    assert "food trailer" in block
    for stocked in ("Dump", "Livestock", "Utility"):
        assert stocked not in block


def test_unstocked_block_reports_a_fully_stocked_catalogue(monkeypatch):
    monkeypatch.setattr("src.domain.categories._advertised_categories", lambda: tuple(CANONICAL_CATEGORIES))
    assert unstocked_categories() == ()
    assert "None" in unstocked_categories_block()


def test_prompt_carries_the_not_in_stock_rule_and_its_data():
    prompt = _respond_prompt()
    assert "WE DO NOT CURRENTLY STOCK THESE" in prompt
    assert "Concession (they may call it:" in prompt
    assert "food trailer" in prompt
    # The three beats the rule demands, in order.
    assert "not have that type in stock" in prompt
    assert "What we DO carry" in prompt
    assert prompt.count(PHONE) >= 2


def test_prompt_forbids_qualifying_for_an_unstocked_type():
    """The failure that would embarrass us is working someone through four questions for a
    trailer we cannot sell them."""
    prompt = _respond_prompt()
    assert "Never qualify them for a type on this list" in prompt


# --- Guardrail 2: offer a human ------------------------------------------------


def test_prompt_carries_the_sales_rep_rule():
    prompt = _respond_prompt()
    assert "THE SALES-REP LINE" in prompt
    assert "You cannot answer, cannot check, or cannot do what they asked" in prompt
    # The trigger the request was really about: a refusal must never be the end of a reply.
    assert 'must\n  NEVER be the end of a reply' in prompt
    for trigger in ("showing listings", "do not stock what they asked for", "speak to a person"):
        assert trigger in prompt


def test_listings_ending_rule_now_permits_the_sales_rep_line():
    """This rule used to ban "calls" outright after listings - the exact opposite of the new
    guardrail. If both survived, the model would be obeying a coin flip."""
    prompt = _respond_prompt()
    ending = prompt[prompt.index("-- 5D. HOW TO END A REPLY THAT SHOWS LISTINGS --") :][:600]
    assert "the SALES-REP LINE, then ONE closing question" in ending
    assert PHONE in ending
    # The old blanket ban listed "calls" among the forbidden extras; it must not still.
    forbidden_list = ending[ending.index("Nothing else after the listings") :]
    assert "calls" not in forbidden_list


def test_sales_rep_line_is_bounded_so_it_cannot_become_a_brush_off():
    prompt = _respond_prompt()
    rule = prompt[prompt.index("-- 5G. THE SALES-REP LINE") :]
    assert "ONCE per reply, and never as the entire reply" in rule
    assert "Never on a plain qualification turn" in rule
    assert "never alongside the CLOSING line" in rule


# --- Alerting the team when we fall short --------------------------------------


def test_every_canned_reply_for_a_turn_we_could_not_settle_names_the_number():
    """The deterministic half of guardrail 2.

    These strings fire on turns we could NOT handle ourselves, and they arrive as an ORDER, so
    the model treats them as the answer and drops the sales-rep line it would have written.
    Measured live before this was fixed: the phone number appeared on only ~5 of 9 such turns.
    A canned string is not model output, so putting the number here makes it certain.
    """
    from src.domain.canned_responses import CANNED_RESPONSES

    for key in (
        "generic_team_request",
        "escalation",
        "contact_human",
        "financing",
        "trade_in",
        "service_parts",
        "listing_interest_fallback",
    ):
        assert PHONE in CANNED_RESPONSES[key], f"{key} leaves the customer with no way to reach anyone"


def test_analyze_prompt_requires_an_alert_when_we_cannot_help():
    """Respond has no tools - it returns text. The alert is Analyze's email_triggers."""
    from src.llm.analyze import build_analyze_prompt

    state = new_session_state("s1")
    state["category"] = "Dump"
    system, _ = build_analyze_prompt(state)
    assert "ANYTHING WE CANNOT ANSWER RAISES A TRIGGER" in system
    # both halves must survive: the concrete list AND the catch-all test behind it
    assert "AND ANYTHING ELSE THAT PASSES THIS TEST" in system
    for case in ("can you beat 8k", "delivery scheduling", "trade-in -> faq/trade_in"):
        assert case in system
    # escalation must stay the complaint/urgency signal, not the label for every price question.
    # Asserted whitespace-insensitively: the meaning matters, not where the line wraps.
    assert "keep escalation for complaints and urgency" in " ".join(system.split()).lower()


def test_analyze_prompt_knows_which_types_are_unstocked():
    """Without this block Analyze cannot tell a not-in-stock ask from an unknown word, so the
    lead most likely to be lost is the one nobody hears about."""
    from src.llm.analyze import build_analyze_prompt

    system, _ = build_analyze_prompt(new_session_state("s1"))
    assert "TYPES WE RECOGNISE BUT DO NOT CURRENTLY STOCK" in system
    assert "Concession" in system
    assert "email_triggers MUST contain one entry, kind=team_request" in system
    assert "phrased as a QUESTION" in system


def test_closing_line_and_sales_rep_line_are_distinguished():
    """Two different phone lines with different triggers; without this they blur together."""
    prompt = _respond_prompt()
    # The two lines share section 5G, which opens by naming them as distinct.
    assert "These are TWO DIFFERENT lines with different triggers" in prompt
    assert "Use one or the other in a reply, NEVER both" in prompt


# --- A complaint is not a sales opportunity -------------------------------------


def test_escalation_canned_text_never_offers_to_keep_selling():
    """Seen live: "I have a complaint against you guys" was answered with the full 13-item
    category list and "which type do you want to go with?". The canned text itself carried
    "In the meantime, I can keep helping you narrow down the right trailer", and canned text
    reaches the respond model as an order."""
    from src.domain.canned_responses import CANNED_RESPONSES

    escalation = CANNED_RESPONSES["escalation"].lower()
    for pitch in ("narrow down", "keep helping", "right trailer"):
        assert pitch not in escalation, f"escalation text still pitches a trailer: {pitch!r}"
    assert PHONE in CANNED_RESPONSES["escalation"]


def test_escalation_turn_asks_no_qualification_question():
    """qualification_node emits "What type of trailer are you looking for?" whenever no
    category is settled - regardless of what the customer actually said."""
    from src.graph.nodes.qualification import qualification_node

    state = {"category": None, "slots": {}, "turn_outcome": {"escalation_owns_turn": True}}
    qualification_node(state)
    assert not state["turn_outcome"].get("next_question")
    assert state["qualification_complete"] is False

    # ...and the ordinary no-category turn is untouched.
    plain = {"category": None, "slots": {}, "turn_outcome": {}}
    qualification_node(plain)
    assert plain["turn_outcome"]["next_question"] == "What type of trailer are you looking for?"


def test_escalation_owns_the_reply_in_the_respond_prompt():
    from src.llm.respond import build_respond_prompt
    from tests.unit.llm_helpers import sample_analysis

    state = {"messages": [{"role": "user", "content": "I have a complaint"}], "slots": {}}
    system, _ = build_respond_prompt(
        state, sample_analysis(), {"escalation_owns_turn": True, "canned_keys": ["escalation"]}
    )
    assert "THIS OWNS THE REPLY" in system
    assert "do NOT list, recommend, or bullet trailer types" in system
    # ...but the door is left open, as a statement rather than a question.
    assert "if they are looking for a trailer as well" in system


def test_analyze_prompt_requires_a_trigger_for_future_stock_questions():
    """Seen live: "when will your new stock of trailers come" was labelled general_question
    and raised NO email trigger, so the lead vanished - the customer then handed over name
    and email and the team was never told. The trigger table was a closed list of examples
    and this case was not on it."""
    from src.llm.analyze import build_analyze_prompt

    system, _ = build_analyze_prompt(new_session_state("s1"))
    assert "AND ANYTHING ELSE THAT PASSES THIS TEST" in system
    assert "when new stock arrives" in system
    assert "paperwork, titling, registration" in system
    assert "seeing, viewing, visiting" in system
    # the other half: an ordinary request for a category we stock must NOT alert the team
    assert "DO NOT RAISE ONE WHEN THE ANSWER IS ALREADY YOURS TO GIVE" in system
    assert "no trigger" in system
    # the intent that let it slip must point at the rule
    assert "This intent still raises an email trigger" in system


# --- Axles: count and capacity are fields; the axle's TYPE is a feature ---------


def test_axle_count_words_map_to_numbers():
    from src.domain.slot_map import parse_axle_count_answer

    for word, expected in (
        ("single", 1), ("SA", None), ("one axle", 1),
        ("tandem", 2), ("double", 2), ("dual", 2), ("two axles", 2),
        ("triple", 3), ("tri", 3), ("three axles", 3),
        ("quad", 4), ("quadruple", 4), ("four axles", 4),
    ):
        if expected is not None:
            assert parse_axle_count_answer(word) == expected, f"{word!r} -> {expected}"


def test_axle_count_and_capacity_never_become_features():
    from src.domain.slot_map import sanitize_non_metadata_features

    for junk in ("10k axles", "tandem axles", "single axle", "triple axles",
                 "two 3500 lb axles", "axle capacity", "7000 lb axle"):
        kept, _ = sanitize_non_metadata_features([junk])
        assert kept == [], f"{junk!r} leaked into non_metadata_features as {kept}"


def test_axle_type_survives_as_a_feature():
    """We hold NO metadata field for the axle's construction, so the feature list is the only
    place "torsion axles" can do any work. The old guard dropped every phrase containing the
    word "axle" and threw it away with the counts."""
    from src.domain.slot_map import sanitize_non_metadata_features

    for real in ("torsion axles", "drop axles", "spring axles"):
        kept, _ = sanitize_non_metadata_features([real])
        assert kept, f"{real!r} was discarded but it is a real feature"

    # one phrase can carry both: the count/capacity goes to its field, the type stays a feature
    kept, _ = sanitize_non_metadata_features(["torsion axles", "7000 lb axles"])
    assert kept == ["torsion axles"]


def test_electric_brakes_are_not_read_as_a_triple_axle_count():
    """Without word boundaries "tri" matches inside "electric" and "one" inside "stone"."""
    from src.domain.slot_map import axle_phrase_is_count_or_capacity

    assert axle_phrase_is_count_or_capacity("torsion axles with electric brakes") is False
    assert axle_phrase_is_count_or_capacity("tandem axles") is True
