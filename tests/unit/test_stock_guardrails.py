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
    # Length rather than cargo_size: concession trailers are bought by length.
    assert spec.required == ["use_case", "trailer_length_ft"]
    assert "serving" in spec.questions["use_case"]


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
    ending = prompt[prompt.index("=== HOW TO END A REPLY THAT SHOWS LISTINGS ===") :][:600]
    assert "the SALES-REP LINE, then ONE closing question" in ending
    assert PHONE in ending
    # The old blanket ban listed "calls" among the forbidden extras; it must not still.
    forbidden_list = ending[ending.index("Nothing else after the listings") :]
    assert "calls" not in forbidden_list


def test_sales_rep_line_is_bounded_so_it_cannot_become_a_brush_off():
    prompt = _respond_prompt()
    rule = prompt[prompt.index("=== THE SALES-REP LINE") :]
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
    assert "ANYTHING WE CANNOT DO FOR THEM ALWAYS RAISES A TRIGGER" in system
    for case in ("can you beat 8k", "delivery scheduling", "trade-in -> faq/trade_in"):
        assert case in system
    # escalation must stay the complaint/urgency signal, not the label for every price question.
    assert "keep\n  escalation for complaints and urgency" in system


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
    assert "This is the CLOSING line. The SALES-REP line below is a different" in prompt
