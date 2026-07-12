"""The turn where a customer hands over contact details must do exactly one thing: send the
emails that were waiting on it. Live, it also invented a brand preference from the listing on
screen, re-ran Pinecone under that brand, banked six listings the customer never saw, and
mailed the team the same lead twice."""
from __future__ import annotations

from src.domain.brands import brand_mentioned_in_text
from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.build import should_search
from src.graph.nodes.email_actions import email_actions_node
from src.graph.nodes.respond import _record_shown_listings
from src.graph.state import new_session_state
from src.llm.schemas import ReplyOutput
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted, say


def contact(name=None, email=None, phone=None) -> dict:
    return {"name": name, "email": email, "phone": phone}


def trigger(kind, faq_key=None, ref=None) -> dict:
    return {"kind": kind, "faq_key": faq_key, "listing_reference": ref, "description": f"{kind} request"}


def after_results() -> dict:
    state = new_session_state("s1")
    state["contact_gate_closed"] = True
    state["category"] = "Dump"
    state["qualification_complete"] = True
    state["search_pending"] = False
    state["shown_listings"] = [{"title": "2026 Iron Bull Trailers Dump", "url": "https://x/2"}]
    state["shown_urls"] = ["https://x/2"]
    return state


def turn(state, text, **updates):
    say(state, text)
    updates.setdefault("contact", contact())
    updates.setdefault("extracted", _empty_extracted())
    updates.setdefault("slot_answers", [])
    state["turn"] = sample_analysis(**updates)
    apply_analysis_to_state(state)
    email_actions_node(state)
    return state["turn_outcome"]


def test_a_brand_read_off_a_listing_is_not_a_brand_preference():
    state = after_results()
    turn(
        state,
        "sure, my name is Ibrahim and email is ibrahim@esided.ai",
        intent="contact_info_provided",
        contact=contact(name="Ibrahim", email="ibrahim@esided.ai"),
        extracted={**_empty_extracted(), "brand_preference": "Iron Bull Trailers"},
    )
    assert state["brand_preference"] is None
    # ...and with no phantom brand, nothing about the search changed, so none runs.
    assert should_search(state) is False


def test_a_brand_the_customer_actually_names_is_kept():
    state = after_results()
    turn(state, "do you have any Iron Bull ones?", intent="requirement_change",
         extracted={**_empty_extracted(), "brand_preference": "Iron Bull Trailers"})
    assert state["brand_preference"] == "Iron Bull Trailers"


def test_brand_mentioned_in_text_tolerates_typos_but_not_the_word_trailer():
    assert brand_mentioned_in_text("Diamond C", "I want a dimond c please")
    assert brand_mentioned_in_text("Iron Bull Trailers", "any iron bull?")
    # "trailer" appears in half our messages — it must not match "Iron Bull Trailers".
    assert not brand_mentioned_in_text("Iron Bull Trailers", "I need a trailer for my tools")
    assert not brand_mentioned_in_text("Iron Bull Trailers", "my email is a@b.c")


def with_listings() -> dict:
    state = after_results()
    state["customer_name"] = "Ibrahim"
    state["customer_email"] = "a@b.c"
    state["shown_listings"] = [
        {"title": "2026 Iron Bull Trailers Dump - 11350", "url": "https://x/1", "make": "Iron Bull Trailers"},
        {"title": "2025 Diamond C LPT210 - 93663", "url": "https://x/2", "make": "Diamond C"},
    ]
    state["shown_urls"] = ["https://x/1", "https://x/2"]
    return state


def logged_item(outcome) -> str | None:
    for event in outcome.get("outbox_events", []) or []:
        if event.get("item_of_interest"):
            return event["item_of_interest"]
    return None


def test_a_make_used_to_point_at_a_listing_is_not_a_brand_preference():
    # "I like the Iron Bull one" is how people pick a trailer off a list. It names the make,
    # but it is a finger, not a filter — recording it would narrow every later search to that
    # one manufacturer because they liked a single trailer.
    state = with_listings()
    outcome = turn(state, "I like the iron bull one", intent="listing_interest", listing_reference=None,
                   extracted={**_empty_extracted(), "brand_preference": "Iron Bull Trailers"},
                   email_triggers=[trigger("listing_interest")])
    assert state["brand_preference"] is None
    assert should_search(state) is False
    # ...and we still work out WHICH trailer they meant, so the lead names it.
    assert logged_item(outcome) == "2026 Iron Bull Trailers Dump - 11350"


def test_a_listing_referenced_by_stock_number_resolves():
    state = with_listings()
    outcome = turn(state, "tell me more about the 93663", intent="listing_interest", listing_reference=None,
                   email_triggers=[trigger("listing_interest")])
    assert logged_item(outcome) == "2025 Diamond C LPT210 - 93663"


def test_an_ambiguous_make_logs_the_interest_without_guessing_the_trailer():
    state = with_listings()
    state["shown_listings"].append(
        {"title": "2026 Iron Bull Trailers Dump - 11351", "url": "https://x/3", "make": "Iron Bull Trailers"}
    )
    outcome = turn(state, "I like the iron bull one", intent="listing_interest", listing_reference=None,
                   email_triggers=[trigger("listing_interest")])
    assert outcome["emails_sent"] == ["Listing Interest"]
    # Two Iron Bulls on screen: better to log interest with no trailer than to log the wrong one.
    assert logged_item(outcome) is None


def test_contact_details_never_re_search_once_results_are_on_screen():
    state = after_results()
    state["search_pending"] = True  # even if something did change
    turn(state, "my email is a@b.c", intent="contact_info_provided", contact=contact(email="a@b.c"))
    assert should_search(state) is False


def test_the_same_listing_interest_is_not_emailed_twice():
    state = after_results()
    # Turn 1: they like one, we hold the email while we ask for their details.
    turn(state, "I like the 2nd one", intent="listing_interest", listing_reference=1,
         email_triggers=[trigger("listing_interest", ref=1)])
    assert len(state["pending_email_actions"]) == 1
    # Turn 2: details arrive AND the extractor re-emits the same trigger. One lead, one email.
    outcome = turn(state, "Ibrahim, ibrahim@esided.ai",
                   intent="contact_info_provided", contact=contact(name="Ibrahim", email="ibrahim@esided.ai"),
                   email_triggers=[trigger("listing_interest", ref=1)])
    assert outcome["emails_sent"] == ["Listing Interest"]


def test_only_listings_the_reply_actually_showed_count_as_shown():
    state = after_results()
    outcome = {
        "listings": [{"title": "A", "url": "https://x/10"}, {"title": "B", "url": "https://x/11"}],
        "outbox_events": [{"event_type": "results_shown", "event_key": "results_shown:0"}],
        "emails_sent": ["Results Shown to User"],
    }
    reply = ReplyOutput(assistant_text="here is one", cited_listing_urls=["https://x/10"])
    _record_shown_listings(state, outcome, reply)
    assert set(state["shown_urls"]) == {"https://x/2", "https://x/10"}
    assert [item["url"] for item in state["shown_listings"]] == ["https://x/2", "https://x/10"]


def test_a_reply_that_shows_nothing_does_not_tell_the_team_we_showed_results():
    state = after_results()
    outcome = {
        "listings": [{"title": "A", "url": "https://x/10"}],
        "outbox_events": [{"event_type": "results_shown", "event_key": "results_shown:0"}],
        "emails_sent": ["Results Shown to User"],
    }
    reply = ReplyOutput(assistant_text="thanks for your email!", cited_listing_urls=[])
    _record_shown_listings(state, outcome, reply)
    assert set(state["shown_urls"]) == {"https://x/2"}   # x/10 stays available for "show me more"
    assert outcome["outbox_events"] == []
    assert outcome["emails_sent"] == []
