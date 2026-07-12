from __future__ import annotations

from src.graph.apply_analysis import apply_analysis_to_state
from src.graph.nodes.email_actions import email_actions_node
from src.graph.state import new_session_state
from tests.unit.llm_helpers import sample_analysis
from tests.unit.test_apply_analysis import _empty_extracted


def contact(name=None, email=None, phone=None) -> dict:
    return {"name": name, "email": email, "phone": phone}


def trigger(kind, faq_key=None, ref=None) -> dict:
    return {"kind": kind, "faq_key": faq_key, "listing_reference": ref, "description": f"{kind} request"}


def declined_customer() -> dict:
    """Someone who brushed off the opening contact invite but is still shopping."""
    state = new_session_state("s1")
    state["contact_gate_closed"] = True
    state["contact_declined"] = True
    state["category"] = "Equipment"
    state["qualification_complete"] = True
    state["shown_listings"] = [{"title": "Diamond C LPX207", "url": "https://x/1"}]
    state["shown_urls"] = ["https://x/1"]
    return state


def turn(state, text, **updates):
    state.setdefault("messages", []).append({"role": "user", "content": text})
    updates.setdefault("contact", contact())
    state["turn"] = sample_analysis(extracted=_empty_extracted(), slot_answers=[], **updates)
    apply_analysis_to_state(state)
    email_actions_node(state)
    return state["turn_outcome"]


def test_listing_interest_reopens_the_contact_question_after_an_early_decline():
    # Brushing off the opening invite is not a standing refusal to ever be contacted. When
    # they later ASK us to act — "I like this one, set up a call" — we cannot pass it to the
    # team without a way to reach them, so we ask again.
    state = declined_customer()
    outcome = turn(
        state,
        "the 1st one catches my eye. I want to set up a call",
        intent="listing_interest",
        listing_reference=1,
        email_triggers=[trigger("listing_interest", ref=1), trigger("team_request")],
    )
    assert outcome["emails_sent"] == []
    assert outcome["contact_followup_missing"] == ["name", "email or phone"]
    assert len(state["pending_email_actions"]) == 2  # held, not lost
    assert state["contact_declined"] is False


def test_giving_the_details_then_fires_everything_that_was_waiting():
    state = declined_customer()
    turn(state, "the 1st one catches my eye. set up a call",
         intent="listing_interest", listing_reference=1,
         email_triggers=[trigger("listing_interest", ref=1), trigger("team_request")])
    outcome = turn(state, "Ibrahim, 03304388550",
                   intent="contact_info_provided", contact=contact(name="Ibrahim", phone="03304388550"))
    assert outcome["emails_sent"] == ["Listing Interest", "Team Request"]
    assert state["pending_email_actions"] == []


def test_declining_the_re_ask_drops_the_batch_silently():
    state = declined_customer()
    turn(state, "the 1st one catches my eye", intent="listing_interest", listing_reference=1,
         email_triggers=[trigger("listing_interest", ref=1)])
    outcome = turn(state, "no, I'd rather not share that", intent="contact_declined")
    assert outcome["emails_sent"] == []
    assert outcome["email_status"] == "skipped (user declined)"
    assert state["pending_email_actions"] == []
    assert state["contact_declined"] is True


def test_an_faq_also_reopens_the_contact_question():
    state = declined_customer()
    outcome = turn(state, "do you offer financing?", intent="faq",
                   email_triggers=[trigger("faq", faq_key="financing")])
    assert outcome["contact_followup_missing"] == ["name", "email or phone"]
    # They still get their answer this turn — only the email waits.
    assert "financing" in outcome["canned_keys"]


def test_results_shown_never_chases_the_customer_for_contact():
    # The system alert fired by showing results is ours, not theirs. It must never turn into
    # a "can I get your email?" — that is the one trigger that stays silent.
    state = declined_customer()
    state["turn"] = sample_analysis(intent="skip_all_show_results", contact=contact(),
                                    extracted=_empty_extracted(), slot_answers=[], email_triggers=[])
    state.setdefault("messages", []).append({"role": "user", "content": "just show me some results"})
    apply_analysis_to_state(state)
    state["turn_outcome"]["system_email_triggers"] = [{"kind": "results_shown", "description": "6 results"}]
    email_actions_node(state)
    outcome = state["turn_outcome"]
    assert not outcome.get("contact_followup_missing")
    assert outcome["emails_sent"] == []
    assert state["contact_declined"] is True  # untouched


def test_a_customer_who_never_declined_is_asked_for_only_what_is_missing():
    state = declined_customer()
    state["contact_declined"] = False
    state["customer_name"] = "Ibrahim"
    outcome = turn(state, "I like the first one", intent="listing_interest", listing_reference=1,
                   email_triggers=[trigger("listing_interest", ref=1)])
    assert outcome["contact_followup_missing"] == ["email or phone"]
