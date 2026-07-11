from __future__ import annotations

from tests.conftest import FakeLLM

from src.domain.canned_responses import CANNED_RESPONSES
from src.llm.respond import build_respond_prompt, respond_turn, respond_with_all_listings
from src.llm.schemas import ReplyOutput
from tests.unit.llm_helpers import sample_analysis


def _listings(*urls: str) -> list[dict]:
    return [{"title": f"Trailer {i}", "url": url} for i, url in enumerate(urls, 1)]


def test_respond_prompt_embeds_canned_strings_and_listings():
    analysis = sample_analysis()
    state = {"category": "Dump", "messages": [{"role": "user", "content": "finance and call me"}], "customer_name": "John"}
    outcome = {
        "canned_keys": ["financing", "escalation"],
        "listings": [
            {
                "title": "Diamond C Dump",
                "price_display": "$9,999",
                "length": "14",
                "width": "7",
                "hitch_type": "Bumper Pull",
                "make": "Diamond C",
                "url": "https://example.test/listing",
            }
        ],
        "email_status": "deferred - ask once for the missing contact piece(s): email or phone",
    }
    system, _ = build_respond_prompt(state, analysis, outcome)
    assert CANNED_RESPONSES["financing"] in system
    assert CANNED_RESPONSES["escalation"] in system
    assert "Diamond C Dump" in system
    assert "https://example.test/listing" in system
    assert "deferred - ask once" in system
    assert "We carry:" in system


def test_respond_prompt_inventory_match_statuses_and_suppression():
    analysis = sample_analysis(intent="inventory_lookup")
    for status in ("exact", "no_exact", "ambiguous"):
        system, _ = build_respond_prompt(
            {"messages": [{"role": "user", "content": "stock 12345"}]},
            analysis,
            {"inventory_match_status": status, "contact_invite_suppressed": True},
        )
        assert f"Inventory lookup result: {status}" in system
        assert "exact -> present" in system
        assert "no_exact -> say we do not currently show" in system
        assert "ambiguous -> ask which model they mean" in system
        assert "do NOT ask for name/email/phone" in system


def test_respond_prompt_contact_ask_and_fake_llm():
    output = ReplyOutput(assistant_text="Hi John", cited_listing_urls=[])
    llm = FakeLLM([output])
    analysis = sample_analysis()
    result = respond_turn(llm, {"messages": [{"role": "user", "content": "hello"}]}, analysis, {"contact_ask": True})
    assert result == output
    assert "First-turn contact invite" in llm.calls[0]["system"]
    assert llm.calls[0]["schema"] is ReplyOutput


def test_dropped_listings_trigger_one_repair_retry():
    # The model self-filters on perceived fit (e.g. drops a hay trailer from a cattle-trailer
    # list). Ranking is the reranker's job, so a short reply is a defect: name what it left
    # out and re-ask once.
    listings = _listings("https://x.test/a", "https://x.test/b", "https://x.test/c")
    short = ReplyOutput(assistant_text="Here are 2", cited_listing_urls=["https://x.test/a", "https://x.test/b"])
    full = ReplyOutput(assistant_text="Here are 3", cited_listing_urls=[l["url"] for l in listings])
    llm = FakeLLM([short, full])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 3}
    )

    assert result == full
    assert len(llm.calls) == 2
    repair_system = llm.calls[1]["system"]
    assert "YOUR PREVIOUS DRAFT WAS REJECTED" in repair_system
    assert "https://x.test/c" in repair_system


def test_complete_reply_is_returned_without_a_retry():
    listings = _listings("https://x.test/a", "https://x.test/b")
    full = ReplyOutput(assistant_text="Here are 2", cited_listing_urls=[l["url"] for l in listings])
    llm = FakeLLM([full])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 2}
    )

    assert result == full
    assert len(llm.calls) == 1


def test_retry_that_still_drops_listings_falls_back_to_the_first_draft():
    # Never loop or fail the turn on a stubborn model — one retry, then ship the best draft.
    listings = _listings("https://x.test/a", "https://x.test/b")
    short = ReplyOutput(assistant_text="Just one", cited_listing_urls=["https://x.test/a"])
    still_short = ReplyOutput(assistant_text="Still one", cited_listing_urls=["https://x.test/a"])
    llm = FakeLLM([short, still_short])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 2}
    )

    assert result == short
    assert len(llm.calls) == 2
