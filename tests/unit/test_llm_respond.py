from __future__ import annotations

from tests.conftest import FakeLLM

from src.domain.canned_responses import CANNED_RESPONSES
from src.llm.respond import build_respond_prompt, respond_turn, respond_with_all_listings
from src.llm.schemas import ReplyOutput
from tests.unit.llm_helpers import sample_analysis


def _listings(*urls: str) -> list[dict]:
    return [{"title": f"Trailer {i}", "url": url} for i, url in enumerate(urls, 1)]


def _reply(*cited: str, lead: str = "Here they are:") -> ReplyOutput:
    """A reply that actually PRESENTS the listings it cites.

    A listing only counts as shown when its URL is in assistant_text — the only field the customer
    ever sees. So a fixture that cites a URL has to render it too, or it is modelling the very bug
    respond_with_all_listings exists to catch.
    """
    cards = "\n".join(f"{i}. [Trailer {i}]({url})" for i, url in enumerate(cited, 1))
    return ReplyOutput(assistant_text=f"{lead}\n{cards}", cited_listing_urls=list(cited))


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


def test_no_search_this_turn_is_not_presented_as_out_of_stock():
    # A turn with no search used to render as "LISTINGS ... (0)" / "none", which the model
    # read as an out-of-stock report and told a mid-qualification customer we had no utility
    # trailers. Empty must say "no search ran", and the already-shown listings must stay in
    # the reference block rather than being offered up as this turn's results.
    analysis = sample_analysis()
    state = {
        "category": "Utility",
        "messages": [{"role": "user", "content": "I'd rather keep it 18ft"}],
        "shown_listings": _listings("https://example.test/old-livestock"),
    }
    system, _ = build_respond_prompt(state, analysis, {"next_question": "What will you be hauling?"})
    assert "NO SEARCH RAN" in system
    assert "NOT an out-of-stock signal" in system
    assert "REFERENCE ONLY" in system
    assert "https://example.test/old-livestock" in system


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


def test_contact_gate_ask_owns_the_whole_turn():
    # The opening contact ask must not share the turn with a qualification question — one of
    # the two always gets ignored, and it is usually ours.
    output = ReplyOutput(assistant_text="Hi! Who am I speaking with?", cited_listing_urls=[])
    llm = FakeLLM([output])
    analysis = sample_analysis()
    state = {"messages": [{"role": "user", "content": "I need a dump trailer"}], "contact_asks": 1}
    outcome = {
        "contact_ask": True,
        "contact_gate_missing": ["name", "email or phone"],
        "next_question": "What material will you be hauling?",  # must be suppressed
    }
    result = respond_turn(llm, state, analysis, outcome)
    assert result == output
    system = llm.calls[0]["system"]
    assert "ASK ONLY FOR CONTACT DETAILS THIS TURN: their name and email or phone" in system
    assert "What material will you be hauling?" not in system
    assert llm.calls[0]["schema"] is ReplyOutput


def test_dropped_listings_trigger_one_repair_retry():
    # The model self-filters on perceived fit (e.g. drops a hay trailer from a cattle-trailer
    # list). Ranking is the reranker's job, so a short reply is a defect: name what it left
    # out and re-ask once.
    listings = _listings("https://x.test/a", "https://x.test/b", "https://x.test/c")
    short = _reply("https://x.test/a", "https://x.test/b")
    full = _reply(*[l["url"] for l in listings])
    llm = FakeLLM([short, full])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 3}
    )

    assert result == full
    assert len(llm.calls) == 2
    repair_system = llm.calls[1]["system"]
    assert "YOUR PREVIOUS DRAFT WAS REJECTED" in repair_system
    assert "https://x.test/c" in repair_system


def test_announced_but_unrendered_listings_trigger_a_retry():
    # Seen live: the reply said "Here are some trailers that match your requirements:" and stopped,
    # yet reported all the URLs in cited_listing_urls. The customer sees assistant_text and nothing
    # else, so those trailers were never shown — and were then recorded as shown, locking them out
    # of "show me more". A cited URL that is not in the reply text is a dropped listing.
    listings = _listings("https://x.test/a", "https://x.test/b")
    announced = ReplyOutput(
        assistant_text="Here are some trailers that match your requirements:",
        cited_listing_urls=[l["url"] for l in listings],
    )
    full = _reply(*[l["url"] for l in listings])
    llm = FakeLLM([announced, full])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(),
        {"listings": listings, "search_ran": True, "result_count": 2},
    )

    assert result is full
    assert len(llm.calls) == 2
    assert "assistant_text" in llm.calls[1]["system"]


def test_complete_reply_is_returned_without_a_retry():
    listings = _listings("https://x.test/a", "https://x.test/b")
    full = _reply(*[l["url"] for l in listings])
    llm = FakeLLM([full])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 2}
    )

    assert result == full
    assert len(llm.calls) == 1


def test_retry_that_still_drops_listings_falls_back_to_the_first_draft():
    # Never loop or fail the turn on a stubborn model — one retry, then ship the best draft.
    listings = _listings("https://x.test/a", "https://x.test/b")
    short = _reply("https://x.test/a", lead="Just one")
    still_short = _reply("https://x.test/a", lead="Still one")
    llm = FakeLLM([short, still_short])

    result = respond_with_all_listings(
        llm, {"messages": [{"role": "user", "content": "show me"}]}, sample_analysis(), {"listings": listings, "search_ran": True, "result_count": 2}
    )

    assert result == short
    assert len(llm.calls) == 2


def test_listing_reference_counts_within_the_batch_on_screen():
    # Seen live: after a "show me more", "I like the 5th one" was resolved against the CUMULATIVE
    # 12 listings instead of the 6 on screen — and respond, never told the reference at all, quoted
    # a trailer the customer had not picked. The index means the 5th of what they can see.
    batch_one = _listings(*[f"https://x.test/old{i}" for i in range(1, 7)])
    batch_two = _listings(*[f"https://x.test/new{i}" for i in range(1, 7)])
    state = {
        "category": "Livestock",
        "shown_listings": batch_one + batch_two,
        "last_shown_listings": batch_two,
        "messages": [{"role": "user", "content": "I like the 5th one"}],
    }
    analysis = sample_analysis(intent="listing_interest", listing_reference=5)

    system, _ = build_respond_prompt(state, analysis, {})

    assert "https://x.test/new5" in system
    # ...and not the 5th of the whole history, nor the last one it happened to see.
    assert "ALREADY RESOLVED" in system
    resolved = system.split("ALREADY RESOLVED")[1].split("\n")[0]
    assert "old5" not in resolved and "new6" not in resolved


def test_no_category_and_nothing_to_go_on_asks_plainly_instead_of_recommending():
    # Seen live: the customer handed over their name and email and got four trailer types
    # recommended back. They had told us NOTHING about the trailer, so there was nothing to
    # recommend from — the type question is a plain question until they give us something.
    state = {"messages": [{"role": "user", "content": "It's Ibrahim, ibrahim@x.test"}], "slots": {}}
    analysis = sample_analysis(intent="contact_info_provided", category_mentioned=None)

    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you after?"})

    assert "Ask it as ONE plain sentence" in system
    assert "do NOT list, suggest, or bullet any trailer types" in system


def test_a_feature_with_no_category_gets_the_structured_recommendation():
    state = {"messages": [{"role": "user", "content": "something with a rear ramp"}], "slots": {}, "non_metadata_features": ["rear ramp"]}
    analysis = sample_analysis(intent="feature_request_no_category", category_mentioned=None)

    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you after?"})

    assert "RECOMMENDING TRAILER TYPES" in system
    assert "given us something to go on, so RECOMMEND" in system


def test_asking_for_a_recommendation_gets_the_structured_recommendation():
    state = {"messages": [{"role": "user", "content": "I can't decide, what do you recommend?"}], "slots": {}}
    analysis = sample_analysis(intent="recommendation_request", category_mentioned=None)

    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you after?"})

    assert "so RECOMMEND" in system


def test_relaxed_search_results_are_presented_as_alternatives():
    # These listings came back only because we dropped the hard filters, so they are the closest
    # we have, not matches. Presenting them silently as matches would be a lie the customer only
    # discovers on the listing page.
    listings = _listings("https://x.test/a")
    state = {"category": "Dump", "messages": [{"role": "user", "content": "24ft gooseneck"}]}
    outcome = {
        "listings": listings,
        "search_ran": True,
        "result_count": 1,
        "filters_relaxed": True,
        "relaxed_filters_dropped": ["length", "hitch type"],
    }

    system, _ = build_respond_prompt(state, sample_analysis(), outcome)

    assert "ALTERNATIVES, NOT EXACT MATCHES" in system
    assert "length, hitch type" in system


def test_missing_listing_fields_are_absent_from_the_block_not_rendered_as_none():
    # Seen live: a trailer with no make, price or length on file was handed to the model as
    # "None - None x None", and the card came back advertising "Make: None / Length: None".
    # It cannot omit what it is never shown.
    listing = {
        "title": "2026 Gooseneck Livestock - 91632",
        "url": "https://x.test/91632",
        "hitch_type": "Gooseneck",
        "make": None,
        "price_display": None,
        "length": None,
        "width": None,
    }
    state = {"category": "Livestock", "messages": [{"role": "user", "content": "show me"}]}

    system, _ = build_respond_prompt(state, sample_analysis(), {"listings": [listing], "search_ran": True, "result_count": 1})

    block_line = next(line for line in system.splitlines() if line.startswith("1. TITLE:"))
    assert "None" not in block_line
    assert "Length" not in block_line and "Make" not in block_line
    assert "Hitch type: Gooseneck" in block_line
    # The stock number is part of the title and is how the customer and the team name the trailer.
    assert "TITLE: 2026 Gooseneck Livestock - 91632" in block_line
    assert "COPY THE TITLE EXACTLY" in system


def test_search_ran_zero_results_is_a_no_more_matches_reply_not_no_search():
    # A "show me more" that excludes every remaining listing used to render as NO SEARCH RAN,
    # which forbade mentioning inventory — so the model replayed old listings from history.
    analysis = sample_analysis()
    state = {
        "category": "Dump",
        "messages": [{"role": "user", "content": "show me more"}],
        "shown_urls": ["https://example.test/dump-1"],
        "shown_listings": _listings("https://example.test/dump-1"),
    }
    system, _ = build_respond_prompt(state, analysis, {"search_ran": True, "result_count": 0, "listings": []})
    assert "SEARCH RAN AND FOUND NO NEW MATCHES" in system
    assert "already seen every match" in system
    assert "979-532-1486" in system
    assert "(NONE - NO SEARCH RAN)" not in system


def test_foreign_history_urls_are_retried_and_stripped():
    # No search ran, but the draft replays two old listings from history — the exact
    # stale-replay pattern from the live audit. The guard retries once; when the retry
    # still replays them, their cards are stripped from the reply.
    analysis = sample_analysis()
    state = {
        "category": "Equipment",
        "messages": [{"role": "user", "content": "switch me to equipment"}],
        "shown_listings": _listings("https://example.test/dump-1", "https://example.test/dump-2"),
        "shown_urls": ["https://example.test/dump-1", "https://example.test/dump-2"],
    }
    bad = _reply("https://example.test/dump-1", "https://example.test/dump-2", lead="Here are our Equipment trailers:")
    llm = FakeLLM(outputs=[bad, bad.model_copy()])
    reply = respond_with_all_listings(llm, state, analysis, {"listings": []})
    assert len(llm.calls) == 2
    assert "YOUR PREVIOUS DRAFT WAS REJECTED" in llm.calls[1]["system"]
    assert "https://example.test/dump-1" not in reply.assistant_text
    assert "https://example.test/dump-2" not in reply.assistant_text
    assert reply.cited_listing_urls == []


def test_single_history_url_answering_a_question_is_left_alone():
    # Quoting ONE old listing back (answering "how much was that one?") is legitimate.
    analysis = sample_analysis()
    state = {
        "category": "Dump",
        "messages": [{"role": "user", "content": "how much was the first one?"}],
        "shown_listings": _listings("https://example.test/dump-1", "https://example.test/dump-2"),
    }
    ok = ReplyOutput(
        assistant_text="The [Trailer 1](https://example.test/dump-1) is $9,995. Would you like to see more?",
        cited_listing_urls=["https://example.test/dump-1"],
    )
    llm = FakeLLM(outputs=[ok])
    reply = respond_with_all_listings(llm, state, analysis, {"listings": []})
    assert len(llm.calls) == 1
    assert "https://example.test/dump-1" in reply.assistant_text
