from __future__ import annotations

from tests.conftest import FakeLLM

from src.domain.canned_responses import CANNED_RESPONSES
from src.llm.respond import (
    _contact_only_turn,
    _has_recommendation_basis,
    build_respond_prompt,
    respond_turn,
    respond_with_all_listings,
)
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


def test_respond_prompt_carries_analyst_digest_and_settled_category_guard():
    analysis = sample_analysis(turn_summary="They want a dump trailer for random things.")
    state = {"category": "Dump", "messages": [{"role": "user", "content": "a dump trailer for random things"}]}
    system, _ = build_respond_prompt(state, analysis, {"next_question": "What's the rough haul weight per load?"})
    # The digest is a high-priority decision line, but decided actions still outrank it.
    assert 'WHAT THE CUSTOMER JUST SAID AND WANTS' in system
    assert "They want a dump trailer for random things." in system
    # Seen live: category settled + pending question, yet the model recommended trailer types.
    assert "The trailer category is SETTLED: Dump" in system
    assert "do NOT use the RECOMMENDING TRAILER TYPES format" in system

    # No settled category -> no guard line (the category question flow owns that case).
    no_cat_state = {"messages": [{"role": "user", "content": "hi"}]}
    system, _ = build_respond_prompt(no_cat_state, analysis, {"next_question": "What type of trailer are you looking for?"})
    assert "trailer category is SETTLED" not in system


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


def test_no_category_and_nothing_to_go_on_gets_the_we_carry_paragraph():
    # They have told us NOTHING about the trailer, so there is nothing to recommend from —
    # the type question names 5-6 of our types inline ("...and many more"), never a
    # structured recommendation list.
    state = {"messages": [{"role": "user", "content": "It's Ibrahim, ibrahim@x.test"}], "slots": {}}
    analysis = sample_analysis(
        intent="contact_info_provided",
        category_mentioned=None,
        slot_answers=[],
        extracted={
            "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
            "payload_lbs": None, "axle_capacity_lbs": None, "hitch_type": None, "haul_item": None,
            "brand_preference": None, "non_metadata_features": [], "numeric_no_preference": [],
        },
    )

    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you after?"})

    assert "and many more" in system
    assert "NO bullets" in system
    assert "so RECOMMEND" not in system


def test_asking_the_type_question_again_switches_to_the_full_lineup_list():
    # Seen live: the bot sent the identical we-carry sentence twice in a row — the customer
    # said "show me more" and got a copy of the last reply. The second ask must give them
    # something NEW (the full lineup as bullets), never the same paragraph.
    state = {
        "messages": [
            {"role": "user", "content": "hi, I need a trailer"},
            {
                "role": "assistant",
                "content": "We carry Equipment, Dump, Enclosed, Utility, Flatbed, and Livestock trailers, "
                "and many more - which type would you like to go with?",
            },
            {"role": "user", "content": "show me more"},
        ],
        "slots": {},
    }
    analysis = sample_analysis(
        intent="recommendation_request",
        category_mentioned=None,
        slot_answers=[],
        extracted={
            "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
            "payload_lbs": None, "axle_capacity_lbs": None, "hitch_type": None, "haul_item": None,
            "brand_preference": None, "non_metadata_features": [], "numeric_no_preference": [],
        },
        haul_classification={"is_lightweight_utility_load": False, "needs_width_question": False, "haul_item_matched": None},
    )

    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you looking for?"})

    assert "ALREADY ASKED" in system
    assert "DO NOT send the same sentence again" in system
    assert "so we cannot recommend" not in system  # the fixed paragraph branch must not also fire


def test_respond_prompt_carries_the_coherence_guardrails():
    state = {"messages": [{"role": "user", "content": "hello"}], "slots": {}}
    system, _ = build_respond_prompt(state, sample_analysis(), {})
    assert "READ THE CONVERSATION BEFORE YOU WRITE" in system
    assert "NEVER send the same or nearly the same message twice in a row" in system
    assert "MORE OF WHATEVER YOUR LAST MESSAGE OFFERED" in system


def test_respond_prompt_carries_the_critical_rules_section():
    state = {"messages": [{"role": "user", "content": "hello"}], "slots": {}}
    system, _ = build_respond_prompt(state, sample_analysis(), {})
    assert "HARD RULES - NEVER BROKEN" in system
    assert "AFTER A CATEGORY CHANGE" in system
    assert "INVENTORY EXISTS ONLY IN THE LISTINGS BLOCK" in system
    assert "THIS TURN'S ORDERS" in system
    assert "HOW TO BUILD THE REPLY - DO THESE STEPS IN ORDER" in system


def test_decided_question_is_first_order_and_digest_comes_last():
    # Seen live (4o-mini): led by the analyst digest, the model followed the digest's story
    # and asked its own question instead of the decided one. The question line now leads the
    # orders as a MUST-END-WITH directive and the digest trails as context.
    state = {"category": "Flatbed", "messages": [{"role": "user", "content": "no"}], "slots": {}}
    system, _ = build_respond_prompt(
        state, sample_analysis(), {"next_question": "What will you be hauling on the flatbed?"}
    )
    assert 'THE ONE QUESTION TO ASK - your reply MUST END with it: "What will you be hauling on the flatbed?"' in system
    assert "FAILED reply" in system
    question_pos = system.index("THE ONE QUESTION TO ASK")
    digest_pos = system.index("WHAT THE CUSTOMER JUST SAID AND WANTS")
    assert question_pos < digest_pos


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
    # Kept, not stripped as a foreign card - but named in plain text, since a turn with no
    # listings of its own is talking about a trailer already on the customer's screen.
    assert "Trailer 1" in reply.assistant_text
    assert "](" not in reply.assistant_text
    assert reply.cited_listing_urls == ["https://example.test/dump-1"]


def test_no_listings_turn_names_the_referenced_trailer_in_plain_text():
    # Messenger renders no markdown, so a linked title arrives as literal brackets and a raw
    # URL. The prompt orders plain text; this is the guarantee when the model ignores it.
    analysis = sample_analysis()
    state = {
        "category": "Livestock",
        "messages": [{"role": "user", "content": "I'm interested in the first one"}],
        "shown_listings": _listings("https://example.test/cattle-1"),
    }
    linked = ReplyOutput(
        assistant_text=(
            "I see you are interested in the 2026 Galyean Cattle Trailer - 15131:\n"
            "1. [2026 Galyean Cattle Trailer - 15131](https://example.test/cattle-1)\n"
            "   - Price: $35,250\n"
            "Feel free to check out our website at https://trailerplace.com."
        ),
        cited_listing_urls=["https://example.test/cattle-1"],
    )
    llm = FakeLLM(outputs=[linked])
    reply = respond_with_all_listings(llm, state, analysis, {"listings": []})
    assert "](" not in reply.assistant_text
    assert "https://example.test/cattle-1" not in reply.assistant_text
    # The numbered marker goes with the link - a lone "1." reads like a list that never comes.
    assert "\n2026 Galyean Cattle Trailer - 15131\n" in reply.assistant_text
    assert "$35,250" in reply.assistant_text
    # The channel still gets the URL to build its own card from.
    assert reply.cited_listing_urls == ["https://example.test/cattle-1"]


def test_unlinking_spares_the_website_link_and_the_trailer_type_list():
    analysis = sample_analysis()
    state = {"messages": [{"role": "user", "content": "what types do you have?"}]}
    reply_in = ReplyOutput(
        assistant_text=(
            "Based on what you need to haul, here are the types worth looking at:\n"
            "1. **Equipment Trailer** - for machinery.\n"
            "2. **Dump Trailer** - for loose material.\n"
            "See [our website](https://trailerplace.com) for more."
        ),
        cited_listing_urls=[],
    )
    llm = FakeLLM(outputs=[reply_in])
    reply = respond_with_all_listings(llm, state, analysis, {"listings": []})
    assert reply.assistant_text == reply_in.assistant_text


def test_a_turn_that_presents_listings_keeps_its_hyperlinked_cards():
    # The card link is how the customer opens the trailer, and it is what Messenger turns
    # into a generic-template card. Only turns with NO listings are unlinked.
    analysis = sample_analysis()
    state = {"category": "Dump", "messages": [{"role": "user", "content": "show me dump trailers"}]}
    llm = FakeLLM(outputs=[_reply("https://example.test/dump-1")])
    reply = respond_with_all_listings(llm, state, analysis, {"listings": _listings("https://example.test/dump-1")})
    assert "[Trailer 1](https://example.test/dump-1)" in reply.assistant_text


def test_contact_only_turn_gets_a_plain_type_question_not_recommendations():
    # Seen live: a chat opened with just a name and email was answered with a 3-category
    # recommendation list because the analyzer mislabeled the turn. Contact-only turns have
    # no recommendation basis, whatever the intent label says.
    analysis = sample_analysis(
        intent="recommendation_request",  # the mislabel
        category_mentioned=None,
        slot_answers=[],
        user_question_to_answer=None,
        contact={"name": "Ibrahim", "email": "ibrahim@x.ai", "phone": None},
        extracted={
            "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
            "payload_lbs": None, "axle_capacity_lbs": None, "hitch_type": None, "haul_item": None,
            "brand_preference": None, "non_metadata_features": [], "numeric_no_preference": [],
        },
    )
    state = {"category": None, "slots": {}, "messages": [{"role": "user", "content": "I'm Ibrahim, ibrahim@x.ai"}]}
    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you looking for?"})
    assert "and many more" in system
    assert "so RECOMMEND" not in system


def test_brand_question_lines_render_single_and_multi():
    analysis = sample_analysis(intent="general_question", category_mentioned=None)
    base_state = {"category": None, "slots": {}, "messages": [{"role": "user", "content": "do you carry Iron Bull?"}]}
    multi = dict(base_state, pending_brand_categories={"brand": "Iron Bull Trailers", "categories": ["Dump", "Equipment"]})
    system, _ = build_respond_prompt(multi, analysis, {})
    assert "BRAND QUESTION" in system
    assert "Dump, Equipment" in system
    single = dict(base_state, pending_brand_categories={"brand": "Delco", "categories": ["Livestock"]})
    system, _ = build_respond_prompt(single, analysis, {})
    assert "ONE category: Livestock" in system
    assert "yes/no" in system


def test_options_ask_with_nothing_known_gets_paragraph_not_bullets():
    # "What are my options?" with nothing told to us is not a recommendation basis —
    # the reply is the we-carry paragraph, not a tailored structured list.
    analysis = sample_analysis(
        intent="recommendation_request",
        category_mentioned=None,
        slot_answers=[],
        extracted={
            "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
            "payload_lbs": None, "axle_capacity_lbs": None, "hitch_type": None, "haul_item": None,
            "brand_preference": None, "non_metadata_features": [], "numeric_no_preference": [],
        },
    )
    state = {"category": None, "slots": {}, "messages": [{"role": "user", "content": "what are my options?"}]}
    system, _ = build_respond_prompt(state, analysis, {"next_question": "What type of trailer are you looking for?"})
    assert "and many more" in system
    assert "so RECOMMEND" not in system


def test_brands_from_inventory_are_in_the_respond_prompt():
    analysis = sample_analysis()
    state = {"category": None, "slots": {}, "messages": [{"role": "user", "content": "which brands do you carry?"}]}
    system, _ = build_respond_prompt(state, analysis, {})
    assert "OUR BRANDS/MAKES" in system
    assert "Gooseneck," not in system.split("OUR BRANDS/MAKES")[1].split("\n")[0]


def test_keep_drop_question_owns_the_reply_and_quotes_units():
    # Live failure (2026-07-15, session d1f0ba7f): the old wording ("Confirm which to carry
    # over ... just ask") was soft enough that the model asked a generic "any other features?"
    # question instead — the keep/drop question never reached the customer. The line must own
    # the reply outright and quote each carried value with its unit.
    analysis = sample_analysis()
    state = {
        "category": "Livestock",
        "slots": {"payload_lbs": 9062.0},
        "messages": [{"role": "user", "content": "I am also looking for a 50ft trailer for my livestock"}],
        "pending_category_change": {"new_category": "Livestock", "dimensions": {"payload": 9062.0}},
    }
    system, _ = build_respond_prompt(state, analysis, {"category_just_changed": "Livestock"})
    assert "CATEGORY-CHANGE KEEP/DROP QUESTION (this owns the reply)" in system
    assert "payload capacity (9062 lbs)" in system
    assert "exactly ONE question mark" in system
    assert "do NOT ask about any other feature" in system


def test_keep_drop_line_formats_hitch_and_lengths():
    analysis = sample_analysis()
    state = {
        "category": "Equipment",
        "slots": {},
        "messages": [{"role": "user", "content": "switch me to an equipment trailer"}],
        "pending_category_change": {
            "new_category": "Equipment",
            "dimensions": {"length": 20.0, "hitch": ["Gooseneck"]},
        },
    }
    system, _ = build_respond_prompt(state, analysis, {})
    assert "length (20 ft)" in system
    assert "hitch type (Gooseneck)" in system


def test_no_listings_turn_that_keeps_fabricating_falls_back_to_the_question():
    # Live failure (2026-07-16, session 719c3eb9): after a keep/drop answer, the decided reply
    # was the Livestock length question — but both drafts fabricated Diamond C FMAX listing
    # cards replayed from history. Stripping them left the shell "Here are some Livestock
    # trailers: / Do any of these align...?" with zero listings and the question never asked.
    # On a no-listings turn the honest floor is the decided question itself.
    fabricated = _reply(
        "https://example.test/fmax-1",
        "https://example.test/fmax-2",
        lead="Here are some Livestock trailers that you might find suitable:",
    )
    client = FakeLLM([fabricated, fabricated])
    state = {
        "category": "Livestock",
        "slots": {},
        "shown_listings": [],
        "messages": [{"role": "user", "content": "i'd like to drop it"}],
    }
    outcome = {"next_question": "What trailer length are you looking for?"}
    reply = respond_with_all_listings(client, state, sample_analysis(), outcome)
    assert reply.assistant_text == "What trailer length are you looking for?"
    assert reply.cited_listing_urls == []
    # The retry was told, in the repair note, that the turn has no listings and what to ask.
    retry_system = client.calls[1]["system"]
    assert "THIS TURN HAS NO LISTINGS AT ALL" in retry_system
    assert "What trailer length are you looking for?" in retry_system


def test_category_suggestion_line_owns_the_reply():
    # Live failure (2026-07-17, session 66d2618b): "a tractor" on a Flatbed raised the Equipment
    # suggestion, but the old soft wording lost to the RECOMMENDING TRAILER TYPES shape — the
    # model bulleted flatbed "types" and asked "Which type would you like to go with?". The line
    # must own the reply like the keep/drop line does.
    analysis = sample_analysis()
    state = {
        "category": "Flatbed",
        "slots": {"haul_item": "a tractor"},
        "messages": [{"role": "user", "content": "a tractor"}],
        "pending_category_suggestion": {
            "suggested_category": "Equipment",
            "from_category": "Flatbed",
            "cargo": "a tractor",
        },
    }
    system, _ = build_respond_prompt(state, analysis, {})
    assert "CATEGORY SWITCH SUGGESTION (this owns the reply" in system
    assert "switch to Equipment, or stay with Flatbed?" in system
    assert "do NOT use the RECOMMENDING TRAILER TYPES format" in system
    assert "exactly ONE question mark" in system
    assert "FAILED reply" in system


def test_suggestion_turn_that_keeps_fabricating_falls_back_to_switch_or_stay():
    # Same live failure, second half: the first draft replayed 5 FMAX URLs from history, and the
    # repair note had NO question to re-anchor the retry on (a suggestion turn decides no
    # next_question), so "reply briefly" won and the switch-or-stay question never shipped.
    fabricated = _reply(
        "https://example.test/fmax-1",
        "https://example.test/fmax-2",
        lead="Here are some features to consider in flatbed trailers:",
    )
    client = FakeLLM([fabricated, fabricated])
    state = {
        "category": "Flatbed",
        "slots": {"haul_item": "a tractor"},
        "shown_listings": [],
        "messages": [{"role": "user", "content": "a tractor"}],
        "pending_category_suggestion": {
            "suggested_category": "Equipment",
            "from_category": "Flatbed",
            "cargo": "a tractor",
        },
    }
    reply = respond_with_all_listings(client, state, sample_analysis(), {})
    assert reply.assistant_text == (
        "For hauling a tractor, our Equipment trailers are usually the better fit - "
        "would you like to switch to Equipment, or stay with Flatbed?"
    )
    assert reply.cited_listing_urls == []
    retry_system = client.calls[1]["system"]
    assert "THIS TURN HAS NO LISTINGS AT ALL" in retry_system
    assert "switch to Equipment, or stay with Flatbed?" in retry_system


def test_no_listings_fabrication_without_a_pending_question_still_strips():
    # No decided question to fall back on (e.g. an FAQ turn): stripping remains the repair.
    fabricated = _reply("https://example.test/fmax-1", "https://example.test/fmax-2", lead="Options:")
    client = FakeLLM([fabricated, fabricated])
    state = {"category": "Dump", "slots": {}, "shown_listings": [], "messages": []}
    reply = respond_with_all_listings(client, state, sample_analysis(), {})
    assert "example.test" not in reply.assistant_text
    assert reply.cited_listing_urls == []


def test_contact_plus_an_axle_requirement_is_not_a_contact_only_turn():
    """"I'm Ibrahim, and the trailer should have 7,000 lb axles" states a real requirement.

    Regression guard: an axle rating used to register here only by accident, as the junk
    feature "10k axles". Once that leak was fixed, the axle number was the ONLY signal left -
    and it was not checked, so the turn counted as saying nothing about trailers and the
    customer got the generic we-carry paragraph instead of a tailored recommendation.
    """
    analysis = sample_analysis(
        category_mentioned=None,
        slot_answers=[],
        contact={"name": "Ibrahim", "email": "ibrahim@google.com", "phone": None, "declined": False},
        extracted={
            "trailer_length_ft": None, "trailer_width_ft": None, "trailer_height_ft": None,
            "payload_lbs": None, "axle_capacity_lbs": 7000.0, "hitch_type": None,
            "haul_item": None, "brand_preference": None,
            "non_metadata_features": [], "numeric_no_preference": [],
        },
    )
    assert _contact_only_turn(analysis) is False
    assert _has_recommendation_basis({"slots": {"axle_capacity_lbs": 7000.0}}, analysis) is True
