"""Opt-in live diagnostic for every canonical trailer-category flow.

Run with:
    $env:RUN_LIVE_VAGUE_QNA='1'
    uv run pytest tests/test_live_vague_qna_conversations.py -s -q

This intentionally calls the real LLMs and Pinecone. It is excluded from normal
test runs unless RUN_LIVE_VAGUE_QNA=1.

Per category, two phases are exercised:
  * phase1_reactive_mixed  -> the customer states only the category, then
                              answers each question the bot asks one at a
                              time, alternating between vague and clear/good
                              answers, all the way to a search.
  * phase2_volunteer_email -> the customer front-loads several details in the
                              opening message (the "volunteer" style), runs to
                              a search and logs interest in a listing, then
                              (after results) asks two more general questions
                              that should each trigger an outbound email -
                              one phrased vaguely, one phrased clearly.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from src.chatbot import service
from src.chatbot.categories import CANONICAL_CATEGORIES
from src.models import ChatRequest


_TEST_CATEGORIES = tuple(CANONICAL_CATEGORIES)

_DYNAMIC_WIDTH_CARGO = {
    "Car Hauler": "I will use it for a full-size pickup truck.",
    "Equipment": "I will be hauling a skid steer.",
    "Race Trailer": "I will be hauling a full-size race car.",
    "Tilt": "I will be hauling a tractor.",
}

_INTERRUPTION_SCENARIOS = {
    "Utility": (
        "counter_question",
        "Before I answer that, what are utility trailers generally used for?",
    ),
    "Dump": (
        "faq_email",
        "Before I answer that, do you offer financing for trailers?",
    ),
    "Livestock": (
        "skip_questions",
        "Can you please stop asking questions and just show me the results?",
    ),
}

# Every category maps to which email an unprompted post-results question
# should trigger; both a vaguely-phrased and a clearly-phrased version of that
# question are sent in phase 2.
_POST_RESULTS_KIND = {
    "Aluminum": "escalation_email",
    "Car Hauler": "faq_email",
    "Equipment": "faq_email",
    "Enclosed": "faq_email",
    "Utility": "faq_email",
    "Fiber": "escalation_email",
    "Race Trailer": "escalation_email",
    "Roll Off": "faq_email",
    "Diesel Tank": "escalation_email",
    "Flatbed": "faq_email",
    "Dump": "faq_email",
    "Tilt": "escalation_email",
    "Livestock": "escalation_email",
}

_POST_RESULTS_MESSAGES = {
    "faq_email": (
        "I don't really know the details but do you guys maybe do some kind "
        "of payment plan or financing thing for these trailers?",
        "Do you offer financing options for trailers?",
    ),
    "escalation_email": (
        "I think I'm leaning toward moving forward somehow, could someone "
        "maybe reach out and help me sort out next steps?",
        "Please call me to schedule a purchase for this trailer.",
    ),
}


_VAGUE_ANSWERS = {
    "haul_item": "Farm gear, toolboxes, and assorted things that change week to week.",
    "haul_material": "Broken concrete, yard waste, and whatever debris the job produces.",
    "haul_weight_lbs": "Probably seven to nine thousand pounds, roughly.",
    "haul_length_ft": "Somewhere from nineteen to twenty-three feet ought to work.",
    "vehicle_type": "A mix of compact cars and midsize crossovers.",
    "vehicle_length_ft": "Approximately 15 to 18 feet long.",
    "item_or_trailer_width_ft": "Whatever standard width you recommend; I do not know a measurement.",
    "trailer_length_ft": "Maybe nineteen to twenty-two feet, give or take.",
    "hitch_type": "Bumper pull or gooseneck—either one is acceptable.",
    "loading_style": "I am flexible as long as loading is straightforward.",
    "open_vs_covered": "Either open or enclosed could work for me.",
    "trailer_size": "A moderate size, perhaps 19 by 8 feet.",
    "sides_gate_storage": "Useful rails and somewhere to secure loose gear would help.",
    "dump_mechanism": "No particular mechanism; dependable operation matters most.",
    "tilt_style": "I am open to whichever tilt design fits the load.",
    "use_case": "General deliveries plus an occasional mobile workspace.",
    # Enclosed cargo_size: length only, no width, per test design.
    "cargo_size": "Somewhere around 19 feet long, not too sure on anything else.",
    "ac_windows_cabinets": "Basic power and storage would be useful, but I can be flexible.",
    "finished_interior": "A simple finish is fine, though unfinished could also work.",
    "gate_preferences": "Any practical everyday gate arrangement is acceptable.",
    "package_scope": "The trailer and several bins together would be ideal.",
    "bin_size": "A medium bin, perhaps around 15 to 20 yards.",
    "deck_style": "Either a step deck or standard deck would suit the work.",
    "cdl_concern": "Staying below CDL limits would be helpful, but I am flexible.",
    "fuel_type": "It will mainly carry diesel, possibly other fuel occasionally.",
    "tank_capacity": "Approximately 450 to 650 gallons.",
    "fiber_use_case": "Primarily field splicing, with a little desk work sometimes.",
    "crew_size": "Usually two to five people depending on the job.",
    "fiber_amenities": "Just practical basics—cooling, power, and a decent work surface.",
    "race_amenities": "A modest work area and some storage would be useful.",
}

_GOOD_ANSWERS = {
    "haul_item": "Lawn mowers and hand tools.",
    "haul_material": "Broken concrete and dirt.",
    "haul_weight_lbs": "8000 lbs.",
    "haul_length_ft": "20 feet.",
    "vehicle_type": "A half-ton pickup truck.",
    "vehicle_length_ft": "17 feet.",
    "item_or_trailer_width_ft": "8 feet wide.",
    "trailer_length_ft": "20 feet.",
    "hitch_type": "Gooseneck.",
    "loading_style": "Rear ramp gate.",
    "open_vs_covered": "Enclosed.",
    "trailer_size": "20 by 8 feet.",
    "sides_gate_storage": "Solid sides with a rear gate.",
    "dump_mechanism": "Hydraulic scissor hoist.",
    "tilt_style": "Hydraulic tilt bed.",
    "use_case": "Daily local deliveries.",
    # Enclosed cargo_size: length only, no width, per test design.
    "cargo_size": "19 feet long.",
    "ac_windows_cabinets": "AC unit and a couple of windows.",
    "finished_interior": "Fully finished interior.",
    "gate_preferences": "Single rear swing gate.",
    "package_scope": "Just the trailer itself.",
    "bin_size": "20 yard bin.",
    "deck_style": "Standard flat deck.",
    "cdl_concern": "Needs to stay under CDL limits.",
    "fuel_type": "Diesel.",
    "tank_capacity": "500 gallons.",
    "fiber_use_case": "Field splicing work.",
    "crew_size": "3 people.",
    "fiber_amenities": "AC and a workbench.",
    "race_amenities": "Generator hookup and storage cabinets.",
}


def _answer_for(slot: str | None, question: str, style: str = "vague") -> str:
    pool = _VAGUE_ANSWERS if style == "vague" else _GOOD_ANSWERS
    if slot in pool:
        return pool[slot]
    text = question.lower()
    if style == "vague":
        if "weight" in text or "payload" in text:
            return _VAGUE_ANSWERS["haul_weight_lbs"]
        if "length" in text or "long" in text:
            return "Somewhere around 16 feet, but I am flexible."
        if "size" in text or "dimension" in text:
            return "A medium size, roughly 16 by 7 feet or thereabouts."
        if "prefer" in text:
            return "I do not have a strict preference; either common option is fine."
        return "Something fairly standard and flexible; I am open to suitable options."
    if "weight" in text or "payload" in text:
        return _GOOD_ANSWERS["haul_weight_lbs"]
    if "length" in text or "long" in text:
        return "16 feet."
    if "size" in text or "dimension" in text:
        return "16 by 7 feet."
    if "prefer" in text:
        return "The standard option is fine."
    return "The standard option is fine."


def _send(
    session_id: str,
    message: str,
    *,
    name: str | None = None,
    email: str | None = None,
):
    return service.handle_chat(
        ChatRequest(
            session_id=session_id,
            sales_phase="main",
            message=message,
            customer_full_name=name,
            customer_email=email,
        )
    )


def _complete_listing_interest_if_missing_contact(session_id: str, response):
    tool_events = (response.thinking_context or {}).get("tool_events") or []
    deferred_missing_contact = any(
        event.get("tool") == "send_interested_listing_email"
        and (event.get("result") or {}).get("status") == "deferred_missing_contact"
        for event in tool_events
        if isinstance(event, dict)
    )
    if deferred_missing_contact:
        assert "Before we get started" not in response.assistant_text
        saved = service._get_session(session_id)
        missing_contact = (
            "ibrahim@esided.ai"
            if saved.get("customer_full_name")
            else "My name is Ibrahim"
        )
        return _send(session_id, missing_contact)

    tools = {event.get("tool") for event in tool_events if isinstance(event, dict)}
    if "send_interested_listing_email" in tools:
        return response

    saved = service._get_session(session_id)
    needs_name = not saved.get("customer_full_name")
    needs_contact = not (saved.get("customer_email") or saved.get("customer_phone"))
    if needs_name and needs_contact:
        return _send(session_id, "My name is Ibrahim and my email is ibrahim@esided.ai")
    if needs_name:
        return _send(session_id, "My name is Ibrahim")
    if needs_contact:
        return _send(session_id, "ibrahim@esided.ai")
    return response


# ---------------------------------------------------------------------------
# Phase 1: category-only opener, then answer every asked question reactively,
# alternating vague and clear/good answers turn by turn.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _TEST_CATEGORIES, ids=_TEST_CATEGORIES)
def test_live_phase1_reactive_mixed_answers(category: str):
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    session_id = str(uuid.uuid4())
    user_text = (
        f"My name is Ibrahim and my email is ibrahim@esided.ai. "
        f"I am looking for a {category} trailer."
    )

    completed = False
    awaiting_history: list[str | None] = []
    interruption = _INTERRUPTION_SCENARIOS.get(category)
    interruption_sent = False
    interruption_verified = False
    interrupted_slot = None
    final_slots: dict = {}
    final_skipped: list = []

    for turn in range(1, 12):
        response = _send(session_id, user_text)
        state = service._get_session(session_id)
        response_tool_events = (response.thinking_context or {}).get("tool_events") or []
        awaiting_history.append(state.get("awaiting_slot"))
        print(json.dumps({
            "phase": "phase1_reactive_mixed",
            "category": category,
            "session_id": session_id,
            "turn": turn,
            "user": user_text,
            "assistant": response.assistant_text,
            "resolved_category": state.get("trailer_category"),
            "awaiting_slot": state.get("awaiting_slot"),
            "slots_collected": state.get("slots_collected") or {},
            "tool_events": response_tool_events,
        }, ensure_ascii=False, default=str))

        if interruption_sent and not interruption_verified:
            kind = interruption[0]
            current_slot = str(state.get("awaiting_slot") or "").strip() or None
            if kind == "skip_questions":
                assert current_slot is None
                assert any(
                    event.get("tool") == "pinecone_search"
                    for event in response_tool_events
                    if isinstance(event, dict)
                )
            else:
                assert current_slot == interrupted_slot
            if kind == "faq_email":
                assert any(
                    event.get("tool") == "send_non_sales_faq_email"
                    for event in response_tool_events
                    if isinstance(event, dict)
                )
            interruption_verified = True

        if state.get("last_listings") or any(
            event.get("tool") == "pinecone_search"
            for event in response_tool_events
            if isinstance(event, dict)
        ):
            completed = True
            break

        slot = str(state.get("awaiting_slot") or "").strip() or None
        if not slot:
            completed = True
            break
        if interruption and not interruption_sent:
            interrupted_slot = slot
            user_text = interruption[1]
            interruption_sent = True
            continue
        style = "vague" if turn % 2 == 1 else "good"
        user_text = _answer_for(slot, response.assistant_text, style)

    assert completed
    recommendation_state = service._get_session(session_id)
    assert recommendation_state.get("last_listings"), (
        f"{category} did not produce listings for phase1_reactive_mixed"
    )
    final_slots = recommendation_state.get("slots_collected") or {}
    final_skipped = recommendation_state.get("slots_skipped") or []
    final_filters = recommendation_state.get("metadata_filters_collected") or {}

    if interruption:
        assert interruption_verified
    if category in ("Utility", "Flatbed", "Tilt"):
        assert recommendation_state.get("trailer_category") == category
    if category == "Equipment":
        assert "hitch_type" in final_skipped
        assert "hitch_type" not in final_slots
    if category == "Enclosed":
        assert awaiting_history.count("cargo_size") == 1
        assert final_filters.get("length_ft") == "19 ft"

    print("PHASE1_SUMMARY=" + json.dumps({
        "category": category,
        "session_id": session_id,
        "completed": completed,
        "awaiting_history": awaiting_history,
        "final_slots": final_slots,
        "final_skipped": final_skipped,
    }, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Phase 2: volunteer several details up front, run to a search, log interest
# in a listing, then (after results) ask two more general questions that
# should each trigger an email - one vague, one clear.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _TEST_CATEGORIES, ids=_TEST_CATEGORIES)
def test_live_phase2_volunteer_then_email(category: str):
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    session_id = str(uuid.uuid4())
    request_text = f"I am looking for a {category} trailer."
    if category in _DYNAMIC_WIDTH_CARGO:
        request_text = f"{request_text} {_DYNAMIC_WIDTH_CARGO[category]}"
    user_text = (
        f"My name is Ibrahim and my email is ibrahim@esided.ai. {request_text}"
    )

    completed = False
    awaiting_history: list[str | None] = []
    interruption = _INTERRUPTION_SCENARIOS.get(category)
    interruption_sent = False
    interruption_verified = False
    interrupted_slot = None

    for turn in range(1, 10):
        response = _send(session_id, user_text)
        state = service._get_session(session_id)
        response_tool_events = (response.thinking_context or {}).get("tool_events") or []
        awaiting_history.append(state.get("awaiting_slot"))
        print(json.dumps({
            "phase": "phase2_volunteer_email",
            "category": category,
            "session_id": session_id,
            "turn": turn,
            "user": user_text,
            "assistant": response.assistant_text,
            "resolved_category": state.get("trailer_category"),
            "awaiting_slot": state.get("awaiting_slot"),
            "metadata_filters": state.get("metadata_filters_collected") or {},
            "tool_events": response_tool_events,
        }, ensure_ascii=False, default=str))

        if interruption_sent and not interruption_verified:
            kind = interruption[0]
            current_slot = str(state.get("awaiting_slot") or "").strip() or None
            if kind == "skip_questions":
                assert current_slot is None
                assert any(
                    event.get("tool") == "pinecone_search"
                    for event in response_tool_events
                    if isinstance(event, dict)
                )
            else:
                assert current_slot == interrupted_slot
            if kind == "faq_email":
                assert any(
                    event.get("tool") == "send_non_sales_faq_email"
                    for event in response_tool_events
                    if isinstance(event, dict)
                )
            interruption_verified = True

        if state.get("last_listings") or any(
            event.get("tool") == "pinecone_search"
            for event in response_tool_events
            if isinstance(event, dict)
        ):
            completed = True
            break

        slot = str(state.get("awaiting_slot") or "").strip() or None
        if not slot:
            completed = True
            break
        if interruption and not interruption_sent:
            interrupted_slot = slot
            user_text = interruption[1]
            interruption_sent = True
            continue
        user_text = _answer_for(slot, response.assistant_text, "vague")

    assert completed
    recommendation_state = service._get_session(session_id)
    assert recommendation_state.get("last_listings"), (
        f"{category} did not produce listings for phase2_volunteer_email"
    )
    if category in _DYNAMIC_WIDTH_CARGO:
        final_filters = recommendation_state.get("metadata_filters_collected") or {}
        final_skipped = recommendation_state.get("slots_skipped") or []
        assert "item_or_trailer_width_ft" in awaiting_history
        assert "item_or_trailer_width_ft" in final_skipped
        assert "width_ft" not in final_filters

    interest_message = "I am interested in the first trailer."
    interest_response = _send(session_id, interest_message)
    interest_response = _complete_listing_interest_if_missing_contact(
        session_id,
        interest_response,
    )
    interest_tool_events = (interest_response.thinking_context or {}).get("tool_events") or []
    interest_tools = {
        event.get("tool") for event in interest_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "phase": "phase2_volunteer_email",
        "category": category,
        "session_id": session_id,
        "scenario": "listing_interest",
        "assistant": interest_response.assistant_text,
        "tool_events": interest_tool_events,
    }, ensure_ascii=False, default=str))
    assert "send_interested_listing_email" in interest_tools

    kind = _POST_RESULTS_KIND[category]
    vague_message, good_message = _POST_RESULTS_MESSAGES[kind]
    expected_tool = (
        "send_non_sales_faq_email" if kind == "faq_email" else "send_escalation_alert_email"
    )

    for style, message in (("vague", vague_message), ("good", good_message)):
        post_response = _send(session_id, message)
        post_tool_events = (post_response.thinking_context or {}).get("tool_events") or []
        post_tools = {
            event.get("tool") for event in post_tool_events if isinstance(event, dict)
        }
        print(json.dumps({
            "phase": "phase2_volunteer_email",
            "category": category,
            "session_id": session_id,
            "scenario": f"post_results_{kind}_{style}",
            "user": message,
            "assistant": post_response.assistant_text,
            "tool_events": post_tool_events,
        }, ensure_ascii=False, default=str))
        assert expected_tool in post_tools, (
            f"{category} ({style} phrasing) did not trigger {expected_tool}"
        )

    print("PHASE2_SUMMARY=" + json.dumps({
        "category": category,
        "session_id": session_id,
        "completed": completed,
        "post_results_kind": kind,
    }, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Brand -> category live coverage
#
# The category resolver maps a handful of *brand* names to a category via the
# cargo tier of categories._CARGO_TERMS. Today the only brand->category mappings
# that exist are Livestock brands (galyean / star trailer / calico trailer), so
# those are the cases: mentioning the brand (with no explicit "livestock" word)
# must still resolve the conversation onto the Livestock track.
#
# Each brand is exercised in two phases:
#   * phase1_reactive  -> the customer only ever answers the exact question that
#                         was asked; it never volunteers a spec/feature before
#                         the bot asks for it.
#   * phase2_volunteer -> the customer front-loads several specs/features in the
#                         opening message (before any question is asked), so the
#                         extractor should pre-fill slots/filters and skip the
#                         questions it already has answers for.
# Both phases run the full flow to a search, then the interest-logging scenario,
# then a general (non-FAQ) question that must NOT trigger any tool/email.
# ---------------------------------------------------------------------------

_BRAND_CATEGORY_CASES = {
    "star trailer": "Livestock",
    "galyean": "Livestock",
    "calico trailer": "Livestock",
}

# Specs volunteered up front in phase 2 (before the bot asks anything). Length is
# the most reliably-extracted metadata filter, so it is the one we assert landed.
_BRAND_VOLUNTEERED_SPECS = (
    "It should be around 20 feet long, 8 feet wide, gooseneck hitch, "
    "and rated for roughly 14000 lbs to haul cattle."
)

_BRAND_GENERAL_QUESTION = "What are livestock trailers commonly used for?"

# FAQ question -> non-sales FAQ email; sales/action asks -> escalation alert email.
_BRAND_FAQ_QUESTION = "Do you offer financing options for trailers?"
_BRAND_ESCALATION_MESSAGE = (
    "Please call me and set up a meeting, and send me a formal quotation."
)

_BRAND_PHASES = ("phase1_reactive", "phase2_volunteer")

_BRAND_PARAMS = [
    (brand, category, phase)
    for brand, category in _BRAND_CATEGORY_CASES.items()
    for phase in _BRAND_PHASES
]


@pytest.mark.parametrize(
    "brand, expected_category, phase",
    _BRAND_PARAMS,
    ids=[f"{brand}-{phase}" for brand, _cat, phase in _BRAND_PARAMS],
)
def test_live_brand_to_category_conversations(
    brand: str, expected_category: str, phase: str
):
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    session_id = str(uuid.uuid4())
    volunteered = phase == "phase2_volunteer"

    # Seed the conversation with the BRAND only (no explicit "livestock" word) so
    # the brand->category cargo-tier mapping is what does the resolution. In the
    # volunteer phase we also front-load specs before any question is asked.
    request_text = f"I am looking for a {brand}."
    if volunteered:
        request_text = f"{request_text} {_BRAND_VOLUNTEERED_SPECS}"
    user_text = (
        f"My name is Ibrahim and my email is ibrahim@esided.ai. {request_text}"
    )

    completed = False
    awaiting_history: list[str | None] = []
    asked_slots: list[str] = []
    for turn in range(1, 10):
        response = _send(session_id, user_text)
        state = service._get_session(session_id)
        response_tool_events = (response.thinking_context or {}).get("tool_events") or []
        awaiting_history.append(state.get("awaiting_slot"))
        print(json.dumps({
            "brand": brand,
            "phase": phase,
            "session_id": session_id,
            "turn": turn,
            "user": user_text,
            "assistant": response.assistant_text,
            "resolved_category": state.get("trailer_category"),
            "awaiting_slot": state.get("awaiting_slot"),
            "slots_collected": state.get("slots_collected") or {},
            "metadata_filters": state.get("metadata_filters_collected") or {},
            "tool_events": response_tool_events,
        }, ensure_ascii=False, default=str))

        # The brand must have resolved onto the expected category as soon as the
        # graph has classified the request.
        resolved = state.get("trailer_category")
        if resolved:
            assert resolved == expected_category, (
                f"brand {brand!r} resolved to {resolved!r}, expected {expected_category!r}"
            )

        if state.get("last_listings") or any(
            event.get("tool") == "pinecone_search"
            for event in response_tool_events
            if isinstance(event, dict)
        ):
            completed = True
            break

        slot = str(state.get("awaiting_slot") or "").strip() or None
        if not slot:
            completed = True
            break
        asked_slots.append(slot)
        # Phase 1 and phase 2 both answer questions reactively from here on; the
        # phase difference is purely whether specs were volunteered up front.
        user_text = _answer_for(slot, response.assistant_text)

    assert completed
    recommendation_state = service._get_session(session_id)
    assert recommendation_state.get("trailer_category") == expected_category
    assert recommendation_state.get("last_listings"), (
        f"{brand} ({phase}) did not produce listings"
    )

    if volunteered:
        # A spec given before it was asked must have been captured as a filter,
        # and the bot must not have re-asked for the trailer length it was handed.
        length_filter = str(
            (recommendation_state.get("metadata_filters_collected") or {}).get("length_ft")
            or ""
        )
        assert "20" in length_filter, (
            f"volunteered length not captured as a filter (got {length_filter!r})"
        )
        assert not any("length" in (slot or "") for slot in asked_slots), (
            f"length was re-asked despite being volunteered: {asked_slots}"
        )

    # --- Interest logging scenario (both phases) ---
    interest_message = "I am interested in the first trailer."
    interest_response = _send(session_id, interest_message)
    interest_response = _complete_listing_interest_if_missing_contact(
        session_id,
        interest_response,
    )
    interest_tool_events = (interest_response.thinking_context or {}).get("tool_events") or []
    interest_tools = {
        event.get("tool") for event in interest_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "brand": brand,
        "phase": phase,
        "scenario": "listing_interest",
        "assistant": interest_response.assistant_text,
        "tool_events": interest_tool_events,
    }, ensure_ascii=False, default=str))
    assert "send_interested_listing_email" in interest_tools

    # --- General (non-FAQ) question scenario (both phases) ---
    general_response = _send(session_id, _BRAND_GENERAL_QUESTION)
    general_tool_events = (general_response.thinking_context or {}).get("tool_events") or []
    general_tools = {
        event.get("tool") for event in general_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "brand": brand,
        "phase": phase,
        "scenario": "general_question",
        "assistant": general_response.assistant_text,
        "tool_events": general_tool_events,
    }, ensure_ascii=False, default=str))
    assert general_response.assistant_text.strip()
    assert "pinecone_search" not in general_tools
    assert "send_interested_listing_email" not in general_tools
    assert "send_non_sales_faq_email" not in general_tools
    assert "send_escalation_alert_email" not in general_tools

    # --- FAQ email scenario (both phases) ---
    faq_response = _send(session_id, _BRAND_FAQ_QUESTION)
    faq_tool_events = (faq_response.thinking_context or {}).get("tool_events") or []
    faq_tools = {
        event.get("tool") for event in faq_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "brand": brand,
        "phase": phase,
        "scenario": "faq_email",
        "assistant": faq_response.assistant_text,
        "tool_events": faq_tool_events,
    }, ensure_ascii=False, default=str))
    assert "send_non_sales_faq_email" in faq_tools

    # --- Escalation alert scenario (both phases): call me / meeting / quotation ---
    escalation_response = _send(session_id, _BRAND_ESCALATION_MESSAGE)
    escalation_tool_events = (escalation_response.thinking_context or {}).get("tool_events") or []
    escalation_tools = {
        event.get("tool") for event in escalation_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "brand": brand,
        "phase": phase,
        "scenario": "escalation_email",
        "assistant": escalation_response.assistant_text,
        "tool_events": escalation_tool_events,
    }, ensure_ascii=False, default=str))
    assert "send_escalation_alert_email" in escalation_tools

    print("BRAND_CATEGORY_SUMMARY=" + json.dumps({
        "brand": brand,
        "phase": phase,
        "session_id": session_id,
        "resolved_category": recommendation_state.get("trailer_category"),
        "asked_slots": asked_slots,
        "awaiting_history": awaiting_history,
        "volunteered": volunteered,
    }, ensure_ascii=False, default=str))
