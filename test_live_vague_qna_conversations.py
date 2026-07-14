"""Opt-in live diagnostic for every canonical trailer-category flow.

Run with:
    $env:RUN_LIVE_VAGUE_QNA='1'
    uv run pytest tests/test_live_vague_qna_conversations.py -s -q

This intentionally calls the real LLMs and Pinecone. It is excluded from normal
test runs unless RUN_LIVE_VAGUE_QNA=1.
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

_REQUEST_OVERRIDES = {
    "Aluminum": "I want an aluminum trailer",
    "Car Hauler": "I need a car hauler for a full-size pickup truck",
    "Equipment": "I am looking for an equipment trailer for a skid steer",
    "Enclosed": "I need an enclosed trailer",
    "Utility": "I am looking for a utility trailer",
    "Fiber": "I am looking for a fiber splicing trailer",
    "Race Trailer": "I need a race trailer for a full-size race car",
    "Roll Off": "I am looking for a roll off trailer",
    "Diesel Tank": "I am looking for a diesel tank trailer",
    "Flatbed": "I am looking for a hotshot trailer",
    "Dump": "I need a dump trailer",
    "Tilt": "I am looking for a tilt trailer for a tractor",
    "Livestock": "I am looking for a livestock trailer",
}

_DYNAMIC_WIDTH_CARGO = {
    "Car Hauler": "I will use it for a full-size pickup truck.",
    "Equipment": "I will be hauling a skid steer.",
    "Race Trailer": "I will be hauling a full-size race car.",
    "Tilt": "I will be hauling a tractor.",
}

_GENERAL_QUERY_SCENARIOS = {
    "Aluminum": (
        "start",
        "basic_qna",
        "Before I choose the type, what are aluminum trailers good for?",
    ),
    "Car Hauler": (
        "middle",
        "basic_qna",
        "Before I answer that, what is a car hauler usually used for?",
    ),
    "Equipment": (
        "middle",
        "basic_qna",
        "Before I answer that, what are equipment trailers commonly used for?",
    ),
    "Enclosed": (
        "start",
        "faq_email",
        "Before I answer that, do you offer financing?",
    ),
    "Utility": (
        "middle",
        "counter_question",
        "Before I answer that, what are utility trailers generally used for?",
    ),
    "Fiber": (
        "middle",
        "basic_qna",
        "Before I answer that, what is a fiber splicing trailer used for?",
    ),
    "Race Trailer": (
        "middle",
        "basic_qna",
        "Before I answer that, what are the main benefits of race trailers?",
    ),
    "Roll Off": (
        "start",
        "faq_email",
        "Before I answer that, do you offer delivery?",
    ),
    "Diesel Tank": (
        "middle",
        "basic_qna",
        "Before I answer that, what are diesel tank trailers used for?",
    ),
    "Flatbed": (
        "start",
        "basic_qna",
        "Before I answer that, what types of loads are flatbed trailers best suited for?",
    ),
    "Dump": (
        "middle",
        "faq_email",
        "Before I answer that, do you offer financing for trailers?",
    ),
    "Tilt": (
        "middle",
        "basic_qna",
        "Before I answer that, what can tilt trailers typically carry?",
    ),
    "Livestock": (
        "start",
        "skip_questions",
        "Can you please stop asking questions and just show me the results?",
    ),
}

_POST_RESULTS_SCENARIOS = {
    "Aluminum": ("escalation_email", "Please send me a formal quote for an aluminum trailer."),
    "Car Hauler": ("faq_email", "What financing options do you offer?"),
    "Equipment": ("basic_qna", "What are equipment trailers commonly used for?"),
    "Enclosed": ("basic_qna", "What kinds of cargo can enclosed trailers carry?"),
    "Utility": ("faq_email", "Do you offer trailer financing?"),
    "Fiber": ("escalation_email", "Please schedule a pickup appointment for a fiber trailer."),
    "Race Trailer": ("basic_qna", "What are the main benefits of race trailers?"),
    "Roll Off": ("faq_email", "Do you offer delivery for trailers?"),
    "Diesel Tank": ("escalation_email", "Please send me an invoice for a diesel tank trailer."),
    "Flatbed": ("basic_qna", "What types of loads are flatbed trailers best suited for?"),
    "Dump": ("faq_email", "Can you explain your financing options?"),
    "Tilt": ("basic_qna", "What can tilt trailers typically carry?"),
    "Livestock": ("escalation_email", "Please reserve a livestock trailer for me."),
}


_VAGUE_ANSWERS = {
    "base_category": "Utility would work best.",
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
    "cargo_size": "Roughly 19 by 8 feet; the height can be flexible.",
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


def _answer_for(slot: str | None, question: str) -> str:
    if slot in _VAGUE_ANSWERS:
        return _VAGUE_ANSWERS[slot]
    text = question.lower()
    if "weight" in text or "payload" in text:
        return _VAGUE_ANSWERS["haul_weight_lbs"]
    if "length" in text or "long" in text:
        return "Somewhere around 16 feet, but I am flexible."
    if "size" in text or "dimension" in text:
        return "A medium size, roughly 16 by 7 feet or thereabouts."
    if "prefer" in text:
        return "I do not have a strict preference; either common option is fine."
    return "Something fairly standard and flexible; I am open to suitable options."


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


def test_live_first_question_after_declining_contact_for_aluminum_and_hotshot():
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    scenarios = [
        (
            "Aluminum",
            "I want an aluminum trailer",
            "What type of trailer are you looking for in aluminum",
            "rough total weight",
        ),
        (
            "Flatbed",
            "I am looking for a hotshot trailer",
            "What will you be hauling on the flatbed",
            "approximate load weight",
        ),
    ]
    for category, request_text, expected_question, skipped_question in scenarios:
        session_id = str(uuid.uuid4())
        initial = _send(session_id, request_text)
        assert "Before we get started" in initial.assistant_text

        response = _send(session_id, "no")
        state = service._get_session(session_id)
        print(json.dumps({
            "scenario": "first_question_after_declining_contact",
            "category": category,
            "session_id": session_id,
            "request": request_text,
            "assistant": response.assistant_text,
            "resolved_category": state.get("trailer_category"),
            "awaiting_slot": state.get("awaiting_slot"),
            "slots_collected": state.get("slots_collected") or {},
            "metadata_filters": state.get("metadata_filters_collected") or {},
        }, ensure_ascii=False, default=str))

        assert state.get("trailer_category") == category
        assert expected_question.lower() in response.assistant_text.lower()
        assert skipped_question.lower() not in response.assistant_text.lower()


@pytest.mark.parametrize("category", _TEST_CATEGORIES, ids=_TEST_CATEGORIES)
def test_live_vague_answers_for_every_category(category: str):
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    session_id = str(uuid.uuid4())
    request_text = _REQUEST_OVERRIDES.get(category) or f"I am looking for a {category} trailer."
    if category in _DYNAMIC_WIDTH_CARGO and category not in _REQUEST_OVERRIDES:
        request_text = f"{request_text} {_DYNAMIC_WIDTH_CARGO[category]}"
    prefetched_response = None
    if category in {"Aluminum", "Car Hauler"}:
        initial_contact = _send(session_id, request_text)
        assert "Before we get started" in initial_contact.assistant_text
        partial_text = (
            "My name is Ibrahim"
            if category == "Aluminum"
            else "My email is ibrahim@esided.ai"
        )
        partial = _send(session_id, partial_text)
        expected_missing = (
            "email address or phone number"
            if category == "Aluminum"
            else "your name"
        )
        assert expected_missing in partial.assistant_text
        assert "Before we get started" not in partial.assistant_text
        user_text = "I would rather not provide anything else."
        prefetched_response = _send(session_id, user_text)
    else:
        user_text = (
            f"My name is Ibrahim and my email is ibrahim@esided.ai. {request_text}"
        )

    completed = False
    awaiting_history = []
    interruption = _GENERAL_QUERY_SCENARIOS.get(category)
    interruption_sent = False
    interruption_verified = False
    interrupted_slot = None
    answered_slot_count = 0
    for turn in range(1, 10):
        response = prefetched_response or _send(session_id, user_text)
        prefetched_response = None
        state = service._get_session(session_id)
        response_tool_events = (response.thinking_context or {}).get("tool_events") or []
        awaiting_history.append(state.get("awaiting_slot"))
        print(json.dumps({
            "category": category,
            "session_id": session_id,
            "turn": turn,
            "user": user_text,
            "assistant": response.assistant_text,
            "resolved_category": state.get("trailer_category"),
            "awaiting_slot": state.get("awaiting_slot"),
            "slots_collected": state.get("slots_collected") or {},
            "slots_skipped": state.get("slots_skipped") or [],
            "metadata_filters": state.get("metadata_filters_collected") or {},
            "tool_events": response_tool_events,
        }, ensure_ascii=False, default=str))

        if interruption_sent and not interruption_verified:
            _, kind, _ = interruption
            current_slot = str(state.get("awaiting_slot") or "").strip() or None
            if kind == "skip_questions":
                assert current_slot is None
                assert any(
                    event.get("tool") == "pinecone_search"
                    for event in response_tool_events
                    if isinstance(event, dict)
                )
                assert not any(
                    event.get("tool") == "send_escalation_alert_email"
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
            print(json.dumps({
                "scenario": kind,
                "category": category,
                "active_slot_preserved": current_slot,
                "assistant": response.assistant_text,
            }, ensure_ascii=False, default=str))

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
            stage, _, message = interruption
            if stage == "middle" and answered_slot_count == 0:
                user_text = _answer_for(slot, response.assistant_text)
                answered_slot_count += 1
                continue
            interrupted_slot = slot
            user_text = message
            interruption_sent = True
            continue
        user_text = _answer_for(slot, response.assistant_text)
        answered_slot_count += 1

    assert completed
    recommendation_state = service._get_session(session_id)
    assert recommendation_state.get("last_listings"), (
        f"{category} did not produce listings for the interest scenario"
    )

    interest_message = "I am interested in the first trailer."
    interest_response = _send(session_id, interest_message)
    initial_interest_events = (interest_response.thinking_context or {}).get("tool_events") or []
    if any(
        event.get("tool") == "send_interested_listing_email"
        and (event.get("result") or {}).get("status") == "deferred_missing_contact"
        for event in initial_interest_events
        if isinstance(event, dict)
    ):
        assert "Before we get started" not in interest_response.assistant_text
        saved = service._get_session(session_id)
        missing_contact = (
            "ibrahim@esided.ai"
            if saved.get("customer_full_name")
            else "My name is Ibrahim"
        )
        interest_response = _send(session_id, missing_contact)
    interest_state = service._get_session(session_id)
    interest_tool_events = (interest_response.thinking_context or {}).get("tool_events") or []
    interest_tools = {
        event.get("tool") for event in interest_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "category": category,
        "session_id": session_id,
        "scenario": "listing_interest",
        "user": interest_message,
        "assistant": interest_response.assistant_text,
        "resolved_category": interest_state.get("trailer_category"),
        "awaiting_slot": interest_state.get("awaiting_slot"),
        "tool_events": interest_tool_events,
    }, ensure_ascii=False, default=str))
    assert "send_interested_listing_email" in interest_tools

    post_kind, post_message = _POST_RESULTS_SCENARIOS[category]
    post_response = _send(session_id, post_message)
    post_state = service._get_session(session_id)
    post_tool_events = (post_response.thinking_context or {}).get("tool_events") or []
    post_tools = {
        event.get("tool") for event in post_tool_events if isinstance(event, dict)
    }
    print(json.dumps({
        "category": category,
        "session_id": session_id,
        "scenario": post_kind,
        "user": post_message,
        "assistant": post_response.assistant_text,
        "resolved_category": post_state.get("trailer_category"),
        "awaiting_slot": post_state.get("awaiting_slot"),
        "tool_events": post_tool_events,
    }, ensure_ascii=False, default=str))
    if post_kind == "escalation_email":
        assert "send_escalation_alert_email" in post_tools
    elif post_kind == "faq_email":
        assert "send_non_sales_faq_email" in post_tools
    else:
        assert post_response.assistant_text.strip()
        assert "pinecone_search" not in post_tools
        assert "send_interested_listing_email" not in post_tools
        assert "send_non_sales_faq_email" not in post_tools
        assert "send_escalation_alert_email" not in post_tools

    final_session = service._get_session(session_id)
    final_state = {
        "category": final_session.get("trailer_category"),
        "slots": final_session.get("slots_collected") or {},
        "skipped": final_session.get("slots_skipped") or [],
        "metadata_filters": final_session.get("metadata_filters_collected") or {},
        "awaiting_history": awaiting_history,
        "interruption": f"{interruption[0]}:{interruption[1]}" if interruption else None,
        "interruption_verified": interruption_verified,
        "listing_interest_verified": True,
        "post_results_scenario": post_kind,
        "post_results_verified": True,
    }
    print("VAGUE_QNA_SUMMARY=" + json.dumps({
        "category": category,
        "session_id": session_id,
        "completed": completed,
        "final_state": final_state,
    }, ensure_ascii=False, default=str))

    if interruption:
        assert interruption_verified
    if category in ("Utility", "Flatbed", "Tilt"):
        assert final_state["category"] == category
    if category == "Equipment":
        assert "hitch_type" in final_state["skipped"]
        assert "hitch_type" not in final_state["slots"]
        assert "hitch_type" not in final_state["metadata_filters"]
    if category == "Enclosed":
        assert "make" not in final_state["metadata_filters"]
        assert final_state["metadata_filters"].get("length_ft") == "19 ft"
        assert final_state["metadata_filters"].get("width_ft") == "8 ft"
        assert awaiting_history.count("cargo_size") == 1
    if category == "Roll Off":
        assert "package_scope" not in final_state["skipped"]
        package_scope = str(final_state["slots"].get("package_scope") or "").lower()
        assert "trailer" in package_scope
        assert "bin" in package_scope
    if category == "Car Hauler":
        assert final_state["slots"].get("vehicle_length_ft") == "15 ft"
        assert final_state["metadata_filters"].get("length_ft") == "15 ft"
    if category in _DYNAMIC_WIDTH_CARGO:
        assert "item_or_trailer_width_ft" in awaiting_history
        assert "item_or_trailer_width_ft" in final_state["skipped"]
        assert "width_ft" not in final_state["metadata_filters"]
