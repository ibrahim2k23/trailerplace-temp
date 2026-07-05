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
    "haul_item": "A bit of everything—tools, machines, and whatever else comes up.",
    "haul_material": "Usually cleanup waste, branches, and mixed jobsite stuff.",
    "haul_weight_lbs": "I'd guess between six and eight thousand pounds.",
    "haul_length_ft": "Maybe eighteen to twenty-four feet should do.",
    "vehicle_type": "Mostly regular sedans and the occasional small SUV.",
    "vehicle_length_ft": "They are likely somewhere between 14 and 17 feet long.",
    "item_or_trailer_width_ft": "A normal flexible width should be fine; I have no specific measurement.",
    "trailer_length_ft": "Around eighteen feet sounds right, though I can adjust.",
    "hitch_type": "Either hitch works for me; I don't lean one way.",
    "loading_style": "Whichever setup makes loading less of a headache.",
    "open_vs_covered": "Open or covered is fine—I can make either work.",
    "trailer_size": "Nothing huge, perhaps about 18 by 8.",
    "sides_gate_storage": "Some practical storage or side rails would be handy.",
    "dump_mechanism": "I have no strong preference as long as it dumps reliably.",
    "tilt_style": "Any common tilt arrangement should be alright.",
    "use_case": "Mostly moving supplies, with some light workshop use now and then.",
    "cargo_size": "About 18 by 8 feet; height is not particularly important.",
    "ac_windows_cabinets": "A few useful comforts would be nice, but none are essential.",
    "finished_interior": "Either unfinished or simply finished would suit me.",
    "gate_preferences": "Whatever gate is easiest for normal day-to-day use.",
    "package_scope": "I need the trailer together with a few bins, ideally.",
    "bin_size": "A medium bin, perhaps around 15 to 20 yards.",
    "deck_style": "A step deck or regular deck would both be acceptable.",
    "cdl_concern": "Staying below CDL limits would be helpful, but I am flexible.",
    "fuel_type": "It will mainly carry diesel, possibly other fuel occasionally.",
    "tank_capacity": "Somewhere in the 400 to 600 gallon neighborhood.",
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


def _send(session_id: str, message: str):
    return service.handle_chat(
        ChatRequest(
            session_id=session_id,
            sales_phase="main",
            message=message,
            customer_full_name="Ibrahim",
            customer_email="ibrahim@esided.ai",
        )
    )


@pytest.mark.parametrize("category", _TEST_CATEGORIES, ids=_TEST_CATEGORIES)
def test_live_vague_answers_for_every_category(category: str):
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    session_id = str(uuid.uuid4())
    user_text = (
        f"My name is Ibrahim and my email is ibrahim@esided.ai. "
        f"I am looking for a {category} trailer."
    )
    if category in _DYNAMIC_WIDTH_CARGO:
        user_text = f"{user_text} {_DYNAMIC_WIDTH_CARGO[category]}"

    completed = False
    awaiting_history = []
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
            kind = interruption[0]
            current_slot = str(state.get("awaiting_slot") or "").strip() or None
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
            interrupted_slot = slot
            user_text = interruption[1]
            interruption_sent = True
            continue
        user_text = _answer_for(slot, response.assistant_text)

    assert completed
    recommendation_state = service._get_session(session_id)
    assert recommendation_state.get("last_listings"), (
        f"{category} did not produce listings for the interest scenario"
    )

    interest_message = "I am interested in the first trailer."
    interest_response = _send(session_id, interest_message)
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
        "interruption": interruption[0] if interruption else None,
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
        assert final_state["metadata_filters"].get("length_ft") == "18 ft"
        assert final_state["metadata_filters"].get("width_ft") == "8 ft"
        assert awaiting_history.count("cargo_size") == 1
    if category == "Roll Off":
        assert "package_scope" not in final_state["skipped"]
        package_scope = str(final_state["slots"].get("package_scope") or "").lower()
        assert "trailer" in package_scope
        assert "bin" in package_scope
    if category == "Car Hauler":
        assert final_state["slots"].get("vehicle_length_ft") == "14 ft"
        assert final_state["metadata_filters"].get("length_ft") == "14 ft"
    if category in _DYNAMIC_WIDTH_CARGO:
        assert "item_or_trailer_width_ft" in awaiting_history
        assert "item_or_trailer_width_ft" in final_state["skipped"]
        assert "width_ft" not in final_state["metadata_filters"]
