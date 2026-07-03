"""Opt-in live diagnostic for vague answers across every trailer category.

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


_VAGUE_ANSWERS = {
    "haul_item": "Mostly assorted equipment and other general things.",
    "haul_material": "A mixture of random materials and general debris.",
    "haul_weight_lbs": "Somewhere around 5,000 pounds, give or take.",
    "haul_length_ft": "Probably around 16 to 20 feet.",
    "vehicle_type": "Different ordinary cars, nothing very specific.",
    "vehicle_length_ft": "Roughly 15 to 18 feet or thereabouts.",
    "trailer_length_ft": "Something around 16 feet, but I am flexible.",
    "hitch_type": "Either bumper pull or gooseneck would probably be fine.",
    "loading_style": "Whatever loading setup is generally easiest.",
    "open_vs_covered": "I could work with either open or covered.",
    "trailer_size": "A medium-sized one, maybe around 16 by 7.",
    "sides_gate_storage": "Some useful sides or storage would be nice.",
    "dump_mechanism": "Any dependable dump mechanism should be okay.",
    "tilt_style": "Either tilt style is fine as long as it works well.",
    "use_case": "General cargo and perhaps some occasional work use.",
    "cargo_size": "Roughly 16 by 7 feet, with flexible height.",
    "ac_windows_cabinets": "Maybe some of those amenities, but I am flexible.",
    "finished_interior": "A basic finish is fine; I am open either way.",
    "gate_preferences": "Any practical gate style would work.",
    "package_scope": "Probably the trailer and some bins, but I am flexible.",
    "bin_size": "A medium bin, perhaps around 15 to 20 yards.",
    "deck_style": "Either a step deck or standard deck is okay.",
    "cdl_concern": "I would prefer flexibility around CDL requirements.",
    "fuel_type": "Mostly diesel, though general fuel use is possible.",
    "tank_capacity": "A few hundred gallons, maybe around 500.",
    "fiber_use_case": "Mostly splicing and perhaps occasional office use.",
    "crew_size": "A small crew, perhaps three or four people.",
    "fiber_amenities": "Basic useful amenities such as AC and a workbench.",
    "race_amenities": "Some cabinets and workspace would probably help.",
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


def test_live_vague_answers_for_every_category():
    if os.getenv("RUN_LIVE_VAGUE_QNA") != "1":
        pytest.skip("Set RUN_LIVE_VAGUE_QNA=1 to run real LLM/Pinecone conversations")

    summaries = []
    for category in CANONICAL_CATEGORIES:
        # Durable persistence stores session IDs as UUID columns.
        session_id = str(uuid.uuid4())
        user_text = (
            f"My name is Ibrahim and my email is ibrahim@esided.ai. "
            f"I am looking for a {category} trailer."
        )

        completed = False
        for turn in range(1, 10):
            response = _send(session_id, user_text)
            state = service._get_session(session_id)
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
                "tool_events": state.get("tool_events") or [],
            }, ensure_ascii=False, default=str))

            if state.get("last_listings") or any(
                event.get("tool") == "pinecone_search"
                for event in (state.get("tool_events") or [])
                if isinstance(event, dict)
            ):
                completed = True
                break

            slot = str(state.get("awaiting_slot") or "").strip() or None
            if not slot:
                completed = True
                break
            user_text = _answer_for(slot, response.assistant_text)

        summaries.append({
            "category": category,
            "session_id": session_id,
            "completed": completed,
            "final_state": {
                "category": service._get_session(session_id).get("trailer_category"),
                "slots": service._get_session(session_id).get("slots_collected") or {},
                "skipped": service._get_session(session_id).get("slots_skipped") or [],
            },
        })

    print("VAGUE_QNA_SUMMARY=" + json.dumps(summaries, ensure_ascii=False, default=str))
    assert [item["category"] for item in summaries] == list(CANONICAL_CATEGORIES)
    assert all(item["completed"] for item in summaries)
