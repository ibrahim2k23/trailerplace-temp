from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.chatbot.constants import DYNAMIC_WIDTH_EXCLUDED_CATEGORIES

logger = logging.getLogger(__name__)

# Human-readable, capitalized rendering of the shared exclusion set for prompts.
_WIDTH_EXCLUDED_LABEL = ", ".join(
    sorted(c.title() for c in DYNAMIC_WIDTH_EXCLUDED_CATEGORIES)
)

LIGHTWEIGHT_KEYWORDS: tuple[str, ...] = (
    "golf cart", "golf carts", "golf equipment",
    "atv", "atvs", "utv", "utvs", "side-by-side", "side by side",
    "dirt bike", "dirt bikes", "motorcycle", "motorcycles",
    "lawn mower", "lawn mowers", "zero turn", "zero-turn",
    "riding mower", "push mower",
    "gardening tools", "landscaping tools", "lawn equipment", "lawn tools",
    "small generator", "small equipment",
    "canoe", "canoes", "kayak", "kayaks", "small watercraft",
    "bicycle", "bicycles", "e-bike", "e-bikes", "ebike", "ebikes",
    "small furniture", "camping gear", "light cargo", "hobby equipment",
)

HEAVY_DUTY_KEYWORDS: tuple[str, ...] = (
    "excavator", "mini excavator", "dozer", "bulldozer", "backhoe", "trackhoe",
    "skid steer", "telehandler", "forklift", "loader", "wheel loader",
    "tractor", "combine", "harvester", "roller", "compactor",
    "scissor lift", "boom lift", "lift", "heavy machinery", "heavy equipment",
    "oversize", "oversized", "wide load",
)


class HaulClassificationDecision(BaseModel):
    is_lightweight_utility_load: bool = False
    needs_width_question: bool = False
    matched_item: Optional[str] = None
    reason: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


@lru_cache(maxsize=1)
def _haul_classifier_llm():
    model = (
        os.getenv("HAUL_CLASSIFIER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        HaulClassificationDecision,
        method="function_calling",
    )


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
    low = text.lower()
    for keyword in keywords:
        if keyword in low:
            return keyword
    return None


def _context_text(
    *,
    user_message: str,
    slots_collected: dict[str, Any],
    recent_messages: list[dict[str, Any]] | list[str],
) -> str:
    slot_values = " ".join(str(v) for v in (slots_collected or {}).values() if v not in (None, ""))
    recent_values = []
    for item in recent_messages or []:
        if isinstance(item, dict):
            if item.get("role") == "user":
                recent_values.append(str(item.get("content") or ""))
        else:
            recent_values.append(str(item))
    return " ".join([user_message or "", slot_values, *recent_values]).strip()


def fallback_haul_classification(
    *,
    category: str | None,
    user_message: str,
    recent_messages: list[dict[str, Any]] | list[str] | None = None,
    slots_collected: dict[str, Any] | None = None,
    metadata_filters_collected: dict[str, Any] | None = None,
) -> HaulClassificationDecision:
    del metadata_filters_collected
    text = _context_text(
        user_message=user_message,
        slots_collected=slots_collected or {},
        recent_messages=recent_messages or [],
    )
    cat = str(category or "").strip().lower()

    lightweight = _contains_keyword(text, LIGHTWEIGHT_KEYWORDS)
    if cat == "utility" and lightweight:
        return HaulClassificationDecision(
            is_lightweight_utility_load=True,
            needs_width_question=False,
            matched_item=lightweight,
            reason="Matched lightweight Utility keyword.",
            confidence="high",
        )

    heavy = _contains_keyword(text, HEAVY_DUTY_KEYWORDS)
    if cat not in DYNAMIC_WIDTH_EXCLUDED_CATEGORIES and heavy:
        return HaulClassificationDecision(
            is_lightweight_utility_load=False,
            needs_width_question=True,
            matched_item=heavy,
            reason="Matched heavy-duty keyword.",
            confidence="high",
        )

    return HaulClassificationDecision(reason="No keyword match.", confidence="low")


def classify_haul_requirements(
    *,
    category: str | None,
    user_message: str,
    recent_messages: list[dict[str, Any]] | list[str] | None = None,
    slots_collected: dict[str, Any] | None = None,
    metadata_filters_collected: dict[str, Any] | None = None,
) -> HaulClassificationDecision:
    context = {
        "category": category,
        "latest_user_message": user_message,
        "recent_messages": recent_messages or [],
        "slots_collected": slots_collected or {},
        "metadata_filters_collected": metadata_filters_collected or {},
    }
    system_prompt = (
        "Classify whether trailer qualification questions should be adjusted.\n"
        "Rules:\n"
        "- Read latest_user_message first and use recent context only to resolve references.\n"
        "- matched_item must contain the specific cargo/item the user says they will haul, "
        "using a concise phrase grounded in the user's words.\n"
        "- CRITICAL: a trailer category names the requested trailer type, not its cargo. Never return a category "
        "name or category phrase as matched_item merely because that category was requested. Return matched_item=null "
        "for 'I want an equipment trailer' or 'show utility trailers'. Accept a category-like term only when explicitly "
        "framed as cargo, such as 'I need to haul equipment', or as a direct answer to an active haul-item question.\n"
        "- A recommendation question such as 'which trailer is suitable for a dirt bike?' "
        "still explicitly states the hauled item; return that item in matched_item.\n"
        "- Never return is_lightweight_utility_load=true or needs_width_question=true with "
        "matched_item=null when an item is stated in the latest message or context.\n"
        "- Do not use a trailer category, trailer description, size, make, or model as matched_item.\n"
        "- Gooseneck and Bumper Pull are strictly hitch types. Never return either as matched_item and do not use "
        "either to infer a trailer category or haul item.\n"
        "- For Utility only, mark is_lightweight_utility_load when the haul item is likely 1500 lbs or less.\n"
        f"- For categories except {_WIDTH_EXCLUDED_LABEL}, mark needs_width_question when "
        "the item is very large, wide, heavy-duty, or a vehicle such as a car or tractor.\n"
        f"- Do not request a width question for {_WIDTH_EXCLUDED_LABEL}.\n"
        "- Use medium or high confidence only when the item is clear."
    )
    try:
        decision = _haul_classifier_llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Return structured classification for:\n{json.dumps(context, default=str)}"),
            ]
        )
        if (
            not str(decision.matched_item or "").strip()
            and (
                decision.is_lightweight_utility_load
                or decision.needs_width_question
                or decision.confidence in {"medium", "high"}
            )
        ):
            decision = _haul_classifier_llm().invoke(
                [
                    SystemMessage(
                        content=(
                            system_prompt
                            + "\nThe first classification set a flag without a matched_item. Re-evaluate. "
                            "If the conversation does NOT explicitly state a specific cargo/item, return "
                            "matched_item=null AND is_lightweight_utility_load=false AND "
                            "needs_width_question=false with low confidence — that is the correct, expected "
                            "answer, not a failure. Only when a specific item is explicitly stated should "
                            "matched_item contain it."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Conversation context:\n"
                            f"{json.dumps(context, default=str)}\n\n"
                            "Incomplete first classification:\n"
                            f"{json.dumps(decision.model_dump(), default=str)}"
                        )
                    ),
                ]
            )
        # Deterministic invariant: a flag without a concrete cargo item is not
        # actionable and previously came from the model inventing an item under
        # retry pressure. If matched_item is still empty, clear the flags rather
        # than let a fabricated item flow into the qualification slots/filters.
        if not str(decision.matched_item or "").strip() and (
            decision.is_lightweight_utility_load or decision.needs_width_question
        ):
            decision.is_lightweight_utility_load = False
            decision.needs_width_question = False
            decision.confidence = "low"
        return decision
    except Exception:
        logger.exception("Haul classifier LLM failed; returning an unclassified result")
        return HaulClassificationDecision(
            reason="The LLM classifier was unavailable.",
            confidence="low",
        )
