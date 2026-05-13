from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    if cat not in {"utility", "enclosed"} and heavy:
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
        "lightweight_keywords": LIGHTWEIGHT_KEYWORDS,
        "heavy_duty_keywords": HEAVY_DUTY_KEYWORDS,
    }
    try:
        return _haul_classifier_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Classify whether trailer qualification questions should be adjusted.\n"
                        "Rules:\n"
                        "- For Utility only, mark is_lightweight_utility_load when the haul item is likely 1500 lbs or less.\n"
                        "- For categories except Utility and Enclosed, mark needs_width_question when the item is very large, wide, heavy0duty or a item of a vehicle brand or similar nature such as cars, tractors etc.\n"
                        "- Treat the keyword lists as strong examples, but similar items may qualify.\n"
                        "- Do not request a width question for Utility or Enclosed.\n"
                        "- Use medium or high confidence only when the item is clear."
                    )
                ),
                HumanMessage(content=f"Return structured classification for:\n{json.dumps(context, default=str)}"),
            ]
        )
    except Exception:
        logger.exception("Haul classifier LLM failed; using keyword fallback")
        return fallback_haul_classification(
            category=category,
            user_message=user_message,
            recent_messages=recent_messages,
            slots_collected=slots_collected,
            metadata_filters_collected=metadata_filters_collected,
        )
