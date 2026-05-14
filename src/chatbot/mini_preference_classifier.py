from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_NO_PREFERENCE_MARKERS = (
    "any",
    "anything",
    "whatever",
    "no preference",
    "no pref",
    "doesn't matter",
    "does not matter",
    "dont care",
    "don't care",
    "do not care",
    "not picky",
    "no idea",
    "not sure",
    "unsure",
    "i don't know",
    "i dont know",
    "do not know",
    "don't know",
    "flexible",
    "no specific",
    "no particular",
    "surprise me",
)


class PreferenceNullDecision(BaseModel):
    has_no_preference: bool = False
    target_slots: list[str] = Field(default_factory=list)
    target_metadata_filters: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


@lru_cache(maxsize=1)
def _preference_classifier_llm():
    model = (
        os.getenv("PREFERENCE_CLASSIFIER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    ).strip()
    return ChatOpenAI(model=model, temperature=0).with_structured_output(
        PreferenceNullDecision,
        method="function_calling",
    )


def classify_no_preference(
    *,
    category: str | None,
    user_message: str,
    awaiting_slot: str | None = None,
    pending_questions: list[dict[str, Any]] | None = None,
    slots_collected: dict[str, Any] | None = None,
    metadata_filters_collected: dict[str, Any] | None = None,
    allowed_category_slots: list[str] | None = None,
) -> PreferenceNullDecision:
    latest = (user_message or "").strip().lower()
    if not any(marker in latest for marker in _NO_PREFERENCE_MARKERS):
        return PreferenceNullDecision(reason="No explicit no-preference language.", confidence="low")
    if not awaiting_slot and not (pending_questions or []):
        return PreferenceNullDecision(reason="No active qualification question to answer.", confidence="low")

    context = {
        "category": category,
        "latest_user_message": user_message,
        "awaiting_slot": awaiting_slot,
        "pending_questions": pending_questions or [],
        "slots_collected": slots_collected or {},
        "metadata_filters_collected": metadata_filters_collected or {},
        "allowed_category_slots": allowed_category_slots or [],
    }
    try:
        return _preference_classifier_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Decide whether the latest user message means the customer has no preference "
                        "for one or more currently relevant trailer qualification questions or search filters.\n"
                        "Rules:\n"
                        "- Return has_no_preference=true only when the user clearly wants a value left blank, unrestricted, any, flexible, or not filtered.\n"
                        "- Prefer targeting the current awaiting_slot when the message is a direct answer to a question.\n"
                        "- Only answer the current awaiting_slot or one of the pending_questions; do not skip unrelated category slots.\n"
                        "- Use target_slots for category qualification slots that should be considered answered/skipped.\n"
                        "- Use target_metadata_filters for Pinecone filters to remove/leave unset, such as length_ft, width_ft, payload_lbs, max_price, hitch_type, color, subcategory.\n"
                        "- Do not mark no preference if the user gives an explicit value in the latest message for that field.\n"
                        "- Use medium or high confidence only when the no-preference intent is clear."
                    )
                ),
                HumanMessage(
                    content=(
                        "Return structured no-preference decision for:\n"
                        f"{json.dumps(context, ensure_ascii=True, default=str)}"
                    )
                ),
            ]
        )
    except Exception:
        logger.exception("Preference classifier LLM failed; using conservative fallback")
        return PreferenceNullDecision(reason="Preference classifier failed.", confidence="low")
