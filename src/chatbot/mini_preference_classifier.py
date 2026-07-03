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
    active_question: str | None = None,
) -> PreferenceNullDecision:
    if not awaiting_slot and not (pending_questions or []):
        return PreferenceNullDecision(reason="No active qualification question to answer.", confidence="low")

    active_slot = awaiting_slot
    if not active_slot and pending_questions:
        active_slot = str((pending_questions or [{}])[0].get("slot") or "")
    question_text = (active_question or "").strip()
    if not question_text and pending_questions:
        question_text = str((pending_questions or [{}])[0].get("question") or "").strip()

    context = {
        "active_question": question_text,
        "user_answer": user_message,
        "active_slot": active_slot,
    }
    try:
        return _preference_classifier_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Decide whether the user's answer means they have no preference, no fixed requirement, "
                        "or want to leave the active qualification question unrestricted. Return structured data only.\n\n"

                        "## RULES\n"
                        "1. Use only active_question and user_answer. Do not infer from broader conversation.\n"
                        "2. has_no_preference=true for: 'any', 'whatever', 'doesn't matter', 'not sure', "
                        "'I don't know', 'flexible', 'no fixed size', 'no specific preference', 'no preference'.\n"
                        "3. CATEGORY QUESTIONS: also true for 'no idea', 'any type', 'type doesn't matter', "
                        "'I don't care about category', 'any [make] trailer', 'just show me [make]'.\n"
                        "   - A retained make/brand alone ('any Iron Bull trailer') = no category preference; preserve the make.\n"
                        "   - 'Gooseneck' or 'Bumper Pull' while a category question is active = hitch constraint only, "
                        "not a category answer → has_no_preference=true for the category slot.\n"
                        "4. has_no_preference=false when the answer gives any concrete value: category, item, color, "
                        "hitch, length, width, weight, price, or similar.\n"
                        "5. If true: target_slots = [active_slot] only (when active_slot is present). "
                        "target_metadata_filters may include the corresponding search filter when relevant.\n"
                        "6. Use medium or high confidence only when the answer clearly means unrestricted/no preference."
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
