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
                        "do not know, or want to leave the current qualification question unrestricted.\n"
                        "Rules:\n"
                        "- Use only the active_question and user_answer. Do not infer from broader conversation.\n"
                        "- Return has_no_preference=true for answers like any, whatever, no fixed size, no specific preference, not sure, I don't know, flexible, or doesn't matter.\n"
                        "- For a category/type/kind choice question, answers like no idea, no type in mind, any type, no category preference, category doesn't matter, type doesn't matter, I don't care about category, any <make> trailer, or just show me <make> also mean no preference.\n"
                        "- If a category/type/kind question is active, a retained make or brand name is not a category choice by itself; for example, 'any Iron Bull trailer' means no category preference while preserving the make.\n"
                        "- Return has_no_preference=false when the answer provides a concrete value, constraint, item, category, color, hitch, length, width, weight, or price.\n"
                        "- If true, target_slots should contain only active_slot when active_slot is present.\n"
                        "- target_metadata_filters may include the corresponding search filter for that active_slot when relevant.\n"
                        "- Use medium or high confidence only when the answer clearly means unrestricted/no preference for the active question."
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
