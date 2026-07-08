"""Shared mutable working state for the decomposed ``_apply_mind_node`` pipeline.

``_apply_mind_node`` used to be a single ~1,400-line function threading dozens of
mutable locals through four sequential phases (category resolution, active Q&A
turn, question planning, action dispatch). E7 splits it into one phase function
per stage. To keep each phase body byte-for-byte identical to the original
monolith — the transform that provably cannot change behavior — every local that
crosses a phase boundary is carried on this context object instead of as a bare
local. Each phase unpacks the fields it needs into same-named locals, runs the
verbatim body, then repacks the (possibly mutated) locals back onto the context.

``APPLY_MIND_FIELDS`` is the exact set of boundary-crossing names, derived by
static analysis of the original function (stored in one phase, read in a later
one), plus a few loop temporaries that are harmless to carry. Missing a genuine
crossing name would silently propagate a stale value, so the set is intentionally
a safe superset.
"""

from __future__ import annotations

from typing import Any

from src.chatbot.state import ChatbotState

APPLY_MIND_FIELDS: tuple[str, ...] = (
    "active_qna_counter_topic",
    "active_qna_email_action",
    "active_qna_escalation_summary",
    "active_qna_faq_category",
    "active_qna_faq_summary",
    "active_qna_question",
    "active_qna_reply",
    "active_qna_retry_question",
    "active_qna_search_now",
    "active_qna_skip_remaining",
    "active_qna_slot",
    "active_qna_unanswered",
    "active_qna_unsupported_request",
    "active_question_attempts",
    "active_question_was_resolved",
    "active_question_was_unanswered",
    "assistant_text",
    "awaiting_slot",
    "category",
    "category_before",
    "category_changed",
    "category_clarification_key",
    "category_needs_clarification",
    "decision",
    "defaulted_metadata_filters",
    "dynamic_questions",
    "extracted_features",
    "extracted_slots",
    "generic_no_category_missing",
    "invalid_required",
    "invalid_required_slot",
    "key",
    "latest_message",
    "make_category_options",
    "make_changed",
    "make_only_missing",
    "metadata_filters",
    "metadata_filters_before",
    "missing_required",
    "original_mind_action",
    "original_mind_text",
    "pending",
    "question_turn",
    "questions_by_slot",
    "repeated_unanswered_escalation",
    "requested_non_metadata_features",
    "required_slots_override",
    "reset_result_state",
    "skipped_unanswered_slot",
    "slot",
    "slots",
    "slots_before",
    "slots_skipped",
    "slots_skipped_before",
    "state",
    "value",
)

_APPLY_MIND_FIELD_SET = frozenset(APPLY_MIND_FIELDS)


class ApplyMindContext:
    """Mutable bag of the locals shared across ``_apply_mind_node`` phases."""

    __slots__ = APPLY_MIND_FIELDS

    def __init__(self, state: ChatbotState) -> None:
        for name in APPLY_MIND_FIELDS:
            setattr(self, name, None)
        self.state = state

    def unpack(self) -> dict[str, Any]:
        """Return every field as a dict for binding to same-named locals."""
        return {name: getattr(self, name) for name in APPLY_MIND_FIELDS}

    def repack(self, local_vars: dict[str, Any]) -> None:
        """Write back any field that currently exists as a phase local."""
        for name, value in local_vars.items():
            if name in _APPLY_MIND_FIELD_SET:
                setattr(self, name, value)
