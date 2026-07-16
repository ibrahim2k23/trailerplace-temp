"""Human-readable conversation reasoning + state log.

Separate from `src/turn_log.py`, which is a compact JSONL machine feed the cost
audit reads. This one is for a developer/operator to literally watch the bot
think: the raw user message, everything the Analyze LLM extracted and decided,
which tools fired (search / inventory lookup / email) and their results, the
reply, and the full session state as it stood right after the turn — one block
per turn, on the `trailerplace.conversation` logger (console + the daily log
file, wired in `src/log_setup.py`).

Structured (fixed labeled fields, grep-able section markers) but written for a
terminal, not a JSON parser — `src/turn_log.py` already covers the machine-
readable case.
"""
from __future__ import annotations

import logging
from typing import Any

CONVERSATION_LOGGER_NAME = "trailerplace.conversation"

logger = logging.getLogger(CONVERSATION_LOGGER_NAME)

_BAR = "=" * 100
_THIN = "-" * 100


def _fmt(value: Any) -> str:
    """Render a value for a one-line label: value field. Empty containers -> '-'."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return "-" if not value else ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "-" if not value else ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _analysis_lines(analysis: Any) -> list[str]:
    """The Analyze LLM's full structured output — the bot's 'reasoning' for the turn."""
    if analysis is None:
        return ["  (no analysis available — receipt replay or a failed turn)"]

    extracted = analysis.extracted
    haul = analysis.haul_classification
    lookup = analysis.inventory_lookup

    lines = [
        f"  summary: {_fmt(getattr(analysis, 'turn_summary', None))}",
        f"  intent: {analysis.intent}",
        f"  category_mentioned: {_fmt(analysis.category_mentioned)}  (info_only={analysis.is_category_info_only})",
        "  extracted: "
        f"length={_fmt(extracted.trailer_length_ft)} width={_fmt(extracted.trailer_width_ft)} "
        f"height={_fmt(extracted.trailer_height_ft)} payload_lbs={_fmt(extracted.payload_lbs)} "
        f"hitch={_fmt(extracted.hitch_type)} brand={_fmt(extracted.brand_preference)}",
        f"  haul_item: {_fmt(extracted.haul_item)}  features: {_fmt(extracted.non_metadata_features)}"
        f"  no_preference: {_fmt(extracted.numeric_no_preference)}",
        f"  slot_answers: {_fmt([f'{a.slot_name}={a.raw_answer!r}' for a in analysis.slot_answers])}",
        "  haul_classification: "
        f"lightweight={haul.is_lightweight_utility_load} needs_width={haul.needs_width_question} "
        f"matched={_fmt(haul.haul_item_matched)}",
    ]
    if lookup.is_lookup:
        lines.append(
            "  inventory_lookup: "
            f"year={_fmt(lookup.year)} make={_fmt(lookup.make)} model={_fmt(lookup.model_text)} "
            f"stock={_fmt(lookup.stock_number)} wants={_fmt(lookup.wants)} confidence={lookup.confidence}"
        )
    if analysis.email_triggers:
        triggers = ", ".join(
            trigger.kind + (f"[{trigger.faq_key}]" if trigger.faq_key else "") for trigger in analysis.email_triggers
        )
        lines.append(f"  email_triggers: {triggers}")
    contact = analysis.contact
    if contact.name or contact.email or contact.phone:
        lines.append(
            f"  contact_in_message: name={_fmt(contact.name)} email={_fmt(contact.email)} phone={_fmt(contact.phone)}"
        )
    if analysis.listing_reference:
        lines.append(f"  listing_reference: #{analysis.listing_reference}")
    if not analysis.answered_current_question and analysis.user_question_to_answer:
        lines.append(f'  interruption: "{analysis.user_question_to_answer}" (pending question not answered)')
    if analysis.dropped_fields or analysis.keep_fields_answer:
        lines.append(
            f"  requirement_change: dropped={_fmt(analysis.dropped_fields)} "
            f"keep_answer={_fmt(analysis.keep_fields_answer)} kept={_fmt(analysis.kept_fields)}"
        )
    if analysis.category_confirm_answer:
        lines.append(f"  category_confirm_answer: {analysis.category_confirm_answer}")
    return lines


def _decision_lines_block(turn_outcome: dict) -> list[str]:
    """The exact 'WHAT THE SYSTEM ALREADY DECIDED' lines the respond prompt was built with.

    Stashed by build_respond_prompt — this is what the reply model was TOLD to do, so a reply
    that ignored its instructions can be diagnosed from the log alone.
    """
    raw = (turn_outcome or {}).get("decision_lines_log")
    if not raw:
        return []
    lines = ["DECISION LINES (Respond prompt):"]
    lines.extend(f"  {line}" for line in str(raw).splitlines())
    lines.append(_THIN)
    return lines


def _tool_lines(turn_outcome: dict) -> list[str]:
    tools: list[str] = []
    if turn_outcome.get("search_ran"):
        tools.append(
            f"search(results={turn_outcome.get('result_count', 0)}, "
            f"brand_relaxed={turn_outcome.get('brand_relaxed', False)})"
        )
    if turn_outcome.get("inventory_lookup_ran"):
        tools.append(f"inventory_lookup(status={turn_outcome.get('inventory_match_status')})")
    if turn_outcome.get("emails_sent"):
        tools.append(f"email({', '.join(turn_outcome['emails_sent'])})")
    lines = [f"TOOLS FIRED: {', '.join(tools) if tools else '-'}"]
    if turn_outcome.get("email_status"):
        lines.append(f"EMAIL STATUS: {turn_outcome['email_status']}")
    return lines


def _state_lines(state: dict) -> list[str]:
    """Every field a support ticket would need to answer 'what does the bot think it knows'."""
    contact = (
        f"name={_fmt(state.get('customer_name'))} email={_fmt(state.get('customer_email'))} "
        f"phone={_fmt(state.get('customer_phone'))} declined={state.get('contact_declined', False)}"
    )
    pending_slot = state.get("pending_question_slot")
    pending_line = (
        f"  pending_question: {pending_slot} (repeats={state.get('pending_question_repeats', 0)})"
        if pending_slot
        else "  pending_question: -"
    )
    return [
        f"  category: {_fmt(state.get('category'))}   clarification_key: {_fmt(state.get('clarification_key'))}",
        f"  slots: {_fmt(state.get('slots'))}",
        f"  skipped_slots: {_fmt(state.get('skipped_slots'))}   "
        f"qualification_complete: {state.get('qualification_complete', False)}",
        pending_line,
        f"  brand_preference: {_fmt(state.get('brand_preference'))}   "
        f"non_metadata_features: {_fmt(state.get('non_metadata_features'))}",
        f"  contact: {contact}",
        f"  pending_category_change: {_fmt(state.get('pending_category_change'))}",
        f"  pending_category_suggestion: {_fmt(state.get('pending_category_suggestion'))}",
        f"  pending_email_actions: {len(state.get('pending_email_actions') or [])}   "
        f"contact_followup_pending: {_fmt(state.get('contact_followup_pending'))}",
        f"  shown_listings: {len(state.get('shown_listings') or [])}   "
        f"shown_urls: {len(state.get('shown_urls') or [])}",
    ]


def log_conversation_turn(
    *,
    session_id: str,
    turn_id: str,
    user_message: str,
    analysis: Any,
    assistant_text: str | None,
    cited_listing_urls: list[str] | None,
    listings_returned: int,
    turn_outcome: dict,
    state: dict,
    error: str | None = None,
) -> None:
    """Log one full, human-readable turn block. Never raises."""
    try:
        lines: list[str] = [
            _BAR,
            f"TURN  session={session_id}  turn={turn_id}",
            _THIN,
            f"USER: {user_message}",
            _THIN,
            "REASONING (Analyze):",
            *_analysis_lines(analysis),
            _THIN,
            *_decision_lines_block(turn_outcome or {}),
            *_tool_lines(turn_outcome or {}),
            _THIN,
        ]
        if error:
            lines.append(f"ERROR: {error}")
        else:
            lines.append(f"ASSISTANT: {assistant_text or ''}")
            if cited_listing_urls:
                lines.append(f"  cited_listing_urls: {_fmt(cited_listing_urls)}")
            lines.append(f"  listings_returned: {listings_returned}")
        lines.extend([_THIN, "STATE AFTER TURN:", *_state_lines(state or {}), _BAR])
        logger.info("\n".join(lines))
    except Exception:  # noqa: BLE001 - observability must never break the response
        logger.exception("conversation logging failed for session %s", session_id)
