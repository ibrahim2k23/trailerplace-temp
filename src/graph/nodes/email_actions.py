from __future__ import annotations

import logging
from typing import Any

from src import conversation_store
from src.tools import email_sender

logger = logging.getLogger(__name__)

# Reason / event_type / canned-key mapping per milestone.md M7 step 2 & step 3.
# `reason` feeds the email body + subject and `turn_outcome.emails_sent`.
# `event_type` is the chatbot_outbox row type the drain handler dispatches on.
_CUSTOMER_KIND_META = {
    "escalation": {"reason": "Escalation", "event_type": "escalation_alert", "canned_key": "escalation"},
    "team_request": {"reason": "Team Request", "event_type": "team_request", "canned_key": "generic_team_request"},
    # faq / listing_interest are resolved dynamically (faq_key / listing reference).
}
_SYSTEM_KIND_META = {
    "results_shown": {"reason": "Results Shown to User", "event_type": "results_shown"},
    "unanswered_question": {"reason": "Unanswered Question", "event_type": "unanswered_question"},
}


def _listing_get(listing: Any, key: str, default: Any = "") -> Any:
    return listing.get(key, default) if isinstance(listing, dict) else getattr(listing, key, default)


def _resolve_listing_interest(state: dict, trigger: Any) -> dict[str, Any]:
    """Pick the selected/unselected/fallback canned variant (spec §Tools, M7 step 3)."""
    shown = state.get("shown_listings") or []
    ref = getattr(trigger, "listing_reference", None)
    base = {"kind": "listing_interest", "reason": "Listing Interest", "event_type": "interested_listing", "is_system": False}
    if ref and 1 <= ref <= len(shown):
        listing = shown[ref - 1]
        title = _listing_get(listing, "title") or "a trailer"
        url = _listing_get(listing, "url") or ""
        return {
            **base,
            "canned_key": "listing_interest_selected",
            "description": f"{trigger.description} ({title} — {url})",
            "item_of_interest": title,
        }
    if shown:
        return {**base, "canned_key": "listing_interest_unselected", "description": trigger.description, "item_of_interest": None}
    return {**base, "canned_key": "listing_interest_fallback", "description": trigger.description, "item_of_interest": None}


def _resolve_customer_trigger(state: dict, trigger: Any) -> dict[str, Any]:
    if trigger.kind == "faq":
        faq_key = trigger.faq_key or "contact_human"
        return {
            "kind": "faq",
            "reason": f"FAQ – {faq_key}",
            "event_type": "non_sales_faq",
            "canned_key": faq_key,
            "description": trigger.description,
            "is_system": False,
            "item_of_interest": None,
        }
    if trigger.kind == "listing_interest":
        return _resolve_listing_interest(state, trigger)
    meta = _CUSTOMER_KIND_META[trigger.kind]
    return {
        "kind": trigger.kind,
        "reason": meta["reason"],
        "event_type": meta["event_type"],
        "canned_key": meta["canned_key"],
        "description": trigger.description,
        "is_system": False,
        "item_of_interest": None,
    }


def _resolve_system_trigger(trigger: dict[str, Any]) -> dict[str, Any]:
    meta = _SYSTEM_KIND_META[trigger["kind"]]
    return {
        "kind": trigger["kind"],
        "reason": meta["reason"],
        "event_type": meta["event_type"],
        "canned_key": None,  # system alerts contribute NO canned text (silent, M7 step 3)
        "description": trigger.get("description", ""),
        "is_system": True,
        "item_of_interest": None,
    }


def _contact_complete(state: dict) -> bool:
    # Gate check computed from state fields — never the contact_status column (M7 step 3).
    return bool(state.get("customer_name")) and bool(state.get("customer_email") or state.get("customer_phone"))


def _missing_piece(state: dict) -> str | None:
    if not state.get("customer_name"):
        return "name"
    if not (state.get("customer_email") or state.get("customer_phone")):
        return "contact_method"
    return None


def _missing_label(missing: str | None) -> str:
    return "name" if missing == "name" else "email or phone"


def _emit_event(state: dict, outcome: dict, resolved: dict[str, Any]) -> None:
    """Build one gate-approved email. Persistence-on: queued as a chatbot_outbox row
    by the route inside the durable transaction. Persistence-off: sent immediately."""
    idx = len(outcome["outbox_events"])
    subject = email_sender.render_subject(
        reason=resolved["reason"], name=state.get("customer_name"), session_id=state.get("session_id", "")
    )
    body = email_sender.render_email_body(
        name=state.get("customer_name"),
        email=state.get("customer_email"),
        phone=state.get("customer_phone"),
        reason=resolved["reason"],
        description=resolved["description"],
    )
    event = {
        "event_key": f"{resolved['kind']}:{idx}",
        "event_type": resolved["event_type"],
        "payload": {"subject": subject, "body": body},
        "reason": resolved["reason"],
        "is_system": resolved["is_system"],
    }
    if resolved.get("item_of_interest"):
        event["item_of_interest"] = resolved["item_of_interest"]
        outcome["lead_item_of_interest"] = resolved["item_of_interest"]
    outcome["outbox_events"].append(event)
    outcome["emails_sent"].append(resolved["reason"])

    logger.info(
        "TOOL email: session=%s reason=%r kind=%s system=%s subject=%r %s",
        state.get("session_id"), resolved["reason"], resolved["kind"], resolved["is_system"], subject,
        "(queued to outbox)" if conversation_store.persistence_enabled() else "(sending directly)",
    )

    if not conversation_store.persistence_enabled():
        sent = email_sender.send_email(subject, body)
        logger.info(
            "TOOL email: session=%s reason=%r %s", state.get("session_id"), resolved["reason"],
            "sent" if sent else "FAILED (see email_sender log for the cause)",
        )
        # Lead upgrade is durable-only; a no-op when persistence is off.
        conversation_store.promote_lead_to_hard(state.get("session_id", ""))


def email_actions_node(state: dict) -> dict:
    """Process every email trigger (customer + system + stashed) under the contact gate.

    Runs on BOTH graph passes (before and after search/lookup). It consumes the
    triggers it processes so the second pass only handles newly-generated results
    alerts (milestone.md M4 email_actions² / M7 step 3 idempotency).
    """
    outcome = state.setdefault("turn_outcome", {})
    outcome.setdefault("canned_keys", [])
    outcome.setdefault("emails_sent", [])
    outcome.setdefault("outbox_events", [])
    turn = state.get("turn")

    # 1. Resolve this pass's new triggers, then consume the source lists.
    resolved_new: list[dict[str, Any]] = []
    if turn is not None:
        for trigger in turn.email_triggers:
            resolved_new.append(_resolve_customer_trigger(state, trigger))
        state["turn"] = turn.model_copy(update={"email_triggers": []})
    for sys_trigger in outcome.get("system_email_triggers", []) or []:
        resolved_new.append(_resolve_system_trigger(sys_trigger))
    outcome["system_email_triggers"] = []

    # 2. FAQ / non-FAQ canned answers are delivered immediately, gate or not
    #    (the user gets their answer; only the email waits). System alerts add none.
    for resolved in resolved_new:
        key = resolved.get("canned_key")
        if key and key not in outcome["canned_keys"]:
            outcome["canned_keys"].append(key)

    stashed: list[dict[str, Any]] = list(state.get("pending_email_actions") or [])

    # 3. Declined → drop the whole batch silently, never ask again (spec §Contact-Info Gate).
    if state.get("contact_declined"):
        state["pending_email_actions"] = []
        state["contact_followup_pending"] = None
        outcome["email_status"] = "skipped (user declined)"
        return state

    # 4. Contact complete → send the whole batch (stashed + new); else stash + ask once.
    if _contact_complete(state):
        to_process = stashed + resolved_new
        state["pending_email_actions"] = []
        state["contact_followup_pending"] = None
        for resolved in to_process:
            _emit_event(state, outcome, resolved)
        customer_reasons = [r["reason"] for r in to_process if not r["is_system"]]
        if customer_reasons:
            outcome["email_status"] = f"sent: {customer_reasons}"
    else:
        state["pending_email_actions"] = stashed + resolved_new
        has_customer = any(not r["is_system"] for r in resolved_new)
        missing = _missing_piece(state)
        # Only customer-initiated triggers ask for contact; system alerts wait silently.
        if has_customer and missing:
            state["contact_followup_pending"] = missing
            outcome["email_status"] = f"deferred — ask once for the missing contact piece(s): {_missing_label(missing)}"

    return state
