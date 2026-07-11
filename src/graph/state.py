from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, TypedDict

from src.llm.schemas import TurnAnalysis

STATE_SCHEMA_VERSION = 1


class SessionState(TypedDict, total=False):
    session_id: str
    lead_id: str | None
    customer_name: str | None
    customer_email: str | None
    customer_phone: str | None
    contact_prompted_initial: bool
    contact_followup_pending: str | None
    contact_declined: bool
    pending_email_actions: list[dict]
    category: str | None
    clarification_key: str | None
    slots: dict[str, Any]
    slot_sources: dict[str, str]
    skipped_slots: list[str]
    non_metadata_features: list[str]
    brand_preference: str | None
    pending_question_slot: str | None
    pending_question_repeats: int
    qualification_complete: bool
    pending_category_change: dict | None
    pending_category_suggestion: dict | None
    shown_listings: list[dict]
    shown_urls: list[str]
    last_search_filters: dict | None
    injected_required_slots: list[str]
    turn: TurnAnalysis | None
    turn_outcome: dict
    messages: list[dict]


_sessions: dict[str, SessionState] = {}

# Serializes concurrent turns for one session in-process (M8 §4). When persistence is
# on, durable_turn's pg_advisory_xact_lock already does this across processes; this
# lock is what gives the persistence-off path the same guarantee.
_session_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_session_locks_guard = threading.Lock()


def session_lock(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        return _session_locks[session_id]


def new_session_state(session_id: str) -> SessionState:
    return {
        "session_id": session_id,
        "lead_id": None,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_prompted_initial": False,
        "contact_followup_pending": None,
        "contact_declined": False,
        "pending_email_actions": [],
        "category": None,
        "clarification_key": None,
        "slots": {},
        "slot_sources": {},
        "skipped_slots": [],
        "non_metadata_features": [],
        "brand_preference": None,
        "pending_question_slot": None,
        "pending_question_repeats": 0,
        "qualification_complete": False,
        "pending_category_change": None,
        "pending_category_suggestion": None,
        "shown_listings": [],
        "shown_urls": [],
        "last_search_filters": None,
        "injected_required_slots": [],
        "turn": None,
        "turn_outcome": {},
        "messages": [],
    }


def _get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = new_session_state(session_id)
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def to_snapshot(state: SessionState) -> dict[str, Any]:
    snapshot = {
        "state_schema_version": STATE_SCHEMA_VERSION,
        **{key: value for key, value in dict(state).items() if key not in {"turn", "turn_outcome"}},
    }
    snapshot["customer_full_name"] = state.get("customer_name")
    snapshot["contact_status"] = "contact_available" if (state.get("customer_email") or state.get("customer_phone")) else "missing_contact"
    return snapshot


def from_snapshot(snapshot: dict[str, Any]) -> SessionState:
    state = new_session_state(str(snapshot.get("session_id") or ""))
    for key, value in snapshot.items():
        if key not in {"state_schema_version", "turn", "turn_outcome"}:
            state[key] = value
    state["turn"] = None
    state["turn_outcome"] = {}
    return state
