from __future__ import annotations

from typing import Any

# The opening contact gate.
#
# Before we start qualifying or searching, we ask ONCE for the customer's name and a way to
# reach them (email or phone). Their actual request is not lost — analyze/apply still record
# the category and any features they mentioned — it just waits one turn.
#
# The gate is deliberately cheap to escape. It closes for good as soon as any of these is
# true, and we never ask again:
#   - we have a name AND an email or phone,
#   - they declined,
#   - they ignored the ask (replied with anything that carried no contact detail).
# If they give us only part of it (just a name, just a phone), we ask once more for the
# missing half — never a third time.

MAX_CONTACT_ASKS = 2


def contact_complete(state: dict[str, Any]) -> bool:
    return bool(state.get("customer_name")) and bool(state.get("customer_email") or state.get("customer_phone"))


def missing_contact_pieces(state: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not state.get("customer_name"):
        missing.append("name")
    if not (state.get("customer_email") or state.get("customer_phone")):
        missing.append("email or phone")
    return missing


def contact_gate_pending(state: dict[str, Any]) -> bool:
    """True while the opening contact ask still owes the customer a turn."""
    if state.get("contact_gate_closed"):
        return False
    if state.get("contact_declined") or contact_complete(state):
        return False
    return True


def update_contact_gate(state: dict[str, Any], gave_contact_this_turn: bool) -> None:
    """Decide, after this turn's message, whether the gate stays open."""
    if state.get("contact_gate_closed"):
        return
    if state.get("contact_declined") or contact_complete(state):
        state["contact_gate_closed"] = True
        return
    asks = int(state.get("contact_asks", 0) or 0)
    if asks == 0:
        return  # we have not asked yet
    if not gave_contact_this_turn:
        # They answered something else, or nothing at all. That is a no — drop it.
        state["contact_gate_closed"] = True
    elif asks >= MAX_CONTACT_ASKS:
        # They gave us part of it and we have already asked for the rest. Stop there.
        state["contact_gate_closed"] = True


def contact_ask_outstanding(state: dict[str, Any]) -> bool:
    """True while we are waiting on a contact ask we made — the opening gate, or the
    follow-up an email trigger needs. A qualification question left unanswered during one of
    OUR detours must not be charged the full two strikes for it."""
    return bool(state.get("contact_followup_pending")) or contact_gate_pending(state)
