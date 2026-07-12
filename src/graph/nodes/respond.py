from __future__ import annotations

from datetime import datetime, timezone

from src.graph.contact_gate import contact_gate_pending, missing_contact_pieces
from src.llm.client import LLMClient
from src.llm.respond import respond_with_all_listings


def make_respond_node(client: LLMClient):
    def respond_node(state: dict) -> dict:
        outcome = state.setdefault("turn_outcome", {})
        # The opening contact ask — deferred when an inventory lookup fired this turn
        # (keyed off the durable inventory_lookup_ran signal, not the stub).
        if contact_gate_pending(state) and not outcome.get("inventory_lookup_ran"):
            outcome["contact_ask"] = True
            outcome["contact_gate_missing"] = missing_contact_pieces(state)
            state["contact_asks"] = int(state.get("contact_asks", 0) or 0) + 1
            state["contact_prompted_initial"] = True
        reply = respond_with_all_listings(client, state, state["turn"], outcome)
        state.setdefault("messages", []).append(
            {
                "role": "assistant",
                "content": reply.assistant_text,
                "listings": outcome.get("listings"),
                "user_feedback": None,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        outcome["reply"] = reply
        return state

    return respond_node
