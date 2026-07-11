from __future__ import annotations

from datetime import datetime, timezone

from src.llm.client import LLMClient
from src.llm.respond import respond_with_all_listings


def make_respond_node(client: LLMClient):
    def respond_node(state: dict) -> dict:
        outcome = state.setdefault("turn_outcome", {})
        # First-turn contact invite — deferred when an inventory lookup fired this
        # turn (keyed off the durable inventory_lookup_ran signal, not the stub).
        if not state.get("contact_prompted_initial") and not outcome.get("inventory_lookup_ran"):
            outcome["contact_ask"] = True
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
