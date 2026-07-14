from __future__ import annotations

from datetime import datetime, timezone

from src.graph.contact_gate import contact_gate_pending, missing_contact_pieces
from src.llm.client import LLMClient
from src.llm.respond import respond_with_all_listings

RESULTS_SHOWN_REASON = "Results Shown to User"


def _url_key(url) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _record_shown_listings(state: dict, outcome: dict, reply) -> None:
    """A listing counts as SHOWN only once it is actually in the reply we send.

    Search used to bank its results as shown the moment it found them. When respond then
    dropped them (or the turn was never about listings at all), we had told the team we
    presented trailers the customer never saw, and permanently excluded those trailers from
    the next "show me more".
    """
    listings = outcome.get("listings") or []
    if not listings:
        return
    cited = {_url_key(url) for url in (reply.cited_listing_urls or [])}
    shown = [item for item in listings if _url_key(item.get("url")) in cited]
    already = set(state.get("shown_urls") or [])
    fresh = [item for item in shown if item.get("url") and item["url"] not in already]
    state.setdefault("shown_listings", []).extend(fresh)
    state["shown_urls"] = sorted(already | {item["url"] for item in fresh})
    if shown:
        # What "the 5th one" refers to: the batch currently on their screen, not everything we
        # have ever sent. Numbered against the cumulative list, a reference to the second batch
        # silently resolved to a trailer from the first.
        state["last_shown_listings"] = shown
    if not shown:
        # Nothing reached the customer, so there is nothing to tell the team about.
        outcome["outbox_events"] = [
            event for event in outcome.get("outbox_events", []) or [] if event.get("event_type") != "results_shown"
        ]
        outcome["emails_sent"] = [
            reason for reason in outcome.get("emails_sent", []) or [] if reason != RESULTS_SHOWN_REASON
        ]


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
        _record_shown_listings(state, outcome, reply)
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
