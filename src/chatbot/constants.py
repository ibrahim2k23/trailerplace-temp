"""Shared tuning constants and context-compaction helpers.

Kept dependency-free so any module (graph, service, tools) can import it without
risking a circular import. Centralizes the conversation-window sizing that was
previously duplicated as ad-hoc slice literals across many call sites.
"""

from __future__ import annotations

from typing import Any

# How many of the most recent conversation turns to include in an LLM context
# window. Full history is always persisted; only what is *shown to a model* is
# windowed.
RECENT_WINDOW = 8

# Max characters kept per message in that window. Long assistant listing blocks
# otherwise dominate the context JSON and crowd out the system instructions,
# degrading instruction adherence on the hot path.
RECENT_CHAR_CAP = 500

# Categories for which the dynamic "how wide is the item?" question is never
# added. Single source of truth so the haul-classifier prompt, its deterministic
# fallback, and the graph gate cannot drift apart (they previously disagreed:
# the prompt listed only 3 while the code gate excluded 6).
DYNAMIC_WIDTH_EXCLUDED_CATEGORIES = frozenset(
    {"utility", "enclosed", "livestock", "aluminum", "flatbed", "dump"}
)


def compact_recent_messages(
    messages: list[dict[str, Any]] | None,
    *,
    window: int = RECENT_WINDOW,
    char_cap: int = RECENT_CHAR_CAP,
) -> list[dict[str, Any]]:
    """Return the last ``window`` messages with each content capped to ``char_cap``."""
    compact: list[dict[str, Any]] = []
    for item in (messages or [])[-window:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        if len(content) > char_cap:
            content = content[:char_cap] + "…"
        compact.append({"role": item.get("role"), "content": content})
    return compact


def compact_listings(listings: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Project listings to just the fields a planner needs to reason about them.

    Sending full listing objects into the planner distracts it (it keys off stale
    listings after a topic change) and inflates latency/cost. Title + url + price
    plus a 1-based index is enough for the planner to resolve references; the
    deterministic search/format path owns the full objects.
    """
    compact: list[dict[str, Any]] = []
    for index, listing in enumerate(listings or [], 1):
        if not isinstance(listing, dict):
            continue
        compact.append(
            {
                "index": index,
                "title": listing.get("title"),
                "url": listing.get("url"),
                "price": listing.get("price_display") or listing.get("price"),
            }
        )
    return compact
