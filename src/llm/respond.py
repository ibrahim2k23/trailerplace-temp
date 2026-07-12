from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.domain.canned_responses import CANNED_RESPONSES
from src.domain.categories import advertised_categories_line, category_reference_block
from src.llm.analyze import MAX_CONTEXT_TURNS, _recent_messages, _state_get
from src.llm.client import LLMClient
from src.llm.schemas import ReplyOutput, TurnAnalysis

logger = logging.getLogger(__name__)


def _outcome_get(turn_outcome: Any, key: str, default: Any = None) -> Any:
    return turn_outcome.get(key, default) if isinstance(turn_outcome, dict) else getattr(turn_outcome, key, default)


def _listing_get(listing: Any, key: str, default: Any = "") -> Any:
    return listing.get(key, default) if isinstance(listing, dict) else getattr(listing, key, default)


def _listing_line(idx: int, listing: Any) -> str:
    def get(key: str, default: str = "") -> Any:
        return _listing_get(listing, key, default)

    return f"{idx}. {get('title')} - {get('price_display') or get('price')} - {get('length')} x {get('width')} - {get('hitch_type')} - {get('make')} - {get('url')}"


def _url_key(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _missing_listings(listings: list[Any], cited_urls: list[str] | None) -> list[Any]:
    """Listings the reply failed to cite — i.e. ones it silently dropped."""
    cited = {_url_key(url) for url in (cited_urls or [])}
    return [
        listing
        for listing in listings
        if _url_key(_listing_get(listing, "url")) and _url_key(_listing_get(listing, "url")) not in cited
    ]


def _contact_gate_lines(state: Any, turn_outcome: Any) -> list[str]:
    """The opening contact ask OWNS the turn — it is the only thing in the reply.

    Returned on its own (the caller uses these lines and nothing else), because a reply that
    asks for their email AND fires off the qualification question gets one of the two
    answered and the other silently dropped.
    """
    missing = _outcome_get(turn_outcome, "contact_gate_missing", []) or []
    wanted = " and ".join(missing) or "name and email or phone"
    first_ask = int(_state_get(state, "contact_asks", 0) or 0) <= 1
    lines = [
        f"- ASK ONLY FOR CONTACT DETAILS THIS TURN: their {wanted}. This is the whole reply.",
        "- Greet them warmly, say ONE short line showing you heard what they're after (their trailer type, "
        "size, cargo - whatever they told us), promise you'll get right to it, and then ask for it. "
        "Make clear it's optional and it's so the team can follow up.",
        "- Do NOT ask any qualification question, do NOT present or mention listings, do NOT mention stock or "
        "availability, and do NOT answer anything else. Their request is safely recorded - it is handled next turn.",
    ]
    if not first_ask:
        lines.append(
            f"- They already gave us part of it, so this is the last time we ask: request only their {wanted}, "
            "briefly and without pressure. If they skip it, we drop the subject for good."
        )
    return lines


def _decision_lines(state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> list[str]:
    if _outcome_get(turn_outcome, "contact_gate_missing"):
        return _contact_gate_lines(state, turn_outcome)
    lines: list[str] = []
    if _outcome_get(turn_outcome, "clarification_question"):
        lines.append(f'- Clarification question to ask: "{_outcome_get(turn_outcome, "clarification_question")}"')
    if _outcome_get(turn_outcome, "next_question"):
        lines.append(f'- Next qualification question to ask: "{_outcome_get(turn_outcome, "next_question")}"')
        lines.append(
            "  Your reply MUST end by asking exactly this question, once. It is the only question in the reply."
        )
    changed_to = _outcome_get(turn_outcome, "category_just_changed")
    if changed_to and _outcome_get(turn_outcome, "next_question"):
        lines.append(
            f"- The category just changed to {changed_to}, so we are starting its questions from the top and we "
            f"know NOTHING about their {changed_to} needs yet. Confirm the switch in ONE short sentence, then ask "
            "the question above and STOP. No listings, no summary of what you have on file, no talk of stock or "
            "availability, no second question, no offer to show options - just the switch and the question."
        )
    if _outcome_get(turn_outcome, "next_question") and not _state_get(state, "category"):
        lines.append(
            "- No trailer category chosen yet: briefly acknowledge anything they told us (name, size, etc.), then simply ask "
            "what TYPE of trailer they're looking for. Mention a handful of example categories from our lineup "
            f"({advertised_categories_line()}) and add that we carry many more. "
            "Do NOT mention listings, stock, or availability, and do NOT say we do or don't have something - no search has run yet."
        )
    if int(_state_get(state, "pending_question_repeats", 0) or 0) == 1:
        lines.append("- The user did not answer it last time - acknowledge their message first, then re-ask casually, once.")
    suggestion = _state_get(state, "pending_category_suggestion")
    if suggestion and isinstance(suggestion, dict):
        suggested = suggestion.get("suggested_category")
        from_category = suggestion.get("from_category")
        cargo = suggestion.get("cargo") or "what they want to haul"
        lines.append(
            f"- Category switch SUGGESTION (ask, do not assume): they are on {from_category}, but they mentioned "
            f'"{cargo}", and our {suggested} trailers are the ones best suited to haul that. '
            f"Briefly say WHY {suggested} suits that load, then ask a clear yes/no: switch to {suggested}, "
            f"or stay with {from_category}? Do NOT show listings and do NOT ask any other question this turn."
        )
    pending_change = _state_get(state, "pending_category_change")
    if pending_change:
        dims = pending_change.get("dimensions", {}) if isinstance(pending_change, dict) else {}
        new_cat = pending_change.get("new_category", "the new category") if isinstance(pending_change, dict) else "the new category"
        dim_labels = {"length": "length", "width": "width", "payload": "payload capacity"}
        offered = ", ".join(f"{dim_labels.get(name, name)} ({value})" for name, value in dims.items()) or "none"
        lines.append(
            f"- Category change to {new_cat}: we're dropping all previous preferences except these measurements. "
            f"Confirm which to carry over (they can keep all, drop some, or change a value): {offered}. "
            "Do NOT show listings this turn; just ask."
        )
    if _outcome_get(turn_outcome, "search_ran"):
        count = _outcome_get(turn_outcome, "result_count", 0)
        lines.append(
            f"- Search ran and returned {count} listing(s), already filtered and ranked for this customer. "
            f"Your reply MUST contain exactly {count} numbered listings (1 to {count}) and cited_listing_urls MUST "
            f"contain exactly {count} URLs - the same ones, in the same order as the LISTINGS block below. "
            "Do NOT drop, add, reorder, or judge whether a listing's size, style or features 'fit' - that ranking "
            "already happened and it is not your job. A listing that looks like a different sub-style than the others "
            "(a hay trailer among cattle trailers, a shorter one, a pricier one) STILL gets shown. "
            "A listing missing an optional field (width, payload, hitch) is normal: show the fields it has and skip the "
            "missing lines - never omit the listing itself. Count them before you finish."
        )
    if _outcome_get(turn_outcome, "inventory_match_status"):
        lines.append(f"- Inventory lookup result: {_outcome_get(turn_outcome, 'inventory_match_status')}.")
        lines.append("  exact -> present the match(es) warmly, then ask whether they're interested in any models shown.")
        lines.append("  no_exact -> say we do not currently show the requested exact trailer, then present closest alternatives; never invent specs.")
        lines.append("  ambiguous -> ask which model they mean, naming the candidates; do not state prices yet.")
    if _outcome_get(turn_outcome, "contact_invite_suppressed"):
        lines.append("- Contact invite suppressed this turn (inventory lookup fired) - do NOT ask for name/email/phone in this reply.")
    canned_keys = _outcome_get(turn_outcome, "canned_keys", []) or []
    if canned_keys:
        canned = [CANNED_RESPONSES[key] for key in canned_keys if key in CANNED_RESPONSES]
        lines.append(f"- Canned text(s) - EACH must appear verbatim or near-verbatim: {canned}")
    if _outcome_get(turn_outcome, "email_status"):
        lines.append(f"- Email status: {_outcome_get(turn_outcome, 'email_status')}.")
    if analysis.user_question_to_answer:
        lines.append(f'- Interruption to answer first: "{analysis.user_question_to_answer}"')
        if _outcome_get(turn_outcome, "next_question"):
            lines.append(
                "  Answer it in 1-2 sentences, then re-ask the pending qualification question in the SAME reply. "
                "Never end the turn without re-asking it - an unanswered question they were never re-asked is a "
                "question we lose."
            )
    return lines


def _reference_block(state: Any, listings: list[Any]) -> str:
    """Listings already presented on earlier turns, for answering questions about them.

    Kept strictly apart from this turn's results: a turn with no search has NOTHING to
    present, and feeding it the back catalogue as "listings available" is what made it
    re-list old inventory under a new category heading.
    """
    shown = _state_get(state, "shown_listings", []) or []
    this_turn = {_url_key(_listing_get(item, "url")) for item in listings}
    lines = [
        _listing_line(idx, item)
        for idx, item in enumerate(shown, 1)
        if _url_key(_listing_get(item, "url")) not in this_turn
    ]
    return "\n".join(lines) or "none"


def build_respond_prompt(
    state: Any, analysis: TurnAnalysis, turn_outcome: Any, repair_note: str | None = None
) -> tuple[str, list[dict]]:
    listings = _outcome_get(turn_outcome, "listings", None) or []
    # An empty results block means "no search ran", NOT "we have nothing in stock" — say so
    # in words. Rendered as a count of 0 it reads like an out-of-stock report, and the model
    # duly told a customer mid-qualification that we had no utility trailers (we have plenty;
    # we simply had not looked yet).
    if listings:
        listing_header = f"LISTINGS TO PRESENT THIS TURN ({len(listings)} - EVERY ONE MUST APPEAR IN YOUR REPLY, AND NOTHING ELSE MAY)"
        listing_block = "\n".join(_listing_line(idx, listing) for idx, listing in enumerate(listings, 1))
    else:
        listing_header = "LISTINGS TO PRESENT THIS TURN (NONE - NO SEARCH RAN)"
        listing_block = (
            "No inventory search ran this turn, so you have NO results and know NOTHING about what is or "
            "is not in stock. This is NOT an out-of-stock signal. Present no listings, and do not mention "
            "inventory, availability, or stock at all - not even to say you have none to show. Simply reply "
            "and ask the pending question above."
        )
    reference_block = _reference_block(state, listings)
    collected = _state_get(state, "slots", None) or _state_get(state, "collected_slots", {}) or {}
    customer_name = _state_get(state, "customer_name")
    decision_lines = "\n".join(_decision_lines(state, analysis, turn_outcome)) or "- No special turn outcome lines."
    repair_block = f"\n=== CORRECTION - YOUR PREVIOUS DRAFT WAS REJECTED ===\n{repair_note}\n" if repair_note else ""

    system = f"""You are the TrailerPlace sales assistant - a friendly trailer lead specialist for a dealership
in Wharton, TX (979-532-1486, https://trailerplace.com; financing and delivery available).
Write the next assistant reply. Warm, concise, conversational; never pushy or repetitive.
{repair_block}
=== WHAT THE SYSTEM ALREADY DECIDED THIS TURN (do not contradict) ===
{decision_lines}

=== RULES ===
- Answer the user's question FIRST, then re-ask the pending qualification question once, in the same reply. Answering without re-asking loses the question.
- When a "Next qualification question" is given above, the reply ends with it. Ask that one question and no other.
- Never re-ask anything already collected, skipped, or marked no-preference.
- When presenting listings: show EVERY listing in the block below, in the exact order given - never omit, add, or reorder any, and never filter by how well a size or feature matches. For EVERY listing shown, put its exact URL in cited_listing_urls.
- Optional fields (width, payload, hitch, height) are missing on many trailers - that is expected. Show the fields that are present and skip the missing lines; a missing field is NEVER a reason to drop a listing.
- Present listings ONLY from the "LISTINGS TO PRESENT THIS TURN" block. When it says NO SEARCH RAN, we have not looked yet - that says NOTHING about our stock. Present no listing cards, do not re-list anything from the reference block, and NEVER say we have nothing / no listings / none available for a category. Saying "I don't have any listings to show you for utility trailers" is WRONG and forbidden: we almost certainly have them, we just haven't searched. Answer the customer and ask the pending question instead.
- The reference block is memory, not inventory to show: use it only to answer a question about a listing the customer refers back to ("the 81382", "the second one"), quoting its real fields and URL. Never re-list it, renumber it, or restate it under a different category.
- Never invent inventory, prices, or policies. Store facts: Wharton TX, 979-532-1486, financing available, delivery available, {settings.trailerplace_website or "https://trailerplace.com"}.
- Gooseneck and Bumper Pull are HITCH TYPES, not categories and (unless the customer says "the Gooseneck brand") not makes. Quote a listing's hitch from its own data; never assume one.
- Sizes are in feet and weights in pounds. Quote back the number we recorded, never a vaguer phrase than they gave.
- We carry: {advertised_categories_line()}.
- 2-6 sentences unless presenting listings.

=== OUR CATEGORIES AND WHAT EACH IS BEST FOR ===
Use this to answer "which trailer suits X?" and to explain why a suggested switch makes sense.
Never name a category outside this list. Gooseneck and Bumper Pull are HITCH TYPES, not categories.
{category_reference_block()}
If the customer only ASKED which trailer suits a job, answer the question - do not assume they
have chosen that category and do not start qualifying them for it.

=== {listing_header} ===
{listing_block}

=== ALREADY SHOWN ON EARLIER TURNS (REFERENCE ONLY - NEVER RE-LIST THESE) ===
{reference_block}

When showing listings, use this structure (repeat for EVERY listing in the block, numbered in order):
1. [Trailer title hyperlinked to exact URL]
   - Category: [category]
   - Make: [make](show if present)
   - Price: [price] (show if present)
   - Length: [length](show if present)
   - Width: [width](show if present)
   - Payload: [payload](show if present)
   - Hitch type: [hitch_type](show if present)
   

Then ask whether the customer is interested.

=== CONTEXT ===
Category: {_state_get(state, "category") or _state_get(state, "selected_category")}; collected: {collected}; customer: {customer_name or "unknown"} (use their first name naturally when known).
Latest analysis intent: {analysis.intent}; user question to answer: {analysis.user_question_to_answer}
"""
    return system, _recent_messages(state, MAX_CONTEXT_TURNS)


def respond_turn(
    client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any, repair_note: str | None = None
) -> ReplyOutput:
    system, messages = build_respond_prompt(state, analysis, turn_outcome, repair_note)
    return client.structured(system=system, messages=messages, schema=ReplyOutput)


def respond_with_all_listings(client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> ReplyOutput:
    """Reply, and if the model silently dropped listings from the block, make it try again.

    The prompt tells it to show every listing, but a small model still self-filters on
    perceived fit (dropping a hay trailer from a cattle-trailer list). Ranking is the
    reranker's job, so a short reply is a defect, not a judgment call: name the listings
    it left out and re-ask once. Keep whichever draft dropped fewer.
    """
    reply = respond_turn(client, state, analysis, turn_outcome)
    listings = _outcome_get(turn_outcome, "listings", None) or []
    if not listings:
        return reply

    missing = _missing_listings(listings, reply.cited_listing_urls)
    if not missing:
        return reply

    logger.warning(
        "respond dropped %d of %d listing(s); retrying once: %s",
        len(missing), len(listings), [_listing_get(item, "url") for item in missing],
    )
    dropped = "\n".join(
        f"- {_listing_get(item, 'title')} ({_listing_get(item, 'url')})" for item in missing
    )
    note = (
        f"You left {len(missing)} of the {len(listings)} listings out of your reply:\n{dropped}\n"
        "Every listing in the LISTINGS block is a valid recommendation - it is already ranked, and it is "
        "not your job to decide one is a poor fit. Rewrite the reply with ALL "
        f"{len(listings)} listings, numbered 1 to {len(listings)} in block order, and put all "
        f"{len(listings)} URLs in cited_listing_urls."
    )
    retry = respond_turn(client, state, analysis, turn_outcome, repair_note=note)
    if len(_missing_listings(listings, retry.cited_listing_urls)) < len(missing):
        return retry
    logger.warning("respond retry did not recover the dropped listing(s); keeping the first draft")
    return reply
