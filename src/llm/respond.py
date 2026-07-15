from __future__ import annotations

import logging
import re
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


# The fields a listing card can carry, in the order they are shown. Every one of them is missing
# on some trailer in the catalogue, so none of them is guaranteed.
_LISTING_FIELDS: tuple[tuple[str, str], ...] = (
    ("Make", "make"),
    ("Price", "price_display"),
    ("Length", "length"),
    ("Width", "width"),
    ("Payload", "payload_capacity"),
    ("Hitch type", "hitch_type"),
)


def _listing_line(idx: int, listing: Any) -> str:
    """One listing, with the fields it does NOT have left out entirely.

    A missing field used to be rendered as the literal "None" ("Length: None", "Make: None"), and
    the model dutifully copied that onto the card - a trailer whose length we simply do not have on
    file was advertised to the customer as having a length of None. It cannot omit what it is never
    shown, so the omission happens here.
    """
    title = str(_listing_get(listing, "title") or "").strip()
    price = _listing_get(listing, "price_display") or _listing_get(listing, "price")
    parts = [f"{idx}. TITLE: {title}"]
    for label, key in _LISTING_FIELDS:
        value = price if key == "price_display" else _listing_get(listing, key)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value if str(item or "").strip())
        text = str(value or "").strip()
        if text and text.lower() not in {"none", "null", "n/a"}:
            parts.append(f"{label}: {text}")
    parts.append(f"URL: {str(_listing_get(listing, 'url') or '').strip()}")
    return " | ".join(parts)


def _url_key(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _is_first_reply(state: Any) -> bool:
    """True on the very first reply of the conversation - nothing has been said back to them yet.

    Decided here rather than left to the model: the opening line is a fixed phrase the business
    wants word for word, and "have I spoken to them yet?" is a fact we already hold.
    """
    messages = _state_get(state, "messages", []) or []
    return not any(
        (message.get("role") if isinstance(message, dict) else getattr(message, "role", None)) == "assistant"
        for message in messages
    )


def _missing_listings(listings: list[Any], reply: Any) -> list[Any]:
    """Listings the customer will not actually see in this reply.

    A listing counts as shown only when its URL appears in the REPLY TEXT. cited_listing_urls is
    the model's own report of what it presented, and it lies: seen live, a reply opened with "Here
    are some trailers that match your requirements:", stopped dead, and still handed back all six
    URLs in cited_listing_urls. Checking the self-report against itself found nothing missing, so
    no retry fired, and the six trailers the customer never saw were recorded as shown - locking
    them out of "show me more" for the rest of the conversation.
    """
    # _url_key drops the trailing slash, so it stays a substring of the linked URL in the text.
    text = str(_outcome_get(reply, "assistant_text", "") or "").lower()
    cited = {_url_key(url) for url in (_outcome_get(reply, "cited_listing_urls", None) or [])}
    missing: list[Any] = []
    for listing in listings:
        url = _url_key(_listing_get(listing, "url"))
        if url and not (url in cited and url in text):
            missing.append(listing)
    return missing


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


def _referenced_listing(state: Any, analysis: TurnAnalysis) -> Any | None:
    """The exact listing the customer just pointed at, resolved here rather than by the model.

    Respond was never given listing_reference at all: it saw only the reference block and had to
    guess which trailer "the 5th one" meant. It guessed wrong, quoting a trailer the customer had
    not picked. The index counts into the batch on screen — the same list Analyze numbered.
    """
    ref = analysis.listing_reference
    shown = _state_get(state, "last_shown_listings", None) or _state_get(state, "shown_listings", []) or []
    if not ref or not 1 <= ref <= len(shown):
        return None
    return shown[ref - 1]


# The three ways a customer asks us to pick a type FOR them: an outright request ("what do you
# recommend?", "I can't decide"), a description of the trailer with no type named, or a question
# about which type suits a job.
_RECOMMENDATION_INTENTS = frozenset({
    "recommendation_request",
    "feature_request_no_category",
    "category_exploration",
})


def _contact_only_turn(analysis: TurnAnalysis) -> bool:
    """The message handed over contact details and said nothing about trailers.

    Judged from what the extractor actually pulled out, not from the intent label: the analyzer
    sometimes labels a bare "I'm John, john@x.com" as a recommendation-ish intent, and that label
    alone was enough to trigger a category list at someone who never mentioned a trailer.
    """
    extracted = analysis.extracted
    gave_contact = bool(analysis.contact.name or analysis.contact.email or analysis.contact.phone)
    said_anything_else = bool(
        extracted.haul_item
        or extracted.non_metadata_features
        or extracted.brand_preference
        or extracted.hitch_type
        or extracted.trailer_length_ft is not None
        or extracted.trailer_width_ft is not None
        or extracted.trailer_height_ft is not None
        or extracted.payload_lbs is not None
        or analysis.slot_answers
        or analysis.user_question_to_answer
    )
    return gave_contact and not said_anything_else


def _has_recommendation_basis(state: Any, analysis: TurnAnalysis) -> bool:
    """Is there any reason to put a list of trailer TYPES in front of them?

    Only two: they asked for one (or said they want a trailer without naming a type), or they have
    told us something about the job - cargo, a size, a feature - that a recommendation can be built
    from. A customer who has just handed over their name and email has told us nothing about
    trailers, and answering that with four category suggestions is recommending into thin air. Ask
    them which type they want instead.
    """
    if _contact_only_turn(analysis):
        return False
    if analysis.intent in _RECOMMENDATION_INTENTS:
        return True
    if _state_get(state, "non_metadata_features", None):
        return True
    slots = _state_get(state, "slots", None) or _state_get(state, "collected_slots", {}) or {}
    return any(value not in (None, "") for value in slots.values())


def _category_question_line(state: Any, analysis: TurnAnalysis) -> str:
    """The ONE question when no category is chosen - listed as types, or asked plainly."""
    closing = (
        "Do NOT mention listings, stock, or availability, and do NOT say we do or don't have something "
        "- no search has run yet."
    )
    if _has_recommendation_basis(state, analysis):
        return (
            "- No trailer category chosen yet, so the ONE question above is which TYPE of trailer they want - and "
            "they have given us something to go on, so RECOMMEND. Ask it using the RECOMMENDING TRAILER TYPES "
            "format below: 3-4 types from our lineup that suit what they told us, one short line each, then ask "
            "which they want to go with and note that we carry more. That list IS the question - do not also ask "
            f"it in a sentence of its own. {closing}"
        )
    return (
        "- No trailer category chosen yet, so the ONE question above is which TYPE of trailer they want. Ask it as "
        "ONE plain sentence and nothing else. They have NOT told us what they haul or what they need, and have NOT "
        "asked us to recommend, so we have nothing to base a recommendation on: do NOT list, suggest, or bullet any "
        f"trailer types this turn. {closing}"
    )


def _decision_lines(state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> list[str]:
    if _outcome_get(turn_outcome, "contact_gate_missing"):
        return _contact_gate_lines(state, turn_outcome)
    lines: list[str] = []
    if _outcome_get(turn_outcome, "clarification_question"):
        lines.append(f'- Clarification question to ask: "{_outcome_get(turn_outcome, "clarification_question")}"')
    if _outcome_get(turn_outcome, "next_question"):
        lines.append(f'- The ONE thing to find out this turn: "{_outcome_get(turn_outcome, "next_question")}"')
        lines.append(
            "  Ask for it ONCE, at the end of your reply, in your own words - the wording above is the "
            "information we need, not a script, so phrase it the way an experienced salesperson would in "
            "this conversation. Do NOT lead up to it with a paraphrase of the same question and then repeat "
            "it verbatim: the reply contains exactly one question mark, and no other question."
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
        lines.append(_category_question_line(state, analysis))
    if int(_state_get(state, "pending_question_repeats", 0) or 0) == 1:
        lines.append("- The user did not answer it last time - acknowledge their message first, then re-ask casually, once.")
    pending_brand = _state_get(state, "pending_brand_categories")
    if pending_brand and isinstance(pending_brand, dict):
        brand = pending_brand.get("brand") or "that brand"
        categories = list(pending_brand.get("categories") or [])
        if len(categories) == 1:
            lines.append(
                f"- BRAND QUESTION (this owns the reply): they asked about {brand} without naming a trailer type, "
                f"and our {brand} stock is all in ONE category: {categories[0]}. Say exactly that in one short "
                f"line, then ask a clear yes/no: would they like to go with {categories[0]}? Do NOT show or "
                "mention listings, do NOT list other categories, and ask no other question."
            )
        elif categories:
            listed = ", ".join(categories)
            lines.append(
                f"- BRAND QUESTION (this owns the reply): they asked about {brand} without naming a trailer type. "
                f"We carry {brand} in exactly these categories: {listed}. Present ONLY these as a bulleted list "
                "(bold category name, em dash, one short line on what it is best for), then ask which one they "
                "want to go with. Do NOT add other categories, do NOT show or mention listings, and ask no other "
                "question."
            )
    if _outcome_get(turn_outcome, "brand_offer_declined"):
        declined_brand = _outcome_get(turn_outcome, "brand_offer_declined")
        lines.append(
            f"- They declined the {declined_brand} category we offered, so we dropped the {declined_brand} "
            "preference. Acknowledge in one short line (no apology), then ask plainly which type of trailer "
            "they are looking for. Do not mention the brand again unless they do."
        )
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
        dim_labels = {"length": "length", "width": "width", "payload": "payload capacity", "hitch": "hitch type"}
        offered = ", ".join(
            f"{dim_labels.get(name, name)} ({', '.join(str(v) for v in value) if isinstance(value, (list, tuple)) else value})"
            for name, value in dims.items()
        ) or "none"
        lines.append(
            f"- Category change to {new_cat}: we're dropping all previous preferences except these. "
            f"Confirm which to carry over (they can keep all, drop some, or change a value): {offered}. "
            "Do NOT show listings this turn; just ask."
        )
    referenced = _referenced_listing(state, analysis)
    if referenced is not None:
        lines.append(
            f"- The listing they are pointing at is ALREADY RESOLVED for you - it is #{analysis.listing_reference} "
            f'of the batch on screen: "{_listing_get(referenced, "title")}" ({_listing_get(referenced, "url")}). '
            "Talk about THIS trailer and no other. Do not count down the list yourself, do not pick a different "
            "one, and do not quote a trailer from an earlier batch. Quote only its real fields, and put ONLY its "
            "URL in cited_listing_urls."
        )
    if _outcome_get(turn_outcome, "search_ran") and not _outcome_get(turn_outcome, "result_count", 0):
        seen_before = bool(_state_get(state, "shown_urls", None))
        reason = (
            "they have already seen every match we have for their current requirements"
            if seen_before
            else "nothing in our current stock matches their requirements"
        )
        lines.append(
            f"- SEARCH RAN AND FOUND NOTHING NEW: {reason}. Say exactly that in one honest, friendly sentence - "
            "do NOT repeat, re-list, or link any trailer already shown, and do NOT invent listings. "
            "Offer to adjust their requirements (a different size, hitch, or feature) to open up more options, "
            "and close with the website/phone line: \"Feel free to check out our website for more info, or give "
            "our sales team a call at 979-532-1486 - they'll be happy to help.\""
        )
    if _outcome_get(turn_outcome, "search_ran") and _outcome_get(turn_outcome, "result_count", 0):
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
    if _outcome_get(turn_outcome, "filters_relaxed"):
        dropped = ", ".join(_outcome_get(turn_outcome, "relaxed_filters_dropped", []) or [])
        on = f" on {dropped}" if dropped else ""
        lines.append(
            f"- THESE ARE ALTERNATIVES, NOT EXACT MATCHES. Nothing in stock met every requirement they gave us, so "
            f"we searched their trailer category again without the constraint(s){on}, and these are the closest we "
            "have. Open with ONE short, matter-of-fact line saying we do not have an exact match on that right now "
            "and these are the nearest options - then show EVERY listing in full, exactly as normal. Never call them "
            "exact matches, never imply they meet the requirement they miss, and never apologise more than once."
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
    followup_missing = _outcome_get(turn_outcome, "contact_followup_missing", []) or []
    if followup_missing:
        wanted = " and ".join(followup_missing)
        lines.append(
            f"- They asked us to DO something (log their interest, set up a call, answer an FAQ) and we cannot "
            f"pass it to the team without their {wanted}. Do the thing they asked FIRST (answer them, confirm the "
            f"listing, give the store number), THEN ask for their {wanted} in one short sentence so the team can "
            "follow up. Ask for every missing piece together, not one per turn. If they say no, we drop it and "
            "never chase them - so ask once, warmly, and do not pressure."
        )
    if _outcome_get(turn_outcome, "email_status") == "skipped (user declined)":
        lines.append(
            "- They declined to share contact details, so nothing was sent to the team. Answer them normally and "
            "helpfully (store number, website, next steps they can take themselves). Do NOT ask for their details "
            "again this turn, and do not imply anyone will call them back."
        )
    if analysis.user_question_to_answer:
        lines.append(f'- Interruption to answer first: "{analysis.user_question_to_answer}"')
        if _outcome_get(turn_outcome, "next_question"):
            lines.append(
                "  Answer it in 1-2 sentences, then ask the pending qualification question - once, at the end of "
                "the SAME reply. Never end the turn without asking it: an unanswered question they were never "
                "asked again is a question we lose."
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
    elif _outcome_get(turn_outcome, "search_ran"):
        listing_header = "LISTINGS TO PRESENT THIS TURN (NONE - SEARCH RAN AND FOUND NO NEW MATCHES)"
        listing_block = (
            "The inventory search DID run this turn and found nothing new to show. Present NO listing cards "
            "and NEVER re-list a trailer from the reference block - the customer has already seen those. "
            "Follow the SEARCH RAN AND FOUND NOTHING NEW line above."
        )
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
    opening_block = (
        "\n=== OPENING LINE - THIS IS THE FIRST REPLY OF THE CONVERSATION ===\n"
        "Your reply MUST begin with this exact sentence, word for word, as its very first line:\n"
        "Thank you for contacting TrailerPlace.\n"
        "Then leave a blank line and write the rest of your reply under it.\n"
        if _is_first_reply(state)
        else ""
    )

    system = f"""You are the TrailerPlace sales assistant - an experienced trailer sales and lead specialist for a
dealership in Wharton, TX (979-532-1486, https://trailerplace.com).
Write the next assistant reply.
{repair_block}{opening_block}
=== WHO YOU ARE ===
A seasoned salesperson who knows trailers and wants to earn this sale. Professional, confident, and
helpful. You keep the customer engaged and moving forward: every reply gives them something and then
takes the next step. Clear and to the point - never pushy, never repetitive, never chatty.
BANNED: praise and filler of any kind - "Great choice", "Perfect", "Awesome", "Excellent", "That's
helpful", "Thanks for sharing", exclamation marks, emojis. You do not cheer the customer on; you find
out what they need and put the right trailer in front of them.

=== WHAT THE SYSTEM ALREADY DECIDED THIS TURN (do not contradict) ===
{decision_lines}

=== RULES ===
- Answer the user's question FIRST, then ask the pending qualification question - once, at the END of the
  same reply, in your own natural words. ONE question per reply: never ask it twice (paraphrased and then
  verbatim), never tack on a second question. The reply contains exactly one question mark.
- DO NOT REACT TO A QUALIFICATION ANSWER: no praise, no agreement, no repeating it back, no recap of what
  we have collected. It is recorded - go straight to the next question.
  ONE EXCEPTION: they cannot answer or have no preference ("no idea", "doesn't matter", "skip that") -
  give ONE short easing line ("No problem - we can keep that flexible and let the trailer decide.") and
  move on. Two sentences, no more.
  Neither rule ever shortens a reply with listings: when the LISTINGS block below has trailers, every one
  is written out in full.
- THE CUSTOMER SEES ONLY assistant_text. Every listing card, bullet, and link must be written out IN FULL
  there. cited_listing_urls is a machine field they never see - a URL only there is a trailer they never
  saw. Announcing listings and stopping ("Here are some trailers that match:") shows them NOTHING.
- Never re-ask anything already collected, skipped, or marked no-preference.
- Present listings ONLY from the "LISTINGS TO PRESENT THIS TURN" block: EVERY listing, in the exact order
  given - never omit, add, reorder, or filter by how well a size or feature fits - and put every shown
  listing's exact URL in cited_listing_urls. Optional fields (width, payload, hitch, height) are missing
  on many trailers: skip the missing lines, never the listing.
- When that block says NO SEARCH RAN, we have not looked yet - that says NOTHING about our stock. Show no
  cards, and NEVER say we have nothing / no listings / none available for a category ("I don't have any
  listings to show you for utility trailers" is WRONG and forbidden). Answer the customer and ask the
  pending question instead.
- The reference block is memory, not inventory to show: use it ONLY to answer a question about a listing
  the customer refers back to ("the 81382", "the second one"), quoting its real fields and URL. Never
  re-list it, renumber it, or restate it under a different category.
- Never invent inventory, prices, or policies. Store facts: Wharton TX, 979-532-1486, financing available, delivery available, {settings.trailerplace_website or "https://trailerplace.com"}.
- When it genuinely fits (you answered an FAQ, the conversation is wrapping up, they seem unsure, or we had
  nothing more to show), end the reply with: "Feel free to check out our website for more info, or give our
  sales team a call at 979-532-1486 - they'll be happy to help." Never append it to a reply that presents
  listings or asks a qualification question, and never use it twice in a row.
- Gooseneck and Bumper Pull are HITCH TYPES, not categories and (unless the customer says "the Gooseneck brand") not makes. Quote a listing's hitch from its own data; never assume one.
- Sizes are in feet and weights in pounds. Quote back the number we recorded, never a vaguer phrase than they gave.
- We carry: {advertised_categories_line()}.
- 2-6 sentences, unless you are presenting listings or a bulleted list - those have their own shape below.

=== OUR CATEGORIES AND WHAT EACH IS BEST FOR ===
Use this to answer "which trailer suits X?" and to explain why a suggested switch makes sense.
Never name a category outside this list. Gooseneck and Bumper Pull are HITCH TYPES, not categories.
{category_reference_block()}
If the customer only ASKED which trailer suits a job, answer the question - do not assume they
have chosen that category and do not start qualifying them for it.

=== RECOMMENDING TRAILER TYPES (STRUCTURED, NEVER A PARAGRAPH) ===
DO THIS ONLY when BOTH are true:
  (a) no category is settled ("Category" in CONTEXT below is empty, or they are moving off the one they
      had); AND
  (b) they gave us something to recommend FROM: their cargo, a feature, a job - or they asked us to
      recommend, said they cannot decide, or said they want a trailer without naming any type.
NEVER OTHERWISE. A name, an email, a phone number, a hello, or an FAQ tells us nothing about the trailer
they need - recommending off it is guessing. Just ask which type of trailer they are after, in one plain
sentence.

When you DO recommend, use 3 or 4 types from OUR CATEGORIES above, laid out like this:

Based on what you need to haul, here are the types worth looking at:

1. **Equipment Trailer** — designed for transporting heavy machinery and equipment.
2. **Dump Trailer** — great for loose materials and can handle heavy loads.
3. **Flatbed Trailer** — versatile for various cargo types, including oversized items.

Which type would you like to go with? We carry more types as well if you would like to explore.

- 3 or 4 types, never fewer, never more, only from OUR CATEGORIES above, picked to suit what they told us
  (most common ones if they told us nothing).
- Each line: bold type name, em dash, ONE short line on what it is best for. Close by asking which type
  they want, plus the note that we carry more types.
- IF THE CATEGORY IS ALREADY SETTLED and they are not asking about types, do NOT do this - they have
  chosen. Just ask the question.

=== ANY OTHER ANSWER THAT IS REALLY A LIST GETS THE SAME SHAPE ===
If the honest answer is a SET of things (use cases, hitch options, deck/gate styles, size considerations),
give it as bullets: bolded name, em dash, one short line each. Two or more items means bullets, never a
paragraph.

=== {listing_header} ===
{listing_block}

=== ALREADY SHOWN ON EARLIER TURNS (REFERENCE ONLY - NEVER RE-LIST THESE) ===
{reference_block}

When showing listings, use this structure (repeat for EVERY listing in the block, numbered in order):
1. [full TITLE, hyperlinked to its exact URL]
   - Category: [category]
   - Make: [Make, only if the block gave one]
   - Price: [Price, only if the block gave one]
   - Length: [Length, only if the block gave one]
   - Width: [Width, only if the block gave one]
   - Payload: [Payload, only if the block gave one]
   - Hitch type: [Hitch type, only if the block gave one]
   - One line description highlighting the strengths of the trailer we have shown.

COPY THE TITLE EXACTLY as it appears after "TITLE:" in the block, including the stock number on the end
("2026 Gooseneck Livestock - 91632", not "2026 Gooseneck Livestock") - the stock number is how everyone
refers to that exact trailer.

A LISTING ONLY HAS THE FIELDS ITS BLOCK LINE LISTS. If a field is not on that listing's line, DELETE THAT
BULLET ENTIRELY - never write "None", "N/A", "Not specified", "Call for price", or a blank, and never
carry a value across from a different listing. A card with three bullets is correct if the block gave
three fields.

The last bullet is a one-sentence sales pitch for THAT trailer, built ONLY from its own fields above and
what the customer told us they need - never invent a feature, spec, condition, or price. Write a
different one for each listing.

=== HOW TO END A REPLY THAT SHOWS LISTINGS ===
After the last listing, ask ONE closing question and then STOP: whether any of these interest them, or
whether they would like to see more results. For example: "Do any of these look like a fit, or would you
like to see more options?"
Nothing else goes after the listings. Do NOT bring up or offer financing, delivery, trade-ins,
warranties, a call, a visit, the store, or their contact details. Do NOT add tips, next steps, or a
second question. One closing question, then stop.

=== CONTEXT ===
Category: {_state_get(state, "category") or _state_get(state, "selected_category")}; collected: {collected}; customer: {customer_name or "unknown"} (use their first name naturally when known).
Latest analysis intent: {analysis.intent}; user question to answer: {analysis.user_question_to_answer}
"""
    return system, _recent_messages(state, MAX_CONTEXT_TURNS)


# Runs on the ASSISTANT'S OWN reply text only — never on anything the customer typed.
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def _urls_in_reply(reply: Any) -> list[str]:
    """Every URL the reply carries — in the text the customer reads and in cited_listing_urls."""
    text = str(_outcome_get(reply, "assistant_text", "") or "")
    found = [url.rstrip(".,;:") for url in _URL_RE.findall(text)]
    for url in _outcome_get(reply, "cited_listing_urls", None) or []:
        if str(url or "").strip():
            found.append(str(url))
    return found


def _foreign_reply_urls(state: Any, analysis: TurnAnalysis, listings: list[Any], reply: Any) -> list[str]:
    """Listing URLs this reply has no right to show.

    The respond model, given no fresh results, replays whole batches of old listing URLs from
    conversation history — or invents placeholder ones. Allowed this turn: the LISTINGS block, the
    one listing the customer pointed at, and the store website. Anything else is foreign, with one
    concession: a single URL from real shown history is tolerated when no fresh batch exists,
    because answering "how much was the Iron Bull one?" legitimately quotes one old listing.
    """
    allowed = {_url_key(_listing_get(item, "url")) for item in listings}
    referenced = _referenced_listing(state, analysis)
    if referenced is not None:
        allowed.add(_url_key(_listing_get(referenced, "url")))
    allowed.add(_url_key(settings.trailerplace_website or "https://trailerplace.com"))
    allowed.add(_url_key("https://trailerplace.com"))
    allowed.discard("")

    history = {
        _url_key(_listing_get(item, "url"))
        for item in (_state_get(state, "shown_listings", []) or [])
    }
    foreign: list[str] = []
    seen: set[str] = set()
    for url in _urls_in_reply(reply):
        key = _url_key(url)
        if not key or key in allowed or key in seen:
            continue
        seen.add(key)
        foreign.append(url)
    if listings:
        return foreign  # a fresh batch means NOTHING else may appear
    fabricated = [url for url in foreign if _url_key(url) not in history]
    if fabricated:
        return foreign
    # All foreign URLs are real history. One is a plausible answer about an old listing;
    # two or more is the replay pattern.
    return foreign if len(foreign) >= 2 else []


def _strip_listing_blocks(text: str, bad_urls: list[str]) -> str:
    """Remove each listing card (title line + its bullet lines) built around a bad URL.

    Operates on OUR reply text (a format our own prompt dictates), never on customer input.
    """
    bad = [url.strip() for url in bad_urls if url.strip()]
    out: list[str] = []
    skipping = False
    for line in text.split("\n"):
        if any(url in line for url in bad):
            skipping = True
            continue
        if skipping:
            stripped = line.strip()
            if stripped.startswith("-") or not stripped:
                continue
            skipping = False
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _without_foreign(reply: Any, foreign: list[str]) -> Any:
    bad_keys = {_url_key(url) for url in foreign}
    cited = [url for url in (_outcome_get(reply, "cited_listing_urls", None) or []) if _url_key(url) not in bad_keys]
    text = _strip_listing_blocks(str(_outcome_get(reply, "assistant_text", "") or ""), foreign)
    return reply.model_copy(update={"assistant_text": text, "cited_listing_urls": cited})


def respond_turn(
    client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any, repair_note: str | None = None
) -> ReplyOutput:
    system, messages = build_respond_prompt(state, analysis, turn_outcome, repair_note)
    return client.structured(system=system, messages=messages, schema=ReplyOutput)


def _repair_note(listings: list[Any], missing: list[Any], foreign: list[str]) -> str:
    parts: list[str] = []
    if missing:
        dropped = "\n".join(
            f"- {_listing_get(item, 'title')} ({_listing_get(item, 'url')})" for item in missing
        )
        parts.append(
            f"You left {len(missing)} of the {len(listings)} listings out of your reply:\n{dropped}\n"
            "Every listing in the LISTINGS block is a valid recommendation - it is already ranked, and it is "
            "not your job to decide one is a poor fit. Rewrite the reply with ALL "
            f"{len(listings)} listings, numbered 1 to {len(listings)} in block order, and put all "
            f"{len(listings)} URLs in cited_listing_urls.\n"
            "The listing cards go in assistant_text, WRITTEN OUT IN FULL, using the listing structure from the "
            "prompt. Announcing them and stopping ('Here are some trailers that match:') is not presenting them - "
            "the customer sees ONLY assistant_text, so a URL listed in cited_listing_urls but missing from "
            "assistant_text is a trailer they never saw."
        )
    if foreign:
        bad = "\n".join(f"- {url}" for url in foreign)
        parts.append(
            f"Your reply presented listing URL(s) you were NOT given this turn:\n{bad}\n"
            "These come from earlier turns or from nowhere at all - showing them again is re-listing stale "
            "inventory the customer has already seen (or inventing inventory). Rewrite the reply WITHOUT them: "
            "present ONLY what the LISTINGS block gives you (if it gives you nothing, present no listings and "
            "no URLs), and cited_listing_urls must contain only those same URLs."
        )
    return "\n\n".join(parts)


def respond_with_all_listings(client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> ReplyOutput:
    """Reply, and repair the two ways a small model betrays the LISTINGS block.

    It silently DROPS listings it judges a poor fit (ranking is the reranker's job, so a short
    reply is a defect), and it ADDS listings it was never given - replaying URLs from conversation
    history as if a search had run, or inventing placeholder ones. Either way: name the offence,
    re-ask once, keep the better draft. If foreign URLs survive the retry, cut those cards out of
    the text ourselves - a shorter honest reply beats a fabricated inventory list.
    """
    reply = respond_turn(client, state, analysis, turn_outcome)
    listings = _outcome_get(turn_outcome, "listings", None) or []

    missing = _missing_listings(listings, reply) if listings else []
    foreign = _foreign_reply_urls(state, analysis, listings, reply)
    if not missing and not foreign:
        return reply

    logger.warning(
        "respond draft defective (missing=%d foreign=%d of %d listing(s)); retrying once: missing=%s foreign=%s",
        len(missing), len(foreign), len(listings),
        [_listing_get(item, "url") for item in missing], foreign,
    )
    retry = respond_turn(
        client, state, analysis, turn_outcome, repair_note=_repair_note(listings, missing, foreign)
    )
    retry_missing = _missing_listings(listings, retry) if listings else []
    retry_foreign = _foreign_reply_urls(state, analysis, listings, retry)
    # Foreign URLs outrank missing ones: showing the customer stale/invented inventory is worse
    # than showing them a shorter list.
    if (len(retry_foreign), len(retry_missing)) < (len(foreign), len(missing)):
        reply, foreign = retry, retry_foreign
    else:
        logger.warning("respond retry did not improve; keeping the first draft")
    if foreign:
        logger.warning("stripping %d foreign listing URL(s) from the reply: %s", len(foreign), foreign)
        reply = _without_foreign(reply, foreign)
    return reply
