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


def _has_recommendation_basis(state: Any, analysis: TurnAnalysis) -> bool:
    """Is there any reason to put a list of trailer TYPES in front of them?

    Only two: they asked for one, or they have told us something about the job - cargo, a size, a
    feature - that a recommendation can be built from. A customer who has just handed over their
    name and email has told us nothing about trailers, and answering that with four category
    suggestions is recommending into thin air. Ask them which type they want instead.
    """
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
    referenced = _referenced_listing(state, analysis)
    if referenced is not None:
        lines.append(
            f"- The listing they are pointing at is ALREADY RESOLVED for you - it is #{analysis.listing_reference} "
            f'of the batch on screen: "{_listing_get(referenced, "title")}" ({_listing_get(referenced, "url")}). '
            "Talk about THIS trailer and no other. Do not count down the list yourself, do not pick a different "
            "one, and do not quote a trailer from an earlier batch. Quote only its real fields, and put ONLY its "
            "URL in cited_listing_urls."
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
- Answer the user's question FIRST, then ask the pending qualification question once, in the same reply. Answering without asking it loses the question.
- ONE question per reply. When a question is given above, the reply ends with it, asked a single time and in your own natural words - never the same ask twice (once paraphrased, once verbatim), never a second question tacked on. Any question we listed is the information we need, not a script to recite.
- DO NOT REACT TO A QUALIFICATION ANSWER. When the customer answers one of our questions, say NOTHING
  about their answer: no praise, no agreement, no repeating it back, no explaining what it means for the
  trailer, no recap of what we have collected so far. It is recorded. Your whole reply is the next
  question - go straight to it.
  THE ONE EXCEPTION: the customer has no preference, is not sure, cannot answer, or wants to skip
  ("no idea", "whatever you recommend", "doesn't matter", "skip that"). THEN give ONE short, engaging
  line that puts them at ease and keeps the momentum ("No problem - we can keep that flexible and let
  the trailer decide."), and move straight on to the next question. Two sentences, no more.
  NONE OF THIS SHORTENS A REPLY THAT HAS LISTINGS. If the LISTINGS block below has trailers in it, that
  easing line is followed by every listing, written out in full. "Brief" never means dropping them.
- THE CUSTOMER SEES ONLY assistant_text. Everything you want them to read - every listing card, every
  bullet, every link - must be written out IN FULL in assistant_text. cited_listing_urls is a machine
  field they never see; putting a URL there does NOT show them the trailer. Announcing listings and then
  stopping ("Here are some trailers that match your requirements:") shows them NOTHING.
- Never re-ask anything already collected, skipped, or marked no-preference.
- When presenting listings: show EVERY listing in the block below, in the exact order given - never omit, add, or reorder any, and never filter by how well a size or feature matches. For EVERY listing shown, put its exact URL in cited_listing_urls.
- Optional fields (width, payload, hitch, height) are missing on many trailers - that is expected. Show the fields that are present and skip the missing lines; a missing field is NEVER a reason to drop a listing.
- Present listings ONLY from the "LISTINGS TO PRESENT THIS TURN" block. When it says NO SEARCH RAN, we have not looked yet - that says NOTHING about our stock. Present no listing cards, do not re-list anything from the reference block, and NEVER say we have nothing / no listings / none available for a category. Saying "I don't have any listings to show you for utility trailers" is WRONG and forbidden: we almost certainly have them, we just haven't searched. Answer the customer and ask the pending question instead.
- The reference block is memory, not inventory to show: use it only to answer a question about a listing the customer refers back to ("the 81382", "the second one"), quoting its real fields and URL. Never re-list it, renumber it, or restate it under a different category.
- Never invent inventory, prices, or policies. Store facts: Wharton TX, 979-532-1486, financing available, delivery available, {settings.trailerplace_website or "https://trailerplace.com"}.
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
DO THIS ONLY when BOTH of these are true:
  (a) no category is settled - "Category" in CONTEXT below is empty, or they are moving away from the
      one they had; AND
  (b) they have given us something to recommend FROM: they described what they will haul or a feature
      they want but never named a trailer type, they asked what we recommend or which type suits a job,
      or they said they cannot decide.
NEVER DO THIS OTHERWISE. In particular, do NOT list trailer types when they merely gave us their name,
email or phone, said hello, asked an FAQ, or said anything else that tells us nothing about the trailer
they need. They have given us no basis for a recommendation, so recommending would be guessing at them.
Just ask which type of trailer they are after, in one plain sentence.

When you DO recommend, do not answer in prose. Recommend 3 or 4 types from OUR CATEGORIES above, laid
out like this:

Based on what you need to haul, here are the types worth looking at:

1. **Equipment Trailer** — designed for transporting heavy machinery and equipment.
2. **Dump Trailer** — great for loose materials and can handle heavy loads.
3. **Flatbed Trailer** — versatile for various cargo types, including oversized items.

Which type would you like to go with? We carry more types as well if you would like to explore.

- 3 or 4 types, never fewer, never more. Only categories from OUR CATEGORIES above.
- Pick the ones that genuinely suit what they told us (their cargo, their job, their size). If they have
  told us nothing yet, pick the most common ones.
- Each line: the type name in bold, an em dash, then ONE short line on what it is best for.
- Always close by asking which type they want to go with, plus the note that we carry more types.
- IF THE CATEGORY IS ALREADY SETTLED and they are not asking about types, do NOT do this. They have
  chosen - listing types back at them makes us look like we were not listening. Just ask the question.

=== ANY OTHER ANSWER THAT IS REALLY A LIST GETS THE SAME SHAPE ===
If the honest answer to their question is a SET of things - what a category is used for, the use cases
for a trailer type, hitch options, deck styles, gate styles, loading options, what to consider at a
given size - give it as a bulleted list: bolded name, em dash, one short line each. Never bury three or
four options inside a paragraph. Two or more items means bullets.

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

COPY THE TITLE EXACTLY as it appears after "TITLE:" in the block - every word and the stock number on
the end ("2026 Gooseneck Livestock - 91632", not "2026 Gooseneck Livestock"). The stock number is how
the customer and our team refer to that exact trailer; a title with it trimmed off points at nothing.

A LISTING ONLY HAS THE FIELDS ITS BLOCK LINE LISTS. Some trailers have no price, no make, no length on
file - that is normal. If a field is not on that listing's line, DELETE THAT BULLET ENTIRELY. Never
write "None", "N/A", "Not specified", "Call for price", or a blank - and never carry a value across
from a different listing. A card with three bullets is correct if the block gave you three fields.

The last bullet is a one-sentence sales pitch for THAT trailer: engaging, attractive, and about what its
size, payload, hitch, or make lets the customer do. Build it ONLY from that listing's own fields above
and what the customer told us they need - never invent a feature, spec, condition, or price. Write a
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

    missing = _missing_listings(listings, reply)
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
        f"{len(listings)} URLs in cited_listing_urls.\n"
        "The listing cards go in assistant_text, WRITTEN OUT IN FULL, using the listing structure from the "
        "prompt. Announcing them and stopping ('Here are some trailers that match:') is not presenting them - "
        "the customer sees ONLY assistant_text, so a URL listed in cited_listing_urls but missing from "
        "assistant_text is a trailer they never saw."
    )
    retry = respond_turn(client, state, analysis, turn_outcome, repair_note=note)
    if len(_missing_listings(listings, retry)) < len(missing):
        return retry
    logger.warning("respond retry did not recover the dropped listing(s); keeping the first draft")
    return reply
