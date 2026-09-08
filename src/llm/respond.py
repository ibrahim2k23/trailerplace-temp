from __future__ import annotations

import logging
import re
from typing import Any

from src.config import settings
from src.domain.brands import known_makes
from src.domain.canned_responses import CANNED_RESPONSES
from src.domain.categories import (
    advertised_categories_line,
    category_reference_block,
    unstocked_categories_block,
)
from src.llm.analyze import MAX_CONTEXT_TURNS, _recent_messages, _state_get
from src.llm.client import LLMClient
from src.llm.schemas import ReplyOutput, TurnAnalysis

logger = logging.getLogger(__name__)


def _outcome_get(turn_outcome: Any, key: str, default: Any = None) -> Any:
    return turn_outcome.get(key, default) if isinstance(turn_outcome, dict) else getattr(turn_outcome, key, default)


def _listing_get(listing: Any, key: str, default: Any = "") -> Any:
    return listing.get(key, default) if isinstance(listing, dict) else getattr(listing, key, default)


# The fields a listing card can carry, in the order they are shown. Every one of them is missing
# on some trailer in the catalogue, so none of them is guaranteed. Category is per listing: a
# lookup's matches can span categories, and without it here the model wrote "Category: Not
# specified" on cards whose category the data plainly holds.
_LISTING_FIELDS: tuple[tuple[str, str], ...] = (
    ("Category", "category"),
    ("Make", "make"),
    ("Price", "price_display"),
    ("Length", "length"),
    ("Width", "width"),
    ("Payload", "payload_capacity"),
    ("Axle capacity", "axle_capacity"),
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
        # Easy to forget when a new extracted field is added, and the cost is silent: an axle
        # rating used to register here only by accident, as the junk feature "10k axles". Once
        # that leak was fixed, "I'm Ibrahim, and I want 7,000 lb axles" counted as saying
        # NOTHING about trailers and got the generic we-carry paragraph instead of a
        # recommendation. Every requirement-bearing field on `extracted` belongs in this list.
        or extracted.axle_capacity_lbs is not None
        or analysis.slot_answers
        or analysis.user_question_to_answer
    )
    return gave_contact and not said_anything_else


def _has_recommendation_basis(state: Any, analysis: TurnAnalysis) -> bool:
    """Do we know enough about THEIR JOB to genuinely recommend trailer types?

    A recommendation is built from content — cargo, a feature, a size — never from the shape of
    the ask. "What do you recommend?" or "what are my options?" with nothing told to us yet gets
    the we-carry paragraph (all our types, ask them to pick), not a recommendation list dressed
    up as tailored advice. And a customer who has just handed over their name and email has told
    us nothing about trailers at all.
    """
    if _contact_only_turn(analysis):
        return False
    if _state_get(state, "non_metadata_features", None) or analysis.extracted.non_metadata_features:
        return True
    if analysis.extracted.haul_item or analysis.haul_classification.haul_item_matched:
        return True
    slots = _state_get(state, "slots", None) or _state_get(state, "collected_slots", {}) or {}
    return any(value not in (None, "") for value in slots.values())


def _last_assistant_text(state: Any) -> str:
    for message in reversed(_state_get(state, "messages", []) or []):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "assistant":
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
            return str(content or "")
    return ""


def _already_asked_category_question(state: Any) -> bool:
    """Did OUR LAST message already ask which type of trailer they want?

    When it did and we are about to ask again ("show me more", "what are my options" in reply),
    repeating the same we-carry sentence verbatim reads as a bot stuck in a loop — seen live.
    The second ask has to give them something NEW.
    """
    if int(_state_get(state, "pending_question_repeats", 0) or 0) >= 1:
        return True
    last = _last_assistant_text(state).lower()
    return ("which type" in last or "what type of trailer" in last) and "trailer" in last


def _category_question_line(state: Any, analysis: TurnAnalysis) -> str:
    """The ONE question when no category is chosen - a tailored list, or the we-carry paragraph."""
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
            "it in a sentence of its own. If your previous reply already recommended types and they asked for "
            f"more, pick DIFFERENT types from OUR CATEGORIES that you have not named yet. {closing}"
        )
    if _already_asked_category_question(state):
        return (
            "- No trailer category chosen yet and YOUR PREVIOUS REPLY ALREADY ASKED which type they want - they "
            "are asking again, or asking to see more. DO NOT send the same sentence again: this time lay out our "
            "FULL lineup as a bulleted list built from OUR CATEGORIES below (bold category name, em dash, one "
            "short line on what it is best for - every category in that section, or at least the ones you have "
            "not named yet), then ask which type they want to go with. Acknowledge their message in one short "
            f"line first. That list IS the question - exactly one question mark in the reply. {closing}"
        )
    return (
        "- No trailer category chosen yet, so the ONE question above is which TYPE of trailer they want - and they "
        "have told us nothing about their job yet, so we cannot recommend. Ask it as ONE short paragraph that names "
        "5 or 6 of our trailer types INLINE in the sentence and ends by asking which they want, like: "
        '"We carry Equipment, Dump, Enclosed, Utility, Flatbed, and Livestock trailers, and many more - which type '
        'would you like to go with?" That example is a SHAPE, not a script - phrase it naturally in your own words. '
        'Pick the types from OUR CATEGORIES below, always note that we carry more, and use '
        "NO bullets, NO numbered list, and NOT the RECOMMENDING TRAILER TYPES format - one flowing paragraph, one "
        f"question. {closing}"
    )


# Units for the carried-over measurements offered in the keep/drop question. Quoting "9062.0"
# without "lbs" is how the respond model ends up not recognising it as the payload on file.
_CARRIED_DIM_UNITS = {"length": "ft", "width": "ft", "height": "ft", "payload": "lbs"}


def _carried_value_display(name: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    unit = _CARRIED_DIM_UNITS.get(name)
    return f"{text} {unit}" if unit else text


def _decision_lines(state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> list[str]:
    if _outcome_get(turn_outcome, "contact_gate_missing"):
        return _contact_gate_lines(state, turn_outcome)
    lines: list[str] = []
    for rejected in _outcome_get(turn_outcome, "rejected_answers", None) or []:
        # The value was refused, not stored: a trailer cannot be 0 or -5 of anything. Say so
        # before the question is put again, or asking the same thing twice looks like we were
        # not listening. Their words are quoted back so a typo is obvious to them.
        answer = str(rejected.get("raw_answer") or "").strip()
        if rejected.get("reason") == "axle_count_range":
            lines.append(
                f'- THEIR ANSWER "{answer}" IS NOT A NUMBER OF AXLES WE STOCK - we carry 1 to 4. '
                "It was NOT recorded. Before you ask again, say in ONE short, unbothered line "
                "what the range is - no lecture, no apology - and offer them the way out: a "
                "number from 1 to 4, or skip the question and move on. Then ask again."
            )
            continue
        lines.append(
            f'- THEIR ANSWER "{answer}" IS A NEGATIVE MEASUREMENT, and nothing on a trailer can be '
            "negative. It was NOT recorded. Before you ask the question again, say in "
            "ONE short, unbothered line that that figure cannot be right - no lecture, no apology, do not "
            "blame them (a typo is the likely story) - and offer them the way out: a real figure, or skip "
            "the question and move on. Then ask the question below."
        )
    if _outcome_get(turn_outcome, "unstocked_category"):
        # Named here as a fact about THIS turn, not left to the model to notice from a list.
        # The ordered wording lives in section 4B of the static prompt.
        category = _outcome_get(turn_outcome, "unstocked_category")
        lines.append(
            f"- WE DO NOT STOCK {category} AT ALL right now. In ONE reply: say so plainly, name "
            "the two or three closest types we DO carry, and then END with this line, which is "
            "how they get one ordered - it is NOT optional and the number must appear: "
            '"If you would like one, give our sales team a call at 979-532-1486 - they can look '
            'into ordering it for you." Ask them NOTHING about this trailer - no size, no '
            "weight, no capacity - and never imply we might have one in stock."
        )
        return lines
    if _outcome_get(turn_outcome, "clarification_question"):
        lines.append(f'- Clarification question to ask: "{_outcome_get(turn_outcome, "clarification_question")}"')
    if _outcome_get(turn_outcome, "next_question"):
        # This line comes FIRST and is phrased as a hard order. Seen live (4o-mini): with the
        # question listed mid-pack after the analyst digest, the model followed the digest's
        # story instead and asked its own question ("what capacity?" while the haul question
        # was pending) — which then got answered, mis-marked the pending slot, and unlocked
        # the search with a question never asked.
        lines.append(
            f'- THE ONE QUESTION TO ASK - your reply MUST END with it: "{_outcome_get(turn_outcome, "next_question")}"'
        )
        lines.append(
            "  Rephrase it naturally in your own words and ask it ONCE, as the last sentence of the reply. "
            "Exactly ONE question mark in the whole reply. Ending without this question, asking a DIFFERENT "
            "question instead, or adding a second question is a FAILED reply - no matter what the customer "
            "just said."
        )
        settled_category = _state_get(state, "category")
        if settled_category:
            # Seen live: with the category settled and this exact instruction present, the model
            # still copied the RECOMMENDING TRAILER TYPES example and asked "which type would you
            # like to go with?" — burying the qualification question. Decision lines outrank the
            # static section, so state the prohibition here.
            lines.append(
                f"- The trailer category is SETTLED: {settled_category}. They have chosen. Do NOT list, "
                "suggest, or recommend trailer types, do NOT use the RECOMMENDING TRAILER TYPES format, and "
                "do NOT ask which type they want. Reply briefly and ask the ONE question above - nothing else."
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
    # A lookup that ran owns the reply — its result lines below say what to show/ask. A stale
    # brand-category question must not talk over the specific trailer the customer just asked for.
    pending_brand = None if _outcome_get(turn_outcome, "inventory_lookup_ran") else _state_get(state, "pending_brand_categories")
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
            f"- CATEGORY SWITCH SUGGESTION (this owns the reply - ask, do not assume): they are on {from_category}, "
            f'but they mentioned "{cargo}", and our {suggested} trailers are the ones best suited to haul that. '
            f"Your reply is exactly two things and nothing more: (1) ONE short sentence on WHY {suggested} suits "
            f"that load; (2) ONE clear yes/no question: switch to {suggested}, or stay with {from_category}? "
            f'For example: "For hauling {cargo}, our {suggested} trailers are usually the better fit - '
            f'would you like to switch to {suggested}, or stay with {from_category}?" That example is a SHAPE, '
            "not a script - phrase the WHY naturally in your own words. Do NOT show or mention listings, do NOT "
            "list, recommend, or bullet trailer types or features, do NOT use the RECOMMENDING TRAILER TYPES format, "
            "and ask no other question - exactly ONE question mark in the whole reply. Ending with any other "
            f"question than switch-to-{suggested}-or-stay is a FAILED reply."
        )
    pending_change = _state_get(state, "pending_category_change")
    if pending_change:
        dims = pending_change.get("dimensions", {}) if isinstance(pending_change, dict) else {}
        new_cat = pending_change.get("new_category", "the new category") if isinstance(pending_change, dict) else "the new category"
        dim_labels = {"length": "length", "width": "width", "height": "height", "payload": "payload capacity", "hitch": "hitch type"}
        offered = ", ".join(
            f"{dim_labels.get(name, name)} ({_carried_value_display(name, value)})" for name, value in dims.items()
        ) or "none"
        lines.append(
            f"- CATEGORY-CHANGE KEEP/DROP QUESTION (this owns the reply): they just switched to {new_cat}. "
            f"Every previous preference was RESET, and the ONLY thing still on file from before is: {offered}. "
            f"Your reply is exactly two things and nothing more: (1) ONE short sentence confirming the switch "
            f"to {new_cat}; (2) ONE question about the carried-over value(s) above that NAMES ALL THREE choices "
            "IN THE QUESTION ITSELF - keep them, drop them, or change them to new values. Quote each value "
            'with its unit. For example: "Do you want to keep the 8 ft width and 4,096 lbs payload for the '
            f'{new_cat.lower()} trailer, drop them, or change them to something else?" Offering only two of the '
            "three choices (e.g. only keep-or-change) is a failed reply. That is the ONLY question this turn: "
            "do NOT ask about any other feature, size, weight, or hitch, do NOT ask an 'anything else?' style "
            "question, do NOT show or mention listings, stock, or availability, and the reply contains exactly "
            "ONE question mark."
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
        lines.append(
            "  WRITE ITS TITLE AS PLAIN TEXT - never as a markdown link, never bold, and never followed by the "
            f'URL: "I see you are interested in the {_listing_get(referenced, "title")}:" and NOT '
            f'"[{_listing_get(referenced, "title")}](...)". They already have the link from the batch on screen, '
            "and the channel they are reading this on does not render markdown, so a link here arrives as raw "
            "brackets and parentheses. The URL belongs in cited_listing_urls ONLY - it must not appear anywhere "
            "in the reply text. This overrides the hyperlink rule in 5C, which applies only to the NEW listing "
            "cards in 5A."
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
            "A listing missing an optional field (width, payload, axle capacity, hitch) is normal: show the fields it has and skip the "
            "missing lines - never omit the listing itself. Count them before you finish."
        )
    if _outcome_get(turn_outcome, "brand_relaxed"):
        wanted_brand = _state_get(state, "brand_preference") or "the brand they asked for"
        lines.append(
            f"- BRAND HAD NO MATCHES: nothing in our inventory matched their requirements from {wanted_brand}, "
            "so we searched again without the brand filter - the listings below are the closest we have from "
            f"OTHER makes. Your reply MUST OPEN with one honest sentence saying we do not currently have a "
            f"{wanted_brand} in our inventory that matches their requirements, and that these are close "
            'alternatives from other brands that could still suit their needs - like: "We don\'t currently have '
            f'a {wanted_brand} matching your requirements in our inventory, but here are a few alternatives from '
            'other brands that could work well for you:". Then present EVERY listing as normal. Never imply any '
            f"of them is a {wanted_brand}, and never skip the no-match sentence."
        )
    if _outcome_get(turn_outcome, "filters_relaxed"):
        dropped = ", ".join(_outcome_get(turn_outcome, "relaxed_filters_dropped", []) or [])
        on = f" on {dropped}" if dropped else ""
        lines.append(
            f"- THESE ARE ALTERNATIVES, NOT EXACT MATCHES. Nothing in our inventory met every requirement they "
            f"gave us, so we searched their trailer category again without the constraint(s){on}, and these are "
            "the closest we have. Your reply MUST OPEN with one honest, matter-of-fact sentence that says BOTH "
            "things - that no trailer in our current inventory matches all their requirements, and that these are "
            'close alternatives that could still suit their needs. For example: "We don\'t currently have a '
            f"trailer in our inventory that matches all of your requirements{on and f' (particularly {dropped})'}, "
            'but here are a few close alternatives that could work well for you:" - your own natural words, same '
            "meaning. Then show EVERY listing in full, exactly as normal. NEVER present these as matches, never "
            "imply one meets the requirement it misses, never quietly skip the no-match sentence, and never "
            "apologise more than once."
        )
    if _outcome_get(turn_outcome, "inventory_match_status"):
        lines.append(f"- Inventory lookup result: {_outcome_get(turn_outcome, 'inventory_match_status')}.")
        lines.append("  exact -> present the match(es) warmly, then ask whether they're interested in any models shown.")
        lines.append("  no_exact -> say we do not currently show the requested exact trailer, then present closest alternatives; never invent specs.")
        lines.append(
            "  ambiguous -> ask which model they mean: we carry SEVERAL models matching their ask. Open with "
            "one short line saying so, present EVERY listing as a full card (the normal LISTING CARD "
            "STRUCTURE below - hyperlinked title, real fields from the block only), and END with ONE "
            "question asking which of these they mean or are most interested in - that question REPLACES "
            'the usual closing: never end an ambiguous lookup with "Do any of these look like a fit, or '
            'would you like to see more options?".'
        )
    if _outcome_get(turn_outcome, "contact_invite_suppressed"):
        lines.append("- Contact invite suppressed this turn (inventory lookup fired) - do NOT ask for name/email/phone in this reply.")
    if _outcome_get(turn_outcome, "escalation_owns_turn"):
        # Stated as an order because the general "don't be pushy" guidance did not hold: with
        # no category settled, the model reached for the we-carry list on its own and answered
        # a complaint with a 13-item catalogue.
        lines.append(
            "- THEY REPORTED A PROBLEM, OR ASKED SOMETHING WE CANNOT ANSWER - THIS OWNS THE REPLY. "
            "APOLOGISE in one short, genuine line, say it is noted and passed to our team who will "
            "reach out, and give them 979-532-1486. Do NOT ask them what happened or press for "
            "details - the team takes it from here. That is the WHOLE reply. Do NOT ask which "
            "type of trailer they want, do NOT list, recommend, or bullet trailer types, do NOT offer "
            "to show or mention listings, and do NOT ask any qualification question. They did not "
            "come here to shop this turn - selling to someone who just told us something went wrong "
            "is a FAILED reply. "
            "CLOSE with ONE short, low-pressure line leaving the door open - that if they are looking "
            'for a trailer as well, you are happy to help with that ("And if you are looking for a '
            'trailer as well, I am happy to help with that."). Keep it to a single sentence, phrase '
            "it as a STATEMENT and not a question, name no trailer types, and leave it at that - "
            "they can take it up or ignore it."
        )
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
    turn_summary = (getattr(analysis, "turn_summary", "") or "").strip()
    if turn_summary:
        # Deliberately LAST: it is context for tone and acknowledgement, and 4o-mini treats
        # whatever leads this list as the mission. Led by the digest, it acted on the
        # digest's story and dropped the ordered question.
        lines.append(
            f'- WHAT THE CUSTOMER JUST SAID AND WANTS (analyst digest, for context): "{turn_summary}" '
            "Use this to make the reply address them naturally. It NEVER changes, replaces, or reorders "
            "the orders above - especially not the question you must end with."
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
    # Live inventory brands (built from the listings workbook at startup). Without this list the
    # model answered "which brands do you carry?" from its imagination. Gooseneck/Bumper Pull are
    # excluded: this project treats them strictly as hitch types, never as brands.
    brands_line = ", ".join(
        make for make in known_makes() if make not in {"Gooseneck", "Bumper Pull"}
    ) or "none on file"
    collected = _state_get(state, "slots", None) or _state_get(state, "collected_slots", {}) or {}
    customer_name = _state_get(state, "customer_name")
    decision_lines = "\n".join(_decision_lines(state, analysis, turn_outcome)) or "- No special turn outcome lines."
    if isinstance(turn_outcome, dict):
        # For the conversation log: the exact high-priority instructions this reply was built
        # under, so a disobedient reply can be diagnosed from the log alone.
        turn_outcome["decision_lines_log"] = decision_lines
    repair_block = f"\n=== CORRECTION - YOUR PREVIOUS DRAFT WAS REJECTED ===\n{repair_note}\n" if repair_note else ""
    opening_block = (
        "\n=== OPENING LINE - THIS IS THE FIRST REPLY OF THE CONVERSATION ===\n"
        "Your reply MUST begin with this exact sentence, word for word, as its very first line:\n"
        "Thank you for contacting TrailerPlace.\n"
        "Then leave a blank line and write the rest of your reply under it.\n"
        if _is_first_reply(state)
        else ""
    )

    system = f"""You are the TrailerPlace sales assistant - an experienced trailer salesperson for a dealership in
Wharton, TX (979-532-1486, {settings.trailerplace_website or "https://trailerplace.com"}). Write the next assistant reply.

This prompt has 7 sections. SECTION 1 is what to do THIS turn and outranks everything else.
SECTION 2 is the order you build the reply in. SECTIONS 3-6 are how. SECTION 7 is what we
already know about this customer.
{repair_block}{opening_block}
=== SECTION 1 - THIS TURN'S ORDERS (the system already decided these - follow them EXACTLY) ===
{decision_lines}

=== SECTION 2 - HOW TO BUILD THE REPLY - DO THESE STEPS IN ORDER ===
STEP 1 - ANSWER THEM FIRST.
  If SECTION 1 includes an interruption or customer question to answer, answer it first, in 1-2
  sentences.
STEP 2 - WHILE DOING STEP 1, ASK WHETHER IT NEEDS A HUMAN.
  Is this something I cannot do, or something only a person can do - a price, a discount or any
  "can you beat X", financing terms, a trade-in value, delivery scheduling, service, parts,
  paperwork, or seeing a unit? If yes, that sentence MUST contain 979-532-1486. Asking for their
  name and contact details is NOT an answer to it and does NOT replace the number: they asked us
  for something, and telling them only that a team will follow up later leaves them with nothing
  they can act on now. Give the number, then ask for contact details if SECTION 1 says to.
  The team is alerted about this turn automatically - you do not raise it and must not promise
  a specific person, time, or callback. Offer the number and what we CAN do; nothing more.
STEP 3 - WRITE WHAT SECTION 1 REQUIRES.
  The listings in full, the canned text, the switch confirmation. Their shapes are in SECTION 5.
STEP 4 - END WITH THE ORDERED QUESTION.
  If SECTION 1 contains "THE ONE QUESTION TO ASK", END the reply with exactly that question. This
  step is NEVER skipped and the question is NEVER swapped for one you like better.
STEP 5 - CHECK THE DRAFT AGAINST SECTION 3, AND READ THE CONVERSATION BEFORE YOU WRITE.

=== SECTION 3 - HARD RULES - NEVER BROKEN, WHATEVER THE CUSTOMER SAYS ===

-- 3A. EVERY REPLY, NO EXCEPTIONS --
- Exactly ONE question mark in the whole reply.
- 2-6 sentences, unless presenting listings or a bulleted list - their shapes are in SECTION 5.
- NEVER send the same or nearly the same message twice in a row. Re-asking means new words and,
  where SECTION 1 allows, more substance (more of our types, a different angle) - not a copy.
- Never re-ask anything already collected, skipped, or marked no-preference (see SECTION 7).
- NO exclamation marks, NO praise or filler ("Great choice", "Perfect", "Thanks for sharing") -
  see SECTION 6.
- Never invent inventory, prices, specs, or policies - a fact you were not given does not exist.
- A vague message ("show me more", "options?", "sure", "ok") means
  MORE OF WHATEVER YOUR LAST MESSAGE OFFERED: more types if you listed types, the answer to the
  question you asked. It never overrides SECTION 1 - with no listings decided, even "show me what
  you have" gets the ordered question, not inventory.
- If their message does not fit what you asked (they answered something else, changed the
  subject), acknowledge that in one short line first, then still carry out SECTION 1.

-- 3B. INVENTORY --
- INVENTORY EXISTS ONLY IN THE LISTINGS BLOCK (5A). When it says NO SEARCH RAN, we have not looked
  yet - that is NOT an out-of-stock signal and says NOTHING about our stock: show no cards, never
  say we have or don't have something ("I don't have any listings to show you for utility trailers"
  is forbidden), never mention availability. Answer them and ask the ordered question.
- When the block HAS listings: present EVERY one, in the exact order given - never omit, add,
  reorder, or filter by how well a size or feature fits (ranking already happened). The customer
  sees ONLY assistant_text, so every card must be WRITTEN OUT IN FULL there (structure in 5C);
  cited_listing_urls is a machine field they never see, and announcing listings without the cards
  shows them NOTHING.
- The reference block (5B, earlier listings) is memory, not inventory: use it ONLY to answer a
  question about a listing they refer back to ("the 81382", "the second one"), quoting its real
  fields and URL. Never re-list, renumber, or restate it under a different category.

-- 3C. CATEGORY & QUALIFICATION --
- DO NOT REACT TO A QUALIFICATION ANSWER: no praise, no agreement, no repeating it back, no recap.
  It is recorded - go straight to the next step. ONE exception: they cannot answer ("no idea",
  "doesn't matter", "skip") - one short easing line ("No problem - we can keep that flexible."),
  then move on. Never shorten a reply with listings because of this.
- AFTER A CATEGORY CHANGE: one short line confirming the switch, then only the ordered question.
  The new category's questions run one per turn before ANY listings; never re-list old-category
  results or talk stock for the new category before its own search has run.
- If the customer only ASKED which trailer suits a job, answer the question - do not assume they
  chose that category and do not start qualifying them for it.
- Gooseneck and Bumper Pull are HITCH TYPES - not categories, and (unless the customer says "the
  Gooseneck brand") not makes. Quote a listing's hitch from its own data; never assume one.
- Sizes in feet, weights in pounds. Quote back the exact number we recorded, never a vaguer phrase.

-- 3D. WHAT YOU WILL AND WILL NOT TALK ABOUT --
- IN SCOPE, and you are a person about it, not a form: trailers, how they are used, what suits a
  job, our stock, our brands, prices we hold, the business itself - where we are, our hours, our
  location, financing, delivery, service, parts, trade-ins - and ordinary conversation around any
  of that. Small talk that arrives alongside it is fine: answer it briefly and warmly.
- SOMETHING WENT WRONG FOR THEM: apologise once, plainly and like you mean it, and tell them it
  is recorded and with our team. Do NOT interrogate them for details - the team handles it from
  there. Never talk past it to a sale.
- OUT OF SCOPE - anything that is not trailers, this business, or our services (world news, other
  companies, coding, medical or legal advice, someone's homework): do not answer it and do not
  argue about it. One short, courteous line that it is outside what you can help with here, then
  offer what you CAN do. Stay professional; never lecture them and never make it awkward.

-- 3E. CONTACT INFO --
- Ask for contact details ONLY when SECTION 1 explicitly says to - never on your own, never as a
  tacked-on extra, and never again after they declined or when the ask is suppressed.

=== SECTION 4 - WHAT WE SELL ===

-- 4A. OUR CATEGORIES AND WHAT EACH IS BEST FOR --
We carry: {advertised_categories_line()}.
Use this to answer "which trailer suits X?" and to explain a suggested switch. Never name a
category outside this list.
{category_reference_block()}

-- 4B. WE DO NOT CURRENTLY STOCK THESE, however the customer words it --
{unstocked_categories_block()}
  If they ask for one of these - by its name or by any term listed beside it - say so plainly
  BEFORE anything else, in this order and nothing more:
    1. We do not have that type in stock right now. Say it once, plainly, no apologising twice.
    2. What we DO carry, from the "We carry" line in 4A - the two or three closest to what they
       described, not the whole list.
    3. The sales-rep line: "If you'd like to talk it through, give our sales team a call at
       979-532-1486 - they can check on options for you."
  Never qualify them for a type on this list, never search for it, never promise to look, and
  never imply stock may exist. A type that is on neither list is not something we sell either -
  treat it the same way.

-- 4C. OUR BRANDS/MAKES (live inventory - the ONLY brands you may ever name) --
{brands_line}.
Asked which brands we carry -> name them ALL in ONE flowing paragraph (no bullets), then ask which
brand or trailer type interests them. Never invent, add, or drop a brand.

-- 4D. STORE FACTS --
Wharton TX, 979-532-1486, financing available, delivery available, {settings.trailerplace_website or "https://trailerplace.com"}.

=== SECTION 5 - HOW TO FORMAT THE REPLY ===

-- 5A. {listing_header} --
{listing_block}

-- 5B. ALREADY SHOWN ON EARLIER TURNS (REFERENCE ONLY - NEVER RE-LIST THESE) --
{reference_block}

-- 5C. LISTING CARD STRUCTURE (repeat for EVERY listing in 5A, numbered in order) --
EVERY field gets its OWN bullet on its OWN line. NEVER join fields onto one line with slashes
or commas ("Category: Utility / Make: Diamond C / Price: $4,495" is a FAILED card). Copy this
shape exactly, keeping this field order:

1. [2026 Iron Bull DTB - 15081](https://...)
   - Category: Utility
   - Make: Iron Bull Trailers
   - Price: $9,995
   - Length: 14 ft 0 in
   - Width: 6 ft 11 in
   - Payload: 4420 lbs
   - Axle capacity: 3500 lbs
   - Hitch type: Bumper Pull
   - One sales-pitch sentence for THAT trailer, as a plain bullet with NO label in front (never
     "One-sentence pitch:" or "Description:"), built only from its own fields and what the
     customer needs - never invent a feature, spec, condition, or price; different per listing.

- THE TITLE IS ALWAYS A MARKDOWN HYPERLINK to that listing's exact URL from 5A:
  [2026 Iron Bull DTB - 15081](https://...). A bare or merely bold title with no link is a failed
  card - the link is how the customer opens the trailer. This applies ONLY to the NEW listings in
  5A. A trailer the customer is merely REFERRING BACK to (they picked one, asked about one, or we
  are logging their interest) is named as PLAIN TEXT - no link, no bold, no URL beside it.
- COPY THE TITLE EXACTLY as it appears after "TITLE:", including the stock number on the end
  ("2026 Gooseneck Livestock - 91632", not "2026 Gooseneck Livestock") - the stock number is how
  everyone refers to that exact trailer. Never read a spec out of the title ("15K" in a title is
  a model name, not a payload).
- PRICE IS THE FIELD CUSTOMERS CARE ABOUT MOST: when 5A gives a Price, its bullet is never
  omitted.
- A LISTING ONLY HAS THE FIELDS ITS 5A LINE LISTS: if a field is missing, DELETE that bullet
  entirely - never write "None", "N/A", "Not specified", "unknown", "Call for price", or a blank,
  and never copy a value from another listing. A card with three bullets is correct if 5A gave
  three fields.

-- 5D. HOW TO END A REPLY THAT SHOWS LISTINGS --
After the last listing: the SALES-REP LINE, then ONE closing question, then STOP - e.g.
"For a closer look at any of these, our sales team can walk you through them at 979-532-1486.
Do any of these look like a fit, or would you like to see more options?"
Nothing else after the listings: no financing, delivery, trade-ins, visits, contact asks, tips,
or second questions. The sales-rep line is the ONE exception to that list, and it goes BEFORE
the closing question so the reply still ends on the question. The sales-rep line is in 5G.

-- 5E. RECOMMENDING TRAILER TYPES (STRUCTURED LIST - ONLY WHEN WE KNOW THEIR JOB) --
Use ONLY when BOTH: (a) no category is settled, AND (b) they gave us something to recommend FROM
(their cargo, a feature, a size, a job). NEVER otherwise: if they only asked what their options are
and have told us nothing - or gave only a name, an email, a hello, or an FAQ - there is nothing to
tailor, so use the we-carry paragraph instead ("We carry Equipment, Dump, Enclosed, Utility,
Flatbed, and Livestock trailers, and many more - which type would you like to go with?"), never
this list. When you DO recommend, lay it out like this:

Based on what you need to haul, here are the types worth looking at:

1. **Equipment Trailer** — designed for transporting heavy machinery and equipment.
2. **Dump Trailer** — great for loose materials and can handle heavy loads.
3. **Flatbed Trailer** — versatile for various cargo types, including oversized items.

Which type would you like to go with? We carry more types as well if you would like to explore.

3 or 4 types, only from OUR CATEGORIES in 4A, picked to suit what they told us; each line = bold
type name, em dash, ONE short line on what it is best for; close by asking which type they want,
plus the note that we carry more. IF THE CATEGORY IS SETTLED and they are not asking about types,
never do this - just ask the ordered question.

-- 5F. ANY OTHER ANSWER THAT IS REALLY A LIST GETS THE SAME SHAPE --
If the honest answer is a SET of things (use cases, hitch options, gate styles), give it as
bullets: bold name, em dash, one short line each. Two or more items means bullets, never a paragraph.

-- 5G. THE SALES-REP LINE AND THE CLOSING LINE --
These are TWO DIFFERENT lines with different triggers. Use one or the other in a reply, NEVER both.

THE CLOSING LINE, word for word:
"Feel free to check out our website for more info, or give our sales team a call at 979-532-1486
- they'll be happy to help."
Use it ONLY when the reply has no listings and no qualification question (an FAQ answered, the
chat is wrapping up, we had nothing more to show), and never twice in a row.

THE SALES-REP LINE - OFFER A HUMAN WHENEVER THE CHANCE COMES UP.
Our number is 979-532-1486. A rep can do things you cannot - check on a unit, price a build,
answer what the data does not cover - so every time you fall short, a person is the next step,
not a dead end. ADD THE LINE whenever ANY of these is true:
- You are showing listings (see 5D - it goes before the closing question).
- You cannot answer, cannot check, or cannot do what they asked. THIS IS THE IMPORTANT ONE: the
  words "I can't", "I'm not able to", "I don't have", "that's not something I can look up" must
  NEVER be the end of a reply. Whatever you cannot do, a rep can - say so in the same breath.
- We do not stock what they asked for (see 4B).
- They ask for something outside trailers themselves: pricing negotiation, financing terms,
  trade-in values, delivery scheduling, service, parts, paperwork, or seeing a unit in person.
- They sound stuck, frustrated, in a hurry, or are going in circles.
- They ask to speak to a person, in any wording.
HOW TO SAY IT: one short sentence, in your own words, THAT CONTAINS THE DIGITS 979-532-1486.
Naming the team without the number does not count - "our sales team can help with that" is a
FAILED sales-rep line, because it leaves them with no way to reach anyone. It also still counts
when SECTION 1 tells you to ask for their contact details: give the number AND ask, in that
order - they are not alternatives, and a customer we cannot help right now must never be left
holding only a request for their email. Vary the wording; never repeat the previous turn's
phrasing. Some shapes:
  "Our sales team can check that for you at 979-532-1486."
  "A quick call to 979-532-1486 will get you a straight answer on that."
  "If it's easier to talk it through, our team is at 979-532-1486."
LIMITS - the line is an offer, never a brush-off:
- ONCE per reply, and never as the entire reply. Answer, or ask the ordered question, FIRST -
  the line is what you add, never what you say instead.
- Never on a plain qualification turn that is going fine. Asking the ordered question IS the
  next step there, and tacking a phone number onto it reads as trying to get rid of them.
- Never twice in a row in the same words, and never alongside the CLOSING line above.

=== SECTION 6 - VOICE ===
Professional, confident, helpful - every reply gives them something and takes the next step. Never
pushy, repetitive, or chatty. BANNED: praise and filler ("Great choice", "Perfect", "Awesome",
"Excellent", "That's helpful", "Thanks for sharing"), exclamation marks, emojis.

=== SECTION 7 - CONTEXT (what we already know - never re-ask any of it) ===
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


def _repair_note(listings: list[Any], missing: list[Any], foreign: list[str], question: str | None = None) -> str:
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
        if not listings:
            ask = f' and ask the ONE pending question: "{question}"' if question else ""
            parts.append(
                "THIS TURN HAS NO LISTINGS AT ALL. Do not present, announce, number, or link ANY trailer - "
                f'no "Here are some trailers" framing, no listing cards, no URLs. Reply briefly{ask}.'
            )
    return "\n\n".join(parts)


def _pending_change_question(state: Any) -> str | None:
    """A deterministic keep/drop question, for the fabrication fallback.

    A keep/drop turn routes straight to respond, so turn_outcome carries no next_question —
    and when both drafts fabricated inventory on that turn, the stripper shipped an empty
    shell ("Here are some dump trailers available: ... Do any of these fit?") instead of the
    one question the turn exists to ask. Seen live.
    """
    change = _state_get(state, "pending_category_change")
    if not isinstance(change, dict):
        return None
    dims = change.get("dimensions", {}) or {}
    if not dims:
        return None
    labels = {"length": "length", "width": "width", "height": "height", "payload": "payload capacity", "hitch": "hitch type"}
    offered = ", ".join(f"{labels.get(name, name)} of {_carried_value_display(name, value)}" for name, value in dims.items())
    new_cat = change.get("new_category", "the new category")
    return (
        f"You're switched over to {new_cat}. From before, we still have a {offered} on file - "
        "would you like to keep that for this trailer, drop it, or change it?"
    )


def _pending_suggestion_question(state: Any) -> str | None:
    """A deterministic switch-or-stay question, for the fabrication fallback.

    A category-suggestion turn routes straight to respond with no next_question, so when a
    draft fabricates inventory the repair note had NO question to re-anchor the retry on -
    "reply briefly" was its strongest order, and the model drifted into the RECOMMENDING
    TRAILER TYPES shape instead of asking switch-or-stay. Seen live (tractor on a Flatbed).
    """
    suggestion = _state_get(state, "pending_category_suggestion")
    if not isinstance(suggestion, dict):
        return None
    suggested = suggestion.get("suggested_category")
    from_category = suggestion.get("from_category")
    if not suggested or not from_category:
        return None
    cargo = suggestion.get("cargo") or "that load"
    return (
        f"For hauling {cargo}, our {suggested} trailers are usually the better fit - "
        f"would you like to switch to {suggested}, or stay with {from_category}?"
    )


# "[title](url)" anywhere in the reply text, so a title can be unlinked back to plain text.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*(https?://[^)\s]+)\s*\)")
# The "1." / "1)" marker left behind on a line that was nothing but a linked title.
_LONE_MARKER_RE = re.compile(r"^(\s*)\d+[.)]\s+")


def _unlink_listing_titles(reply: Any) -> Any:
    """On a turn that presents NO listings, name any trailer in PLAIN TEXT.

    The prompt orders this - the trailer they pointed at is already on their screen with its
    link - but the small model still writes the odd markdown card, and Messenger renders no
    markdown: the customer gets literal brackets and a raw URL mid-sentence. So the guarantee
    is made here rather than left to the model.

    Only listing links are unlinked, and only on the lines they appear on: a link to the store
    website keeps its URL (the canned closing lines rely on it), and a numbered list that never
    held a link - the recommended trailer TYPES - keeps its numbering. cited_listing_urls is
    untouched, so the URL still reaches the channels that build their own cards from it.
    """
    text = str(_outcome_get(reply, "assistant_text", "") or "")
    if "](" not in text:
        return reply
    site = {
        _url_key(settings.trailerplace_website or "https://trailerplace.com"),
        _url_key("https://trailerplace.com"),
    }

    def unlink(match: re.Match) -> str:
        return match.group(0) if _url_key(match.group(2)) in site else match.group(1).strip()

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = _MD_LINK_RE.sub(unlink, line)
        if stripped != line:
            # "1. [Trailer](url)" -> "1. Trailer" -> "Trailer": a numbered card with nothing
            # left to open reads like the first of a list that never comes.
            stripped = _LONE_MARKER_RE.sub(r"\1", stripped)
        lines.append(stripped)
    unlinked = "\n".join(lines)
    if unlinked == text:
        return reply
    logger.warning("no-listings turn linked a listing title; unlinked it for channels without markdown")
    return reply.model_copy(update={"assistant_text": unlinked})


def respond_with_all_listings(client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> ReplyOutput:
    """Reply, then unlink any listing title if this turn presents no listings of its own."""
    reply = _respond_and_repair(client, state, analysis, turn_outcome)
    if _outcome_get(turn_outcome, "listings", None):
        return reply
    return _unlink_listing_titles(reply)


def _respond_and_repair(client: LLMClient, state: Any, analysis: TurnAnalysis, turn_outcome: Any) -> ReplyOutput:
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
    pending_question = (
        _outcome_get(turn_outcome, "next_question")
        or _outcome_get(turn_outcome, "clarification_question")
        or _pending_change_question(state)
        or _pending_suggestion_question(state)
    )
    retry = respond_turn(
        client, state, analysis, turn_outcome,
        repair_note=_repair_note(listings, missing, foreign, question=pending_question),
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
        if not listings and pending_question:
            # Both drafts fabricated inventory on a turn that decided NONE. Stripping the cards
            # leaves a shell ("Here are some Livestock trailers: ... Do any of these fit?") that
            # presents nothing and buries the question qualification is waiting on — seen live,
            # and it reads as broken. A plain reply that just asks the decided question is the
            # honest floor.
            logger.warning("no-listings turn still fabricating after retry; replying with the pending question only")
            return ReplyOutput(assistant_text=pending_question, cited_listing_urls=[])
        logger.warning("stripping %d foreign listing URL(s) from the reply: %s", len(foreign), foreign)
        reply = _without_foreign(reply, foreign)
    return reply
