from __future__ import annotations

import json
import re
from typing import Any

from src import conversation_store
from src.domain.brands import brand_mentioned_in_text, categories_for_make
from src.domain.categories import (
    WIDTH_EXCLUDED_CATEGORIES,
    category_clarification_question,
    resolve_category_clarification_answer,
    resolve_category_from_text,
    resolve_category_matches,
)
from src.domain.defaults import defaults_for
from src.domain.normalizer import normalize_category, normalize_make
from src.domain.slot_map import (
    _SLOT_METADATA_FILTER_MAP,
    brand_is_actually_a_hitch,
    can_autofill_slot,
    equivalent_slots,
    is_recognized_slot_value,
    normalize_answer_for_slot,
    normalize_hitch_answer,
    normalize_slot_targets,
    sanitize_non_metadata_features,
    slot_value_kind,
    slots_of_kind,
)
from src.domain.trailer_fields import feature_like_optional_slots, get_trailer_fields
from src.domain.units import parse_dimensions
from src.graph.contact_gate import contact_ask_outstanding, update_contact_gate
from src.llm.schemas import HaulClassification, TurnAnalysis

INJECTED_WIDTH_SLOT = "item_or_trailer_width_ft"
INJECTED_WIDTH_QUESTION = "About how wide is that item or trailer you need to haul?"

# Aluminum is the odd one out: it is the inventory category we stock, and the trailer TYPE the
# customer wants it in ("utility", "enclosed") is a slot underneath it, not a category of its own.
# So a type word spoken inside the Aluminum flow is an ANSWER, never a category change.
ALUMINUM_CATEGORY = "Aluminum"
BASE_CATEGORY_SLOT = "base_category"


def _model_text_is_category_word(model_text: str | None) -> bool:
    """True when the "model" is really a trailer CATEGORY ("dump trailer", "utility").

    Seen live: "I want a Diamond C dump trailer, 7x14, ..." came back as a HIGH-confidence
    lookup with model_text="dump trailer". The lookup hijacked the turn (one arbitrary
    "exact" match instead of a qualified search) and suppressed the Diamond C brand
    preference as "just the lookup's make". A category word names what they are shopping
    for, never a specific unit.
    """
    text = str(model_text or "").strip().lower()
    if not text:
        return False
    stripped = re.sub(r"\b(trailers?)\b", " ", text).strip()
    if not stripped:
        return True  # "trailer" alone is not a model either
    return any(tier == "naming" for _cat, tier in resolve_category_matches(stripped) if _cat) and not re.search(
        r"[a-z]*\d", stripped
    )  # a digit-bearing token ("lpx14", "fhg 24k") is a real model code, category word or not


def lookup_requested(turn: Any) -> bool:
    """A direct-identifier inventory lookup rides along with ANY intent, not just intent=inventory_lookup.

    The analyze prompt deliberately keeps the dominant intent on the bigger action ("here's my
    email, I'm looking for a Diamond C LPX" -> contact_info_provided), so gating on the intent
    silently dropped those lookups. The extracted identifier block is the signal.
    """
    lookup = getattr(turn, "inventory_lookup", None) if turn else None
    if not (lookup and lookup.is_lookup and lookup.confidence in {"medium", "high"}):
        return False
    if lookup.stock_number or (lookup.year and lookup.make):
        return True
    # Make alone is a brand preference, never a lookup — require a real identifier, and a
    # category word posing as the model is category shopping, not an identifier.
    return bool(lookup.model_text) and not _model_text_is_category_word(lookup.model_text)


def _brand_is_lookup_make(analysis: TurnAnalysis, brand: str) -> bool:
    """True when the extracted brand is just the make half of this turn's lookup identifier.

    "I am looking for Iron Bull Dtb" is a request to pull up specific stock, not a standing
    instruction to filter every later search to Iron Bull — recording it as brand_preference
    also fired the which-category-for-that-brand question over the lookup's own results.
    """
    if not lookup_requested(analysis):
        return False
    make = analysis.inventory_lookup.make or ""
    if not make:
        return False
    # "Iron Bull" vs "Iron Bull Trailers": same make, different suffix — containment either way.
    brand_norm = normalize_make(brand).lower()
    make_norm = normalize_make(make).lower()
    return brand_norm in make_norm or make_norm in brand_norm


def enforce_haul_classification_invariant(haul: HaulClassification) -> HaulClassification:
    if (haul.is_lightweight_utility_load or haul.needs_width_question) and not haul.haul_item_matched:
        return haul.model_copy(update={"is_lightweight_utility_load": False, "needs_width_question": False})
    return haul


def _set_slot(state: dict[str, Any], key: str, value: Any, source: str = "user") -> None:
    state.setdefault("slots", {})[key] = value
    state.setdefault("slot_sources", {})[key] = source
    if key in state.setdefault("skipped_slots", []):
        state["skipped_slots"].remove(key)


def required_slots_for_state(state: dict[str, Any]) -> list[str]:
    if not state.get("category"):
        return []
    spec = get_trailer_fields(state["category"])
    required = list(spec.required)
    if INJECTED_WIDTH_SLOT in state.get("injected_required_slots", []) and INJECTED_WIDTH_SLOT not in required:
        pending = state.get("pending_question_slot")
        if pending in required:
            required.insert(required.index(pending) + 1, INJECTED_WIDTH_SLOT)
        else:
            required.append(INJECTED_WIDTH_SLOT)
    return required


def _mark_skipped(state: dict[str, Any], slot: str | None) -> None:
    if not slot:
        return
    if slot not in state.setdefault("skipped_slots", []):
        state["skipped_slots"].append(slot)
    state["slots"].pop(slot, None)
    state["slot_sources"].pop(slot, None)


def _apply_contact(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    contact = analysis.contact
    # contact.declined catches the mixed message the intent field loses: "I'd rather not
    # share that, but I need it for cargo hauling, 6x10" carries a bigger intent, and read
    # only off the intent the refusal vanished - so we asked for their email again, the one
    # thing the spec says never happens after a decline.
    declined = analysis.intent == "contact_declined" or getattr(contact, "declined", False)
    if declined:
        state["contact_declined"] = True
    gave_contact = bool(contact.name or contact.email or contact.phone)
    if declined and not gave_contact:
        state["contact_followup_pending"] = None
        state["contact_repeat_charged"] = False
        update_contact_gate(state, gave_contact_this_turn=False)
        return
    # A decline can still hand over a piece ("I'm Maria, but no email") - record what they
    # gave; the declined flag above stops any further asking.
    if contact.name:
        state["customer_name"] = contact.name
    if contact.email:
        state["customer_email"] = contact.email
    if contact.phone:
        state["customer_phone"] = contact.phone
    if gave_contact:
        state["contact_followup_pending"] = None
        state["contact_repeat_charged"] = False
        conversation_store.update_lead_contact(
            session_id=state["session_id"],
            full_name=state.get("customer_name"),
            email=state.get("customer_email"),
            phone=state.get("customer_phone"),
        )
    update_contact_gate(state, gave_contact_this_turn=gave_contact)


def _current_user_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _swap_shown_urls_for_category(state: dict[str, Any], category: str) -> None:
    """Point the exclude-list at THIS category's shown history.

    shown_urls is what search skips as "already shown". Kept flat across categories it kept
    hiding the old category's inventory forever; per category, a switch stops excluding the
    old category's trailers and a return to it restores its own exclusions.
    """
    buckets = state.setdefault("shown_urls_by_category", {})
    state["shown_urls"] = list(buckets.get(category) or [])


def _start_category(state: dict[str, Any], category: str) -> None:
    state["category"] = category
    state["qualification_complete"] = False
    _swap_shown_urls_for_category(state, category)
    conversation_store.update_lead_item_of_interest(state["session_id"], category)
    for key, value in defaults_for(category).items():
        _set_slot(state, key, value, "default")


# On a category change we drop every collected feature EXCEPT the measurements. A
# measurement can be sitting under ANY of the names its kind is known by — a length is
# `trailer_length_ft` on Livestock, `haul_length_ft` on Equipment, `vehicle_length_ft` on
# Car Hauler — so we look for it under all of them rather than a hand-listed few.
_DIMENSION_KINDS: dict[str, str] = {
    "length": "length_ft",
    "width": "width_ft",
    "height": "height_ft",
    "payload": "payload_lbs",
}
_DIMENSION_SLOT_KEYS: dict[str, tuple[str, ...]] = {
    dim: slots_of_kind(kind) for dim, kind in _DIMENSION_KINDS.items()
}
# Hitch rides along in the keep/drop question: it is as much a standing requirement as a
# length, and silently carrying (or silently dropping) it was wrong both ways.
_DIMENSION_SLOT_KEYS["hitch"] = ("hitch_type",)
# When re-seeding a kept dimension into the new category, write it under the canonical slot;
# _fill_category_slot_aliases then copies it into whatever name the new category asks by.
_DIMENSION_TARGET_SLOTS: dict[str, tuple[str, ...]] = {
    "length": ("trailer_length_ft",),
    "width": ("trailer_width_ft",),
    "height": ("trailer_height_ft",),
    "payload": ("payload_lbs",),
    "hitch": ("hitch_type",),
}


def _carried_dimensions(slots: dict[str, Any], sources: dict[str, Any] | None = None) -> dict[str, Any]:
    """The length/width/payload/hitch values currently held, keyed by canonical dimension.

    Only values the CUSTOMER put there (or kept through an earlier change) carry over. A
    category default (Flatbed's 8 ft deck width, an env-seeded hitch) is our assumption for
    THAT category, not a preference of theirs — offering it back in the keep/drop question
    asked them to confirm a number they never said, and dragged one category's default into
    another category where it does not belong.
    """
    sources = sources or {}
    carried: dict[str, Any] = {}
    for dim, keys in _DIMENSION_SLOT_KEYS.items():
        for key in keys:
            if sources.get(key) == "default":
                continue
            value = slots.get(key)
            if dim == "hitch":
                if value:  # a hitch is text (or a one-item list), never a number
                    carried[dim] = value
                    break
            elif _is_number(value):
                carried[dim] = float(value)
                break
    return carried


def _kept_dimension_names(kept_fields: list[str]) -> set[str]:
    """Map the LLM's kept_fields (canonical names or raw slot keys) to dimension names."""
    names: set[str] = set()
    for field in kept_fields:
        token = (field or "").strip().lower().replace(" ", "_")
        for dim, keys in _DIMENSION_SLOT_KEYS.items():
            if token == dim or token in keys:
                names.add(dim)
    return names


def _switch_category(
    state: dict[str, Any], new_category: str, kept_dims: dict[str, float], source: str = "user"
) -> None:
    """Move to ``new_category``, dropping every feature except the kept measurements.

    Metadata slots, non-metadata features, brand, hitch, haul item, injected questions
    and skips are all cleared; only the length/width/payload the user chose to keep carry
    over. Qualification then restarts for the new category.
    """
    state["category"] = new_category
    # Snapshot what the switch is about to clear: extraction runs AFTER this, and the
    # analyzer routinely re-emits the old category's haul_item/features off the history on
    # the very turn of the change. Stored again, the echo lands in the NEW category's cargo
    # slot via the aliases and silently answers a question that was never asked (seen live:
    # the new category's use-case question got skipped). The snapshot lets extraction tell
    # an echo from a genuinely restated value.
    state.setdefault("turn_outcome", {})["cleared_on_category_change"] = {
        "haul_item": (state.get("slots", {}) or {}).get("haul_item"),
        "features": list(state.get("non_metadata_features", []) or []),
    }
    state["slots"] = {}
    state["slot_sources"] = {}
    state["skipped_slots"] = []
    state["injected_required_slots"] = []
    state["non_metadata_features"] = []
    state["brand_preference"] = None
    state["pending_category_change"] = None
    state["pending_category_suggestion"] = None
    state["pending_question_slot"] = None
    state["pending_question_repeats"] = 0
    state["qualification_complete"] = False
    _swap_shown_urls_for_category(state, new_category)
    for key, value in defaults_for(new_category).items():
        _set_slot(state, key, value, "default")
    for dim, value in kept_dims.items():
        for key in _DIMENSION_TARGET_SLOTS[dim]:
            _set_slot(state, key, value, source)
    # Respond keys off this to keep the reply to the switch + the next question, nothing else.
    state.setdefault("turn_outcome", {})["category_just_changed"] = new_category
    conversation_store.update_lead_item_of_interest(state["session_id"], new_category)


# Marks a measurement that came from the PREVIOUS category rather than from something the
# customer said about the new one. Only these can be dropped by the keep/drop answer.
CARRIED_SLOT_SOURCE = "carried"


def _begin_category_change(state: dict[str, Any], new_category: str) -> None:
    """Move to ``new_category`` now, then ask which old measurements to carry over.

    The switch happens immediately so that anything the customer said in the SAME message
    ("switch me to an equipment trailer, 20 ft") lands in the new category's slots when
    extraction runs later this turn — the old code deferred the switch until the keep/drop
    answer and wiped those values on the way through. The carried measurements are tagged
    ``carried`` so a later "start fresh" can drop exactly those, and nothing the customer
    stated for the new category.
    """
    carried = _carried_dimensions(state.get("slots", {}), state.get("slot_sources", {}))
    state["pending_category_suggestion"] = None
    _switch_category(state, new_category, carried, source=CARRIED_SLOT_SOURCE)
    if carried:
        state["pending_category_change"] = {"new_category": new_category, "dimensions": carried}


def _drop_unkept_carried_dimensions(
    state: dict[str, Any], offered: dict[str, float], kept: set[str]
) -> None:
    """Drop the carried-over measurements the customer did not ask to keep.

    A measurement they have since restated themselves is theirs, not a leftover, so it is
    never dropped here — that is what the ``carried`` source tag distinguishes.

    Each dropped value is recorded in turn_outcome so extraction (which runs AFTER this) can
    refuse to re-store an analyzer echo of it. Seen live: "nope, no specific needs" dropped the
    carried 9062 lbs payload, and the same turn's `extracted` block re-emitted payload_lbs=9062
    copied from the history — silently undoing the drop.
    """
    dropped_values = state.setdefault("turn_outcome", {}).setdefault("dropped_carried_values", {})
    for dim in offered:
        if dim in kept:
            continue
        for key in _DIMENSION_SLOT_KEYS[dim]:
            if state.get("slot_sources", {}).get(key) == CARRIED_SLOT_SOURCE:
                value = state.get("slots", {}).pop(key, None)
                state.get("slot_sources", {}).pop(key, None)
                if value is not None:
                    dropped_values[dim] = value


def _refresh_pending_change_dimensions(state: dict[str, Any]) -> None:
    """Re-read the offered measurements from live slots before we ask about them.

    The customer may have restated one in the very message that changed the category
    ("...and make it 20 ft"); the keep/drop question must offer 20, not the stale 26.
    """
    change = state.get("pending_category_change")
    if not isinstance(change, dict):
        return
    live = _carried_dimensions(state.get("slots", {}), state.get("slot_sources", {}))
    offered = {dim: live[dim] for dim in change.get("dimensions", {}) if dim in live}
    if offered:
        change["dimensions"] = offered
    else:
        state["pending_category_change"] = None


# Intents where the user is actually shopping (so a category may move). Everything else -
# general_question, faq, inventory_lookup, smalltalk, ... - is a question ABOUT trailers
# and must never move the category.
#
# category_exploration is in the list on purpose: "I'm looking for a trailer to haul a
# tractor" is a WANT the model often labels exploration. What separates asking from wanting
# is is_category_info_only, which _wants_category_action checks first — a genuine "which
# trailer suits a tractor?" sets that flag and still cannot move the category.
_CATEGORY_ACTION_INTENTS = {
    "category_selection",
    "category_change",
    "category_exploration",
    "feature_request_no_category",
    "recommendation_request",
    "qualification_answer",
    "requirement_change",
}


# People answer the contact ask and say what they came for in one breath: "My name is Ibrahim, my
# email is ibrahim@x.ai, I need an aluminum trailer." The extractor labels the whole message
# contact_info_provided - the biggest thing in it, as far as it is concerned - so the category they
# just named was thrown on the floor and we turned around and asked them to pick a trailer type.
# Handing over contact details can never CHANGE a category (nothing about it says they changed their
# mind), but when none is chosen yet it can certainly start one.
_CATEGORY_START_INTENTS = _CATEGORY_ACTION_INTENTS | {"contact_info_provided"}


def _wants_category_action(analysis: TurnAnalysis, current: str | None) -> bool:
    """True only when the message expresses what the user WANTS, not what they're asking about.

    "Which trailer is best for hauling a tractor?" is information - the category never
    moves. "I need something to haul a tractor" is a want - it may.
    """
    if analysis.is_category_info_only:
        return False
    allowed = _CATEGORY_ACTION_INTENTS if current else _CATEGORY_START_INTENTS
    return analysis.intent in allowed


def _implied_category_from_cargo(cargo_text: str | None) -> str | None:
    """The category a cargo phrase points at, resolved from the term lists.

    Run on the EXTRACTED haul item as well as the raw message: "I've got a John Deere 5075E to
    move" contains no cargo term the resolver knows, but the extractor's haul_item ("John Deere
    5075E tractor") often does. Missing that meant no suggestion ever fired for the turn.
    """
    for category, _tier in resolve_category_matches(cargo_text or ""):
        return category
    return None


def _cargo_mentioned(analysis: TurnAnalysis) -> str | None:
    return analysis.haul_classification.haul_item_matched or analysis.extracted.haul_item


def _fresh_cargo_mention(state: dict[str, Any], analysis: TurnAnalysis) -> str | None:
    """The cargo this MESSAGE names, or None when it merely echoes the haul_item on file.

    The analyzer re-emits the stored haul_item on turns that never said it (seen live:
    "i'd like to see more" came back with haul_item="some cargo" copied from history, which
    re-fired — as an outright change this time — the Enclosed switch the customer had
    declined two turns earlier). A cargo equal to the one already on file is not a new
    signal; extraction runs after the category rules, so on a turn that genuinely restates
    a NEW cargo the stored slot still holds the old one and the mention passes.
    """
    cargo = _cargo_mentioned(analysis)
    if not cargo:
        return None
    stored = state.get("slots", {}).get("haul_item")
    if stored and str(stored).strip().casefold() == cargo.strip().casefold():
        return None
    return cargo


def _record_declined_suggestion(state: dict[str, Any], suggestion: dict[str, Any]) -> None:
    entry = {
        "category": suggestion.get("suggested_category"),
        "cargo": str(suggestion.get("cargo") or "").strip().casefold(),
    }
    declined = state.setdefault("declined_category_suggestions", [])
    if entry not in declined:
        declined.append(entry)


def _suggestion_declined(state: dict[str, Any], analysis: TurnAnalysis, category: str) -> bool:
    """Did they already turn down a move to ``category`` for the cargo currently in play?

    "Stay with Dump" settles the some-cargo→Enclosed question. The same cargo must not
    raise it again — not as a suggestion, and certainly not as an outright change. A NEW
    cargo that happens to point at the same category is a new question and may ask.
    """
    candidates = {
        str(value).strip().casefold()
        for value in (_cargo_mentioned(analysis), state.get("slots", {}).get("haul_item"))
        if value
    }
    for entry in state.get("declined_category_suggestions", []) or []:
        if entry.get("category") == category and entry.get("cargo") in candidates:
            return True
    return False


def _named_and_implied_categories(state: dict[str, Any], analysis: TurnAnalysis) -> tuple[str | None, str | None]:
    """Split this message's category signals into an explicit choice and an implied one.

    "named"   -> the user said the trailer TYPE ("a tilt trailer").
    "implied" -> the user only said a cargo/task term that maps to a category ("a tractor").
    A message can carry both ("a tilt trailer to haul a tractor"), which is exactly the
    case where we must ask rather than assume.
    """
    named_all: list[str] = []
    text_implied: str | None = None
    for category, tier in resolve_category_matches(_current_user_text(state)):
        if tier == "naming":
            named_all.append(category)
        elif tier == "cargo" and text_implied is None:
            text_implied = category
    named = named_all[0] if named_all else None
    # The EXTRACTED haul item outranks a raw-text cargo scan: the extractor knows which noun
    # is the load. Seen live: "I run a small landscaping outfit ... equipment trailer for my
    # compact tractor" — the text scan hit "landscaping" (a Utility cargo term describing
    # their BUSINESS) and suggested switching an Equipment customer to Utility, when the
    # actual cargo ("compact tractor") maps straight back to Equipment. The text scan stays
    # as the fallback for turns where the extractor pulled no cargo at all.
    #
    # With a category already chosen, only a FRESH cargo mention can imply a move — an
    # analyzer echo of the stored haul_item must not. With none chosen yet the stored cargo
    # is exactly what should pick the category (the contact gate defers the request a turn,
    # and the echo is how it arrives).
    cargo = _fresh_cargo_mention(state, analysis) if state.get("category") else _cargo_mentioned(analysis)
    implied = _implied_category_from_cargo(cargo) or text_implied
    if ALUMINUM_CATEGORY in named_all:
        # "an aluminum utility trailer", "a utility trailer but in aluminum": two type words, and
        # only one of them is a category we stock. Aluminum IS the category; the other type is the
        # base_category they want it built as. Whichever they happened to say first, aluminum wins
        # — taking the other one made us drop Aluminum and qualify them for a steel utility trailer.
        named = ALUMINUM_CATEGORY
    # Fall back to the LLM's category only when it is not just restating the cargo
    # implication (it catches typos/paraphrases the term lists miss).
    if named is None and analysis.category_mentioned and analysis.category_mentioned != implied:
        named = analysis.category_mentioned
    return named, implied


def _suggest_category_switch(state: dict[str, Any], suggested: str, analysis: TurnAnalysis) -> None:
    if _suggestion_declined(state, analysis, suggested):
        return
    state["pending_category_suggestion"] = {
        "suggested_category": suggested,
        "from_category": state.get("category"),
        "cargo": analysis.haul_classification.haul_item_matched or analysis.extracted.haul_item,
    }


def _accept_category_suggestion(state: dict[str, Any], suggestion: dict[str, Any]) -> None:
    """They said yes to "switch to X?" — move them, but do not throw away why we suggested it.

    The cargo they named and the features they voiced are the REASON for the switch, so both
    travel with them (the haul item answers the new category's cargo question through the slot
    aliases). Measurements and hitch get the same one-turn keep/drop question a requested
    category change gets, instead of the silent carry-over this path used to do.
    """
    slots = state.get("slots", {}) or {}
    carried = _carried_dimensions(slots, state.get("slot_sources", {}))
    haul_item = slots.get("haul_item") or suggestion.get("cargo")
    features = list(state.get("non_metadata_features", []) or [])
    new_category = suggestion["suggested_category"]
    _switch_category(state, new_category, carried, source=CARRIED_SLOT_SOURCE)
    if haul_item:
        _set_slot(state, "haul_item", haul_item)
    state["non_metadata_features"] = features
    if carried:
        state["pending_category_change"] = {"new_category": new_category, "dimensions": carried}


# Hitch values masquerade as categories in the workbook for some makes; they are never an
# answerable "which category?" option.
_NON_CATEGORY_CATEGORIES = {"Gooseneck", "Bumper Pull", "Unknown"}


def _brand_available_categories(brand: str) -> list[str]:
    categories = categories_for_make(normalize_make(brand)) or categories_for_make(brand)
    return [category for category in categories if category not in _NON_CATEGORY_CATEGORIES]


def _maybe_ask_brand_categories(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    """A brand named before any category: look up which categories that make comes in and ask.

    One category -> a yes/no ("want to go with Livestock?"); several -> "which of these?".
    The brand_preference itself is recorded by extraction as usual; it is UNRECORDED later
    only if they decline the single category we offered.
    """
    brand = analysis.extracted.brand_preference
    text = _current_user_text(state)
    if not brand or brand_is_actually_a_hitch(brand, text) or not brand_mentioned_in_text(brand, text):
        return
    if lookup_requested(analysis):
        # "I am looking for Iron Bull Dtb" names the make as half of a lookup identifier —
        # the lookup's own results answer the turn; the brand-category ask would talk over them.
        return
    categories = _brand_available_categories(brand)
    if not categories:
        return
    state["pending_brand_categories"] = {"brand": brand, "categories": categories}


def _resolve_pending_brand_categories(state: dict[str, Any], analysis: TurnAnalysis) -> bool:
    """Read the answer to "which category for that brand?". True when the turn is consumed.

    A named/implied category (whether or not it was on the list) adopts it and keeps the brand.
    "yes" on a single-category offer adopts that category. "no" declines it — and the brand
    preference goes with it, since it was only ever recorded to serve that offer. A contact-gate
    answer keeps the question alive for the turn the gate borrowed; anything else lets it drop
    so an ignored question can never wedge the conversation.
    """
    pending = state.get("pending_brand_categories")
    if not pending:
        return False
    if state.get("category"):  # a category arrived some other way — the question is moot
        state["pending_brand_categories"] = None
        return False
    if lookup_requested(analysis):
        # They moved on to a specific trailer — the lookup owns this turn, drop the question.
        state["pending_brand_categories"] = None
        return False
    named, implied = _named_and_implied_categories(state, analysis)
    target = named or implied
    single = pending["categories"][0] if len(pending["categories"]) == 1 else None
    if not target and single and analysis.category_confirm_answer == "yes":
        target = single
    if target:
        state["pending_brand_categories"] = None
        _start_category(state, target)
        return True
    if analysis.category_confirm_answer == "no":
        state["pending_brand_categories"] = None
        state["brand_preference"] = None
        # Extraction must not re-record the brand off this "no" message.
        state["turn_outcome"]["brand_offer_declined"] = pending["brand"]
        return True
    if analysis.intent in {"contact_info_provided", "contact_declined"}:
        return False  # the contact gate borrowed this turn; ask the brand question next
    state["pending_brand_categories"] = None  # they moved on — never wedge the flow on it
    return False


def _apply_clarification(state: dict[str, Any], analysis: TurnAnalysis) -> bool:
    """Deterministic category-clarification (spec: ask clarification before mapping).

    Returns True when clarification handled this turn (caller skips normal category apply).
    Uses only the ported domain resolver on category text — not the LLM intent decision.
    """
    text = _current_user_text(state)
    pending_key = state.get("clarification_key")
    if pending_key:
        resolution = resolve_category_clarification_answer(text, pending_key)
        if resolution.category:
            state["clarification_key"] = None
            state["turn_outcome"].pop("clarification_question", None)
            _start_category(state, resolution.category)
            return True
        if resolution.needs_clarification:
            state["turn_outcome"]["clarification_question"] = (
                resolution.clarification_question or category_clarification_question(pending_key)
            )
            return True
        state["clarification_key"] = None
        return False
    if not state.get("category") and (not analysis.is_category_info_only or _states_a_need(text)):
        # The info_only override: the analyzer labeled "I need an office trailer." an
        # informational ask (seen live), which silently skipped the office-trailer
        # clarification the spec requires. A message that opens by stating a need is a
        # WANT whatever the label says.
        resolution = resolve_category_from_text(text)
        if resolution.needs_clarification:
            state["clarification_key"] = resolution.clarification_key
            state["turn_outcome"]["clarification_question"] = resolution.clarification_question
            return True
    return False


_NEED_PHRASE_RE = re.compile(
    r"\b(i\s+(?:need|want|would\s+like)|i'?m\s+(?:looking|after|in\s+the\s+market)|looking\s+for)\b",
    re.IGNORECASE,
)


def _states_a_need(text: str) -> bool:
    return "?" not in (text or "") and bool(_NEED_PHRASE_RE.search(text or ""))


def _apply_category(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    # (0) Answer to a pending "which category for that brand?" question.
    if _resolve_pending_brand_categories(state, analysis):
        return

    # (a) Answer to a pending "should we switch you to X?" suggestion.
    suggestion = state.get("pending_category_suggestion")
    if suggestion:
        # The analyzer often labels an acceptance that NAMES the category ("yes, switch to
        # equipment") as a plain category_change with no confirm answer. Falling through to
        # the ordinary change path wipes the cargo that motivated the suggestion — so while
        # the yes/no is pending, a non-informational mention of the suggested category IS
        # the yes, and a mention of the category they are already on IS the no.
        mentioned = None if analysis.is_category_info_only else analysis.category_mentioned
        if analysis.category_confirm_answer == "yes" or (
            analysis.category_confirm_answer is None
            and mentioned == suggestion.get("suggested_category")
        ):
            _accept_category_suggestion(state, suggestion)
            return
        # "no", or they moved on without answering -> stay put and never re-ask.
        state["pending_category_suggestion"] = None
        if analysis.category_confirm_answer == "no" or mentioned == suggestion.get("from_category"):
            # Remember the refusal: the same cargo must never raise this switch again,
            # as a suggestion or as an outright change.
            _record_declined_suggestion(state, suggestion)
            return

    # (b) A pending keep/drop question from a category change. The category already moved
    # when the change was requested; all that is left is to honour the drops.
    #
    # It gets exactly ONE turn, answered or not. If the customer replies with something
    # else entirely ("I think 18ft would be good"), we keep the measurements we offered and
    # move on — leaving the question pending would block qualification (and therefore every
    # remaining question) forever.
    if state.get("pending_category_change"):
        change = state["pending_category_change"]
        offered = change.get("dimensions", {})
        answer = analysis.keep_fields_answer
        if answer == "some":
            kept = _kept_dimension_names(analysis.kept_fields)
        elif answer == "none":
            kept = set()
        else:  # "all", or no answer at all — they moved on, so keep what we offered
            kept = set(offered)
        _drop_unkept_carried_dimensions(state, offered, kept)
        state["pending_category_change"] = None
        if answer:
            return
        # No keep/drop answer: this message was about something else, so let the normal
        # category rules below read it.

    # (c) They are ANSWERING our own base-category question — "what type are you looking for in
    # aluminum: utility, equipment, enclosed?". The type they name is the answer to that question.
    # Read as a category choice it looked like they wanted to leave Aluminum, so we switched them
    # to Utility, wiped the Aluminum slots and restarted qualification — off the back of them
    # answering us. While that question is pending, the category cannot move.
    if state.get("pending_question_slot") == BASE_CATEGORY_SLOT:
        return

    # An informational question never moves the category, no matter what it mentions.
    current = state.get("category")

    # A results request that NAMES a different trailer type is a category change wearing the
    # wrong label. Seen live: "show me dump trailers instead" mid-Utility came back as
    # show_more_results — the gate below swallowed the change, and _apply_skips_and_repeats
    # then marked every OLD-category question skipped and searched the old category. Only an
    # explicitly NAMED type moves the category here; a cargo word on a results request stays
    # navigation (that is what _RESULTS_NAV_INTENTS protects).
    if (
        analysis.intent in {"show_more_results", "skip_all_show_results"}
        and not analysis.is_category_info_only
    ):
        named_only = [
            category
            for category, tier in resolve_category_matches(_current_user_text(state))
            if tier == "naming"
        ]
        target = named_only[0] if named_only else None
        if target and target != current:
            if current:
                _begin_category_change(state, target)
            else:
                _start_category(state, target)
            return

    if not _wants_category_action(analysis, current):
        # The analyzer sometimes labels "I'll be hauling a tractor on it" as plain chat
        # (general_question, smalltalk) instead of a shopping intent, and the gate above then
        # swallowed the cargo signal entirely — the suggestion the spec requires never fired.
        # A freshly mentioned cargo is a real signal regardless of the label, as long as the
        # message is a statement and not an informational question.
        if current:
            _apply_cargo_only_signal(state, analysis, current)
        else:
            # No category yet and the gate failed (e.g. "do you carry Iron Bull?" is a
            # question): a brand mention still deserves the which-category-for-that-brand ask.
            _maybe_ask_brand_categories(state, analysis)
        return

    named, implied = _named_and_implied_categories(state, analysis)

    # Rule 1: nothing chosen yet -> adopt whatever they named, or what their cargo implies.
    if not current:
        target = named or implied
        if not target:
            # A brand with no category is a question for the workbook: which categories does
            # that make come in? Ask, instead of the generic "what type of trailer?".
            _maybe_ask_brand_categories(state, analysis)
            return
        _start_category(state, target)
        # "a tilt trailer to haul a tractor": take them at their word (Tilt), but a tractor
        # belongs on an Equipment trailer — offer the switch instead of silently overriding.
        if named and implied and implied != named:
            _suggest_category_switch(state, implied, analysis)
        return

    # Rule 3: results already shown -> a category OR cargo term switches outright,
    # pausing only to ask which measurements to carry over. A cargo-implied move they
    # already declined stays declined; only NAMING the category overrides that.
    if state.get("shown_urls"):
        target = named or implied
        if target and target != current:
            if named is None and _suggestion_declined(state, analysis, target):
                return
            _begin_category_change(state, target)
        return

    # Rule 2: mid-qualification.
    if named and named != current:
        # They explicitly named a different type — that is a change, not a suggestion.
        _begin_category_change(state, named)
        return
    if implied and implied != current:
        # Only a cargo/task term points elsewhere — confirm before moving them.
        _suggest_category_switch(state, implied, analysis)


# Asking to see more of the current results — or pointing at one of them — is the opposite
# of leaving the category. These turns are navigation, never a category signal, whatever
# the extractor happened to put in the cargo fields.
_RESULTS_NAV_INTENTS = {"show_more_results", "skip_all_show_results", "listing_interest"}


def _apply_cargo_only_signal(state: dict[str, Any], analysis: TurnAnalysis, current: str | None) -> None:
    """Honour a cargo mention on a turn whose INTENT failed the category-action gate.

    Never a silent move: pre-results it raises the switch SUGGESTION (a question), post-results
    it starts the keep/drop change the spec calls for. Informational questions stay inert, a
    cargo that maps to the current category is simply an answer, and only a cargo FRESHLY
    stated this turn counts — an analyzer echo of the haul_item already on file is not the
    customer changing their mind (seen live: "i'd like to see more" echoed "some cargo" and
    yanked a Dump customer to Enclosed).
    """
    if not current or analysis.is_category_info_only:
        return
    if analysis.intent in _RESULTS_NAV_INTENTS:
        return
    if state.get("pending_category_change") or state.get("pending_category_suggestion"):
        return
    implied = _implied_category_from_cargo(_fresh_cargo_mention(state, analysis))
    if not implied or implied == current:
        return
    if _suggestion_declined(state, analysis, implied):
        return
    if state.get("shown_urls"):
        _begin_category_change(state, implied)
    else:
        _suggest_category_switch(state, implied, analysis)


def _apply_aluminum_base_category(state: dict[str, Any]) -> None:
    """Fill Aluminum's ``base_category`` from the trailer type the customer named.

    base_category is the type they want the aluminum trailer built as, and it is the subcategory
    filter the search runs on. Two different messages state it — "a utility trailer in aluminum"
    up front, and "utility" in answer to our base-category question — and they are the same fact.
    The extractor cannot be relied on for either: it reads the type word as a category and emits
    no slot answer at all, which left base_category empty (or nulled by the answered-but-unparsed
    fallback) and the search unfiltered.
    """
    if normalize_category(state.get("category") or "") != ALUMINUM_CATEGORY:
        return
    named = [
        category
        for category, tier in resolve_category_matches(_current_user_text(state))
        if tier == "naming"
    ]
    base = next((category for category in named if category != ALUMINUM_CATEGORY), None)
    if not base:
        return
    # Either they said it alongside "aluminum", or they said it while we had the question on the
    # table. A type word in any other message is not an answer to a question we did not ask.
    if ALUMINUM_CATEGORY in named or state.get("pending_question_slot") == BASE_CATEGORY_SLOT:
        _set_slot(state, BASE_CATEGORY_SLOT, base)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _store_dimension_pair(state: dict[str, Any], category: str, slot_name: str, raw_answer: Any) -> None:
    """A size answer like "8x25" or "8x25x6.5" states MULTIPLE dimensions — record each
    under its own canonical slot.

    Otherwise only the slot that was asked about (say trailer_length_ft) would keep a
    value and the width (and height, for a 3-number answer) the customer just gave us
    would be dropped on the floor.
    """
    # Any slot that carries a size: a dimension slot (trailer_length_ft) or a composite
    # size question (trailer_size, cargo_size) whose answer feeds length/width(/height).
    targets = set(_SLOT_METADATA_FILTER_MAP.get(slot_name, ()))
    is_size_slot = slot_value_kind(slot_name) in {"length_ft", "width_ft", "height_ft"} or bool(
        targets & {"length_ft", "width_ft", "height_ft"}
    )
    if not is_size_slot:
        return
    if normalize_category(category or "") == "Roll Off":  # bin sizes are yardage, not WxL
        return
    dims = parse_dimensions(raw_answer)
    if not dims:
        return
    width_ft, length_ft, height_ft = dims
    _set_slot(state, "trailer_width_ft", width_ft)
    _set_slot(state, "trailer_length_ft", length_ft)
    if height_ft is not None:
        _set_slot(state, "trailer_height_ft", height_ft)


def _store_slot_answer(state: dict[str, Any], category: str, slot_name: str, raw_answer: Any) -> None:
    """Store one slot answer.

    For a slot that maps to Pinecone metadata target(s) (e.g. cargo_size -> length_ft,
    width_ft), parse the answer for each target and store the NUMBER under that target
    slot key — never a re-encoded "key=value" string. The slot itself holds its answer
    normalized to its own kind: a numeric slot keeps a number (a range resolves to its
    smallest side), hitch_type/subcategory keep a canonical value, and a genuinely vague
    answer for any of these is null (no preference) rather than raw text — free-text
    slots (haul_item and similar) have no kind and are always stored exactly as given.
    """
    for target_key, parsed in normalize_slot_targets(category or "", slot_name, raw_answer).items():
        if _is_number(parsed):
            _set_slot(state, target_key, parsed)
    _store_dimension_pair(state, category or "", slot_name, raw_answer)
    value = normalize_answer_for_slot(category or "", slot_name, raw_answer)
    if not is_recognized_slot_value(slot_name, value) and is_recognized_slot_value(slot_name, state.get("slots", {}).get(slot_name)):
        # A vague restatement ("something roomy", an unparseable hitch spelling) must not
        # overwrite the clean value another part of this same turn already produced.
        return
    _set_slot(state, slot_name, value)
    if slot_name == "bin_size" and _is_number(value):
        # We carry no real bin-size metadata — the yard number IS the trailer length the
        # customer needs, so mirror it into the canonical slot everything else reads.
        _set_slot(state, "trailer_length_ft", value)


# A bare yes/no/shrug answers the optional question but names no equipment, so there is nothing for
# the feature matcher to match on. Same for anything phrased as a refusal — "no ramps" is not a
# request for ramps, and mirroring it would search for the very thing they turned down.
_NON_FEATURE_ANSWERS = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "fine", "no", "nope", "none", "n/a", "na",
    "maybe", "idk", "not sure", "no preference", "any", "either", "whatever", "doesn't matter",
    "does not matter", "dont matter", "don't matter",
})
_NEGATED_ANSWER_RE = re.compile(r"^(?:no|not|none|nothing|never|without|don'?t|do\s+not)\b", re.IGNORECASE)
# One optional answer often names several pieces of equipment ("AC, windows and cabinets"). They are
# separate features and are matched separately — kept as one phrase, the whole string has to match.
_FEATURE_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|\bplus\b|&)\s*", re.IGNORECASE)


def _mirror_optional_answer_as_feature(
    state: dict[str, Any], category: str, slot_name: str, raw_answer: Any
) -> None:
    """Mirror a descriptive optional answer into the non-metadata feature list.

    "Butterfly gates", "scissor lift", "drive-over fenders", "lined walls" — we hold no metadata
    field for any of them, so stored only under their own slot they never reach the search: nothing
    filters on them and nothing ranks on them. The feature matcher is the one thing that acts on
    them, so the answer has to land there too. Analyze is told to emit these as features itself;
    this is the net for the turns it forgets.
    """
    if slot_name not in feature_like_optional_slots(category or ""):
        return
    text = str(raw_answer or "").strip()
    if text.casefold().strip(" .!") in _NON_FEATURE_ANSWERS or _NEGATED_ANSWER_RE.match(text):
        return
    parts = [part for part in _FEATURE_SPLIT_RE.split(text) if part.strip()]
    features, _ = sanitize_non_metadata_features(parts or [text])
    known = state.setdefault("non_metadata_features", [])
    existing = {feature.casefold() for feature in known}
    for feature in features:
        if feature.casefold() not in existing:
            known.append(feature)
            existing.add(feature.casefold())


def _brand_only_points_at_a_listing(state: dict[str, Any], analysis: TurnAnalysis, brand: str) -> bool:
    """Is this brand name being used to POINT at a listing rather than to ask for that make?

    "I like the Iron Bull one" is how people pick a trailer off a list — it names the make,
    but it is a finger, not a filter. Recording it as a brand preference would narrow every
    later search to that one manufacturer on the strength of them liking a single trailer.
    """
    if analysis.intent != "listing_interest" and analysis.listing_reference is None:
        return False
    for listing in state.get("shown_listings") or []:
        make = listing.get("make") if isinstance(listing, dict) else getattr(listing, "make", None)
        if make and brand_mentioned_in_text(str(make), brand):
            return True
    return False


def _normalized_hitch_values(value: Any) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    return sorted(str(item).strip().casefold() for item in items if item)


def _echoes_dropped_carried(state: dict[str, Any], key: str, value: Any) -> bool:
    """Is this extracted value just an echo of a carried-over value dropped THIS turn?

    The analyzer sometimes re-emits an already-collected value it saw in the history rather
    than in the message (seen live: "nope, no specific needs" came back with
    payload_lbs=9062 in `extracted`). On the turn the keep/drop answer dropped that value,
    re-storing the echo would silently undo the drop — the value only counts again when THIS
    message states a different one.
    """
    outcome = state.get("turn_outcome") or {}
    dropped = dict(outcome.get("dropped_carried_values") or {})
    if outcome.get("category_just_changed"):
        # Same analyzer habit, other side of the change: on the turn the category moves, the
        # carried-over measurements we are ABOUT TO ASK about get re-emitted in `extracted` off
        # the history. Storing that echo re-tags the value from "carried" to "user", and the
        # keep/drop answer next turn can no longer drop it (drops only touch source=="carried").
        change = state.get("pending_category_change")
        if isinstance(change, dict):
            for dim, value in (change.get("dimensions") or {}).items():
                dropped.setdefault(dim, value)
    if not dropped:
        return False
    if "hitch" in dropped and slot_value_kind(key) == slot_value_kind("hitch_type"):
        if _normalized_hitch_values(value) == _normalized_hitch_values(dropped["hitch"]):
            return True
    kind = slot_value_kind(key)
    for dim, kind_name in _DIMENSION_KINDS.items():
        if kind == kind_name and dim in dropped and _is_number(value) and float(value) == float(dropped[dim]):
            return True
    return False


def _echoes_cleared_on_change(state: dict[str, Any], kind: str, value: Any) -> bool:
    """Is this extracted value just an echo of something the category change wiped THIS turn?

    The old category's haul_item and features are cleared by the switch on purpose — the new
    category asks its own cargo/use-case question from scratch. The analyzer re-emitting the
    old value off the history (not off this message) must not sneak it back in and mark that
    question answered. A value the customer genuinely restated differs from the cleared one
    — or is the cleared one, in which case they said the same thing and re-asking is wrong
    anyway only when they actually said it; a verbatim echo is indistinguishable, so the
    cleared value never re-enters on the change turn itself.
    """
    cleared = (state.get("turn_outcome") or {}).get("cleared_on_category_change") or {}
    if not cleared:
        return False
    if kind == "haul_item":
        old = cleared.get("haul_item")
        return bool(old) and str(old).strip().casefold() == str(value or "").strip().casefold()
    if kind == "feature":
        old_features = {str(item).strip().casefold() for item in (cleared.get("features") or [])}
        return str(value or "").strip().casefold() in old_features
    return False


def _apply_extraction(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    category = state.get("category") or ""
    user_text = _current_user_text(state)

    # Features are only the things we cannot filter on. A hitch, a size, a weight or a price
    # parked in this list is a requirement we would silently fail to apply — lift the hitch
    # out and drop the rest (their values already live in their own slots).
    features, hitch_from_features = sanitize_non_metadata_features(analysis.extracted.non_metadata_features)
    for feature in features:
        if _echoes_cleared_on_change(state, "feature", feature):
            continue
        if feature not in state.setdefault("non_metadata_features", []):
            state["non_metadata_features"].append(feature)

    if analysis.extracted.brand_preference:
        brand = analysis.extracted.brand_preference
        if brand_is_actually_a_hitch(brand, user_text):
            # "I want a gooseneck" is a hitch, not the Gooseneck make — reading it as a make
            # would quietly restrict every result to one manufacturer.
            hitch_from_features = hitch_from_features or normalize_hitch_answer(brand)
        elif (
            brand_mentioned_in_text(brand, user_text)
            and not _brand_only_points_at_a_listing(state, analysis, brand)
            and not _brand_is_lookup_make(analysis, brand)
        ):
            state["brand_preference"] = brand
        # Otherwise the extractor read the make off a listing already on screen rather than
        # off the customer. Seen live: a turn that only gave a name and email came back with
        # brand_preference="Iron Bull Trailers" — which then filtered every later search to
        # that one manufacturer. A brand preference has to be something they said, and said
        # as a preference.

    for key in analysis.extracted.numeric_no_preference:
        _set_slot(state, key, None)
    if analysis.extracted.haul_item and not _echoes_cleared_on_change(state, "haul_item", analysis.extracted.haul_item):
        _set_slot(state, "haul_item", analysis.extracted.haul_item)
    if analysis.extracted.hitch_type:
        # A single named type is a real preference; "both" (the model's way of saying
        # "either is fine") is not one — treat it the same as no preference (null).
        hitch = analysis.extracted.hitch_type
        value = list(hitch) if len(hitch) == 1 else None
        if not (value and _echoes_dropped_carried(state, "hitch_type", value)):
            _set_slot(state, "hitch_type", value)
    if hitch_from_features and not state.get("slots", {}).get("hitch_type"):
        _set_slot(state, "hitch_type", hitch_from_features)
    numeric_map = {
        "trailer_length_ft": analysis.extracted.trailer_length_ft,
        "trailer_width_ft": analysis.extracted.trailer_width_ft,
        "trailer_height_ft": analysis.extracted.trailer_height_ft,
        "payload_lbs": analysis.extracted.payload_lbs,
        "haul_weight_lbs": analysis.extracted.payload_lbs,
        "payload_need": analysis.extracted.payload_lbs,
    }
    valid_slots = set(required_slots_for_state(state)) | set(get_trailer_fields(category).optional)
    for key, value in numeric_map.items():
        if value is None or _echoes_dropped_carried(state, key, value):
            continue
        if key in valid_slots or key.startswith("trailer_"):
            _set_slot(state, key, value)
    for answer in analysis.slot_answers:
        # A blank raw_answer is not an answer. Seen live: answering the length question for
        # Livestock, the extractor ALSO emitted haul_item='', haul_weight_lbs='' and
        # hitch_type='' — slots Livestock never asks about. Stored, those mark questions
        # answered that were never asked and put junk like haul_item="" into the search
        # query text. A slot name we understand nowhere (neither this category's spec nor a
        # known kind/filter target) is noise too.
        if not str(answer.raw_answer or "").strip():
            continue
        known = (
            answer.slot_name in valid_slots
            or slot_value_kind(answer.slot_name) is not None
            or answer.slot_name in _SLOT_METADATA_FILTER_MAP
        )
        if not known:
            continue
        if _echoes_dropped_carried(
            state, answer.slot_name, normalize_answer_for_slot(category, answer.slot_name, answer.raw_answer)
        ):
            continue
        # The old cargo re-emitted as a slot answer for the NEW category's cargo slot is the
        # same history echo as extracted.haul_item — it must not answer a question never asked.
        if _echoes_cleared_on_change(state, "haul_item", answer.raw_answer):
            continue
        _store_slot_answer(state, category, answer.slot_name, answer.raw_answer)
        _mirror_optional_answer_as_feature(state, category, answer.slot_name, answer.raw_answer)
    # A question the LLM marked answered must advance even if it was vague/partial and
    # produced no parseable value (spec: loose answer -> null, never re-ask). But ONLY when
    # the message was actually about the pending question: seen live, the respond model asked
    # the wrong question ("what capacity?" while haul_item was pending), the customer answered
    # THAT ("2000 lbs"), the analyzer stamped answered_current_question=true - and this
    # fallback nulled haul_item, closing a question that was never asked and unlocking the
    # search. An answer that filled only OTHER, non-equivalent slots leaves the pending
    # question open to be asked again.
    pending = state.get("pending_question_slot")
    if pending and analysis.answered_current_question and pending not in state.get("slots", {}):
        answered_slots = {
            answer.slot_name for answer in analysis.slot_answers if str(answer.raw_answer or "").strip()
        }
        pending_family = {pending, *equivalent_slots(pending)}
        if not answered_slots or answered_slots & pending_family:
            _set_slot(state, pending, None)


def _fill_category_slot_aliases(state: dict[str, Any]) -> None:
    """Answer this category's questions with measurements the customer ALREADY gave us.

    Every category names the same measurements differently — Equipment asks for
    `haul_length_ft`, Livestock for `trailer_length_ft`, Car Hauler for
    `vehicle_length_ft` — but they are one fact, and search normalizes them to one
    `length_ft` target anyway. Qualification, though, was matching on the NAME: tell us
    "an equipment trailer, 20 ft" and the 20 landed in `trailer_length_ft`, leaving
    `haul_length_ft` empty, so we turned around and asked for a length we had just been
    given. This copies each known measurement into whatever name the current category asks
    by, for a category chosen mid-sentence and for one switched into later alike.

    A skipped slot stays skipped: they declined to answer it, and a number from elsewhere is
    not a change of heart.
    """
    category = state.get("category")
    if not category:
        return
    spec = get_trailer_fields(category)
    slots = state.setdefault("slots", {})
    skipped = state.get("skipped_slots", []) or []
    for slot in list(required_slots_for_state(state)) + list(spec.optional):
        if slot in slots or slot in skipped or not can_autofill_slot(slot):
            continue
        for sibling in equivalent_slots(slot):
            value = slots.get(sibling)
            source = state.get("slot_sources", {}).get(sibling, "user")
            if _is_number(value):
                _set_slot(state, slot, float(value), source)
                break
            # The cargo question is free text — "random things, wood to pipes to furniture"
            # answers Dump's haul_material just as it answered haul_item.
            if isinstance(value, str) and value.strip():
                _set_slot(state, slot, value, source)
                break


# Utility's weight qualification slot — the "what's the rough total weight?" question.
UTILITY_WEIGHT_SLOT = "haul_weight_lbs"


# Every slot that can already tell us how wide the trailer needs to be.
_WIDTH_SLOTS = (INJECTED_WIDTH_SLOT, "trailer_width_ft", "width_ft")


def _width_already_known(state: dict[str, Any]) -> bool:
    """True once the customer has told us a width — as a number, or as an explicit
    no-preference (a slot present with a null value)."""
    slots = state.get("slots", {}) or {}
    return any(key in slots for key in _WIDTH_SLOTS)


def _inject_width_question(state: dict[str, Any], haul: HaulClassification) -> None:
    # needs_width_question is a SIZE judgment: the cargo is large/wide/a vehicle, so we
    # need the trailer wide enough — inject the width question. (Independent of weight.)
    category = state.get("category")
    if not category or category in WIDTH_EXCLUDED_CATEGORIES:
        return
    injected = state.setdefault("injected_required_slots", [])
    if _width_already_known(state):
        # They already gave us a width ("20ft long, 6ft wide"). Asking "how wide is that item
        # or trailer?" anyway reads as if we weren't listening — and, worse, it keeps
        # qualification open on a question that is already answered, so the search never runs.
        if INJECTED_WIDTH_SLOT in injected:
            injected.remove(INJECTED_WIDTH_SLOT)
        return
    if haul.needs_width_question and INJECTED_WIDTH_SLOT not in injected:
        injected.append(INJECTED_WIDTH_SLOT)


def _skip_weight_for_lightweight(state: dict[str, Any], haul: HaulClassification) -> None:
    # is_lightweight_utility_load is a WEIGHT judgment: a Utility load the LLM judged
    # <= ~1500 lbs (golf cart, ATV, mower, kayak, ...). We already know it's light, so
    # don't ask the weight question — mark it skipped. The user can still volunteer a
    # weight later (that path clears the skip and records it), and search simply applies
    # no payload filter, which is correct: any utility trailer handles a light load.
    if state.get("category") != "Utility" or not haul.is_lightweight_utility_load:
        return
    if UTILITY_WEIGHT_SLOT not in state.get("slots", {}):
        _mark_skipped(state, UTILITY_WEIGHT_SLOT)


def _apply_skips_and_repeats(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    pending = state.get("pending_question_slot")
    if analysis.intent == "skip_current":
        _mark_skipped(state, pending)
        state["pending_question_slot"] = None
        state["pending_question_repeats"] = 0
    elif analysis.intent in {"skip_all_show_results", "show_more_results"}:
        if (state.get("turn_outcome") or {}).get("category_just_changed"):
            # The same message that asked for results also moved the category. The NEW
            # category's questions have not been asked yet — mass-skipping them here would
            # search with zero qualification. The change owns the turn; results wait until
            # the new category's questions are answered or the customer skips them THEN.
            return
        for slot in required_slots_for_state(state):
            if slot not in state.get("slots", {}):
                _mark_skipped(state, slot)
        state["qualification_complete"] = True
    elif pending and not analysis.answered_current_question and analysis.intent not in {"category_selection", "qualification_answer"}:
        if contact_ask_outstanding(state):
            # We interrupted them to ask for contact details, so of course they didn't answer
            # the qualification question. Charge that detour ONE strike, not one per turn —
            # otherwise our own detour skips their question out from under them.
            if state.get("contact_repeat_charged"):
                return
            state["contact_repeat_charged"] = True
        state["pending_question_repeats"] = int(state.get("pending_question_repeats", 0)) + 1
        if state["pending_question_repeats"] >= 2:
            _mark_skipped(state, pending)
            state["pending_question_slot"] = None
            state["pending_question_repeats"] = 0
            state["turn_outcome"].setdefault("system_email_triggers", []).append(
                {"kind": "unanswered_question", "description": f"Skipped unanswered question: {pending}"}
            )


def _apply_requirement_changes(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    if analysis.keep_fields_answer:
        # This message answered the keep/drop question, and that machinery already dropped
        # exactly the carried values the customer waved off. The analyzer routinely piles
        # MORE into this turn — seen live twice: intent=drop_requirements with no
        # dropped_fields (the blanket wipe below then destroyed every slot), and
        # dropped_fields listing trailer_length_ft when the 50 ft was the customer's OWN
        # requirement for the new category, stated in the change message and never offered
        # in the keep/drop question. On these turns the keep/drop answer is the ONLY
        # authority on what gets dropped.
        return
    for field in analysis.dropped_fields:
        state.get("slots", {}).pop(field, None)
        state.get("slot_sources", {}).pop(field, None)
    if analysis.intent == "drop_requirements" and not analysis.dropped_fields:
        state["slots"] = {}
        state["slot_sources"] = {}


def _search_inputs(state: dict[str, Any]) -> str:
    """Everything a Pinecone search is built from, as a comparable string."""
    return json.dumps(
        {
            "category": state.get("category"),
            "brand_preference": state.get("brand_preference"),
            "slots": state.get("slots", {}) or {},
            "skipped_slots": sorted(state.get("skipped_slots", []) or []),
        },
        sort_keys=True,
        default=str,
    )


def apply_analysis_to_state(state: dict[str, Any]) -> dict[str, Any]:
    analysis: TurnAnalysis = state["turn"]
    state["turn_outcome"] = {"canned_keys": [], "emails_sent": [], "system_email_triggers": []}
    search_inputs_before = _search_inputs(state)
    _apply_contact(state, analysis)
    haul = enforce_haul_classification_invariant(analysis.haul_classification)
    state["turn"] = analysis.model_copy(update={"haul_classification": haul})
    if not _apply_clarification(state, state["turn"]):
        _apply_category(state, state["turn"])
    _apply_extraction(state, state["turn"])
    _apply_aluminum_base_category(state)
    _fill_category_slot_aliases(state)
    _refresh_pending_change_dimensions(state)
    _inject_width_question(state, haul)
    _skip_weight_for_lightweight(state, haul)
    _apply_requirement_changes(state, state["turn"])
    _apply_skips_and_repeats(state, state["turn"])
    if _search_inputs(state) != search_inputs_before:
        # Something a search is built from moved, so the results on screen are stale. This
        # stays set until a search actually runs (the change may land several turns before
        # qualification finishes), and it is what keeps chat-only turns — a listing the
        # customer likes, their email address — from re-querying Pinecone for nothing.
        state["search_pending"] = True
    return state
