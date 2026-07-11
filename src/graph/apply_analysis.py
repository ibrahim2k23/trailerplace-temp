from __future__ import annotations

from typing import Any

from src import conversation_store
from src.domain.categories import (
    category_clarification_question,
    resolve_category_clarification_answer,
    resolve_category_from_text,
    resolve_category_matches,
)
from src.domain.defaults import defaults_for
from src.domain.normalizer import normalize_category
from src.domain.slot_map import (
    _SLOT_METADATA_FILTER_MAP,
    is_recognized_slot_value,
    normalize_answer_for_slot,
    normalize_slot_value,
    slot_value_kind,
)
from src.domain.trailer_fields import get_trailer_fields
from src.domain.units import parse_dimensions
from src.llm.schemas import HaulClassification, TurnAnalysis

WIDTH_EXCLUDED_CATEGORIES = {"Roll Off", "Enclosed", "Fiber", "Race Trailer", "Diesel Tank"}
INJECTED_WIDTH_SLOT = "item_or_trailer_width_ft"
INJECTED_WIDTH_QUESTION = "About how wide is that item or trailer you need to haul?"


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
    if analysis.intent == "contact_declined":
        state["contact_declined"] = True
        state["contact_followup_pending"] = None
        return
    if contact.name:
        state["customer_name"] = contact.name
    if contact.email:
        state["customer_email"] = contact.email
    if contact.phone:
        state["customer_phone"] = contact.phone
    if contact.name or contact.email or contact.phone:
        state["contact_followup_pending"] = None
        conversation_store.update_lead_contact(
            session_id=state["session_id"],
            full_name=state.get("customer_name"),
            email=state.get("customer_email"),
            phone=state.get("customer_phone"),
        )


def _current_user_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", []) or []):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def _start_category(state: dict[str, Any], category: str) -> None:
    state["category"] = category
    state["qualification_complete"] = False
    conversation_store.update_lead_item_of_interest(state["session_id"], category)
    for key, value in defaults_for(category).items():
        _set_slot(state, key, value, "default")


# On a category change we drop every collected feature EXCEPT these three measurements
# (length, width, payload). Each canonical dimension may live under any of the listed
# slot keys depending on how it was originally captured (direct extraction vs. a slot
# answer that maps to a metadata target).
_DIMENSION_SLOT_KEYS: dict[str, tuple[str, ...]] = {
    "length": ("trailer_length_ft", "length_ft"),
    "width": ("trailer_width_ft", "width_ft", "item_or_trailer_width_ft"),
    "payload": ("payload_lbs", "haul_weight_lbs", "payload_need"),
}
# When re-seeding a kept dimension into the new category, write it under every key that
# search's metadata-filter map reads, so the carried value actually filters results.
_DIMENSION_TARGET_SLOTS: dict[str, tuple[str, ...]] = {
    "length": ("trailer_length_ft",),
    "width": ("trailer_width_ft",),
    "payload": ("payload_lbs", "haul_weight_lbs", "payload_need"),
}


def _carried_dimensions(slots: dict[str, Any]) -> dict[str, float]:
    """The length/width/payload values currently held, keyed by canonical dimension."""
    carried: dict[str, float] = {}
    for dim, keys in _DIMENSION_SLOT_KEYS.items():
        for key in keys:
            if _is_number(slots.get(key)):
                carried[dim] = float(slots[key])
                break
    return carried


def _kept_dimension_names(kept_fields: list[str]) -> set[str]:
    """Map the LLM's kept_fields (canonical names or raw slot keys) to dimension names."""
    names: set[str] = set()
    for field in kept_fields:
        token = (field or "").strip().lower()
        for dim, keys in _DIMENSION_SLOT_KEYS.items():
            if token == dim or token in keys:
                names.add(dim)
    return names


def _switch_category(state: dict[str, Any], new_category: str, kept_dims: dict[str, float]) -> None:
    """Move to ``new_category``, dropping every feature except the kept measurements.

    Metadata slots, non-metadata features, brand, hitch, haul item, injected questions
    and skips are all cleared; only the length/width/payload the user chose to keep carry
    over. Qualification then restarts for the new category.
    """
    state["category"] = new_category
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
    for key, value in defaults_for(new_category).items():
        _set_slot(state, key, value, "default")
    for dim, value in kept_dims.items():
        for key in _DIMENSION_TARGET_SLOTS[dim]:
            _set_slot(state, key, value)
    conversation_store.update_lead_item_of_interest(state["session_id"], new_category)


def _begin_category_change(state: dict[str, Any], new_category: str) -> None:
    """Switch to ``new_category``, pausing to ask which measurements to carry over."""
    carried = _carried_dimensions(state.get("slots", {}))
    state["pending_category_suggestion"] = None
    if carried:
        state["pending_category_change"] = {"new_category": new_category, "dimensions": carried}
    else:
        _switch_category(state, new_category, {})


# Intents where the user is actually shopping (so a category may move). Everything else -
# general_question, category_exploration, faq, inventory_lookup, smalltalk, ... - is a
# question ABOUT trailers and must never move the category.
_CATEGORY_ACTION_INTENTS = {
    "category_selection",
    "category_change",
    "feature_request_no_category",
    "recommendation_request",
    "qualification_answer",
    "requirement_change",
}


def _wants_category_action(analysis: TurnAnalysis) -> bool:
    """True only when the message expresses what the user WANTS, not what they're asking about.

    "Which trailer is best for hauling a tractor?" is information - the category never
    moves. "I need something to haul a tractor" is a want - it may.
    """
    if analysis.is_category_info_only:
        return False
    return analysis.intent in _CATEGORY_ACTION_INTENTS


def _named_and_implied_categories(state: dict[str, Any], analysis: TurnAnalysis) -> tuple[str | None, str | None]:
    """Split this message's category signals into an explicit choice and an implied one.

    "named"   -> the user said the trailer TYPE ("a tilt trailer").
    "implied" -> the user only said a cargo/task term that maps to a category ("a tractor").
    A message can carry both ("a tilt trailer to haul a tractor"), which is exactly the
    case where we must ask rather than assume.
    """
    named: str | None = None
    implied: str | None = None
    for category, tier in resolve_category_matches(_current_user_text(state)):
        if tier == "naming" and named is None:
            named = category
        elif tier == "cargo" and implied is None:
            implied = category
    # Fall back to the LLM's category only when it is not just restating the cargo
    # implication (it catches typos/paraphrases the term lists miss).
    if named is None and analysis.category_mentioned and analysis.category_mentioned != implied:
        named = analysis.category_mentioned
    return named, implied


def _suggest_category_switch(state: dict[str, Any], suggested: str, analysis: TurnAnalysis) -> None:
    state["pending_category_suggestion"] = {
        "suggested_category": suggested,
        "from_category": state.get("category"),
        "cargo": analysis.haul_classification.haul_item_matched or analysis.extracted.haul_item,
    }


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
    if not state.get("category") and not analysis.is_category_info_only:
        resolution = resolve_category_from_text(text)
        if resolution.needs_clarification:
            state["clarification_key"] = resolution.clarification_key
            state["turn_outcome"]["clarification_question"] = resolution.clarification_question
            return True
    return False


def _apply_category(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    # (a) Answer to a pending "should we switch you to X?" suggestion.
    suggestion = state.get("pending_category_suggestion")
    if suggestion:
        if analysis.category_confirm_answer == "yes":
            # Already spent a turn asking; carry the measurements over without a second question.
            _switch_category(state, suggestion["suggested_category"], _carried_dimensions(state.get("slots", {})))
            return
        # "no", or they moved on without answering -> stay put and never re-ask.
        state["pending_category_suggestion"] = None
        if analysis.category_confirm_answer == "no":
            return

    # (b) Answer to a pending keep/drop question from a category change.
    if state.get("pending_category_change") and analysis.keep_fields_answer:
        change = state["pending_category_change"]
        offered = change.get("dimensions", {})
        if analysis.keep_fields_answer == "all":
            kept = dict(offered)
        elif analysis.keep_fields_answer == "some":
            keep_names = _kept_dimension_names(analysis.kept_fields)
            kept = {dim: value for dim, value in offered.items() if dim in keep_names}
        else:  # "none"
            kept = {}
        _switch_category(state, change["new_category"], kept)
        return

    # An informational question never moves the category, no matter what it mentions.
    if not _wants_category_action(analysis):
        return

    named, implied = _named_and_implied_categories(state, analysis)
    current = state.get("category")

    # Rule 1: nothing chosen yet -> adopt whatever they named, or what their cargo implies.
    if not current:
        target = named or implied
        if not target:
            return
        _start_category(state, target)
        # "a tilt trailer to haul a tractor": take them at their word (Tilt), but a tractor
        # belongs on an Equipment trailer — offer the switch instead of silently overriding.
        if named and implied and implied != named:
            _suggest_category_switch(state, implied, analysis)
        return

    # Rule 3: results already shown -> a category OR cargo term switches outright,
    # pausing only to ask which measurements to carry over.
    if state.get("shown_urls"):
        target = named or implied
        if target and target != current:
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
    for target_key in _SLOT_METADATA_FILTER_MAP.get(slot_name, ()):
        parsed = normalize_slot_value(category or "", target_key, raw_answer)
        if _is_number(parsed):
            _set_slot(state, target_key, float(parsed))
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


def _apply_extraction(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    category = state.get("category") or ""
    if analysis.extracted.brand_preference:
        state["brand_preference"] = analysis.extracted.brand_preference
    for feature in analysis.extracted.non_metadata_features:
        if feature not in state.setdefault("non_metadata_features", []):
            state["non_metadata_features"].append(feature)
    for key in analysis.extracted.numeric_no_preference:
        _set_slot(state, key, None)
    if analysis.extracted.haul_item:
        _set_slot(state, "haul_item", analysis.extracted.haul_item)
    if analysis.extracted.hitch_type:
        # A single named type is a real preference; "both" (the model's way of saying
        # "either is fine") is not one — treat it the same as no preference (null).
        hitch = analysis.extracted.hitch_type
        _set_slot(state, "hitch_type", list(hitch) if len(hitch) == 1 else None)
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
        if value is not None and (key in valid_slots or key.startswith("trailer_")):
            _set_slot(state, key, value)
    for answer in analysis.slot_answers:
        _store_slot_answer(state, category, answer.slot_name, answer.raw_answer)
    # A question the LLM marked answered must advance even if it was vague/partial and
    # produced no parseable value (spec: loose answer -> null, never re-ask).
    pending = state.get("pending_question_slot")
    if pending and analysis.answered_current_question and pending not in state.get("slots", {}):
        _set_slot(state, pending, None)


# Utility's weight qualification slot — the "what's the rough total weight?" question.
UTILITY_WEIGHT_SLOT = "haul_weight_lbs"


def _inject_width_question(state: dict[str, Any], haul: HaulClassification) -> None:
    # needs_width_question is a SIZE judgment: the cargo is large/wide/a vehicle, so we
    # need the trailer wide enough — inject the width question. (Independent of weight.)
    category = state.get("category")
    if not category or category in WIDTH_EXCLUDED_CATEGORIES:
        return
    if haul.needs_width_question and INJECTED_WIDTH_SLOT not in state.setdefault("injected_required_slots", []):
        state["injected_required_slots"].append(INJECTED_WIDTH_SLOT)


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
    elif analysis.intent == "skip_all_show_results":
        for slot in required_slots_for_state(state):
            if slot not in state.get("slots", {}):
                _mark_skipped(state, slot)
        state["qualification_complete"] = True
    elif pending and not analysis.answered_current_question and analysis.intent not in {"category_selection", "qualification_answer"}:
        state["pending_question_repeats"] = int(state.get("pending_question_repeats", 0)) + 1
        if state["pending_question_repeats"] >= 2:
            _mark_skipped(state, pending)
            state["pending_question_slot"] = None
            state["pending_question_repeats"] = 0
            state["turn_outcome"].setdefault("system_email_triggers", []).append(
                {"kind": "unanswered_question", "description": f"Skipped unanswered question: {pending}"}
            )


def _apply_requirement_changes(state: dict[str, Any], analysis: TurnAnalysis) -> None:
    for field in analysis.dropped_fields:
        state.get("slots", {}).pop(field, None)
        state.get("slot_sources", {}).pop(field, None)
    if analysis.intent == "drop_requirements" and not analysis.dropped_fields:
        state["slots"] = {}
        state["slot_sources"] = {}


def apply_analysis_to_state(state: dict[str, Any]) -> dict[str, Any]:
    analysis: TurnAnalysis = state["turn"]
    state["turn_outcome"] = {"canned_keys": [], "emails_sent": [], "system_email_triggers": []}
    _apply_contact(state, analysis)
    haul = enforce_haul_classification_invariant(analysis.haul_classification)
    state["turn"] = analysis.model_copy(update={"haul_classification": haul})
    if not _apply_clarification(state, state["turn"]):
        _apply_category(state, state["turn"])
    _apply_extraction(state, state["turn"])
    _inject_width_question(state, haul)
    _skip_weight_for_lightweight(state, haul)
    _apply_requirement_changes(state, state["turn"])
    _apply_skips_and_repeats(state, state["turn"])
    return state
