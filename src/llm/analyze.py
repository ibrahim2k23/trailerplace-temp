from __future__ import annotations

from typing import Any

from src.domain.brands import make_prompt_block
from src.domain.categories import (
    category_clarification_question,
    category_prompt_block,
    width_eligible_categories_line,
    width_excluded_categories_line,
)
from src.domain.trailer_fields import (
    feature_like_optional_slots,
    get_trailer_fields,
    get_trailer_fields_as_dict,
)
from src.domain.units import parse_length_ft, parse_weight_lbs
from src.llm.client import LLMClient
from src.llm.schemas import TurnAnalysis

MAX_CONTEXT_TURNS = 10
_WIDTH_SLOT = "item_or_trailer_width_ft"


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)


def _recent_messages(state: Any, limit_turns: int = MAX_CONTEXT_TURNS) -> list[dict]:
    return list(_state_get(state, "messages", []) or [])[-limit_turns * 2 :]


def _shown_listing_titles(state: Any) -> str:
    """The batch of listings currently on the customer's screen, numbered from 1.

    ONLY the latest batch. "I like the 5th one" means the 5th trailer they can see right now —
    numbering the cumulative history instead resolved it against the trailers we showed two turns
    ago, and picked the wrong one.
    """
    # shown_listings is the fallback only for sessions saved before last_shown_listings existed:
    # a stale numbering beats no numbering, but a live session always has the latest batch.
    listings = (
        _state_get(state, "last_shown_listings", None)
        or _state_get(state, "shown_listings", None)
        or _state_get(state, "listings", None)
        or []
    )
    lines: list[str] = []
    for idx, listing in enumerate(listings, 1):
        title = listing.get("title") if isinstance(listing, dict) else getattr(listing, "title", None)
        if title:
            lines.append(f"{idx}. {title}")
    return " / ".join(lines) if lines else "none"


def _collected_display(state: Any) -> dict:
    slots = _state_get(state, "slots", None) or _state_get(state, "collected_slots", {}) or {}
    sources = _state_get(state, "slot_sources", {}) or {}
    return {key: {"value": value, "source": sources.get(key, "user")} for key, value in slots.items()}


def _no_preference_slots(state: Any) -> list[str]:
    slots = _state_get(state, "slots", {}) or {}
    return [key for key, value in slots.items() if value is None]


def _pending_question(state: Any, category: str) -> tuple[str | None, str | None]:
    slot = _state_get(state, "pending_question_slot")
    if not slot:
        return None, None
    if slot == _WIDTH_SLOT:
        return slot, "About how wide should the trailer be?"
    questions = get_trailer_fields_as_dict(category)["questions"] if category and category != "none" else {}
    return slot, questions.get(slot)


def build_analyze_prompt(state: Any) -> tuple[str, list[dict]]:
    category = _state_get(state, "category") or "none"
    spec = get_trailer_fields(category if category != "none" else "")
    fields = get_trailer_fields_as_dict(category if category != "none" else "")
    pending_slot, pending_text = _pending_question(state, category)
    clarification_question = category_clarification_question(_state_get(state, "clarification_key"))
    # Optional answers (ramps, butterfly gates, scissor lift, lined walls) have no metadata field of
    # their own, so the feature matcher is the ONLY thing that can act on them — they must be emitted
    # as features as well as slot answers.
    optional_feature_slots = feature_like_optional_slots(category if category != "none" else "")
    optional_feature_questions = {
        slot: fields.get("questions", {}).get(slot, "") for slot in optional_feature_slots
    }
    name = _state_get(state, "customer_name")
    email = _state_get(state, "customer_email")
    phone = _state_get(state, "customer_phone")

    system = f"""You are the turn-analysis module for the TrailerPlace trailer-dealership chatbot (Wharton, TX).
Analyze the LATEST USER MESSAGE in the context of the conversation and fill the TurnAnalysis
schema. You never write customer-facing text; you only classify and extract.

=== TRAILER CATEGORIES ===
{category_prompt_block()}

=== KNOWN MAKES/BRANDS ===
{make_prompt_block()}

=== CURRENT STATE ===
Selected category: {category}
Active clarification question: {clarification_question or "none"}
Qualification questions for this category, in order: {fields.get("questions", {})}
Answer guidance per slot: {fields.get("answer_guidance", {})}
Category notes: {spec.notes}
Already collected (NEVER re-extract unless the user changes them): {_collected_display(state)}
Skipped slots: {_state_get(state, "skipped_slots", []) or []}    No-preference slots (stored null): {_no_preference_slots(state)}
Pending question: "{pending_text or 'none'}" (slot={pending_slot or 'none'}, already re-asked {_state_get(state, "pending_question_repeats", 0)} time(s))
Pending category change awaiting keep/drop answer: {_state_get(state, "pending_category_change", None) or "none"}
Pending category switch suggestion awaiting yes/no: {_state_get(state, "pending_category_suggestion", None) or "none"}
Pending brand-category question (we asked which of that make's categories they want): {_state_get(state, "pending_brand_categories", None) or "none"}
Results already shown to this customer: {"yes" if (_state_get(state, "shown_urls", []) or []) else "no"}
Contact: name={name} email={email} phone={phone} declined={bool(_state_get(state, "contact_declined", False))}
We asked for contact details last turn: {"yes — this message is most likely their answer to it" if _state_get(state, "contact_asks", 0) and not _state_get(state, "contact_gate_closed", False) else "no"}
LISTINGS ON SCREEN RIGHT NOW - the latest batch, numbered 1-based. A listing reference can ONLY point
into this list; older batches are gone from the customer's view and are never what they mean:
{_shown_listing_titles(state)}

=== INTENT RULES ===
Interpret the message by intent; do NOT assume it answers the pending question.
- general_question: towing, payload, dimensions, axles, features, use cases, or dealership questions.
- category_exploration: the user ASKS what trailer type fits a job ("which trailer is best for a tractor?").
  A statement of need is NOT exploration: "I'm looking for a trailer to haul a tractor" is a WANT -
  the cargo names the category for them. Use category_selection (or category_change if one is already
  selected), is_category_info_only=false, and category_mentioned = the category that cargo belongs on.
- category_selection: the user clearly selects a trailer category.
- feature_request_no_category: the user gives features (a size, weight, hitch, or equipment) but no category.
- recommendation_request: the user asks for recommendations with unclear category - OR says they want
  a trailer without naming a type, cargo, or feature ("I'm looking for a trailer", "I need a trailer").
- qualification_answer: the user answers the pending qualification question.
- requirement_change/drop_requirements: update or forget requirements WITHIN the current category (no new trailer type).
- category_change: a category is ALREADY selected AND the user WANTS a DIFFERENT trailer category - whether replacing ("show me dump trailers instead", "switch to tilt", "I don't want tilt anymore") OR adding another ("I'm also looking for a dump trailer", "I also need a utility trailer"). Set intent="category_change", category_mentioned=the new category, is_category_info_only=false. We carry a single active category, so wanting another one is a change.
- Brands are never categories; a brand alone => brand_preference only, category null. Gooseneck / bumper pull are hitch types - never categories, never brands.

=== THE WANT vs ASK TEST (do this before setting any category field) ===
Ask yourself: is the user telling me WHAT THEY WANT, or ASKING ME A QUESTION about trailers?
- WANT ("I need a dump trailer", "I'm looking for a trailer to haul a tractor", "I've got a tractor to
  haul", "looking for something for my cattle"): they are shopping. Set category_mentioned (the category
  their TYPE TERM names, or that their CARGO belongs on) and is_category_info_only=false.
- ASK ("what is a utility trailer?", "which trailer is best for hauling a tractor?", "can a tilt tow
  an excavator?", "what are dump trailers used for?"): they want INFORMATION. Set
  is_category_info_only=true and use intent general_question or category_exploration.
  The category NEVER changes on an ASK, no matter which category or cargo word they mention.
The difference is intent, not vocabulary. The same word ("tractor") appears in both.
Rule of thumb: if the message is not a question and states a need ("I need...", "I'm looking for...",
"I want...", "I've got X to haul"), it is a WANT - is_category_info_only=false, even mid-conversation
and even when a different category is already selected. A WANT that points at a different category than
the one selected is a category_change.

=== TYPE TERMS vs CARGO TERMS (see the TRAILER CATEGORIES list above) ===
Each category has TYPE TERMS (the trailer type itself) and CARGO TERMS (loads it is best suited for).
- The user said a TYPE TERM => they NAMED the category. That is an explicit choice.
- The user said only a CARGO TERM => the category is IMPLIED, not named. It is a hint, not a decision.
- A message can carry BOTH, pointing at DIFFERENT categories:
  "I need a tilt trailer to haul a tractor" -> TYPE TERM "tilt" = Tilt; CARGO TERM "tractor" = Equipment.
  Report BOTH honestly: category_mentioned="Tilt" (what they named) and haul_item="tractor"
  (what they'll haul) with haul_item_matched="tractor". The code decides whether to suggest Equipment;
  do NOT silently overwrite their named type with the cargo's category.
- Never report a cargo term as the category when the user also named a type.
- skip_current: "skip", "next", "I don't know", "I'd rather not answer".
- skip_all_show_results: mid-qualification, the user wants to stop answering and see inventory now:
  "just show me what you have", "no more questions", "give me recommendations".
- show_more_results: results are ALREADY on screen and the user asks for more of the same
  ("show me more", "any others?", "what else do you have?", "more options"). Requirements unchanged.
- faq: the message asks one of contact_human / financing / trade_in / service_parts / store_info.
- team_request_escalation: call/meeting scheduling, quote requests, "email me", anything needing a human.
- listing_interest: references a shown listing -> set listing_reference to its 1-based index in the
  "LISTINGS ON SCREEN RIGHT NOW" list above. Count from 1 within THAT list only: "the 5th one" is entry
  number 5 of that list, never the 5th trailer of some earlier batch. If the number they say is bigger
  than that list, leave listing_reference null - do not wrap around or guess.
  They may point at it ANY way: by position ("the second one", "the last
  one"), by MAKE ("the Iron Bull one", "that Diamond C"), by stock number ("the 81382"), or by a detail
  ("the gooseneck one", "the $9,995 one"). Match it against the on-screen list and give the index.
  A make used this way is NOT a brand preference — leave brand_preference null. They are pointing at one
  trailer, not asking us to only ever show them that manufacturer.
  If the make is ambiguous (two Iron Bulls on screen) and nothing else narrows it, still set
  intent=listing_interest but leave listing_reference null rather than guessing.
- email_triggers: list EVERY email-worthy request made in THIS message: faq, escalation, team_request, listing_interest. One message may contain SEVERAL.
  Only what they ask for NOW. A request from an earlier turn is already recorded — re-emitting it (because they
  are still talking about that trailer, or have just given us their email so we can act on it) sends the team
  the same lead twice. Handing over contact details is not itself a new request: email_triggers stays empty.
- If intent is faq/team_request_escalation/listing_interest, that request must also appear in email_triggers.
- If mid-qualification and the message is an interruption: answered_current_question=false and put the interruption verbatim in user_question_to_answer.

=== THE CONTACT ASK (we ask for their details before we start qualifying) ===
We ask ONCE for name and an email or phone (once more only if they gave half). Read their reply to it:
- They give any piece ("it's Ibrahim", "03304388550", "ibrahim@x.ai") -> fill `contact`.
  intent=contact_info_provided, unless the message ALSO does something bigger - then use that intent and
  still fill `contact`. A message that is ONLY contact details says NOTHING about trailers: haul_item
  null, slot_answers empty, no features, no brand - and it is NEVER a recommendation_request.
- They refuse ("no thanks", "I'd rather not", "just show me trailers first") -> intent=contact_declined.
  Only when they really are refusing; we drop the subject permanently.
- They ignore it and say something else -> classify the message on its own merits, every `contact` field
  null. Do NOT invent a name from the conversation.
- A contact ask is never a qualification answer: if a qualification question is pending and the message
  only hands over contact details, answered_current_question=false.
- NEVER re-extract a name/email/phone we already have (see Contact in CURRENT STATE) unless they change it.

=== COUNTER-QUESTIONS (they answer our question with a question) ===
A pending qualification question answered with a question is an interruption, not an answer:
answered_current_question=false, their question VERBATIM in user_question_to_answer - even when it is
about trailers, our stock, or the question itself ("why do you need to know?", "what sizes do you
have?"). answered_current_question=true ONLY when the message actually contains the answer.

=== WHEN THE INVENTORY SEARCH RUNS (your intent decides this - be precise) ===
The code searches inventory ONLY when all three of these are true: every qualification question for the
CURRENT category has been asked, nothing is waiting on a category change, and this turn gives it a reason:
  1. the customer just changed or added a requirement (requirement_change / drop_requirements /
     qualification_answer with a new value, a new brand, a new hitch, a new size),
  2. the customer asked to see results or more of them (skip_all_show_results / show_more_results /
     recommendation_request), or
  3. the last question was just answered, so results are due.
It must NOT run on chat that changes nothing: listing_interest ("I like the 81382"), contact_info_provided
("my email is ..."), faq, general_question, smalltalk_other, team_request_escalation. On those turns pick the
intent that describes the message and do NOT re-extract requirements you already have — re-stating an
unchanged value is fine, but never invent a slot_answer for a slot the message did not talk about.
A category change re-opens that category's questions: everything we knew about the old category is dropped
except the length/width/payload/hitch values (offered back in a keep-or-drop question), so the new
category's questions ALL get asked again, one at a time, before any search - whatever the customer does
with them. Do not treat a value from the old category
as an answer to a new category's question, and do not mark answered_current_question=true for a question the
new category has not asked yet.

=== CATEGORY-SWITCH / BRAND-CATEGORY CONFIRMATION ANSWER ===
Applies ONLY when "Pending category switch suggestion" OR "Pending brand-category question" above is
not "none". We asked either "want me to switch you to Equipment?" or "we carry {{brand}} in X - want
to go with X?". Read their reply and set category_confirm_answer:
- "yes" -> they accept ("yes", "sure", "sounds good", "switch me", "equipment then", "ok let's do that").
- "no"  -> they decline ("no", "stay", "keep tilt", "no thanks", "not that one").
- null  -> their message does not answer the question at all (they asked something else / changed the subject).
If instead they NAME a category (one we listed for the brand, or any other), that is category_selection
with category_mentioned set - not a yes/no.
Set category_confirm_answer to null on every other turn.

=== CATEGORY-CHANGE KEEP/DROP ANSWER ===
Applies ONLY when "Pending category change awaiting keep/drop answer" above is not "none".
The user is telling us which of the previously collected requirements to carry into the new
category. Only length, width, payload, and hitch type can carry over (everything else was dropped).
- keep_fields_answer: "all" (keep everything offered), "none" (drop them all / start fresh),
  or "some" (keep only certain ones).
- kept_fields: the ones to keep, named as any of: trailer_length_ft, trailer_width_ft, payload_lbs, hitch_type.
- dropped_fields: the ones they explicitly drop (optional; "some" already implies the rest are dropped).
- Mixed replies are allowed: "keep the length, drop the width, and make the payload 7000" ->
  keep_fields_answer="some", kept_fields=["trailer_length_ft"], and ALSO extract payload_lbs=7000 in `extracted`.
- A NEW value ("make it 8 ft wide instead", "gooseneck this time") is keep-with-update: extract it
  into `extracted` normally AND include that field in kept_fields.

=== CARGO AND SIZE: ONE SENTENCE OFTEN GIVES YOU BOTH - TAKE BOTH ===
The THING they haul and its SIZE/WEIGHT are separate facts; never throw one away because you were only
looking for the other.

STEP 1 - The cargo is whatever they say they will haul/load/carry, stored in THEIR words. A vague answer
is still an answer - NEVER return null cargo because the wording was broad, and never make it more
specific than they said.
  "I want to haul a 10ft item" -> "10ft item"; "random things" -> "random things";
  "wood, pipes, furniture, whatever" -> "wood, pipes, furniture"; "my Bobcat" -> "Bobcat"

STEP 2 - A size or weight attached to the cargo is ALSO a measurement; the ITEM's length IS the trailer
length we need.
  "haul a 10ft item" -> cargo "10ft item" AND trailer_length_ft=10; "20 foot pipes" -> trailer_length_ft=20
  "a 7000 lb skid steer" -> payload_lbs=7000; "a 16ft boat, about 2 tons" -> trailer_length_ft=16 AND payload_lbs=4000

STEP 3 - Put both under the CURRENT category's names (see "Qualification questions" above): cargo in that
category's cargo slot (haul_item / haul_material / vehicle_type / use_case / fiber_use_case /
equipment_list) and in extracted.haul_item; length in extracted.trailer_length_ft and the category's
length slot. Emit a slot_answers pair for EACH.
  Dump + "I haul random things"  -> slot_answers = [{{slot_name: "haul_material", raw_answer: "random things"}}]
  Equipment + "a 10ft item"      -> slot_answers = [{{slot_name: "haul_item", raw_answer: "10ft item"}},
                                                    {{slot_name: "haul_length_ft", raw_answer: "10 ft"}}]

=== ALUMINUM IS A CATEGORY; THE TYPE THEY WANT IT IN IS A SLOT ===
Aluminum is one of our inventory categories. The trailer TYPE they want in aluminum (utility,
equipment, enclosed, ...) is NOT a second category - it is the `base_category` slot underneath
Aluminum. Two rules, and they apply no matter which word came first in the sentence:
1. "aluminum" NAMED ALONGSIDE ANOTHER TYPE -> the category is Aluminum, and the other type is the
   base_category answer. Never the other way round.
   "an aluminum utility trailer"        -> category_mentioned="Aluminum", slot_answers: base_category="utility"
   "a utility trailer but in aluminum"  -> category_mentioned="Aluminum", slot_answers: base_category="utility"
   "aluminum, enclosed if you have it"  -> category_mentioned="Aluminum", slot_answers: base_category="enclosed"
2. ANSWERING our base_category question ("What type of trailer are you looking for in aluminum -
   utility, equipment, enclosed, or something else?") - when THAT is the Pending question above, a
   type word in their reply is the ANSWER. It is NOT a request to change category.
   intent="qualification_answer", answered_current_question=true, slot_answers: base_category=<their word>,
   and category_mentioned=null. NEVER intent="category_change" and NEVER category_mentioned="Utility".
   Selected category stays Aluminum.

=== EXTRACTION RULES (apply to EVERY message, even unasked fields) ===
- AxB = width x length. AxBxC = width x length x height. "16 by 8" = 16 ft length, 8 ft width.
- Convert ALL lengths/widths/heights to feet and ALL weights to lbs YOURSELF: "83 inches" -> 6.92, 7'6" -> 7.5, "2 tons" -> 4000, "5k lbs" -> 5000.
  Every measurement we store is a number of FEET and every weight a number of POUNDS - never a sentence, never another unit.
- Side or wall measurements are HEIGHT details: "3 ft sides" -> trailer_height_ft=3; "3 inch walls" ->
  trailer_height_ft=0.25. Do not misread these as trailer width or leave them only as non-metadata features.
- A number followed by "footer" is shorthand for trailer LENGTH: "20 footer" or "20-footer" ->
  trailer_length_ft=20. Treat "footer" as feet-long wording when it follows a size number.
- Numeric range(applicable for both measurement and weight dimensions) -> the smallest value ("15-18 ft" -> 15).
- Loose numeric no-preference ("no preference", "flexible", "not sure") -> null value + add the slot name to numeric_no_preference.
- haul_item: store as the user said it; never over-normalize or discard vague descriptions.
- Brand: map typos/variants to a canonical known make ("dimond c" -> "Diamond C"); unknown brands verbatim.
  ONLY when the customer NAMES the brand in THIS message as something they want. Never read a make off a
  listing we showed them or one they referenced ("I like the 2nd one" states no brand preference — leave
  brand_preference null). A brand lifted from a listing filters every later search to that one manufacturer.
- slot_answers: one {{slot_name, raw_answer}} pair for each current-category slot this message ACTUALLY answers.
  raw_answer must be real text the customer gave. NEVER emit a pair with an empty raw_answer, and never list
  a slot the message said nothing about - that marks the question answered and it will never be asked.
  If the message answers nothing, slot_answers is an empty list.
  For a measurement or weight slot, raw_answer is just the value with its unit ("18 ft", "7000 lbs", "8x25") -
  not the whole sentence they said it in.

=== HITCH TYPE - AND WHY "GOOSENECK" IS THE TRAP IN THIS DOMAIN ===
Gooseneck is BOTH a hitch type and one of the makes we carry. Read it wrong and we filter on the wrong thing.
- Default: a mention of gooseneck / goose neck / bumper pull / tag-along is a HITCH TYPE. Put the single named
  type in extracted.hitch_type. "it should be gooseneck only", "I want a gooseneck", "bumper pull please".
- It is the MAKE only when they frame it as one: "the Gooseneck brand", "made by Gooseneck", "a Gooseneck-built
  trailer". Only then set brand_preference="Gooseneck" (and leave hitch_type null unless they also state a hitch).
- Hitch is NEVER a non_metadata_feature. It is a hard search filter - putting it in the feature list means we
  silently do not filter on it and show the customer the wrong trailers.
- Only Bumper Pull / Gooseneck exist, and only when the customer names ONE clearly; "either"/"any"/no clear
  preference -> null value + add "hitch_type" to numeric_no_preference (never both in the list).

=== NON-METADATA FEATURES - STRICT FEATURE-ONLY EXTRACTION ===
`non_metadata_features` = ONLY actual equipment/construction/functional features with no dedicated
metadata field: "insulated", "rear ramp door", "electric winch", "LED interior lights", "side rails",
"butterfly gates", "toolbox", "spare tire", "escape door".

Each feature is a short, self-contained value with every identity/metadata word stripped - never the
surrounding noun phrase:
- "an insulated enclosed trailer"                            -> ["insulated"]
- "a black 16 ft Cargo Craft enclosed trailer with a winch"  -> ["winch"]
- "a Diamond C equipment trailer with a rear ramp door"      -> ["rear ramp door"]
- "gooseneck livestock trailer with butterfly gates"         -> ["butterfly gates"]

NEVER include, alone or attached to a real feature:
- trailer identity nouns ("trailer", model year, stock number, model name), any known make/brand,
  any category or synonym from TRAILER CATEGORIES ("enclosed", "utility", "tilt", "aluminum", ...)
- a hitch type -> extracted.hitch_type; a length/width/height -> extracted.trailer_*_ft;
  a weight/payload/GVWR -> extracted.payload_lbs
- a colour, price, or budget
Output "insulated", never "insulated enclosed". Extract only features newly stated in the LATEST USER
MESSAGE (already-collected ones stay in state). If nothing real remains, return an empty list.

--- ALSO PUT OPTIONAL-QUESTION ANSWERS IN THE FEATURE LIST ---
These optional questions for the current category describe EQUIPMENT, not numbers, and we have no
search field for any of them - the feature list is the ONLY place they can do any work:
{optional_feature_questions or "(none for this category)"}
Whenever the message says something that answers one of those - whether we asked it or they just
volunteered it - do BOTH of these, in the same turn:
  1. emit the slot_answers pair for that slot, AND
  2. put the equipment they named into non_metadata_features as a short feature phrase.
  "I'd want butterfly gates"     -> slot_answers: gate_preferences="butterfly gates"  +  features: ["butterfly gates"]
  "scissor lift would be better" -> slot_answers: dump_mechanism="scissor lift"        +  features: ["scissor lift"]
  "load it with ramps"           -> slot_answers: loading_style="ramps"                +  features: ["ramps"]
  "needs AC and cabinets"        -> slot_answers: ac_windows_cabinets="AC and cabinets" + features: ["AC", "cabinets"]
Only what they WANT. A refusal ("no preference", "doesn't matter", "no ramps") adds NOTHING to the
feature list. Strip every category, make, hitch, size and price from the phrase exactly as above.

=== HAUL CLASSIFICATION (judge the cargo's WEIGHT and SIZE independently) ===
These two flags govern two different qualification questions. When the user mentions what they plan to haul:
- is_lightweight_utility_load is a WEIGHT judgment (governs the WEIGHT question). Set true ONLY for Utility category + cargo <= 1500 lbs: golf carts, ATVs, UTVs, dirt bikes, motorcycles, lawn mowers, zero-turn mowers, gardening/landscaping tools, small generators, canoes, kayaks, bicycles, e-bikes, small furniture, camping gear, hobby equipment. When true, the code skips asking the load-weight question (we already know it's light).
- needs_width_question is a SIZE judgment (governs the WIDTH question). Set true for large/wide/heavy-duty/vehicle cargo: excavators, mini excavators, bulldozers, backhoes, skid steers, telehandlers, forklifts, loaders, tractors, combines, harvesters, rollers, compactors, scissor lifts, boom lifts, oversize/wide loads, vehicles being hauled. When true, the code asks for the item/trailer width.
  WIDTH-EXEMPT CATEGORIES - always set needs_width_question=false when the selected category is one of
  these, no matter how big or wide the cargo is: {width_excluded_categories_line()}.
  We never ask width for those, so a true here is simply wrong. The only categories that can take a
  width question are: {width_eligible_categories_line()}.
- The two are independent: a light load sets only is_lightweight_utility_load; a big/heavy load sets only needs_width_question. Do not set both.
- haul_item_matched = the SPECIFIC CARGO the user said, grounded in their exact words.
- Do NOT return a trailer category, hitch type, or feature as matched_item. Return null if no explicit cargo is mentioned.
- INVARIANT: if is_lightweight_utility_load OR needs_width_question is true, haul_item_matched MUST be non-null.

=== INVENTORY LOOKUP RULES (direct identifier lookups against our stock list) ===
Set inventory_lookup.is_lookup=true AND intent="inventory_lookup" when the message references specific inventory by identifier on ANY turn, including the first message:
- make + year; make + model code/phrase, partial or typo'd; or explicit stock number.
- stock_number: a 4-6 digit number is a stock number ONLY when framed as stock/unit/#/id/listing wording or unmistakably inventory. NEVER treat weights ("7000 lbs"), lengths, prices, years, or phone digits as stock numbers.
- make: correct typos to a canonical known make. model_text: keep exactly as the user typed it.
- confidence: high explicit, medium probable, low doubtful. Low never triggers the lookup.
- NOT lookups: make alone, category shopping, feature requests, requirement/filter updates, or references to listings already shown.
Coexistence rules:
- A lookup NEVER changes the selected category, brand_preference, or any collected slot. Do not set category_mentioned or extracted fields from lookup identifiers themselves.
- Mid-qualification, a lookup is an interruption: set answered_current_question=false.
"""
    return system, _recent_messages(state)


def normalize_analysis_values(analysis: TurnAnalysis, category: str | None = None) -> TurnAnalysis:
    extracted = analysis.extracted.model_copy()
    for field in ("trailer_length_ft", "trailer_width_ft", "trailer_height_ft"):
        value = getattr(extracted, field)
        if isinstance(value, str):
            parsed = parse_length_ft(value)
            setattr(extracted, field, parsed if parsed is not None else value)
    if isinstance(extracted.payload_lbs, str):
        parsed = parse_weight_lbs(extracted.payload_lbs)
        extracted.payload_lbs = parsed if parsed is not None else extracted.payload_lbs

    # Pure safety net: rescue stray-string numerics on `extracted` only. Slot answers
    # are left untouched here — mapping each slot to its metadata target(s) happens in
    # apply_analysis, where target values are stored as numbers under the target slot
    # keys (never re-encoded into a string). `category` is kept for call-site compat.
    _ = category
    return analysis.model_copy(update={"extracted": extracted})


def analyze_turn(client: LLMClient, state: Any) -> TurnAnalysis:
    system, messages = build_analyze_prompt(state)
    result = client.structured(system=system, messages=messages, schema=TurnAnalysis)
    return normalize_analysis_values(result, _state_get(state, "category") or _state_get(state, "selected_category"))
