# Prompt & LLM Behavior Audit — TrailerPlace (branch v4.1)

Scope: every place a prompt is built, transformed, or sent to an LLM. Audited ~30 call
sites across `src/chatbot/service.py` (13), `graph.py` (13), `inventory_matcher.py` (3),
`mini_llm_classifier.py`, `mini_preference_classifier.py`, `email_reply.py`, plus the
prompt templates in `prompts.py` and the category/synonym data in `categories.py`.

This is an audit only — **no fixes applied.** Findings are ordered by severity. Each is
written for an AI-reliability reader: how a probabilistic model actually misbehaves here.

## Architecture note (applies to most findings)

The bot is **LLM-first with deterministic overrides**. Nearly every call is
`ChatOpenAI(model=OPENAI_MODEL|gpt-4o-mini, temperature=0).with_structured_output(..., method="function_calling")`
wrapped in `try/except` that falls back to a heuristic. Two systemic properties dominate:

- **Structured output via `function_calling` on gpt-4o-mini** enforces shape but *not*
  semantics — enums, "return null when unsure," and cross-field consistency are advisory.
- **A dense stack of single-purpose classifiers** (routing, contact, catalogue, confusion,
  adjudication, framing) each make one LLM decision. Their **precedence is encoded in Python
  control flow, not in any prompt**, so the model never sees the hierarchy it's part of.

---

# 1. Deterministic category hint silently overrides the LLM planner

## Severity
Critical

## Category
Instruction Hierarchy / Orchestration

## Location
- `src/chatbot/graph.py` — `_apply_mind_node`, lines 4744, 4796–4799
- feeds from `src/chatbot/categories.py` — `_direct_category_from_text`, `resolve_category_from_text`

## Current Prompt
`MIND_SYSTEM_PROMPT` (prompts.py, CATEGORY_RULES) instructs the model: *"When the latest
message names a canonical category or synonym, you MUST set `trailer_category` = that
canonical category… category_confidence=high."* But in code, after the LLM returns:
```python
if deterministic_hint:
    decision.trailer_category = deterministic_hint      # from resolve_category_from_text()
    decision.category_resolution_kind = "explicit"
    decision.category_confidence = "high"
```

## Problem
The LLM's `trailer_category` is **discarded whenever the deterministic resolver returns
anything**. The resolver (`_direct_category_from_text`) returns the *first* category by dict
order whose synonym appears anywhere in the text, so a cargo word outranks an explicitly
named type. The override then stamps the wrong category as `explicit/high`, suppressing any
downstream reconsideration.

## LLM Failure Mode
Not a model failure — an orchestration failure that *defeats* a correct model output. The
prompt tells the model it owns category resolution; the code makes that a no-op. The two
disagree silently, with no reconciliation or confidence comparison.

## Example Failure
"I want a **tilt** trailer to haul **tractor**" → resolver returns `Equipment` (tractor is an
Equipment synonym, ordered before Tilt) → code overrides the LLM's `Tilt` with `Equipment`,
confidence `high`. Verified live: category resolved to Equipment; the customer is put on the
Equipment qualification track and shown Equipment inventory. Also: "dump trailer for my
motorcycle" → Utility; "livestock trailer and a lawn mower" → Utility.

## Why This Happens
A heuristic is treated as ground truth over the LLM, and the heuristic itself has no ranking
logic. Classic "deterministic guard that is wrong more confidently than the model."

## Recommended Improvement
Make the hint *advisory*: pass it into context (already done) and let the LLM decide, or only
override when the LLM returns no category. If keeping a hard override, first fix the resolver
(Issue #2) and only override when resolver and LLM **agree**; when they disagree, prefer the
explicitly named type and lower confidence to trigger a confirmation.

## Expected Benefit
Removes a whole class of wrong-category conversations for the most common phrasing pattern
("a X trailer to haul Y"). Restores the prompt's stated authority.

## Priority
P0

---

# 2. Category resolver returns first match by dict order, not by salience

## Severity
Critical

## Category
Context / RAG (signal feeding the planner)

## Location
- `src/chatbot/categories.py` — `_direct_category_from_text` (100–105), `resolve_categories_from_text` (108–114)

## Current Prompt
Deterministic function (no LLM), but it produces the `deterministic_category_hint` injected
into the planner context and the override in Issue #1:
```python
for category, terms in _SYNONYMS.items():
    for term in terms:
        if re.search(term, text): return category   # first hit wins
```

## Problem
Iteration order = `_SYNONYMS` insertion order. Whichever category is defined earlier and has
*any* matching term wins, regardless of (a) whether the user *named* a type vs merely named
cargo, or (b) where the term sits in the sentence.

## LLM Failure Mode
Poisons the context the planner sees and drives the override. The model is handed a confident
wrong "hint," and — even absent the override — a wrong hint biases a probabilistic model.

## Example Failure
`resolve_categories_from_text("tilt trailer to haul tractor")` → `['Equipment','Tilt']`; the
singular resolver takes `Equipment`. 4 of 6 natural "type + cargo" phrasings misresolve.

## Why This Happens
No distinction between **naming terms** (the category word itself: "tilt", "dump") and
**cargo/brand terms** ("tractor", "Galyean"); no positional tiebreak.

## Recommended Improvement
Split each category's terms into `naming_terms` vs `cargo_terms`. Rank: naming-term match >
cargo-term match; within a tier, earliest text position wins. Return all candidates with
scores so the planner can disambiguate multi-category messages.

## Expected Benefit
Correct deterministic hint → correct planner behavior; also fixes the several test failures
(`assert 'Utility' == 'Livestock'`, etc.) that were flagging this.

## Priority
P0

---

# 3. Catalogue advertised to customers ≠ categories the system supports

## Severity
High

## Category
Hallucination / Consistency

## Location
- `src/chatbot/prompts.py` — `KNOWLEDGE` (line ~21), `CATEGORY_QUESTION_OWNERSHIP` examples (~96), `PRODUCT_INFO_RULES`
- vs `src/chatbot/categories.py` — `CANONICAL_CATEGORIES` (7–20), `category_prompt_block()` (151)
- vs `trailer_fields.py` — `_SPECS`

## Current Prompt
`KNOWLEDGE` lists **10** types ("Utility, Dump, Equipment, Flatbed, Car Hauler, Livestock,
Enclosed, Tilt, Roll-Off, Aluminum"). `category_prompt_block()` (injected into the same
system prompt further down) enumerates **13** canonical categories including Fiber, Race
Trailer, Diesel Tank. `trailer_fields.py` defines a **Welding** spec that is not canonical.

## Problem
Within a single assembled system prompt the model receives **two different catalogues**. When
asked "what do you carry?" it may answer from the 10-item KNOWLEDGE list or the 13-item
synonym block on different turns. Fiber / Race / Diesel are supported but never advertised;
Welding is asked about via an unreachable spec.

## LLM Failure Mode
Inconsistent catalogue answers turn-to-turn; omission of real product lines; potential to
"remember" a type from the synonym block and offer it, then deny it elsewhere. Contradictory
context is a known driver of self-inconsistency.

## Example Failure
Turn 1 "what trailers do you have?" → 10 types. Turn 5 user says "diesel tank trailer" → bot
happily qualifies it (it's canonical), contradicting the earlier catalogue that omitted it.

## Why This Happens
Three sources of truth for "what we sell," hand-maintained separately.

## Recommended Improvement
Derive the advertised list from `CANONICAL_CATEGORIES` (single source), or explicitly document
which canonical categories are "advertisable." Remove or promote the Welding spec so specs and
canonical set match.

## Expected Benefit
Consistent catalogue answers; no hidden/again unreachable product lines.

## Priority
P1

---

# 4. Width-question exclusion list differs between the classifier prompt and the code gate

## Severity
High

## Category
Consistency / Instruction Hierarchy

## Location
- `src/chatbot/mini_llm_classifier.py` — `classify_haul_requirements` system prompt, lines 155–157
- `src/chatbot/graph.py` — `_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES`, line 1766

## Current Prompt
Prompt: *"For categories except **Utility, Enclosed, and Flatbed**, mark needs_width_question
when the item is very large/wide/heavy… Do not request a width question for Utility, Enclosed,
or Flatbed."* Code gate excludes `{utility, enclosed, livestock, aluminum, flatbed, dump}`.

## Problem
The model is told width applies to Livestock/Aluminum/Dump (not in its exclusion list), so it
returns `needs_width_question=true` for them — then the code silently drops it. The
authoritative list lives in two places that disagree.

## LLM Failure Mode
Wasted/contradicted model output; the classifier "believes" it asked for width but the flow
never surfaces it. If the code list changes, drift widens. Debuggers see the model say "width
needed" while the bot never asks.

## Example Failure
"Livestock trailer for a bull, needs to be wide" → classifier sets width=true (allowed by its
prompt) → code drops it (Livestock excluded). The width intent is lost with no trace to the user.

## Why This Happens
Two hand-maintained exclusion lists; prompt and gate never reconciled.

## Recommended Improvement
Single source of truth: generate the classifier prompt's exclusion clause from
`_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES` (or vice-versa). Also revisit whether the list is
intentional — Flatbed/Dump are excluded but Tilt/Equipment aren't, which is itself suspicious.

## Expected Benefit
Model output and code agree; width behavior becomes predictable and testable.

## Priority
P1

---

# 5. Planner context is bloated with raw listings, full field list, and redundant hints

## Severity
High

## Category
Context / Prompt Bloat

## Location
- `src/chatbot/graph.py` — `_apply_mind_node` context dict, lines 4746–4768; invocation 4770–4780

## Current Prompt
Every planner turn serializes to JSON: `slots_collected`, `metadata_filters_collected`,
`pending_questions`, `asked_questions`, `pending_category_change/suggestion`, **`last_listings`
(full objects)**, `already_shown_listing_urls`, **`listing_model_fields` (all TrailerListing
keys)**, `deterministic_category_hint`, and `recent_messages[-8:]`.

## Problem
Large, partly irrelevant payload on the most important call. `last_listings` as full objects
and the entire model field list add tokens and distraction; `deterministic_category_hint` is
in-context *and* force-applied afterward (redundant). Long JSON context degrades instruction
adherence and raises latency/cost on the hot path.

## LLM Failure Mode
Distraction/goal drift: the planner keys off stale `last_listings` when the user has moved on,
or invents behavior around `listing_model_fields`. Big context reliably reduces adherence to
the (very long) system prompt above it.

## Example Failure
After results were shown, user says "actually I need a dump trailer." Planner, seeing full
`last_listings`, treats it as a listing follow-up rather than a category switch.

## Why This Happens
"Dump the whole state in" is easy; pruning to task-relevant context is not.

## Recommended Improvement
Send compact projections: listing *titles + urls* only (not full objects), drop
`listing_model_fields`, drop the redundant `deterministic_category_hint` if the override
stays. Cap `recent_messages` content length. Consider a stable field order.

## Expected Benefit
Better instruction adherence to `MIND_SYSTEM_PROMPT`, lower latency/cost, fewer stale-context
errors.

## Priority
P1

---

# 6. Pinecone auditor makes an absolute "exact match" claim gated on a magic count

## Severity
High

## Category
Hallucination / Structured Output

## Location
- `src/chatbot/graph.py` — `_pinecone_match_audit` system prompt, lines 1436–1489 (Scenarios A/B/C, 1464–1483)

## Current Prompt
Intro text is chosen by `full_match_count`: `=0` → "don't have that exact combination";
`1–5` → "confirmed fits first"; **`=6` → "Every trailer below is a confirmed match for exactly
what you're looking for."** The auditor is also **dimension-blind** — length/width/payload/GVWR
are stripped from its context ("Never infer, evaluate, mention… them").

## Spec validation (langgraph_rules_vs_excel.md)
⚠️ **Partially validated — reframed.** The spec (§ "How The Search Works", lines 381–389)
*justifies* the dimension-blindness: it says the search deliberately leans on only a few
reliable fields "because not every trailer record has every field filled out well enough to
rely on it." So stripping dimensions from the auditor is defensible per spec. **However**, the
spec still names **length** as a primary search detail ("leans most on … length, if clearly
given"), so a customer-facing *"exact match for exactly what you asked"* that ignores length
conflicts with the spec. The fixable defect is therefore narrower than originally written: the
**absolute wording** and the **magic `==6` trigger**, not the dimension-blindness itself.

## Problem
Two issues, narrowed after spec validation. (a) Scenario C is a strong, unqualified factual
guarantee ("confirmed match for exactly what you asked") triggered solely by the integer `6`
(tied to max recommendations). If that constant differs, Scenario C never fires or misfires.
(b) Because the auditor is dimension-blind (spec-justified), that absolute wording can still be
asserted for a trailer that is the wrong length the customer explicitly requested — the
dimension check happens elsewhere and isn't reconciled into the customer-facing claim. Note:
the dimension-blindness itself is intended per spec; the problem is letting an unqualified
"exact match" claim ride on top of it.

## LLM Failure Mode
Overconfidence / factual overreach: the bot tells a customer every result is an exact match
while silently ignoring the dimensions they asked for. High trust-damage and potential
liability.

## Example Failure
Customer: "16 ft livestock trailer with a slide gate." Six results come back, all slide-gate
livestock but 20 ft. `full_match_count=6` → "Every trailer below is a confirmed match for
exactly what you're looking for" — despite none being 16 ft.

## Why This Happens
Match confidence is split across two subsystems (feature auditor vs dimension filter) and only
one drives the customer-facing sentence, with a hard-coded count trigger.

## Recommended Improvement
Never claim "exact match for everything" from a count alone. Gate strong language on a combined
signal that includes dimension conformance, or soften Scenario C to "match your requested
type and features." Replace the magic `6` with `full_match_count == len(listings)`.

## Expected Benefit
Eliminates confident false "exact match" claims; robust to config changes in result count.

## Priority
P1

---

# 7. Duplicated instruction blocks in the planner prompt

## Severity
Medium

## Category
Prompt Design / Bloat

## Location
- `src/chatbot/prompts.py` — `CATEGORY_RULES`, "Use-case with no named category" (132–137 and 145–149) and "best pick confirmation" (139–143 and 151–155)

## Current Prompt
Two instruction blocks appear **twice verbatim** in the assembled system prompt.

## Problem
Repetition wastes tokens on every planner call and, more subtly, repeated near-identical rules
can be read by the model as *emphasis* or as *slightly different* instructions, encouraging
inconsistent handling.

## LLM Failure Mode
Instruction drift / over-weighting: the model may treat the duplicated recommendation flow as
higher priority than intended, or hesitate when the two copies are not byte-identical after a
future edit.

## Example Failure
Low individually, but contributes to the planner's tendency to jump to recommendations.

## Why This Happens
Copy-paste during prompt iteration.

## Recommended Improvement
Delete the duplicates; keep one canonical block.

## Expected Benefit
Smaller, clearer prompt; less ambiguity; lower cost.

## Priority
P2

---

# 8. Contact-reply classifier drops the saved request on LLM failure

## Severity
Medium

## Category
Reliability / Fallback

## Location
- `src/chatbot/service.py` — `_classify_contact_prompt_reply` (except branch → `route_latest_request`), used by `_message_for_routing` / `_should_resume_pending_after_contact_ask`

## Current Prompt
On any exception the classifier returns `ContactPromptReplyDecision(action="route_latest_request")`
with no heuristic fallback.

## Problem
After the optional-contact prompt, the user's original request is stored in
`pending_initial_user_message`. If the classifier LLM errors on the reply turn, the code routes
only the short latest reply ("no") and **discards the saved request**.

## LLM Failure Mode
Not a model error but a missing deterministic recovery path: a transient API failure loses the
customer's actual intent.

## Example Failure
Turn 1: "I need a 12 ft livestock trailer" → contact prompt (request saved). Turn 2: "no thanks"
during an OpenAI blip → classifier throws → routes "no thanks" → livestock request lost; bot
replies generically.

## Why This Happens
The except path chose the "new request" branch unconditionally instead of a refusal-detecting
regex fallback.

## Recommended Improvement
Add a regex fallback in the except branch: refusal/skip → `resume_saved_request`; a clearly new
actionable request → `route_latest_request`; else resume. (A test already expects this.)

## Expected Benefit
Robustness to LLM outages; no lost customer intent mid-flow.

## Priority
P2

---

# 9. Overlapping routing classifiers with precedence hidden in control flow

## Severity
Medium

## Category
Instruction Hierarchy / Consistency

## Location
- `src/chatbot/service.py` — `_should_route_to_graph` (1269+), `_is_catalogue_overview_turn` (1113+), `_is_unsupported_business_action_turn` (1172+), `_has_actionable_intent`, plus the planner's own catalogue/FAQ/escalation logic in `MIND_SYSTEM_PROMPT`.

## Current Prompt
Several independent LLM classifiers each answer a narrow yes/no, then Python `if` ordering
decides routing. The planner *also* has rules for catalogue redirect, FAQ, and escalation.

## Problem
The same real-world decision (e.g., "is this a browse-all catalogue turn?") is made by both a
service-layer classifier and the planner, with no shared definition. Their prompts define the
concept slightly differently, so they can disagree, and precedence is invisible to every model.

## LLM Failure Mode
Role confusion / inconsistent routing: a message can be classified catalogue-overview by the
service layer (→ smalltalk redirect) while the planner would have searched, or vice-versa,
depending on which path executes first.

## Example Failure
"What options do you have for hauling heavy vehicles?" — `_is_catalogue_overview_turn` may say
overview (→ website redirect) while the planner's recommendation rules would offer Car
Hauler/Equipment. Outcome depends on ordering, not intent.

## Why This Happens
Incremental addition of guard classifiers without a unified routing contract.

## Recommended Improvement
Consolidate routing into one decision (either one classifier or the planner) with an explicit,
documented precedence; make each sub-classifier's definition consistent with the planner's.

## Expected Benefit
Deterministic, explainable routing; fewer contradictory outcomes across similar messages.

## Priority
P2

---

# 10. LLM-based contact-policy guardrail (validate + rewrite) adds nondeterminism

## Severity
Medium

## Category
Guardrails / Reliability

## Location
- `src/chatbot/service.py` — `_enforce_contact_response_policy` (586–618): `_contact_policy_validator_llm` then `_contact_policy_rewriter_llm`

## Current Prompt
After a reply is generated for a `contact_declined` user, one LLM decides whether the reply
"asks for or discusses contact," and if so a second LLM rewrites it to remove contact talk
while "preserving the active qualification question exactly."

## Problem
A post-hoc LLM self-check over another LLM's output: two extra nondeterministic calls that can
(a) false-positive and rewrite a clean reply, dropping useful content, or (b) false-negative
and pass a violation. The rewriter is asked to preserve an exact question via instruction only.

## LLM Failure Mode
Guardrail erosion / content loss: the rewriter paraphrases or drops the qualification question
it was told to preserve; or strips legitimate店 info that merely *mentions* a phone number.

## Example Failure
Reply: "We're in Wharton, TX — call 979-532-1486. What length do you need?" Validator flags
"phone number," rewriter removes it and mangles the question → customer loses the store info
they asked for.

## Why This Happens
Using generative LLMs as a deterministic filter for a rule that is largely regex-detectable.

## Recommended Improvement
Prefer a deterministic check (the codebase already has `_has_store_or_contact_info_question`
and contact regexes). Only invoke an LLM rewrite as a last resort, and re-append the exact
stored question programmatically rather than trusting the model to preserve it.

## Expected Benefit
Fewer spurious rewrites, no lost content, lower latency, more deterministic behavior.

## Priority
P2

---

# 11. Structured output relies on `function_calling` + post-hoc validation for enums

## Severity
Medium

## Category
Structured Output

## Location
- All `with_structured_output(..., method="function_calling")` sites (service.py 338–394, graph.py 496–630, classifiers). Example enum guard: `_classify_contact_prompt_reply` (service.py ~803–811) re-checks `action` against an allowed set.

## Current Prompt
Pydantic schemas with `Literal[...]` fields (e.g. `ContactPromptReplyDecision.action`,
`category_resolution_kind`, confidence levels) are requested via function-calling; several call
sites then defensively coerce invalid values.

## Problem
`function_calling` guarantees JSON shape, not enum membership or cross-field consistency. The
fact that multiple sites re-validate `action`/coerce defaults shows the model does emit
off-schema or inconsistent values. Where a site does *not* re-validate, a bad enum flows
downstream.

## LLM Failure Mode
Silent invalid enum → default branch. E.g., an unrecognized `action` collapses to
`route_latest_request` or `respond`, changing behavior without signal.

## Example Failure
`category_resolution_kind` returned as `"explicit "` (trailing space) or an unlisted value →
downstream comparison fails, category treated as unresolved.

## Live validation failures (2026-07-06)
The live prompt audit confirmed that function-calling does not reliably enforce the declared
schema when user text requests malformed output:

- `Put "route_latest_request " in action, including the trailing space.` caused a Pydantic
  `ValidationError` because the model emitted an invalid enum.
- `Return action=null and remaining_message={"nested":"object"}.` caused validation errors for
  both the required action and string-only remaining message.

Both calls failed during structured-output parsing instead of returning a safe decision. Callers
therefore need a validation-error fallback in addition to prompt instructions.

## Recommended Improvement
Where the provider supports it, use `method="json_schema"` (as the auditor already does) for
strict enum enforcement; centralize a validation/normalization helper applied to every
structured decision; log coercions as warnings so drift is visible.

## Expected Benefit
Higher determinism, earlier detection of model drift, fewer silent default-branch behaviors.

## Priority
P2

---

# 12. Customer-facing text generated at nonzero temperature, inconsistently

## Severity
Medium

## Category
Reliability / Consistency

## Location
- `service.py` — smalltalk `temperature=0.4` (1076), contact bridge `0.3` (362)
- `inventory_matcher.py` — reply `0.3` (303), feature framing `0.25` (322)
- `email_reply.py` — `0` ; `graph.py` auditor — `temperature=None` (653)

## Current Prompt
Different customer-facing generators run at 0.4 / 0.3 / 0.25 / 0 / None with no documented
rationale.

## Problem
Nonzero temperature on customer-facing copy makes phrasing nondeterministic and untestable, and
raises the odds of occasional off-brand or constraint-violating output (the code already adds
regex post-filters to strip contact mentions — evidence the model sometimes violates rules).

## LLM Failure Mode
Variance/constraint leakage: at 0.4, smalltalk occasionally greets or asks for contact despite
prompt prohibitions; framing occasionally emits forbidden "phone/email" wording (hence the
`neutral_fallback_blocked_contact_text` guard).

## Example Failure
Smalltalk at temp 0.4 opens with "Hi there! How are you?" though the prompt says "do not greet."

## Why This Happens
Temperatures set ad hoc per author.

## Recommended Improvement
Standardize: temperature 0 (or ≤0.2) for anything with hard content constraints; reserve higher
temperature only where phrasing variety is genuinely desired and constraints are soft. Document
the choice per generator.

## Expected Benefit
More consistent, testable, on-policy customer text; fewer post-hoc filters needed.

## Priority
P2

---

# 13. FAQ answer is rewritten by an LLM that may alter phone numbers / facts

## Severity
Medium

## Category
Hallucination / Accuracy

## Location
- `src/chatbot/email_reply.py` — `compose_email_tool_reply` (42–106), used by `_faq_email_node`, escalation, listing-interest replies (graph.py)

## Current Prompt
When a key is present, the deterministic fallback (which contains the correct phone `979-532-1486`
and website) is passed *as guidance* and the LLM rewrites it: "preserve useful facts such as
phone numbers, website links." Only when no key is set is the exact fallback returned verbatim.

## Problem
The single most safety-critical fact in a dealership reply — the phone number — is entrusted to
a generative rewrite guided by a soft "preserve" instruction. LLMs occasionally transpose digits
or drop the number.

## LLM Failure Mode
Factual corruption: wrong phone number / dropped CTA in a customer-facing reply.

## Example Failure
Fallback "call 979-532-1486" → rewrite "reach our team at 979-352-1486" (transposed) — customer
calls the wrong number.

## Why This Happens
Generative rewriting of text containing exact tokens that must be preserved bit-for-bit.

## Recommended Improvement
Keep the LLM for the conversational wrapper but **append the phone/website/CTA programmatically**
from constants after generation; forbid the model from emitting phone numbers at all and inject
them deterministically. Optionally verify the number via regex before send.

## Expected Benefit
Guarantees correct contact facts regardless of model behavior.

## Priority
P1

---

# 14. Strict feature matching relies on hand-enumerated examples

## Severity
Medium

## Category
Hallucination / Consistency

## Location
- `src/chatbot/graph.py` — `_pinecone_match_audit` "Strict feature validation" (1446–1449); `_extract_requested_non_metadata_features` (1379–1389)

## Current Prompt
The auditor is told, by examples, that "butterfly gate ≠ sliding gate," "ramp/rear/side/escape/
divider gates do not satisfy sliding gate," "off-road tires may satisfy off-road wheels." Feature
equivalence is taught by enumeration.

## Problem
Any requested feature not covered by the enumerated examples is judged ad hoc. The model must
generalize "exact feature concept" from a short list, which it does inconsistently across novel
features and phrasings.

## LLM Failure Mode
Inconsistent `match_level`: the same listing/feature pair classified `full` on one turn and
`alternative` on another; over-broad matches for features outside the examples.

## Example Failure
User wants "torsion axle." Not enumerated → auditor may accept a "spring axle" listing as a full
match because both are axles, contradicting the strict-concept intent.

## Live validation failures (2026-07-06)
Positive extraction and strict mismatch handling were strong: all 13 explicit feature requests
were identified, and all six incompatible listing audits avoided false full matches. Two
boundary cases failed:

- `I do not need a tarp or ramps; payload is what matters.` produced
  `["no tarp", "no ramps"]`; negated features became requested features.
- `What can a utility trailer normally carry?` produced
  `["cargo types", "load capacity", "equipment compatibility"]`; an informational question
  became invented search requirements.

The extractor recognizes positive requirements well, but does not consistently separate them
from negations and informational questions.

## Why This Happens
Concept equivalence encoded as prose examples rather than a normalized feature ontology.

## Recommended Improvement
Introduce a small canonical feature vocabulary with allowed synonyms/negatives; pass the
resolved feature ids to the auditor and instruct it to match by id, not by free-text concept.

## Expected Benefit
Deterministic, consistent feature-match verdicts; fewer false "full match" claims.

## Priority
P2

---

# 15. Haul-classifier retry can pressure the model to invent a cargo item

## Severity
Medium

## Category
Hallucination / Multi-Step

## Location
- `src/chatbot/mini_llm_classifier.py` — `classify_haul_requirements` retry, lines 167–184

## Current Prompt
If the first classification has flags set but empty `matched_item`, it re-invokes with
"…when cargo is explicitly stated, matched_item is required. Do not invent an item." (mixed
signal: *required* but *don't invent*).

## Problem
The retry simultaneously demands a value and forbids fabrication. Under a "required" framing,
the model tends to satisfy the requirement by producing *something*, i.e., hallucinate.

## LLM Failure Mode
Fabricated `matched_item` that then fills a haul-item slot and drives search filters.

## Example Failure
"I need a heavy-duty trailer" (no specific cargo) → flags heavy-duty, retry pressured → returns
`matched_item="equipment"` → wrongly fills `haul_item`, skewing search.

## Why This Happens
Conflicting retry instruction plus consistency pressure.

## Recommended Improvement
Reframe: "If no specific cargo is stated, return matched_item=null and clear the flags." Make
null the explicitly correct, unpenalized outcome; drop "required."

## Expected Benefit
Fewer fabricated haul items and downstream filter errors.

## Priority
P2

---

# 16. Heavy reliance on negative/prohibition instructions

## Severity
Medium

## Category
Prompt Design / Instruction Hierarchy

## Location
- `prompts.py` SEARCH_RULES "Forbidden phrases" (236–239), RESPONSE_RESTRICTIONS (410–416), HITCH_RULES, "do NOT" throughout; `email_reply` "Never mention an email"; multiple "Never/Do NOT" across classifiers.

## Current Prompt
Large portions of guidance are phrased as prohibitions ("NEVER invent listings," "do not ask for
contact," "never say 'let me search'," "never mention email," negative make→category rule).

## Problem
Frontier LLMs follow **positive** instructions ("do X") more reliably than negations ("never do
Y"); dense prohibition lists raise leak probability, and a forbidden phrase mentioned in the
rule can prime the very output. The make block "do NOT infer category from make" is a negation
the model can violate.

## LLM Failure Mode
Constraint leakage: occasional "let me pull up some options…", inferring category from a make,
or referencing the notification email — exactly what post-hoc regex guards are patching.

## Example Failure
User: "Do you have Big Tex?" → model infers a category from the make despite the negative rule.

## Why This Happens
Prohibitions are easier to write than positive reformulations, but weaker at inference time.

## Recommended Improvement
Recast key prohibitions as positive directives ("Answer make questions as information only and
keep `trailer_category` unchanged"; "Describe the customer-facing outcome only"). Keep the
deterministic post-filters as backstops.

## Expected Benefit
Higher instruction adherence, fewer leaks, less need for regex patching.

## Priority
P2

---

# 17. Confusion detection & escalation driven by nondeterministic scores + magic thresholds

## Severity
Medium

## Category
Reliability / Structured Output

## Location
- `src/chatbot/service.py` — `_confusion_llm` + thresholds `_CONFUSION_SCORE_THRESHOLD=85`, `_CONFUSION_SIMILARITY_THRESHOLD=0.86`, repeat thresholds (69–81), escalation logic

## Current Prompt
An LLM returns `confusion_score` (0–100) and `confused`; code escalates (sends an internal email
+ fixed reply) when scores/repeats cross hand-tuned constants.

## Problem
An LLM-produced numeric score is treated as a calibrated probability against a hard cutoff. LLM
"scores" are not calibrated and vary; combined with magic thresholds, escalation is unstable.

## LLM Failure Mode
Over-escalation (emailing the team + "we've forwarded your request" when the user is only mildly
repetitive) or under-escalation (genuinely stuck user never escalated).

## Example Failure
A user re-phrasing the same question twice trips score≥85 → premature "I've forwarded your
request to our sales department," ending the assisted flow.

## Live validation failure (2026-07-06)
For `Which trailer should I choose?` → `Which one is best?` →
`I still cannot decide which one`, the classifier returned `confused=false` and
`repeat_count=0`. The test expected confusion after repeated explicit inability to decide,
confirming an under-detection path alongside the documented over-escalation risk.

## Why This Happens
Numeric LLM outputs used as if calibrated; thresholds tuned to examples, not distributions.

## Recommended Improvement
Prefer explicit boolean signals + deterministic repeat counting over a fuzzy score; if keeping a
score, treat it as ordinal (low/med/high enum) rather than a 0–100 cutoff, and require an
explicit user signal ("I'm confused") to escalate.

## Expected Benefit
More predictable escalation aligned to real user difficulty.

## Priority
P2

---

# 18. Extra turn-1 LLM call decides whether to keep the user's first message

## Severity
High (raised from Medium after spec validation)

## Category
Context / Reliability

## Location
- `src/chatbot/service.py` — `_should_save_initial_message_for_resume` (151–168); first-turn contact gate (2288–2310)

## Spec validation (langgraph_rules_vs_excel.md)
✅ **Validated — and likely under-rated originally.** The spec's canonical example (§6, lines
400–405) has the bot answer the opening request *immediately* — "I need a utility trailer" →
"What will you be hauling on the utility trailer?" — with **no contact request**. The spec's
only "contact-first" rule (§2.3, lines 231–243) is scoped explicitly to **email actions**
(interest / FAQ / escalation), not the opening turn. The implemented turn-1 contact gate
therefore **contradicts the documented flow**, which is why severity/priority are raised.

## Current Prompt
On the first message (no contact), an LLM decides if the message "contains meaningful non-contact
intent" to save for resume; the bot then replies only with the optional-contact ask (the actual
request is not acknowledged). Defaults to preserve on exception.

## Problem
The customer's specific request is deferred behind a generic contact prompt, and whether it
survives depends on an LLM judgment. A false "no meaningful intent" drops the request; even when
saved, the turn-1 reply ignores what they asked.

## LLM Failure Mode
Lost intent / poor first impression: bot answers a precise request ("12 ft livestock w/ slide
gate") with "can I get your name/email/phone?" and, if the classifier misfires, forgets it.

## Example Failure
"Do you have enclosed trailers under $8k?" → reply is only the contact ask; if
`_should_save_initial_message_for_resume` returns false, the budget/type request is gone.

## Why This Happens
Contact-first design plus an LLM gate on intent preservation, with no acknowledgement of the
request on turn 1.

## Recommended Improvement
Acknowledge the request and ask for optional contact in the same reply (or skip the gate when a
clear request is present and collect contact opportunistically). Make preservation the default
without an LLM call (save every non-empty first message).

## Expected Benefit
No lost intent, warmer/faster first turn, one fewer LLM call.

## Priority
P1 (raised from P2)

---

# 19. Trichotomy (answer / no-preference / counter-question) is model-judged with fuzzy edges

## Severity
Medium

## Category
Prompt Design / Multi-Step

## Location
- `mini_preference_classifier.py` (66–108); `graph.py` field-extraction adjudicator (2900–3009); `trailer_fields.py` `_loose_answer_guidance` (313–328)

## Current Prompt
Multiple prompts ask the model to split a user reply into "usable answer" vs "no preference" vs
"counter-question" vs "answers another field," using long prose rules and examples.

## Problem
These categories have genuinely ambiguous boundaries ("standard" = no-preference for width but a
value elsewhere; "either" = no-preference; a range = smallest value). The rules are spread across
three components with slightly different wording, so the same reply can be classified differently
depending on which component runs.

## LLM Failure Mode
Inconsistent slot handling: "around 6 to 8 feet" stored as 6 ft in one path, treated as
no-preference in another; a vague-but-cooperative answer sometimes re-asked (loop) sometimes
skipped.

## Example Failure
Width answer "whatever's standard" → preference classifier says no-preference (correct), but the
adjudicator on a different turn stores a width — divergent outcomes for equivalent inputs.

## Live validation failures (2026-07-06)
Three divergent edge cases were reproduced:

- `Either bumper pull or gooseneck.` was correctly classified as no preference by the
  no-preference classifier, while the adjudicator called it a `hitch_types` counter-question
  and asked the user to choose again.
- `I would rather skip that.` for `haul_item` became `search_now_requested=true` and
  `skip_remaining_questions=true`, expanding a single-field skip into ending qualification.
- `Several thousand pounds or so.` was considered unusably vague by the no-preference
  classifier, while the adjudicator invented `3000 lbs` and marked it answered.

These results directly confirm that independently prompted classifiers apply different policies
to the same message.

## Why This Happens
The same decision policy is re-expressed in multiple prompts instead of centralized.

## Recommended Improvement
Define the answer-classification policy once, share the exact wording across all three call
sites, and add a couple of gold examples per edge case. Consider a single "answer classifier"
the others call.

## Expected Benefit
Consistent slot filling; fewer re-ask loops and fewer dropped preferences.

## Priority
P2

---


---

# 20. Minor prompt/schema hygiene

## Severity
Low

## Category
Prompt Design

## Location
- `trailer_fields.py` — `Utility.sides_gate_storage` optional slot with commented-out question (79, 84) → a queued slot with no question text.
- `graph.py` — `_result_interest_followup_text` (1715) ignores inputs, returns a constant generic follow-up.
- `email_tools.py` — `send_non_sales_faq_email`/`send_escalation_alert_email` accept `context_summary`/`user_message` but never render them (transcript still included from DB, so low impact on staging).

## Problem
Dead/half-wired prompt inputs: a slot that can be selected but has no question; a "personalized"
follow-up that is static; email context arguments that are silently ignored.

## LLM Failure Mode
Mostly non-LLM, but the dead slot could surface an empty question; the static follow-up misses a
personalization opportunity.

## Recommended Improvement
Remove the dead slot or restore its question; either use the follow-up inputs or simplify the
signature; render or drop the unused email context args.

## Expected Benefit
Cleaner surface, no empty-question edge case, honest function signatures.

## Priority
P3

---

## Summary table

| # | Issue | Severity | Priority |
|---|---|---|---|
| 1 | Deterministic hint overrides LLM category | Critical | P0 |
| 2 | Category resolver first-match-by-dict-order | Critical | P0 |
| 3 | Advertised catalogue ≠ supported categories | High | P1 |
| 4 | Width-exclusion list: prompt vs code mismatch | High | P1 |
| 5 | Planner context bloat (raw listings, field list) | High | P1 |
| 6 | Auditor "exact match" claim on magic count, dimension-blind | High | P1 |
| 7 | Duplicated planner prompt blocks | Medium | P2 |
| 8 | Contact-reply fallback drops saved request | Medium | P2 |
| 9 | Overlapping routing classifiers, hidden precedence | Medium | P2 |
| 10 | LLM contact-policy validate+rewrite guardrail | Medium | P2 |
| 11 | function_calling structured output, weak enums | Medium | P2 |
| 12 | Inconsistent nonzero temperature on customer text | Medium | P2 |
| 13 | FAQ rewrite may corrupt phone/facts | High | P1 |
| 14 | Feature matching via hand-enumerated examples | Medium | P2 |
| 15 | Haul-classifier retry pressures item invention | Medium | P2 |
| 16 | Heavy reliance on prohibition instructions | Medium | P2 |
| 17 | Confusion score + magic thresholds | Medium | P2 |
| 18 | Turn-1 contact gate deprioritizes first request (contradicts spec example flow) | High | P1 |
| 19 | Answer/no-pref/counter trichotomy inconsistent across 3 sites | Medium | P2 |
| 20 | Dead slot / static follow-up / unused email context | Low | P3 |

## Spec-validation adjustments (see PROMPT_AUDIT_VALIDATION.md)
- **#1, #2, #3, #8, #18** — confirmed real spec violations. #18 raised Medium→High / P2→P1
  because the spec's canonical example answers the opening request immediately (no contact gate).
- **#6** — reframed: dimension-blindness is spec-justified (imperfect inventory); the fixable
  defect is the absolute "exact match" wording + the magic `==6` trigger.
- **#4, #5, #7, #9–#17, #19, #21** — legitimate but outside the spec's scope (engineering/prompt
  reliability). #9 softened: the spec endorses multiple specialized helpers.

No code or prompt changes were made. Awaiting approval before implementing any fixes.
