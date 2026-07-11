# TrailerPlace Chatbot Refactor — Full-Codebase Audit & Implementation Plan

Deliverable structure follows `task.md` exactly: Phase 1 audit (independent, full-codebase observation), Phase 2 refactor plan (prompt architecture / conversation flow / code structure / model usage / rules consolidation / optimizations), Phase 3 implementation spec detailed enough for a separate executing model to follow without re-derivation.

## Context

Production trailer-sales chatbot: Streamlit (`app.py`) → FastAPI (`main.py`) → `src/chatbot/service.py` (session, routing, contact flow, durable persistence) → LangGraph (`src/chatbot/graph.py`) → Pinecone search / local inventory matcher / email tools → deterministic formatting. Known problems per task.md: hallucinations, unnatural conversation flow, prompt sprawl, general code health.

**Coverage statement**: this plan is grounded in a full independent read of the first-party codebase — `graph.py` (7,278 lines), `service.py` (2,654), `app.py` (1,607), `inventory_matcher.py` (1,542), `pinecone_search.py` (835), `prompts.py`, `categories.py`, `mini_llm_classifier.py`, `mini_preference_classifier.py`, `email_reply.py`, `email_tools.py`, `email_sender.py`, `make_resolver.py`, `make_inventory.py`, `formatting.py`, `trailer_fields.py`, `normalizer.py`, `ingest.py`, `models.py`, `state.py`, `conversation_store.py`, `shown_listings_store.py`, `db.py`, `db_models.py`, `thinking_agent.py`, `main.py`, and the `tests/` directory. `PROMPT_AUDIT.md`/`overview.md` were used as reference only; several of their claims were found stale (noted below).

---

# PHASE 1 — AUDIT

## 1A. Architecture map (end-to-end message flow)

```
Streamlit app.py
  │  POST /chat {session_id, turn_id, message, phase, shown urls}
  │  (turn_id = uuid per submit, retained until success → idempotent retry;
  │   session_id in browser sessionStorage + URL query param)
  ▼
FastAPI main.py (/chat, /session/reset, /session/{id}, /health) → service.handle_chat
  ▼
DURABLE WRAPPER handle_chat (service.py:2069–2152) — implemented, not just planned:
  pg_advisory_xact_lock per session · turn-receipt idempotency (ChatbotTurn)
  · state_snapshot JSONB restore · email outbox capture + post-commit drain
  ▼
_handle_chat_in_memory (service.py:2238–2654) gate sequence:
  1. contact extraction            [LLM #S1 + regex merge]
  2. pending results/contact-action delivery  [may early-return; LLM email composer]
  3. listing-reference / direct inventory lookup
       [LLM #S10 gate → LLM #S11 resolver → Python identity check;
        then inventory_matcher extractor LLM #I1 → local Excel match → intro LLMs #I2/#I3]
  4. turn-1 contact gate           [LLM #S4 decides whether to save the request]
  5. contact-reply classification  [LLM #S2 + bridge LLM #S3]
  6. confusion escalation          [regex-gated LLM #S9]
  7. graph-vs-smalltalk routing    [LLMs #S6, #S5, #S7 in worst case]
  8a. smalltalk                    [LLM #S8, temp 0.4]
  8b. LangGraph:
        mind node (planner LLM #G1, gpt-4o-mini, MIND_SYSTEM_PROMPT ≈ 480 lines)
        → apply_mind (1,392-line node): category transition #G9, make verify #G10,
          question adjudicator #G7 → field extractor #G5 → no-preference #M2
          → reconciler #G8 (→ optional retry-repair #G7 again),
          non-recommendation safety #G6, haul classifier #M1,
          office clarification #G11, filter confirmation #G2
        → conditional edge: pinecone_search node │ 3 email nodes │ finalize
  ▼
pinecone_search node: feature extractor #G4 → embed (text-embedding-3-small)
  → Pinecone (top_k=50, hard filters: category/make/hitch/aluminum-subcategory/length_gte)
  → fit rerank → make-preference rerank → slice [:5]
  → MATCH AUDITOR #G12 (the ONLY gpt-5-mini call: json_schema, responses API,
    reasoning effort minimal) → deterministic sanitizers → format_listing_results
email nodes: send (blocking SMTP) → compose_email_tool_reply LLM #E1 → finalize
  ▼
Response assembly → durable commit → outbox drain → ChatResponse → Streamlit cards
```

**Where each model sits (task.md asked to clarify GPT-5-mini's role)**: `gpt-4o-mini` (default via `OPENAI_MODEL`) powers everything conversational — planner, all classifiers, extractors, reply composers. **`gpt-5-mini` has exactly one job**: the Pinecone match auditor (`graph.py:643–659`, `_pinecone_match_audit_llm`) — it verifies each retrieved listing against the user's requested category/features, labels full/partial/alternative, and writes the customer-facing `intro_text` and per-listing blurbs. It is deliberately dimension-blind (dimensions handled by the deterministic rerank) and is the only call using `method="json_schema"` + Responses API + reasoning.

## 1B. Prompt inventory (every LLM call site)

**Service layer** (`service.py`; all `function_calling` structured output unless noted):
| # | Call | Lines | Model/temp | Purpose | Fallback |
|---|---|---|---|---|---|
| S1 | contact extractor | 339–343, 500–599 | `_CHAT_MODEL`/0 | name/email/phone + name confidence from current message | regex extraction |
| S2 | contact-prompt reply classifier | 353–358, 801–869 | env/0 | classify reply after contact ask (7 actions) | `route_latest_request` — **drops saved request** |
| S3 | contact bridge | 361–363, 876–927 | env/**0.3**, plain text | 1–2 sentence acknowledgement opener | canned text |
| S4 | initial-message preservation | 377–380, 189–206 | `_CHAT_MODEL`/0 | is the first message worth saving for resume | `True` |
| S5 | catalogue-overview | 383–388, 1157–1222 | env/0 | "what do you carry" vs specific search | `False` |
| S6 | unsupported-business-action | 391–396, 1225–1283 | env/0 | should escalation reach the graph | `False` |
| S7 | ROUTE/SMALLTALK | inline 1319 | `_CHAT_MODEL`/0, one token | graph relevance for post-results follow-ups | regex `_has_actionable_intent` |
| S8 | smalltalk generator | inline 1090 | `_CHAT_MODEL`/**0.4**, plain | non-graph reply | canned |
| S9 | confusion detector | 346–350, 1494–1569 | `_CHAT_MODEL`/0 | 0–100 confusion score | `(False, 0)` |
| S10 | listing-ref intent gate | 2051–2056 | env/0 | explicit ordinal/identifier/demonstrative? | no-reference |
| S11 | listing-ref resolver | 2043–2048 | env/0 | exact index/title/url | no-reference; Python re-validates identity |
| — | contact-policy validator+rewriter | 365–374 | — | **DEAD CODE** — never invoked; active guard is regex (626–649). PROMPT_AUDIT #10 is stale. | — |

**Graph layer** (`graph.py`; 12 lru-cached factories at 495–659):
| # | Call | Purpose | Notes |
|---|---|---|---|
| G1 | `_mind_llm` (4777) | planner: action/category/slots/text (`MindDecision`, 7 actions) | system prompt = `prompts.py` `MIND_SYSTEM_PROMPT` (~18 concatenated rule blocks + category/make blocks) |
| G2 | category-filter confirmation (5540) | keep/discard/change carried filters | |
| G3 | `_filter_extractor_llm` (517, 3769) | **DEAD** — legacy stack; `_legacy_apply_explicit_filter_extraction` (5453) and `_legacy_field_updates_from_filter_extraction` (3401) are never called | |
| G4 | requested-feature extractor (1377) | non-metadata features from request text | good grounding clause |
| G5 | field-extraction adjudicator (3470) | PRIMARY extractor: filters+slots+features+rejections | ~66-line prompt |
| G6 | non-recommendation turn (5213) | FAQ vs escalation vs continue | skipped when key starts with "test" |
| G7 | question-turn adjudicator (2905) | active-question turn: answered/no-pref/counter/search-now | **largest prompt (~175 lines)**; also drives retry-repair (3160) |
| G8 | active-turn reconciler (2795) | final authority merging G1+G5+G7+M2 | reuses G7 schema |
| G9 | category-transition (2676) | approve/reject category switch | |
| G10 | make verification (2737) | brand really intended? | |
| G11 | office-trailer clarification (2429) | Fiber vs Enclosed | |
| G12 | **Pinecone match auditor** (643–659, 1419–1586) | full/partial/alternative labels + `intro_text` + `sales_blurb` | **gpt-5-mini**, json_schema, temp None, reasoning minimal. Scenario A/B/C wording gated on `full_match_count` with hardcoded `= 6` (1475) while `SEARCH_MAX_RECOMMENDATIONS` defaults to **5** (`pinecone_search.py:738`). Input log (1518–1525) misreports model as `gpt-5.4-mini`/effort `none` vs actual `gpt-5-mini`/`minimal`. |

**Other modules**: E1 email-reply composer (`email_reply.py:42`, env/0, plain text — rewrites replies containing the dealership phone); I1 inventory query extractor (`inventory_matcher.py:284,476`, 0); I2 inventory intro (`301,1212`, **0.3**, plain); I3 feature-framing intro (`315,1107`, **0.25**); M1 haul classifier (`mini_llm_classifier.py:45,122`, 0, with self-retry that says "matched_item is required… do not invent" — conflicting); M2 no-preference classifier (`mini_preference_classifier.py:24,37`, 0). `make_resolver.py` has **no** LLM (the `use_llm_fallback` param is `del`-eted at 181–185 — vestigial).

**Persona definition**: defined once in `prompts.py` `PERSONA` but injected into only some generators (mind, smalltalk, question adjudicator via KNOWLEDGE section); the inventory intros, email composer, and bridge have their own ad-hoc tone instructions — tone is *not* single-sourced.

## 1C. Rules inventory — duplications & contradictions (the consolidation targets)

| Rule | Locations | Status |
|---|---|---|
| "Range → smallest value", "loose answer → null", "width needs a number" | prompts: field extractor (3473–3539), question adjudicator (2907–3082), reconciler (2797–2846), `trailer_fields._loose_answer_guidance` (313–328), `QUALIFICATION_RULES` in prompts.py; code: `_validate_slot_value` (4435) | **5+ restatements**, wording drifts |
| FAQ/escalation trigger definitions | mind prompt (FAQ_RULES/ESCALATION_RULES), G6 prompt (5216), G7 prompt (3033), office-clarification tool_rules (2423), deterministic overrides (6551–6570, 6723–6732), service S6 prompt | **6 places** |
| Dynamic-width exclusions | prompt M1 (155–157): {Utility, Enclosed, Flatbed}; code gate `_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES` (graph.py:1769): 6 categories | **contradict** — model output silently dropped for Livestock/Aluminum/Dump |
| Advertised catalogue | `KNOWLEDGE` (prompts.py:21): 10 types; `category_prompt_block()`: 13; `trailer_fields._SPECS`: 14 (incl. non-canonical **Welding**); `CATEGORY_QUESTION_OWNERSHIP` example (~96): 10 | **3 different catalogues inside one system prompt** |
| Make alias maps | `make_resolver._ALIASES` (22–67), `pinecone_search.MAKE_ALIAS_MAP` (76–97), `normalizer.MAKE_MAP` (31–48) | **triplicated**, short-form vs long-form divergence reconciled only by `$in` filters |
| Category alias maps | `normalizer.CATEGORY_MAP` (8–25) vs `make_inventory._CATEGORY_ALIASES` (28–33) | duplicated; aliases absent from `_SPECS` |
| Unit parsers | `ingest.parse_lbs/parse_length_ft` (96–185) vs `pinecone_search._parse_number/_parse_length_ft` (102–159) | duplicated with **divergent coverage** (yd/metric only in ingest) |
| Listing-card renderers | `formatting.format_listing_results` vs `inventory_matcher._format_listing_block` (983–1020) | two layouts/field-sets for the same concept |
| Canned email replies | `graph.py` `_INTEREST_GENERIC_FALLBACK` etc. (81–91) vs `email_reply.EDITABLE_CUSTOMER_RESPONSE_GUIDANCE` (10–39) | duplicated |
| A×B shorthand orientation | `_dimension_shorthand_updates` (3278) + extractor prompt (3505): width×length; `bare_trailer_inches` branch (5423): bare value → **length** | **internally contradictory** |
| Category hint vs planner | `CATEGORY_RULES` prompt says the model MUST set the category; code (4803–4806) unconditionally overwrites it with the deterministic hint | prompt authority is a **no-op**; resolver itself is first-match-by-dict-order (`categories.py:100–105`) so cargo words beat named types ("tilt trailer to haul tractor" → Equipment) |
| Skip counter for counter-questions | `overview.md` §2.5/§6.6: counter-questions must NOT count as failed answers; code 6187–6203, 6300–6301: they **do count** toward the ≥2 skip threshold (6305) | spec/code contradiction |
| Confusion threshold 85 | encoded in the S9 prompt (1523) AND re-checked in Python (1552) | double-encoded magic number |
| Duplicate planner prompt blocks | `CATEGORY_RULES` "use-case with no named category" (132–137 = 145–149) and "best pick" (139–143 ≈ 151–155) | verbatim duplication in every planner call |

## 1D. Hallucination root causes

Grounding is already strong in most inventory paths (deterministic cards; auditor told "use only supplied listing facts"; `_invalid_pinecone_intro` regex validator (1049) + count-based intro overwrite (1566–1581); inventory intros have contradiction-reverting guardrails (1289–1298)). The remaining real leak paths:

1. **Auditor Scenario C** (`graph.py:1475`): "Every trailer below is a confirmed match for exactly what you're looking for" triggered by `full_match_count = 6` — an absolute claim from a dimension-blind auditor, and the trigger is misaligned with the default result count of 5 (so it currently mis/never fires, and breaks whenever the env knob changes).
2. **`sales_blurb`/`customer_label`** (auditor output, 1489) render into cards via `formatting._safe_sales_blurb` (105–139) — they pass that regex guard but **not** the stronger `_invalid_pinecone_intro` validator that protects `intro_text`.
3. **Email-reply composer** (`email_reply.py:42`) generatively rewrites text containing the dealership phone `979-532-1486` with only a soft "preserve facts" instruction — digit-transposition risk on the most safety-critical fact; the phone is hardcoded in **5+ places** (graph.py 63–74, prompts.py FAQ templates, email_reply guidance).
4. **Haul-classifier retry** (`mini_llm_classifier.py:167–194`): "matched_item is required… do not invent" — the "required" framing pressures fabrication of a cargo item that then drives search filters.
5. **Feature extractor edge cases**: negations ("no tarp") and informational questions become requested features (no deterministic negation filter after G4/G5).
6. **Catalogue inconsistency** (1C): the bot can advertise 10 types, then happily qualify a Diesel Tank — reads as hallucination to users.

## 1E. Conversational flow issues

1. **Turn-1 experience**: a specific request ("12 ft livestock trailer") is answered with only a contact ask; whether the request survives depends on LLM S4. `overview.md` §6.3's canonical flow answers the request immediately.
2. **Contact-reply failure loses intent**: S2's exception fallback (`route_latest_request`, 867–869) discards `pending_initial_user_message` on any API blip — no regex refusal fallback.
3. **History windowing is ad hoc**: full history goes to the graph and the API response (no cap); LLM contexts use `[-8:]`, `[-6:]`, `[-4:]` inconsistently across ~12 sites; no truncation of long assistant listing turns inside those windows except `_compact_recent_context [:1800]`; no summarization for long threads.
4. **Latency = unnatural pauses**: worst case ~8–9 sequential service-layer LLM round-trips before the graph, then up to ~6 more inside `apply_mind`, all sequential, plus blocking SMTP sends inline on the request path (email_sender timeout up to 30 s).
5. **Tone drift**: customer-facing text produced by 6+ different generators at temperatures 0/0.25/0.3/0.4 with independently written style rules.
6. **Planner context bloat** (`_mind_node` 4753–4775): full `last_listings` objects + `listing_model_fields` + a hint that is force-applied afterwards anyway — distraction + stale-listing bias after topic changes.
7. **Counter-questions advance the skip counter** (1C last row) — users who ask a clarifying question twice get their question silently skipped.

## 1F. Code quality issues

**Dead code** (safe deletions, all verified by grep):
- service.py: `_contact_policy_validator_llm`/`_contact_policy_rewriter_llm` + `ContactPolicyDecision`/`ContactPolicyRewrite` (339–396 region, 140–146); `_message_for_routing` (930–938) + `_should_resume_pending_after_contact_ask` (872–873); `_UNANSWERED_QUESTION_REPEAT_THRESHOLD` (100); `_INITIAL_CONTACT_REQUEST` (106–109); `_has_full_initial_details` (609–610, trivial alias).
- graph.py: entire legacy extraction stack — `_filter_extractor_llm` (517), `_extract_filter_decision` (3752), `_metadata_filters_from_extraction` (3805), `_slot_updates_from_extraction` (3844), `_fallback_filter_extraction` (3649), `_legacy_apply_explicit_filter_extraction` (5453), `_legacy_field_updates_from_filter_extraction` (3401), plus `FilterExtractionDecision` and the `FILTER_EXTRACTOR_MODEL` env var; `_result_interest_followup_text` (1715) `del`s its 3 params; `_valid_preference_targets` (4165) `del`s `category`; vestigial `reply_source` constants (7117, 7180).
- app.py: unreachable lines 1453–1456 in `_process_assistant_reply` (after `return False`) — the backend-failure hint never shows.
- email_sender.py: abstract `GraphEmailSender` stub (63–76); unused async `enqueue_faq_email_notification` + `_faq_email_pool` (216–243, 28).
- make_resolver.py: `use_llm_fallback` param + unreachable `"llm"` MatchType (11, 181–185).
- trailer_fields.py: commented-out `sides_gate_storage` slot (84, 90).
- tests: `test_categories.py` is a bare pandas print script, not a test (no assertions; reads Excel at collection).
- inventory_matcher.py: no-op ternary `row.get if isinstance(row, dict) else row.get` (555, 579).

**Structural issues**: `_apply_mind_node` = **1,392 lines** with the ~25-key `return {**state, …}` dict rebuilt in **10 places** (drop-a-key hazard); `_handle_chat_in_memory` = 416 lines with ~7 near-identical `ChatResponse(...)` blocks; slot-validation loop copy-pasted ~5×; near-duplicate helper pairs (`_make_only_missing_slots`/`_generic_no_category_missing_slots`, `_classify_make_category_no_preference`/`_classify_generic_category_no_preference`); contact-bridge assembly copy-pasted in smalltalk + graph branches (2457–2468, 2605–2616); email nodes share a duplicated `continue_search_after_email` tail (7194, 7251) that re-invokes search invisibly to the graph topology.

**Reliability**: session dicts are mutated **outside** the `_sessions` lock (in-memory path races; only the durable path serializes via advisory lock); model selection inconsistent (half the constructors hardcode `_CHAT_MODEL`, half read `OPENAI_MODEL` at call time); lru-cached LLM factories freeze env at first call; every LLM wrapped in bare `except Exception` with fallbacks of differing philosophy (one returns `True`); `ChatbotOutbox.claimed_at` never set, outbox drain is best-effort with no retry scheduler; `make_resolver._valid_make_map` rebuilt on every resolve (115–123, uncached, on the hot prefilter path); Excel loaded twice via two independent caches (`prepared_inventory`, `load_make_inventory`); `shown_listings_store` maintained but never consulted by the service (parallel bookkeeping).

**Tests**: good offline coverage of deterministic guards (rerank, filters, formatting guardrails, email tools, contact flow, metadata extraction — the two big suites are 2,834 and 3,397 lines); live suites (`test_live_prompt_audit_issues.py`, `test_live_vague_qna_conversations.py`) are env-gated and never run by default, so all prompt-behavior guarantees are unexercised in CI.

## 1G. Model allocation sanity check

Current split — `gpt-4o-mini` for all ~28 conversational/classifier calls; `gpt-5-mini` (minimal reasoning, json_schema) only for the match auditor — is **broadly right but inverted in one place**: the highest-stakes semantic decisions (the planner G1 and the active-turn reconciler G8, which arbitrate everything) run on the weakest model with the longest prompts, while significant 4o-mini budget is burned on gate classifiers (S4, S5, S6, S7) whose jobs are mergeable or deterministic. The auditor on gpt-5-mini is correct (verification benefits from reasoning; json_schema gives strict enums). Recommendation in Phase 2 §4.

---

# PHASE 2 — REFACTOR PLAN

## 2.1 New prompt architecture

**Single source of truth, layered.** Create a `src/chatbot/prompts/` package:

```
prompts/
  core.py        # PERSONA, KNOWLEDGE (catalogue derived from CANONICAL_CATEGORIES),
                 # GROUNDING ("Use only data provided in this conversation. If the
                 # answer is not in the provided data, say you don't have that
                 # information."), STYLE (tone, formatting), CONTACT_POLICY
  policies.py    # shared rule blocks rendered from code constants:
                 #  answer_classification_policy() — the range/loose/width/hitch rules,
                 #  faq_escalation_policy(), width_exclusion_clause(), catalogue_line()
  mind.py        # planner prompt = core + policies + action table (dedup'd CATEGORY_RULES)
  turns.py       # question adjudicator / reconciler prompts (share answer_classification_policy())
  extraction.py  # field extractor / feature extractor prompts
  retrieval.py   # match auditor prompt (scenario wording count-relative, injected at call time)
  replies.py     # smalltalk / bridge / email-reply / inventory-intro prompts (share PERSONA+STYLE)
```

Rules that already exist as code constants (`_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES`, `CANONICAL_CATEGORIES`, `_ALLOWED_HITCH_TYPES`, phone/URL) are **rendered into prompt text from those constants**, never retyped. Every customer-facing generator prompt gets the same GROUNDING block and explicit structure (Role / Instructions / Constraints / Output format). Concrete grounding techniques already present (deterministic cards, auditor evidence-only rule, intro validators) are kept and extended: the auditor's per-listing outputs (`sales_blurb`, `customer_label`) go through the same validation as `intro_text`; the phone number is injected programmatically, not generated (see 3.4/3.7 work items).

**Citation-style grounding**: the auditor already receives listings by `position` and returns per-position verdicts that Python re-validates — keep this as the citation mechanism; extend `match_validation` to record which `match_evidence_text` fragment confirmed each feature (one extra schema field, used for logging/debug, not customer display).

## 2.2 Conversation flow fixes

- **State management stays authoritative-in-Python** (it already is; keep it). Fix the windowing: one constant set in a new `src/chatbot/constants.py` — `RECENT_WINDOW = 8`, `RECENT_CHAR_CAP = 500/message` — applied via a single `compact_recent_messages()` helper replacing the 12 ad-hoc `[-8:]/[-6:]/[-4:]` slices. Full history still persisted; only LLM contexts are windowed. Summarization for long threads is **deferred** (durable snapshot already preserves slots/filters, which is the semantically important memory; a rolling summary adds LLM cost for marginal gain here).
- **Turn 1**: always save a non-empty first message (delete S4); acknowledge the request in the same reply as the optional contact ask (deterministic template).
- **Contact-reply resilience**: regex refusal/skip fallback in S2's except branch → resume saved request; never default to discarding it.
- **Counter-questions must not advance the skip counter** (align code 6300–6301 with the documented contract).
- **Topic changes**: keep the category-transition reconciler as the single gate; make the deterministic hint advisory + tiered (naming vs cargo terms) so the planner + reconciler see a trustworthy signal instead of being overwritten.
- **Latency** (biggest "feels scripted/disjointed" driver): see 2.6.

## 2.3 Code structure

Target layout (incremental extraction, no big-bang rewrite):

```
src/chatbot/
  constants.py        # windows, thresholds, phone/URL, width exclusions, hitch set
  llm.py              # make_llm(name) — central factory: model env resolution,
                      # temperature registry, structured-output method, ValidationError-safe
                      # invoke wrapper, coercion logging. All 25+ factories route through it.
  prompts/            # package per 2.1
  routing.py          # service-layer gate order extracted from _handle_chat_in_memory
  contact.py          # contact extraction/status/policy (regex + S1/S2/S3)
  graph/
    __init__.py       # build_chatbot_graph
    mind.py           # _mind_node + context builder
    apply_mind/       # decomposed: category.py, active_turn.py, questions.py, actions.py
    search.py         # pinecone node + audit
    emails.py         # 3 email nodes (shared tail helper)
    state_return.py   # single build_state_return(state, **overrides) replacing 10 dict rebuilds
  retrieval/          # pinecone_search.py, inventory_matcher.py, formatting.py (shared units.py,
                      # single make-alias source)
```

Rationale: `apply_mind` decomposition is what makes every other fix reviewable; the `llm.py` wrapper is what makes model/temperature policy enforceable; `constants.py` is what makes rules single-sourced. Extraction order and risk controls in Phase 3 Stage E.

## 2.4 Model usage

| Call class | Model | Temp | Method | Rationale |
|---|---|---|---|---|
| Planner (G1) + active-turn reconciler (G8) | **gpt-5-mini**, reasoning minimal | (n/a) | json_schema | highest-stakes decisions; strict enums end the coercion whack-a-mole; auditor already proves the pattern works. Env-gated (`MIND_MODEL`, default gpt-4o-mini initially) so it's a config flip after A/B. |
| Match auditor (G12) | gpt-5-mini (keep) | None | json_schema (keep) | already correct |
| Extractors/classifiers (G4,G5,G7,G9,G10,G11,G2,M1,M2,S1,S2,S9,S10,S11) | gpt-4o-mini | 0 | function_calling now; migrate to json_schema opportunistically | cheap, high-volume |
| Customer-facing generators (S8, S3, I2, I3, E1) | gpt-4o-mini | **0–0.2 standardized** (from 0.4/0.3/0.25) | plain | testable, on-policy phrasing; hard constraints don't survive temp 0.4 |
| Gate LLMs S4, S5+S6+S7 | **eliminated / merged** | — | — | see 2.6 |

All model/temperature choices live in one registry in `llm.py` with a one-line comment each. Fix the auditor logging mismatch (`gpt-5.4-mini`) by deriving the log from the same resolver as the factory.

## 2.5 Rules consolidation

One source of truth per rule; enforcement placement:

| Rule | Lives in code | Stated in prompt | Action |
|---|---|---|---|
| Catalogue | `CANONICAL_CATEGORIES` | rendered via `catalogue_line()` | delete 10-type list + hardcoded example; remove/park Welding spec |
| Width exclusions | `_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES` → constants.py | rendered clause in M1 prompt + fallback | one set, two renderings |
| Range-smallest / loose-answer / width-numeric | validators (`_validate_slot_value` extended) | one shared `answer_classification_policy()` block injected into G5/G7/G8/M2 prompts verbatim | ends the 5-way drift |
| FAQ/escalation triggers | deterministic overrides stay as backstop | one `faq_escalation_policy()` block shared by mind/G6/G7/S6 | 6 → 1 definition |
| Category resolution | tiered resolver (naming>cargo, position tiebreak) in categories.py; hint advisory | CATEGORY_RULES (dedup'd) | code+prompt agree; disagreement logged |
| Phone/URL | constants + programmatic injection/verification | prompts told NOT to emit phone digits | bit-exact guarantee |
| Make aliases | single `make_aliases.py` consumed by resolver/search/normalizer | — | 3 → 1 |
| Unit parsing | single `units.py` (superset: yd/metric) used by ingest + search | — | 2 → 1 |
| A×B orientation | one function, one documented convention (width×length per existing prompt+code majority); delete the bare-inch→length override or make it consistent | extractor prompt references the convention | contradiction resolved |
| Skip counter | code: counter-questions don't increment | G7 prompt states the same | spec restored |
| Confusion threshold | Python only; prompt asks for ordinal signals, not a calibrated 0–100 | — | one encoding |

## 2.6 Optimizations (fewer LLMs, fewer calls, lower latency)

1. **Delete 2 dead LLM stacks** (service policy pair, graph legacy extractor) and **remove S4** (always-save) — −3 defined LLMs, −1 call on turn 1.
2. **Merge the routing gates S5+S6+S7 into one structured router call** (`RoutingDecision{route: graph|smalltalk|catalogue_redirect|escalation_route, reason}`) invoked only when the existing deterministic pre-checks are inconclusive — worst case 3 calls → 1, and one shared definition ends the §1C routing overlap.
3. **Merge M2 (no-preference) into G7 (question adjudicator)** — G7's schema already contains `no_preference_for_active_question`; M2 exists only as a second opinion for G8. With the shared policy block (2.5) the second opinion is redundant → −1 call per active-Q&A turn. Keep G8 (reconciler) as the arbiter of G1/G5/G7.
4. **Fold G4 (feature extractor) into G5's schema output** where the search node currently calls both sequentially (G5 already returns `requested_non_metadata_features`; the search node re-extracts) — −1 call per search.
5. **Parallelize the independent inputs to the reconciler** (G5 extractor, G7 adjudicator, M1 haul) with `ThreadPoolExecutor` — they share inputs, not outputs; ~2–3× faster active turns even before merging.
6. **Async email delivery**: the durable path already has an outbox — route the in-memory path's blocking SMTP sends through the existing (currently unused) executor pattern; removes up to 30 s from worst-case turns.
7. **Caching**: cache `make_resolver._valid_make_map` (`lru_cache`); single shared Excel load for `prepared_inventory`/`load_make_inventory`; leave embeddings uncached (queries rarely repeat verbatim).
8. Net effect: worst-case sequential LLM calls per ordinary turn drop from ~14–15 to ~7–8, with the remaining chain partially parallel.

---

# PHASE 3 — IMPLEMENTATION SPEC

## Ground rules for the implementing session
1. Read `overview.md` §10 invariants first; treat them as regression contract. Line numbers below are anchors from branch `v4.4` — relocate by symbol name if drifted.
2. One stage per commit minimum; run `uv run pytest -q` after every work item and compare against a baseline recorded **before any change** (`uv run pytest -q 2>&1 | tail -5`).
3. Surgical edits — do not reformat surrounding code. Preserve feature-gating (missing OPENAI/DB/SMTP keys must never crash).
4. Comment the *why* on every prompt/guard you touch (task.md Phase 3 requirement). Flag every ASSUMPTION marked below in your final summary.
5. Live tests (user-approved): run the named `-k` subsets of `tests/test_live_prompt_audit_issues.py` (env `RUN_LIVE_PROMPT_AUDIT=1`) after Stages B and C only — not repeatedly.

## Stage A — Dead code & hygiene (zero behavior change)
**A1.** Delete the dead code inventoried in §1F (service policy LLMs + schemas, `_message_for_routing` pair, unused constants; graph legacy extraction stack + `FilterExtractionDecision`; email_sender async path + ABC stub; make_resolver `use_llm_fallback`; trailer_fields dead slot; inventory_matcher no-op ternaries; vestigial `reply_source`). For app.py 1453–1456: **move** the hint text before the `return False` (it's a useful message, currently unreachable) rather than deleting.
**A2.** Fix the auditor input-log model mismatch: extract `_pinecone_audit_model_name()` used by both factory (649) and log (1521).
**A3.** Rename/convert `tests/test_categories.py` into a real test or move it to `scripts/inspect_categories.py` (ASSUMPTION: move to scripts).
**Verify:** full offline suite green vs baseline; `rg "_filter_extractor_llm|ContactPolicyDecision|_message_for_routing|enqueue_faq_email_notification"` returns only test monkeypatch references, which must be updated or removed with their tests.

## Stage B — Correctness: category resolution (the P0 pair)
**B1. Tiered resolver** (`categories.py:100–114`): split `_SYNONYMS` into `_NAMING_TERMS` (the type words: tilt, dump, flatbed, enclosed, …) and `_CARGO_TERMS` (tractor, atv, cattle, scissor lift, brand words, …), keep `_SYNONYMS` as merged view for existing imports. Rank matches: naming > cargo, then earliest position, then longest term. `resolve_categories_from_text` returns ranked list. Expose the winning tier on `CategoryResolution` (new field `match_tier`). ASSUMPTION: term classification is mechanical (type-word vs cargo/brand) — list it for review.
**B2. Advisory override** (`graph.py:4803–4806`): keep hint when LLM returned nothing; keep when they agree; on disagreement prefer a **naming-tier** hint, otherwise keep the LLM's canonical category with `category_confidence="medium"` (so G9 reconciliation still guards). Log `category_hint_disagreement`.
**Verify:** new unit cases — "tilt trailer to haul tractor"→Tilt, "dump trailer for my motorcycle"→Dump, "livestock trailer and a lawn mower"→Livestock, "I need to haul a tractor"→Equipment, "utility trailer to haul equipment"→Utility. Run `tests/test_graph_metadata_extraction.py` (3,397-line suite) in full. Live: `-k categor`.

## Stage C — Hallucination guardrails
**C1. Auditor scenarios count-relative** (`graph.py:1464–1483`): inject `len(listings)` into the prompt at call time; Scenario C = `full_match_count == {len(listings)}`, wording softened to claim only audited dimensions: "Every trailer below matches the trailer type and features you asked for." Update the "unless full_match_count = 6" rule line (1482) and any hardcoded 6 in `_sanitize_requested_feature_analysis`/post-validation (1546–1581) to `len(facts)`-relative.
**C2. Blurb validation parity**: run `sales_blurb`/`customer_label` through `_invalid_pinecone_intro`-equivalent checks inside `_sanitize_requested_feature_analysis` (or a sibling `_invalid_sales_blurb`); on failure blank the blurb so `formatting.py` falls back to its deterministic why-line (it already does at 240 when blurb is absent).
**C3. Phone/URL programmatic injection** (`email_reply.py` + constants): constants `TRAILERPLACE_PHONE`/`TRAILERPLACE_URL`; after generation, regex-verify any phone-like token equals the constant (replace if not); if the fallback contained the phone but output lost it, append "You can reach our team at {PHONE}."; same for URL. Guard: only enforce when the fallback contains the constant (don't inject into unrelated replies). Consolidate the 5+ hardcoded copies to the constants.
**C4. Haul-classifier retry reframe** (`mini_llm_classifier.py:179–183`): null-affirming instruction ("if no specific cargo is explicitly stated, matched_item=null AND both flags false AND low confidence is the correct answer"); after retry, deterministically force flags False when `matched_item` is empty.
**C5. Feature negation/informational filter**: after G4/G5 feature extraction, deterministic post-filter dropping features whose source-message context matches negation patterns (`no |don't need|do not need|without`) or when the message is interrogative-informational with no request verb. ASSUMPTION: regex post-filter rather than prompt-only fix.
**C6. Catalogue single-sourcing** (`prompts.py:21`, `~96`; `categories.py`; `trailer_fields.py:250`): derive KNOWLEDGE line and the ownership example from `CANONICAL_CATEGORIES` (+ static description map — ASSUMPTION on Fiber/Race/Diesel descriptions); move the Welding spec behind a comment or delete (ASSUMPTION: park with comment).
**Verify:** `tests/test_pinecone_rerank.py`, `test_listing_formatting.py`, `test_faq_reply_node.py`, `test_email_tools.py`, `test_mini_llm_classifier.py`; new unit tests for C3 (transposed digits corrected; no false injection) and C5 (negated/informational dropped, positive kept). Live: one search conversation + `-k "feature or width"`.

## Stage D — Flow fixes
**D1. Turn-1** (`service.py:189–206`, 2309–2331): delete S4 + its schema; always save non-empty first message; deterministic acknowledge-then-ask template ("Happy to help you find the right trailer. Before we start — could I get your name and phone or email? You're welcome to skip this."). ASSUMPTION: keep the optional contact ask on turn 1 (dropping it entirely is a product decision — flag it).
**D2. S2 fallback** (`service.py:867–869`): regex refusal/skip (`no|nope|skip|rather not|not now|prefer not` + decline phrasing) → resume-saved-request action; clearly-new-request heuristic → route_latest_request; default resume.
**D3. Skip counter** (`graph.py:6187–6203, 6300–6301`): counter-question turns (G7 `counter_question_topic` set, not answered) must not increment `active_question_attempts`. Keep email-interrupt behavior as-is (documented contract says those also shouldn't count — align both; ASSUMPTION flagged).
**D4. Windowing helper** (`constants.py` + `compact_recent_messages(messages, n=8, char_cap=500)`): replace the 12 slice sites in graph.py/service.py; planner context (4753–4775) additionally: listings → `{index,title,url,price}` projection, drop `listing_model_fields`.
**D5. Temperature standardization**: S8 0.4→0.2, S3 0.3→0, I2 0.3→0, I3 0.25→0, each with a rationale comment (via the Stage E registry if it lands first, else in place).
**Verify:** `tests/test_optional_contact_flow.py` (2,834-line suite; update turn-1 expectations), `test_chat_service_confusion_escalation.py`; manual smoke: "I need a 12 ft livestock trailer" → acknowledged + contact ask → "no thanks" → Livestock qualification proceeds; counter-question twice on one slot → slot not skipped.

## Stage E — Consolidation & structure (incremental extractions)
**E1. `constants.py`**: thresholds (confusion 85/0.86/repeat 2-3), windows, phone/URL, `_DYNAMIC_WIDTH_EXCLUDED_CATEGORIES` (move from graph.py:1769; render M1's prompt clause + fallback set from it), `_ALLOWED_HITCH_TYPES`. Remove the 85-threshold from the S9 prompt text (Python is the enforcer; prompt asks for boolean+ordinal signals).
**E2. `llm.py` central factory**: `make_llm(role: str)` resolving model (per-role env → `OPENAI_MODEL` → default), temperature registry, structured-output method, and a `safe_invoke()` that catches `ValidationError` + coerces/logs enum drift (`structured_output_coerced | role=… raw=…`). Migrate all factories (graph 495–659, service 339–396/1090/1319/2043–2056, inventory 284/301/315, mini classifiers) — mechanical, one module at a time.
**E3. Shared policy blocks** (2.5): `answer_classification_policy()` injected into G5/G7/G8 (+M2 while it exists); `faq_escalation_policy()` into mind/G6/G7/S6; delete the duplicated CATEGORY_RULES blocks (prompts.py 145–155).
**E4. Alias/unit unification**: `make_aliases.py` (single map + short/long form handling) consumed by make_resolver/pinecone_search/normalizer; `units.py` (superset parser incl. yd/metric) used by ingest + pinecone_search; resolve the A×B orientation contradiction (keep width×length; delete/align the `bare_trailer_inches` length override at graph.py:5423 — ASSUMPTION flagged).
**E5. Merged router** (2.6 #2): one `RoutingDecision` call replacing S5/S6/S7 invocations inside `_should_route_to_graph`; deterministic pre-checks unchanged; delete the three old prompts after the contact-flow suite passes.
**E6. Reconciler-input parallelization + M2 merge + G4 fold** (2.6 #3–5): first parallelize G5/G7/M1 via executor (no semantic change), then remove the separate M2 call (G7's `no_preference_for_active_question` + shared policy block covers it; G8 unchanged as arbiter), then stop re-extracting features in the search node when state already holds them.
**E7. `build_state_return()`**: single state-assembly helper replacing the 10 `return {**state,…}` sites in `_apply_mind_node`; then split `apply_mind` into `graph/apply_mind/{category,active_turn,questions,actions}.py` preserving call order exactly. This is the largest mechanical step — do it **last**, after all behavioral fixes are green, so diffs stay reviewable.
**E8. Async email** (2.6 #6): in-memory path routes sends through a background executor; durable path unchanged (outbox). Cache `_valid_make_map`; unify the double Excel load behind one loader.
**Verify per item:** full offline suite; after E5/E6 run live `-k "trichotomy or routing"` equivalents plus one full manual conversation each through: qualification→search→show more, FAQ email, escalation, listing interest, category switch.

## Stage F — Deferred (needs product decisions / separate sessions)
- gpt-5-mini for planner+reconciler (config flip prepared by E2; needs A/B evaluation).
- Confusion detection redesign (ordinal enum + deterministic repeats).
- Feature ontology (canonical feature ids for the auditor).
- Rolling summarization for very long threads.
- Unifying the two listing-card renderers into one component.
- Outbox retry scheduler + `claimed_at` usage.
- In-memory session locking (or dropping the in-memory path entirely where DB is guaranteed).

## Verification (end-to-end, after all stages)
1. Offline: `uv run pytest -q` — green vs pre-change baseline (record both).
2. Live (approved): `RUN_LIVE_PROMPT_AUDIT=1 uv run pytest tests/test_live_prompt_audit_issues.py -v` once after Stage C and once after Stage E; one `RUN_LIVE_VAGUE_QNA=1` subset (`-k` two categories) as the final gate.
3. Manual API smokes (backend `uv run python main.py`): (a) "I want a tilt trailer to haul a tractor" → Tilt track; (b) turn-1 acknowledge + decline → request resumes; (c) full qualification → search → intro wording + cards → "show more" → exclusions hold; (d) "how do I contact you" → reply contains exactly 979-532-1486; (e) counter-question twice → slot not skipped; (f) category switch post-results → new cycle.
4. Update `architecture.md`/`details.md` (new modules) and mark fixed items in `PROMPT_AUDIT.md` with a Status column.
5. Final summary must list every ASSUMPTION: B1 term classification, C5 regex filter, C6 descriptions + Welding parking, D1 keeping turn-1 contact ask, D3 email-interrupt counting, E4 orientation convention, A3 test relocation.
