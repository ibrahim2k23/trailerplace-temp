# Milestone 0-9 Details

## `main.py`

Creates the FastAPI app with `create_app()` and runs uvicorn on `settings.chatbot_api_port`, defaulting to `8000`. Since M8 the app carries a `lifespan` handler, so a real uvicorn boot runs migrations, compiles the graph, and probes the DB before `/health` reports `ok`.

## `src/config.py`

Defines `Settings.from_env()` and module-level `settings`. It loads `.env`, parses booleans/ints/floats defensively, and centralizes the env keys listed for Milestone 0. Later modules should import `settings` instead of calling `os.getenv` directly.

`TRAILERPLACE_PERSIST_CHATS` now defaults to enabled unless explicitly disabled; durable persistence still stays off unless the DB settings are complete.

M8 adds two keys: `chat_timeout_seconds` (`CHAT_TIMEOUT_SECONDS`, default `150.0`) bounds every server-side graph run and must stay below the `timeout=180` app.py passes to `requests.post(/chat)`; `chat_max_message_chars` (`CHAT_MAX_MESSAGE_CHARS`, default `4000`) caps an inbound message. Note `Settings` is a **frozen** dataclass — tests must swap the whole object (`monkeypatch.setattr(routes, "settings", dataclasses.replace(...))`), not one field.

M6 fix: `rerank_warn_ratio`/`rerank_extreme_ratio`/`rerank_length_weight`/`rerank_missing_dim_penalty` previously defaulted to `0.0` when unset in `.env` (only `RERANK_WARN_RATIO`/`RERANK_EXTREME_RATIO` are actually set there), which would have zeroed out the fit-rerank's length-weighting and missing-dimension penalty entirely once `search.py` went live. Defaults now match the reference `pinecone_search.py` module constants (`1.35`/`1.9`/`8.0`/`0.35`) so an unconfigured deployment reranks the same way the reference implementation did.

## `src/db.py`

Creates the SQLAlchemy Postgres engine from `HOST`, `PGUSER`, URL-quoted `PASSWORD`, `DATABASE`, and `PORT`. Uses `postgresql+psycopg`, `sslmode=require` for non-local hosts, cached engine/session factory helpers, `database_enabled()`, and `ensure_schema()` via `Base.metadata.create_all()`.

M8 adds `run_migrations()` — builds `alembic.config.Config` from the repo-root `alembic.ini` (resolved off `__file__`, not the cwd) and runs `command.upgrade(cfg, "head")`; `alembic/env.py` already resolves the engine through `get_engine()`, so no URL is threaded through. Called at boot only when `DB_AUTO_CREATE=1`. Also `ping() -> bool`, a `SELECT 1` that logs and returns `False` rather than raising — the `/health` readiness probe.

## `src/db_models.py`

Defines the four durable persistence models: `ChatbotLead`, `ChatbotConversation`, `ChatbotTurn`, and `ChatbotOutbox`. These match the existing Alembic migrations: leads use `psid` as the session lookup key; conversations store simplified `conversation` JSONB plus full `state_snapshot`; turns use `(session_id, turn_id)` as idempotency receipts; outbox rows are unique on `(session_id, turn_id, event_key)`.

## `src/log_setup.py`

Provides `configure_trailerplace_logging()`. It configures stdlib logging once and updates the root log level from `LOG_LEVEL`, defaulting to `INFO`.

## `src/models.py`

Provides `TrailerListing`, a lenient dataclass matching `app.py` card construction/rendering fields. It accepts strings, numbers, and `None` for listing fields so frontend card creation does not silently drop valid listings.

## `src/thinking_agent.py`

Frontend compatibility shim. The thinking agent is out of scope, so all enablement functions return `False`, generation returns an empty successful result, and logging is a no-op.

## `src/conversation_store.py`

Durable persistence facade used by the frontend and later graph code. `persistence_enabled()` requires chats not disabled plus complete DB config. Lead helpers create/update soft leads, contact status, item of interest, and hard promotion. `durable_turn()` takes a Postgres advisory transaction lock, loads the conversation row and receipt, rejects mismatched replay messages, and commits/rolls back atomically. Conversation helpers maintain the simplified turn list, preserve feedback, restore sessions from snapshot or simplified turns, close sessions without deleting rows, and store feedback by frontend turn index. Outbox helpers enqueue transactional events, register handlers, and drain pending/failed rows with `FOR UPDATE SKIP LOCKED`.

**M7 additions**: `promote_lead_to_hard(session_id, *, session=None)` and `update_lead_item_of_interest(session_id, item, *, session=None)` accept an optional caller-owned session so the route can apply the lead upgrade / item update inside the same `durable_turn` transaction as the outbox rows (falling back to a short-lived committing session when called standalone). `register_default_outbox_handlers()` (invoked by `create_app`) maps all six outbox `event_type`s — `interested_listing`, `non_sales_faq`, `escalation_alert`, `team_request`, `results_shown`, `unanswered_question` — to a single handler that calls `src.tools.email_sender.send_email(subject, body)` and raises on a False return so the drain marks the row `failed` (retryable). Every outbox payload is `{subject, body}`, so all event types share one handler.

## `src/llm/schemas.py`

Defines strict pydantic structured-output models for the two LLM calls. `TurnAnalysis` contains intent, email triggers, haul classification, inventory lookup identifiers, category mention, extracted fields, slot answers as pair objects, contact info, listing reference, dropped/kept fields, and interruption flags. `ReplyOutput` contains assistant text and cited listing URLs. All models forbid extra fields and avoid open-ended dicts.

## `src/llm/client.py`

Defines the injectable `LLMClient` protocol with `structured(system, messages, schema)`. `OpenAILLMClient` calls OpenAI structured outputs using `settings.openai_model` and pydantic response parsing.

## `src/llm/analyze.py`

Builds the Analyze system prompt and message context. The builder reads the **real `SessionState` keys** — `category`, `pending_question_slot` (resolved to its question text via the field spec, with the injected-width slot handled), `slots`/`slot_sources`, `skipped_slots`, no-preference slots derived from `None`-valued slots, flat `customer_name`/`customer_email`/`customer_phone`, `clarification_key` (rendered to its question), and `pending_category_change`. The prompt injects category and make blocks, current state, pending question/repeat count, collected/skipped/no-preference slots, contact status, shown listing titles, intent rules, extraction/unit-conversion rules, haul classification rules, and inventory lookup rules. `analyze_turn()` calls the injected client with `TurnAnalysis`. `normalize_analysis_values()` is a pure safety net: it rescues stray-string numerics on `extracted` only and passes `slot_answers` through untouched (metadata-target mapping moved to apply_analysis).

## `src/llm/respond.py`

Builds the Respond system prompt and message context. Decision lines read from the correct sources: `pending_question_repeats` and `pending_category_change` from state, the interruption question from the analysis object, and turn-scoped signals (next question, search/inventory match status, contact suppression, canned keys, email status, contact ask) from `turn_outcome`; customer name and collected slots come from the flat state keys. The prompt includes system-decided turn outcome lines, inventory match-status instructions, contact suppression, canned response text by key, email/contact status, listing summaries, store facts, advertised categories, and customer context. `respond_turn()` calls the injected client with `ReplyOutput`.

## `src/graph/state.py`

Defines `SessionState`, `new_session_state()`, process-local `_sessions`, `_get_session()`, `clear_session()`, `to_snapshot()`, and `from_snapshot()`. Snapshots include durable session fields and omit transient `turn` / `turn_outcome`.

## `src/graph/build.py`

Builds one compiled LangGraph. The graph runs Analyze, Apply Analysis, Email Actions, routing, Qualification/Search/Inventory (real as of M6), post-results Email Actions (the `email_actions_after_results` second pass), and Respond. Routing reads `state.turn.intent`, inventory lookup confidence, clarification/category-change flags, and qualification completion. Topology is unchanged since M4; M7 only made the two `email_actions` node instances real (both call the same `email_actions_node`, which is idempotent across the two passes).

## `src/graph/apply_analysis.py`

Deterministic state mutation layer. It merges contact info, handles contact decline, enforces haul-classification invariant, resolves category clarification, applies category/defaults, preserves the haul-item lock, resolves category-change keep/drop answers, records slot answers/extracted fields, handles numeric no-preference, stores brand/non-metadata features, handles skip-current/skip-all, repeat-once auto-skip with `unanswered_question` trigger, drops requirements, and injects `item_or_trailer_width_ft` when heavy/wide cargo requires it.

Slot answers are stored via `_store_slot_answer`: for a slot with metadata targets (e.g. `cargo_size` → `length_ft`, `width_ft`), each target is parsed with `normalize_slot_value` and the **number** is stored under the target key — never a re-encoded `"key=value"` string — while the category slot keeps the raw answer as its answered marker. A final guard marks any LLM-confirmed-answered pending slot as `None` (no-preference) when it produced no parseable value, so vague/partial answers still advance to the next question.

Category clarification (`_apply_clarification`) uses the ported domain resolver on the latest user message (domain-data resolution, not the LLM intent decision): when an ambiguous trigger (e.g. "office trailer") is seen before a category is chosen, it sets `clarification_key` plus `turn_outcome["clarification_question"]` and skips category application that turn; the next message is run through `resolve_category_clarification_answer` to pick the category (via the shared `_start_category` helper) or re-ask. The router's existing `clarification_key` branch is now live.

## `src/graph/nodes/*`

Node wrappers for the compiled graph. `analyze.py` calls M3 Analyze, `apply_analysis.py` calls deterministic state mutation, `qualification.py` chooses the next unanswered required slot including injected width, and `respond.py` calls M3 Respond and appends timestamped assistant messages. `respond.py` defers the first-turn contact invite when `turn_outcome["inventory_lookup_ran"]` is set (rather than any stub marker). `email_actions.py` is the real M7 node (see its own section below).

`search.py` (real as of M6): asserts `qualification_complete` (the route already enforces this; the assert is a defensive backstop). Builds `metadata_filters` from `state["slots"]` by running each answered slot key through `src.domain.slot_map._SLOT_METADATA_FILTER_MAP` + `normalize_slot_value(category, target_key, value)` — this is the same translation `apply_analysis._store_slot_answer` uses, so a slot like `haul_length_ft` or the Roll Off `bin_size` special case lands on the same `length_ft` target key `pinecone_search.py` expects. A single-item `hitch_type` list (`["Bumper Pull"]`, from `ExtractedFields.hitch_type`) becomes a plain string filter; a two-item list (either acceptable) is left out of the filter entirely. `state["brand_preference"]` becomes the hard `make` filter (Locked Decision). Calls `src.search.pinecone_search.search_pinecone_listings` with `already_shown_urls=state["shown_urls"]` for dedupe and `settings.search_top_k`/`settings.search_max_recommendations`. **Zero-result brand fallback**: if a `make`-filtered search returns nothing, it re-runs once with `make` stripped from the filters and sets `turn_outcome["brand_relaxed"] = True` so Respond can say no matches for that brand were found. Appends results to `state["shown_listings"]`/`shown_urls`, writes `turn_outcome["listings"]`/`search_ran`/`result_count`, and — when results are non-empty — appends a `results_shown` system-alert trigger (`description = "Pinecone search — {n} results — {category} ({filters})"`) to `turn_outcome["system_email_triggers"]` for M7's email-actions node to gate/send. Reuses `src.graph.apply_analysis._current_user_text` (the same "last user message" helper `apply_analysis.py` already has) rather than re-deriving it.

`inventory_lookup.py` (real as of M6): re-asserts the gate `build.py`'s `_route` already enforces (`turn.intent == "inventory_lookup"` and `turn.inventory_lookup.is_lookup` and `confidence in {"medium","high"}`) as a defensive backstop. Calls `src.search.inventory_matcher.lookup_inventory` with the four identifiers straight from `TurnAnalysis.inventory_lookup` (no user-text parsing in this node — Analyze already did that). Writes `turn_outcome["inventory_result"]` (the full `{match_status, matches, requested_label}` dict) and `turn_outcome["inventory_match_status"]`, which `src/llm/respond.py`'s `_decision_lines` already reads to build the exact/no_exact/ambiguous prompt block (built in M3). Appends only the **not-already-shown** matches to `state["shown_listings"]`/`shown_urls` (dedupe by URL) so re-showing the same stock number twice in a session doesn't duplicate cards. **Preserves the M4 stub's exact contract**: `turn_outcome["inventory_lookup_ran"] = True` and `turn_outcome["contact_invite_suppressed"] = True` unconditionally whenever this node runs — `respond_node` (`src/graph/nodes/respond.py:14`) uses `inventory_lookup_ran` (not `contact_prompted_initial` state) to decide whether to skip the first-turn contact ask, which is what makes the invite defer to the next user turn per the Locked Decision. **Side-query invariant**: asserts `category`/`slots`/`brand_preference`/`skipped_slots`/`qualification_complete` are byte-for-byte unchanged from entry to exit — a lookup is a side-question, not a qualification answer, and this assert is the enforcement of that milestone rule. When matches are non-empty, appends a `results_shown` system trigger (`description = "Inventory lookup — {n} results — {requested_label}"`), identical in shape to the search node's trigger so M7's email-actions node handles both uniformly.

## `scripts/smoke_llm.py`

Manual real-OpenAI smoke script. It runs Analyze for: `I want a 7x14 dump trailer for hauling dirt, I'm John, 555-1234` and prints the parsed `TurnAnalysis` JSON. This is not run by the test suite.

## `alembic.ini` and `alembic/*`

Alembic is configured for the existing migration directory. `env.py` imports `src.db.get_engine()` and `src.db_models.Base.metadata`. `20260525_0001_optional_contact_leads.py` creates/updates `chatbot_leads` and `chatbot_conversations`; `20260703_0002_durable_chat_state.py` adds state snapshot/version/closed columns, `chatbot_turns`, and `chatbot_outbox`.

## `src/shown_listings_store.py`

Keeps an in-process map of shown URLs by session id and exposes helpers used by `app.py`. The accumulator handles object listings, dict listings, missing listings, and `listings=None`.

## `src/api/*`

`app.py` contains the FastAPI factory. `routes.py` exposes `/health`, graph-backed `/chat`, text-only `/session/{id}`, `/session/reset`, and (M5) `GET /session/{id}/state`. `/chat` runs in-memory when persistence is off. `_handle_chat_in_memory` (used by both modes) unions the frontend-supplied `already_shown_listing_urls` into backend `shown_urls` and appends timestamped user messages, so dedupe works even with persistence off. When persistence is on, it wraps the turn with `durable_turn()`, replays receipts, loads snapshot state, creates/gets a soft lead, and — only when a lead id backs the FK — saves `state_snapshot`, simplified conversation, state version, and turn receipt, then drains outbox after commit. `schemas.py` defines pydantic models for those routes, plus M5's `DebugStateResponse`.

`GET /session/{id}/state` (M5) returns `{exists, state}` where `state` is `to_snapshot()` of the live in-memory session (`_sessions[session_id]`) — the same shape the durable layer writes to `state_snapshot`, so scenario assertions read exactly the field names used everywhere else in the codebase (`category`, `slots`, `skipped_slots`, `pending_question_slot`, `pending_question_repeats`, `clarification_key`, `brand_preference`, `customer_name`, ...). Gated behind `settings.debug_state_endpoint` (`DEBUG_STATE_ENDPOINT` env var, already defined in `src/config.py` since M0 but unused until now) — returns 404 when the flag is off, so it never appears in a normal deployment. Unknown session id returns `{"exists": false, "state": null}` rather than 404, so the scenario runner can distinguish "endpoint disabled" from "session not created yet."

**M6**: `_handle_chat_in_memory` now filters `turn_outcome["listings"]` against `reply.cited_listing_urls` when `settings.show_only_llm_mentioned_cards` is set (its `.env` default is `true`) — only listings the Respond LLM actually mentioned in its reply text are sent to the frontend as cards, matching the milestone's "only LLM-mentioned cards" behavior for both search and inventory-lookup results (both write `turn_outcome["listings"]` in the same shape, so this filter is agnostic to which node populated it). When the flag is off, or there was no `reply` (should not happen in practice), all of `turn_outcome["listings"]` passes through unfiltered.

**M7**: `create_app()` calls `conversation_store.register_default_outbox_handlers()` at startup so the post-commit drain can send. In `/chat`, after `graph.invoke` and only when a lead id backs the FK, the route reads the live session's `turn_outcome["outbox_events"]` and enqueues one `chatbot_outbox` row per event (via `enqueue_outbox_event`) inside the durable transaction; if any events exist it calls `promote_lead_to_hard(..., session=db_session)`, and if `turn_outcome["lead_item_of_interest"]` is set it calls `update_lead_item_of_interest(..., session=db_session)` — all committed atomically with the snapshot/receipt, then `deliver_pending_outbox()` sends after commit. Persistence-off turns never reach this block; the email-actions node sends directly instead. The debug `GET /session/{id}/state` route additionally merges the live `turn_outcome["emails_sent"]` list onto the returned snapshot (which normally drops transients) so M7 scenarios can assert `expect_emails_sent` against the last turn's decisions.

## `tests/unit/test_scaffold.py`

Covers the health route, frontend shim import surface, settings parsing/defaults, lenient listing construction, and shown-listing URL accumulation.

## `src/domain/categories.py`

Ported category resolver data and helpers. Keeps canonical categories, two-tier naming vs cargo synonyms, clarification rules, `CategoryResolution.match_tier`, prompt block generation, and the Gooseneck/Bumper Pull category guard. Also exposes `make_prompt_block()` as a lazy wrapper around `src.domain.brands.make_prompt_block()` for the milestone import check.

## `src/domain/trailer_fields.py`

Ported dealership question specs. Provides `TrailerFieldSpec`, per-category required/optional slots, questions, answer guidance, `_DEFAULT_SPEC`, merged loose-answer guidance in `get_trailer_fields_as_dict()`, and `list_all_categories()`.

## `src/domain/brands.py`

Ported make inventory logic from `make_inventory.py`. Loads `listings_final_v5.xlsx`, normalizes makes/categories, exposes `known_makes()`, `make_prompt_block()`, `categories_for_make()`, and `make_filter_values()`. If the optional Excel reader dependency is absent, it returns an empty inventory instead of crashing the prompt block.

## `src/domain/make_aliases.py`

Ported alias-to-canonical make map. This is shared data for catalog normalization and later search/rerank logic; it is not user-text regex extraction.

## `src/domain/normalizer.py`

Ported listing-catalog normalizers for category, subcategory, make, color, hitch, condition, category/subcategory display, dealer-note cleanup, and embedding text construction. Imports aliases from `src.domain.make_aliases`.

## `src/domain/units.py`

Ported catalog parsers for length in feet and weight in pounds. These are safety-net/listing-data parsers, not user-input extraction logic.

## `src/domain/defaults.py`

Defines empty `CATEGORY_DEFAULTS` plus `defaults_for(category)`. Includes a commented example for future category defaults. M5 adds an optional import-time merge from `CATEGORY_DEFAULTS_JSON` (a JSON-encoded dict, e.g. `{"Utility": {"hitch_type": "Bumper Pull"}}`) so the operator can seed a default before starting the server for the `defaults-applied` live scenario, without hand-editing this file — unset/empty env var is a strict no-op, so it cannot affect any other scenario or milestone's behavior.

## `src/domain/slot_map.py`

Defines `_SLOT_METADATA_FILTER_MAP` from the Pinecone spec and `normalize_slot_value()`. Handles Roll Off bin-size-as-length and delegates other length/payload parsing to domain unit parsers.

## `src/domain/canned_responses.py`

Contains exact FAQ and non-FAQ canned response strings from the spec, split into `FAQ_CANNED_RESPONSES`, `NON_FAQ_CANNED_RESPONSES`, and combined `CANNED_RESPONSES`.

## `tests/unit/test_domain_*.py`

Covers category salience/guards, slot-map coverage, unit parsing, normalizers, brand fixture loading, defaults, and canned response keys.

## `tests/integration/test_db.py`

Marked `db` and skipped unless `TEST_DATABASE_URL` is set to a database whose name contains `test` or `ALLOW_DESTRUCTIVE_DB_TESTS=1` is explicitly set. These tests reset tables in the target DB, so they must run only against a disposable/local test database. Covers Alembic `upgrade head`, table existence, durable turn receipt replay/mismatch behavior, lead updates, conversation restore/close, and outbox delivery through a registered fake handler.

## `tests/unit/test_apply_analysis.py`

Milestone 4 deterministic state tests. Covers contact merge/decline, defaults and user override, haul-item lock, category-change keep flow, category clarification trigger-then-resolve, skip current/all, repeat-once auto-skip with system trigger, drop requirements, no-preference storage, width injection/exclusion, and invariant enforcement.

## `tests/unit/test_qualification.py`

Milestone 4 qualification tests. Covers spec question order, answered/skipped slots, qualification completion, and injected width question sequencing.

## `tests/integration/test_chat_roundtrip.py`

Milestone 4 FakeLLM integration tests for persistence-off `/chat`, state carryover across turns, text-only restore, reset pop/close, and Equipment + tractor width-question round trip.

## `tests/conftest.py`

Adds `FakeLLM`, a queued structured-output fake that records calls and returns schema instances. It is used by LLM prompt/plumbing tests and avoids real OpenAI calls.

## `tests/unit/test_llm_*.py`

Milestone 3 LLM-layer tests. They cover strict schema round-trips and invalid literals, Analyze prompt required blocks/rules, inventory lookup guard text, FakeLLM plumbing, numeric safety-net normalization, Respond prompt canned strings/listings/match statuses/contact handling, and haul classification invariant behavior. `tests/unit/llm_helpers.py` provides a shared sample `TurnAnalysis` builder.

## `scripts/convo_runner.py`

Milestone 5 live scripted-conversation harness. Split into pure/testable and network layers:

- `load_scenario(path)` / `load_scenarios(dir_or_file)`: `yaml.safe_load` a single file or every `*.yaml` in a directory into `{name, turns: [...]}`; raises `ValueError` if `name`/`turns` are missing.
- `evaluate_assertions(turn, *, reply_text, state, listings)`: pure function returning a list of failure-message strings (empty = pass). Implements every assertion kind named in the milestone:
  - `expect_state`: dict match against the raw snapshot. Two suffix conventions extend plain equality: `"<list_field>_contains"` checks membership in that list field (e.g. `skipped_slots_contains: "haul_material"`, straight from the milestone's own example), and `"slot:<name>"` checks `state["slots"][name]` (e.g. `"slot:trailer_length_ft": 15.0`) — introduced because individual qualification-slot values are nested one level under `slots` in the snapshot and the milestone's own assertion vocabulary is flat top-level keys.
  - `expect_reply_contains_any` / `expect_reply_not_contains`: case-insensitive substring checks.
  - `expect_listings`: bool vs. truthiness of the returned listings list.
  - `expect_emails_sent`: ordered-list-of-Reason-strings compare against `state.get("emails_sent")`; `null` means "no-op unless something is already there." This field is forward-compatible with M7 — nothing sets `emails_sent` in state yet, so it is unused by any current M5 scenario.
- `run_scenario(base_url, scenario, session_id=None)`: the network-touching part — generates a fresh `uuid4` session id, POSTs each turn's `user` text to `{base_url}/chat`, GETs `{base_url}/session/{id}/state` for the post-turn snapshot, and evaluates assertions per turn.
- `main()`: CLI (`argparse`) — accepts a scenario file or a directory (`--all` conceptually implied when a directory is given), prints a pass/fail transcript per turn including the failing assertion text, exits nonzero on any failure. Uses only already-pinned dependencies (`requests`, `pyyaml`).

Not run by the pytest suite (it makes real HTTP calls to a running backend and real OpenAI calls) — its parsing/assertion logic is covered without network by `tests/unit/test_convo_runner_parsing.py`.

## `scripts/scenarios/*.yaml`

The 20 scenarios named in `milestone.md`'s Milestone 5 section, one file each, no more/fewer. Each follows the shape from the milestone's own example (`name` + `turns`, each turn a `user` message plus zero or more `expect_*` keys) and is grounded in the actual M0–M4 implementation it exercises:

- **Contact collection** (`contact-full`, `contact-partial-name`, `contact-decline`, `contact-ignore-asks-trailer`): the four spec §Contact Collection cases plus the "ignore and lead with a trailer question" path — exercise `_apply_contact`'s merge/decline logic in `src/graph/apply_analysis.py`.
- **Category mapping** (`category-direct`, `category-info-vs-select`, `category-exploration`, `feature-request-no-category`, `prefill-from-first-message`): direct selection vs. information-only vs. exploration vs. features-before-category, plus a dense first-message prefill asserting `slot:trailer_length_ft`/`slot:trailer_width_ft`/`slot:haul_material`/`slot:haul_weight_lbs` are all populated with zero redundant questions.
- **Flow control** (`interruption-repeat-once`, `explicit-skip`, `skip-all-show-results`): the milestone's own repeat-once example verbatim, an explicit single-turn skip, and the skip-all-and-show-results gate flip (`qualification_complete: true`).
- **Category resolution edge cases** (`haul-item-lock`, `clarification-term`, `gooseneck-not-category`): tilt-trailer-hauling-a-tractor (proves `categories.py`'s two-tier `_NAMING_TERMS`/`_CARGO_TERMS` salience beats a cargo-term collision), the `office_trailer_use` clarification rule (asks, then resolves via "fiber splicing work"), and the Gooseneck-is-never-a-category guard.
- **Brand handling** (`brand-only`, `brand-plus-category`, `brand-typo`): brand alone (category asked, not assumed), brand + category together (both stored, no re-ask), and typo correction ("dimond c" → `brand_preference: "Diamond C"`) via the live `listings_final_v5.xlsx` inventory workbook already at the repo root.
- **Defaults and loose answers** (`defaults-applied`, `loose-answers`): the former documents in its header comment the `CATEGORY_DEFAULTS_JSON` env var required before starting the server for that run; the latter exercises range→smallest-value and no-preference→null-without-re-ask.

`tests/unit/test_convo_runner_parsing.py` asserts this exact 20-name set exists and that every file parses, so a typo or accidental deletion fails fast in CI without needing the server up.

## `tests/unit/test_convo_runner_parsing.py`

Milestone 5 acceptance tests, no network/LLM. Covers: `load_scenario` round-trip and its `name`/`turns` validation error; `load_scenarios` directory loading in sorted order; every `evaluate_assertions` kind in isolation with fabricated `(reply_text, state, listings)` inputs (equality, `_contains`, `slot:` prefix, contains-any/not-contains, listings bool, emails-sent none/mismatch); a no-assertions turn always passes; and two integration-with-the-real-files checks — every file under `scripts/scenarios/` parses and has a non-empty `turns` list with a `user` key on each turn, and the full set of scenario filenames matches the milestone's named list exactly (catches drift between the doc and the fixtures). M6 extends the expected-name-set assertion with the 13 new M6 scenario names (kept in one set, split into an `# M5` / `# M6` comment for readability).

## `src/search/pinecone_search.py`

Milestone 6 port of the reference `pinecone_search.py`, logic byte-for-byte unchanged — only the four import lines were repointed at `src.domain.*` (M1's `normalizer.py`/`units.py`/`make_aliases.py`, and `src.domain.brands.make_filter_values` in place of the old `src.chatbot.make_inventory`). Public surface: `search_pinecone_listing_result(*, category, slots, metadata_filters=None, user_message, already_shown_urls=None, top_k=None, max_recommendations=None) -> PineconeListingSearchResult` and `search_pinecone_listings(...) -> list[dict]` (same call, strips the internal `match_evidence_text` key from each returned dict).

Three-stage pipeline inside `search_pinecone_listing_result`: (1) embed the query text (`user_message` + category + sorted slots + sorted metadata filters) and query Pinecone with `_metadata_filter(category, slots, metadata_filters)` as a hard filter — category always `$eq`, `make` expanded to `$eq`/`$in` over `make_filter_values()`'s raw workbook variants, `hitch_type` restricted to `{"Gooseneck", "Bumper Pull"}`, `subcategory` only for the Aluminum category, and a `length_ft_num` `$gte` floor; (2) when `RERANK_ENABLED`, `_rerank_listings_by_fit` re-orders by a fit score — any dimension ratio (length/payload-or-gvwr-fallback/width/height) under `1.0` is a hard "fail" that's always sorted behind non-failing candidates, with penalty accumulating past `RERANK_WARN_RATIO`/`RERANK_EXTREME_RATIO` weighted by `RERANK_LENGTH_WEIGHT`, plus `RERANK_MISSING_DIM_PENALTY` for listings missing a required dimension; (3) `_apply_category_make_preference` re-orders by `CATEGORY_MAKE_PREFERENCES` (a hardcoded per-category preferred-brand list, e.g. Dump favors `["Iron Bull Trailers", "Diamond C", "Texas Pride"]`) using a 3/3 (two brands) or 3/2/1 (three+ brands) quota before backfilling remaining slots up to `max_recommendations`. All three stages log their decisions (`pinecone_search`, `rerank_debug`, `make_rerank_debug`/`make_rerank_summary`) for ops debugging.

`_clean_match` is the single place a raw Pinecone match dict becomes a card dict — its key set (`title, condition, price, price_display, category, subcategory, make, model, trim, stock_number, color, hitch_type, year, length, width, height, axles, gvwr, payload_capacity, material, floor, url, relevance_score, match_evidence_text`) is what `search.py`'s node, `routes.py`'s `SHOW_ONLY_LLM_MENTIONED_CARDS` filter, and app.py's `TrailerListing` construction all key off — `src/search/inventory_matcher.py`'s `_row_to_listing` was extended in M6 to produce the same shape from Excel rows.

## `src/search/ingest.py`

Milestone 6 port of the reference `ingest.py`, logic unchanged — same import fix as `pinecone_search.py`, plus an explicit `load_dotenv()` call preserved from the reference (this module reads `os.environ["OPENAI_API_KEY"]`/`os.environ["PINECONE_API_KEY"]` at **import time**, so it cannot rely on some other module having already loaded `.env` first when run standalone via `python -m src.search.ingest`). Reads `listings_final_v5.xlsx` at repo root (drops any `msrp` column), builds one record per row via `build_record` (parses price, normalizes category/make/color/hitch/condition through `src.domain.normalizer`, computes `length_ft_num`/`width_ft_num`/`gvwr_lbs_num`/`payload_lbs_num` via `src.domain.units`, builds `embedding_text` via `build_embedding_text`, truncates to `match_evidence_text` at 3000 chars), embeds in batches of 50 via `OPENAI_EMBEDDING_MODEL`, and upserts to `PINECONE_INDEX_NAME` in batches of 100. CLI flags `--force` (re-index even if the index already has vectors) and `--no-wipe` (with `--force`, skip the delete-all-then-reupsert step). Not run by the pytest suite — it makes real OpenAI/Pinecone calls against the real workbook.

## `src/search/inventory_matcher.py`

Milestone 6 reduced port of the reference `inventory_matcher.py` (1540 lines in the reference, cut substantially). **Deleted entirely** (confirmed with the owner as obsolete, not to be recreated): the three embedded mini-LLMs (`_extractor_llm`, `_inventory_reply_llm`, `_inventory_feature_framing_llm` and their pydantic response models `TrailerQueryExtraction`/`InventoryFeatureFramingDecision`), the whole regex-fallback extraction stack (`_fallback_extraction`, `_extract_year`, `_extract_stock` — whose "any 4-6 digit number is a stock number" heuristic was the source of the "7000 lbs parsed as a stock number" bug the milestone explicitly calls out — `_wants_price/availability/details`, `_fallback_requested_features`, `_model_candidate_from_text`, `_remove_phrase`), the reply-generation formatters (`generate_inventory_response` and everything it calls: `_fallback_inventory_intro`, `_no_exact_inventory_fallback`, `_feature_neutral_fallback`, `_inventory_feature_framing_llm_intro`, `_inventory_intro_llm`, `_format_listing_block(s)`, `_listing_bullets`, `_format_match_line`, `_item_label`, `_price_text`), and the context-reference/orchestration layer (`_ordinal_reference`, `_has_context_reference`, `answer_from_last_listings`, `should_attempt_chat_lookup`, `is_potential_direct_inventory_lookup`, `validated_direct_inventory_extraction`, `search_trailers`, `_lookup_intent`, `_requested_identifiers`, `_has_specific_lookup_identifiers`, `_is_direct_inventory_lookup`, `_best_make_from_query`). All of this is replaced by `TurnAnalysis.inventory_lookup` (Analyze, M3) for identifier extraction and `ReplyOutput`/the Respond prompt's inventory-result block (also M3) for reply wording — this module now only matches and returns card dicts.

**Kept, logic unchanged**: `normalize_text` (mojibake/curly-quote/`&`/x-dimension cleanup for Excel title text), `_clean_scalar`, `_stock_text`, `extract_model_code` (+ `_COMMON_MODEL_WORDS` stopword set), `prepare_inventory`/`prepared_inventory` (`@lru_cache(maxsize=1)`, locates `listings_final_v5.xlsx` via `Path(__file__).resolve().parents[2]` — the same depth pattern `src/domain/brands.py` already uses since both files sit two levels under repo root), `_score_candidate` (rapidfuzz: `0.30*make_score + 0.35*model_code_score + 0.15*model_text_score + 0.10*title_score + 0.10*search_score`), `_no_exact_alternative_rows` (same-make-other-year rows ranked by year proximity first, then same-year rows scoring `overall>=25` as backfill), `_spec_blob_text`/`_inventory_match_evidence_text`/`_filter_same_make_rows`.

**Adapted**: `match_inventory` keeps its threshold logic (stock exact match; year+make with `make_score>=85` and no model mentioned; model rows needing `make_score>=75` and (`model_code_score>=75` or `model_text_score>=78`) and `overall>=58`; a `>=70`-overall possible-model fallback) but its signature changed from `(user_query, extraction: TrailerQueryExtraction, df, limit)` to `(identifiers: _Identifiers, df, limit)` — `_Identifiers` is a small new dataclass (`year`, `possible_make`, `possible_model_code`, `possible_model_text`, `stock_number`) replacing the deleted `TrailerQueryExtraction`. The wants-based direct-lookup gating (`_uses_year_metadata_filter` used to require `year and (wants_price or wants_availability)`) was simplified to `bool(identifiers.year)` alone, because every call into this module is now already a validated direct lookup (Analyze's gate decided that upstream) — there is no more "maybe this isn't really a lookup" signal to gate on. `_row_to_listing` was extended with `color`, `axles`, `material` (from the Excel's `trailer_material` column, renamed to match Pinecone's card key), `floor`, and `relevance_score` (renamed from the reference's `match_score`) for byte-for-byte card-shape parity with `pinecone_search.py`'s `_clean_match` output (milestone.md M6 step 6 explicitly calls for this).

**New public entrypoint** — `lookup_inventory(*, year, make, model_text, stock_number, limit=5) -> {"match_status": "exact"|"no_exact"|"ambiguous"|"none", "matches": [...], "requested_label": str}`. A pure function of identifiers: builds an `_Identifiers` from the four args (`possible_model_code = extract_model_code(model_text)`, mirroring how the reference matcher always derived model_code from model text), calls `match_inventory`, then maps the raw `{entity_type, best_match, top_matches, no_exact_reason}` result to a `match_status`:
- `no_exact_reason` set → `"no_exact"` (matches = the alternative rows `_no_exact_alternative_rows` found).
- no `top_matches` at all → `"none"`.
- exactly one match → `"exact"` (nothing to disambiguate).
- `entity_type` is `_STOCK` or `_YEAR_MAKE` → `"exact"` even with multiple rows — a stock lookup or a "what {year} {make} trailers do you have" query returning several real results is a normal result set, not an ambiguity about which single item the user meant.
- `best_match` resolved (the reference's own `unique_exact`/high-confidence-possible-model logic) → `"exact"`.
- otherwise (multiple distinct model candidates cleared the matching threshold with no clear winner) → `"ambiguous"`, which is what drives Respond's "did you mean the 7210S-BT or the TSB 7K?" clarification block.

`requested_label` joins the non-empty parts of `year`/`make`/`model_text` (e.g. `"2026 Iron Bull Trailers FHG24K"`), falling back to `"that exact trailer"` when all three are blank — feeds Respond's "we do not currently show {requested_label}" wording for `no_exact`.

## `tests/unit/test_search_node.py`

Milestone 6 unit tests for `src/graph/nodes/search.py`. Monkeypatches `search_module.search_pinecone_listings` directly (acting as the "FakePinecone" the milestone's test plan calls for — simpler than mocking the Pinecone SDK's response objects, since the node only ever touches the already-tested `search_pinecone_listings` function boundary) to assert: metadata-filter translation for length/width/payload slots via `_SLOT_METADATA_FILTER_MAP`, the Roll Off `bin_size`→`length_ft` special case, single-item `hitch_type` list → string filter, `brand_preference` → `make` filter, the zero-result brand-filter fallback re-run (asserts exactly 2 calls, second without `make`, `turn_outcome["brand_relaxed"] is True`), `already_shown_urls` pass-through from `state["shown_urls"]` for dedupe, the qualification-incomplete gate (`AssertionError`), the `results_shown` system trigger on non-empty results (and its absence on empty results), and a result-shape check that literally re-runs app.py's `TrailerListing(...)` construction (app.py:1481-1509, copied verbatim into the test) against a sample search result dict to catch silent card-dropping regressions.

## `tests/unit/test_inventory_matcher.py`

Milestone 6 unit tests for `src/search/inventory_matcher.py` against a small 5-row in-memory fixture `DataFrame` (not the real 200KB workbook) — an `autouse` fixture monkeypatches `matcher.prepared_inventory` to return the prepared fixture so every test in the file is fast and deterministic. Covers: stock-number exact match; year+make returning all matching rows (`match_status == "exact"` even with 2 rows, per the `_YEAR_MAKE` special-case in `lookup_inventory`); make+model partial match ("iron bull fhg" + "FHG 24K" finds the right row among two similarly-named FHG models); a narrowed-single-row typo match (`"7210 bt"` against a lone `"7210S-BT"` row, calling `match_inventory` directly with a single-row `df` to keep the typo assertion independent of the separate ambiguous-case test); a no-exact year (`2030` vs. real rows at `2025`/`2026`) falling back to same-make-other-year alternatives; a genuine two-close-models ambiguous case (`"Diamond C" + "7210 BT"` against both `"7210S-BT"` and `"7210-BT"` fixture rows — verified empirically to produce `match_status == "ambiguous"` with `rapidfuzz`'s actual scoring, not just asserted from a theoretical threshold); card-dict shape parity via the same app.py `TrailerListing` construction pattern as `test_search_node.py`; `normalize_text` mojibake/dimension-string cleanup; and `requested_label` composition.

## `tests/unit/test_inventory_lookup_node.py`

Milestone 6 unit tests for `src/graph/nodes/inventory_lookup.py`. Monkeypatches `inventory_lookup_module.lookup_inventory` to control results deterministically. Covers: the gate refusing `confidence="low"` and `is_lookup=False` (both raise `AssertionError`, matching the node's defensive re-assertion of `build.py`'s routing gate); matches appended to `shown_listings`/`shown_urls` with already-shown URLs excluded from the newly-appended set (but still present in `turn_outcome["listings"]` for that turn's response); the side-query invariant — `category`/`slots`/`brand_preference`/`skipped_slots`/`qualification_complete`/`pending_question_slot` are asserted byte-for-byte unchanged after the node runs; the `inventory_lookup_ran`/`contact_invite_suppressed` contract preserved exactly as the M4 stub had it; and the `results_shown` system trigger firing only when matches are non-empty.

## `tests/integration/test_chat_roundtrip.py` (M6 additions)

Two new tests, both monkeypatching `inventory_lookup_module.lookup_inventory` for determinism (no real Excel/rapidfuzz involved — that boundary is already covered by `test_inventory_matcher.py`): `test_first_turn_inventory_lookup_defers_contact_invite` posts a first-turn lookup message, asserts the HTTP response's `listings` contains the matched card and that `_sessions[sid]["contact_prompted_initial"]` is still `False` after that turn, then posts a second turn and asserts it flips to `True` (proving the invite fires on the *next* turn, not the lookup turn itself, per the Locked Decision). `test_show_only_llm_mentioned_cards_filters_uncited_listings` returns two ambiguous-match candidates from the mocked lookup but has the `FakeLLM`'s queued `ReplyOutput` cite only one URL, then asserts the HTTP response's `listings` contains only that cited one — exercising the M6 `routes.py` filter end-to-end.

## `scripts/scenarios/*.yaml` (M6 additions)

13 new scenario files matching milestone.md's M6 test list exactly, following the same `name` + `turns` shape as the M5 scenarios: `search-happy-path` (qualification completes → real search returns listings), `refine-length-change` (a requirement change re-searches the same turn, other slots kept), `more-options-dedupe` ("more options" after skip-all), `category-change-keep-some` (switch category, keep a subset of transferable slots, new category's search fires), `reference-second-listing` (ordinal reference to a shown listing resolves without a new search), `brand-filter-search` (Diamond C + Dump → brand hard-filtered results), `brand-filter-no-match` (obscure/nonexistent brand → zero-result fallback wording), `inventory-stock-lookup` (explicit stock number → cards, bypasses qualification), `inventory-first-message-contact-deferred` (first-message lookup defers the contact invite to the next turn), `inventory-weight-not-stock` ("haul 7000 lbs" must never be parsed as a stock number or trigger a lookup), `inventory-model-typo` (rapidfuzz typo tolerance on a model code), `inventory-ambiguous-clarify` (multiple close model candidates → clarifying question, no price stated yet), `inventory-mid-qna` (a lookup mid-qualification is an interruption — results shown, category/slots untouched, pending question resumes after). `tests/unit/test_convo_runner_parsing.py::test_expected_scenario_set_is_complete` was extended to include these 13 names alongside the existing M5 set.

## `src/tools/email_sender.py` (M7)

The email-delivery boundary. `send_email(subject, body) -> bool` dispatches on `settings.email_backend`:
- **`graph`** (`_send_via_graph`): POSTs a `client_credentials` token request to `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token` with scope `https://graph.microsoft.com/.default`, then POSTs `https://graph.microsoft.com/v1.0/users/{SENDER_EMAIL}/sendMail` with a `Bearer` header and a Text-body message to `RECIPIENT_EMAIL`; HTTP 202 = success. Uses plain `requests` — **no msal** (Locked Decision).
- **`smtp`** (`_send_via_smtp`): `smtplib` STARTTLS to `SMTP_HOST:SMTP_PORT`, optional login with `SMTP_USER`/`SMTP_PASSWORD`, From `SMTP_FROM`, To **`EMAIL_TO`** (the SMTP recipient var — distinct from Graph's `RECIPIENT_EMAIL`).

Any exception is caught, logged, and returns `False` so an email failure never crashes a turn. `render_email_body(name, email, phone, reason, description)` produces the exact spec block (`Full Name:` / `Email:` / `Phone Number:` lines, blank line, then `[{Reason}] {description}`, with missing fields rendered as `Not provided`). `render_subject(reason, name, session_id)` produces `TrailerPlace Lead — {Reason} — {name or session id}`. Both formatters are pure so tests can exact-match them.

## `src/graph/nodes/email_actions.py` (M7)

The real email-actions node, replacing the M4 marker stub. It runs on **both** graph passes (`email_actions` before routing and `email_actions_after_results` after search/inventory) and is idempotent: it resolves and then **consumes** the triggers it processes (rewrites `state["turn"]` with `email_triggers=[]` via `model_copy`, and clears `turn_outcome["system_email_triggers"]`), so the second pass only sees the newly-generated `results_shown` alert.

Each turn it resolves three sources into a common descriptor `{kind, reason, event_type, description, canned_key, is_system}`: customer `turn.email_triggers` (`faq` → `FAQ – {faq_key}` / `non_sales_faq` / canned=faq_key; `escalation` → `Escalation` / `escalation_alert` / canned=`escalation`; `team_request` → `Team Request` / `team_request` / canned=`generic_team_request`; `listing_interest` → `Listing Interest` / `interested_listing` with the selected/unselected/fallback canned variant chosen from `state["shown_listings"]` + `listing_reference`, selected descriptions embedding the listing title+URL), code-generated `turn_outcome["system_email_triggers"]` (`results_shown` → `Results Shown to User`; `unanswered_question` → `Unanswered Question`; both `is_system`, no canned text), and previously-stashed `state["pending_email_actions"]`.

FAQ/non-FAQ canned keys are recorded into `turn_outcome["canned_keys"]` **immediately regardless of the gate** (the user always gets their answer; only the email waits); system alerts contribute none. The contact gate (`Name + (Email|Phone)`, computed from state fields, never the `contact_status` column) then decides the whole batch: **declined** → drop new + stashed silently, never re-ask, `email_status = "skipped (user declined)"`; **complete** → emit every stashed + new descriptor as an email event and clear the stash/followup; **incomplete** → stash the batch, and for customer-initiated triggers set `contact_followup_pending` (`name` or `contact_method`) + `email_status = "deferred — ask once …"`, while system alerts stash silently (no followup, no status). Emitting builds `{event_key: "{kind}:{running_index}", event_type, payload: {subject, body}}` (body rendered with current contact) appended to `turn_outcome["outbox_events"]`, records the Reason in `turn_outcome["emails_sent"]` (system alerts included), and — when persistence is **off** — sends immediately via `email_sender.send_email` (the route handles enqueue + lead-hard when persistence is on). `email_status` only ever reflects customer emails, so the Respond prompt never mentions system alerts.

## `tests/conftest.py` (M7 addition)

Adds `FakeEmailSender` alongside `FakeLLM`: a capturing double with `send_email(subject, body) -> bool` (monkeypatched into the node for persistence-off tests) and `handler(*, subject, body)` (registered as an outbox handler for DB tests). `succeed=False` simulates a delivery failure so the retry/`failed`-status path can be exercised.

## `tests/unit/test_email_sender.py` (M7)

Exercises the sender in isolation: exact `render_email_body`/`render_subject` output (including `Not provided` fills and the en-dash Reason), backend selection by swapping `email_sender.settings` for a `SimpleNamespace`, the graph path's token-request fields and `sendMail` payload/recipient with a mocked `requests.post`, a non-202 graph response → `False`, the smtp path's recipient (`EMAIL_TO`) and login with a fake `smtplib.SMTP`, and a raising transport → `False` without propagating.

## `tests/unit/test_email_actions.py` (M7)

Drives `email_actions_node` directly in persistence-off mode (autouse fixture patches `persistence_enabled → False` and the node's `email_sender.send_email` to a `FakeEmailSender`). Covers the full gate matrix (complete→send, partial→stash+ask with `contact_followup_pending`, stashed→provided→send, declined→drop+no-reask), FAQ canned reply present even when the email is deferred, listing-interest selected/unselected/fallback variant selection (and title+URL in the selected body), multi-trigger ordering (`["FAQ – financing", "Escalation"]`) and blocked-then-sent, system-alert send-but-not-in-canned + stash-silently-without-followup, `unanswered_question` gating, and two-pass idempotency (escalation processed pass 1, only the new `results_shown` sent pass 2).

## `tests/integration/test_email_outbox.py` (M7)

One persistence-off test (`/chat` escalation with full contact sends directly through a monkeypatched sender) plus DB-marked tests (`@pytest.mark.db`, skipped without `TEST_DATABASE_URL`) that run a full escalation `/chat` against a disposable Postgres and assert: the outbox row is created in-transaction and drained (`status="sent"`, handler captured payload, lead upgraded to `hard`); a failing handler leaves the row `failed` and a later drain retries it to `sent`; and a duplicate `turn_id` replay short-circuits on the receipt without enqueuing a second event.

## `scripts/scenarios/*.yaml` (M7 additions)

9 new scenario files matching milestone.md's M7 test list: `faq-financing-with-contact` (`expect_emails_sent: ["FAQ – financing"]`), `escalate-quote-no-contact-then-provide` (stash then send once contact given), `escalate-decline-contact` (`expect_emails_sent: none`, conversation continues), `listing-interest-second-one`, `faq-plus-escalate-one-message` (multi-trigger fan-out), `inventory-lookup-then-interest` (first-turn lookup cards → interest → Listing Interest email + hard lead), `results-shown-alert` (silent alert on each results turn incl. refinement), `unanswered-question-alert` (two non-answers → silent alert, reply never mentions email), and `results-alert-stashed-then-sent` (anonymous lookup stashes the alert; volunteered contact sends it). `tests/unit/test_convo_runner_parsing.py::test_expected_scenario_set_is_complete` was extended with these 9 names.

## `src/api/readiness.py` (M8)

A four-key module-global (`graph`, `db`, `error`) with `mark_graph_ready()`, `mark_db_ready()`, `mark_failed(reason)`, `reset()` (test hook), `is_ready()`, and `readiness_status()`. `readiness_status()` returns `{"status": "error", "detail": …}` when a startup step failed, `{"status": "ok"}` when both flags are set, else `{"status": "starting", "detail": "waiting on: graph, db"}`. It exists so `routes.py` (request handling) and `api/app.py` (wiring) each keep one job.

## `src/api/app.py` (M8 changes)

`create_app()` now attaches `lifespan`, which calls `readiness.reset()` then `run_startup()`. `run_startup()`: (1) if `db.database_enabled()` — run `db.run_migrations()` when `DB_AUTO_CREATE=1`, otherwise log a reminder to run `alembic upgrade head` manually; (2) `ensure_graph()` to compile the LangGraph once, then `mark_graph_ready()` — so the first `/chat` does not pay the build cost inside its own timeout budget; (3) if `conversation_store.persistence_enabled()` and `db.ping()` fails → `mark_failed("database unreachable")` and return, else `mark_db_ready()` (persistence-off is a valid deployment with nothing to probe). Every exception is caught, logged, and recorded via `mark_failed(f"{type}: {exc}")` — the process keeps serving so `/health` can explain itself.

This lives in a lifespan rather than in `create_app()` directly so that `TestClient(create_app())` (no context manager) never opens a database connection; tests that want the startup path enter `with TestClient(app) as client`.

## `src/api/routes.py` (M8 changes)

- `ensure_graph()` — public wrapper over the lazy `_get_graph()`, called by the lifespan.
- `_GRAPH_EXECUTOR` — an 8-worker `ThreadPoolExecutor`; `GraphFailure` — the exception type meaning "this turn's graph run broke or timed out".
- `_require_uuid(value, field)` — parses `session_id` and `turn_id` at the top of `/chat`, raising **422**. Before M8, `uuid.UUID(request.session_id)` raised deep inside the durable path and its `ValueError` was swallowed by the handler that means "turn_id reused", so a malformed id reported **409**. With this guard, 409 is reserved for the genuine conflict.
- `_error_response(session_id)` — the full-contract apology body: `assistant_text = ERROR_ASSISTANT_TEXT`, `listings: []`, `sales_phase: "main"`, `onboarding_api_messages: []`, the three customer fields read best-effort off `_sessions`, `main_prior_messages: None`, `thinking_context: None`. `customer_email` is always a key because app.py probes it with `"customer_email" in data`.
- `health(response)` — returns `readiness.readiness_status()` and sets **503** unless `status == "ok"`. Safe for the frontend: `_wait_for_backend_ready()` checks `response.ok and body["status"] == "ok"` and retries until its own deadline; it never calls `raise_for_status()`.
- `chat(request)` — validates both ids, truncates `message` to `settings.chat_max_message_chars` (logged, via `model_copy` since `ChatRequest` is pydantic), then runs `_run_turn` while holding `session_lock(session_id)`. Exception policy, in order: `HTTPException` re-raised (422 stays 422); `ValueError` → **409** (durable_turn's turn-id conflict); `GraphFailure` → `logger.exception` + **200** `_error_response`. Because `durable_turn` rolls back on exception, **no `ChatbotTurn` receipt is written for a failed turn**, so the frontend's retry with the same `turn_id` re-runs the graph rather than replaying the stored apology.
- `_run_turn(request, turn_id)` — the previous body of `chat()` (persistence-off short-circuit, then the durable transaction: receipt replay, snapshot reload, soft lead, graph, snapshot/receipt/outbox writes, post-commit drain).
- `_invoke_graph(state)` — submits `_get_graph().invoke` to `_GRAPH_EXECUTOR` and waits `settings.chat_timeout_seconds`; a `TimeoutError` or any other exception becomes `GraphFailure`. This is the single point where a broken graph is contained, and it is what keeps the frontend from ever seeing a 500.
- `reset_session` — always **200**; `close_session` is wrapped in try/except so "New Conversation" cannot fail for the user, and `clear_session` already no-ops on an unknown id.

## `src/graph/state.py` (M8 changes)

Adds `_session_locks: defaultdict[str, threading.Lock]` guarded by `_session_locks_guard`, exposed as `session_lock(session_id)`. When persistence is on, `durable_turn`'s `pg_advisory_xact_lock(hashtext(session_id))` already serializes concurrent turns for one session across processes; this lock gives the persistence-off path the same guarantee inside one process.

## `src/conversation_store.py` (M8 changes)

New pure helper `_apply_feedback_to_messages(messages, conversation)`: copies each message (never mutates the caller's list), defaults `listings`/`user_feedback` to `None` so every restored message has the keys app.py reads, then walks the simplified turn list and writes `conversation[i]["feedback"]` onto message `2*i + 1` when that message exists and has `role == "assistant"`. This is the inverse of app.py's `turn_idx = i // 2`, valid because the simplified list holds exactly one entry per user/assistant pair. Empty-string and out-of-range feedback are ignored.

`restore_session` calls it on both message sources (the preferred `state_snapshot["messages"]` and the rebuild-from-`conversation` fallback), so a note saved before a browser refresh reappears in the feedback box instead of looking lost. `close_session` now calls `ensure_persistence_schema()` first, matching its siblings.

## `tests/fixtures/chat_response_contract.json` (M8)

The golden `/chat` body. Its `_comment` key documents why `customer_email` must always be present; the test pops `_comment` and asserts the remaining key set matches the live response exactly.

## `tests/integration/test_api_contract.py` (M8)

Network-free: a `_FakeGraph` class (records the state it saw; can sleep, raise, or return listings) is monkeypatched over `routes._get_graph`, and persistence is off. The `offline` fixture returns an installer that wires the fake graph and builds a `TestClient`. Covers: the golden key set plus `"customer_email" in body`; a verbatim copy of app.py's `TrailerListing(...)` construction block run over a returned listing (a shape regression fails here instead of silently dropping cards in the UI, since app.py swallows the exception); `RuntimeError` from the graph → 200 + `ERROR_ASSISTANT_TEXT` + logged; a 0.05 s `chat_timeout_seconds` against a sleeping graph → same apology; a 32-char cap truncating the message the graph actually sees; malformed `session_id` and `turn_id` → 422 (parametrized); unknown-session reset → 200; closed-session shape; `/health` 503 `starting` before the lifespan and 200 `ok` inside `with TestClient(app)`; and 503 `error` when `persistence_enabled()` is true but `db.ping()` fails.

Both settings-dependent tests swap the whole frozen `Settings` object via `dataclasses.replace`.

## `tests/unit/test_restore_feedback.py` (M8)

Five pure tests over `_apply_feedback_to_messages`: turn 0 → message 1 and turn 3 → message 7 (with untouched assistant turns staying `None`); restored messages always carry `listings`/`user_feedback`; feedback on a turn with no assistant reply yet is ignored; empty-string feedback does not overwrite; the caller's message list is not mutated.

## `tests/integration/test_db.py` (M8 additions)

Three DB-marked tests plus a `_drain_feedback_pool()` helper (submits one no-op per pool worker and waits, since `enqueue_save_user_feedback` writes on a background `ThreadPoolExecutor`). `test_restore_after_backend_restart` writes a snapshot through `durable_turn`, clears `_sessions` to simulate a cold process, and asserts `restore_session` returns the text (listing dicts still live in the snapshot — the *route* is what nulls them for the frontend). `test_feedback_lands_on_turn_0_and_turn_3_and_survives_later_turns` saves feedback on two turns, drains the pool, rewrites the conversation with a later turn to prove `_merge_existing_feedback` preserves it, then asserts it comes back on assistant messages 1 and 7. `test_closed_session_reports_closed` covers the `{exists: true, closed: true}` shape the frontend uses to start fresh.

## `tests/unit/test_scaffold.py` (M8 change)

`test_health` now disables persistence and `database_enabled`, injects a `FakeLLM` graph, and enters `TestClient` as a context manager — `/health` reports `ok` only after the lifespan runs.

## `docs/e2e_checklist.md` (M8)

The manual end-to-end checklist required by the milestone: boot (503 → 200 on `/health`), converse, cards render, refresh mid-conversation (text restores, **cards do not** — expected per the Locked Decision, and "more options" still dedupes because the backend's `shown_urls` is authoritative while the frontend's payload is now empty), restart the backend and continue, save feedback and verify it in the `conversation` JSONB and after a refresh, New Conversation (`closed_at` set, never deleted), logout, plus robustness spot-checks for 422 / 409 / receipt replay / oversized message / a broken search turn returning a 200 apology.

## `src/llm/usage.py` (M9)

Per-turn LLM call/token accounting — the data the cost audit asserts on.

`TurnUsage` is a mutable dataclass counting `chat_completions`, `embeddings`, `prompt_tokens`, `completion_tokens`, and the distinct `models` seen; `total_tokens` is a derived property and `as_dict()` inlines it for logging. A module-level `contextvars.ContextVar` holds the active instance. `usage_scope()` is a context manager that installs a fresh `TurnUsage` and resets the token on exit; `current_usage()` returns it or `None`.

`record_completion(model, prompt_tokens, completion_tokens)` and `record_embedding(model, prompt_tokens)` **no-op when called outside a scope**. That is deliberate: `ingest.py`, `smoke_llm.py`, and the pytest suite all call the same client/search code paths without a turn, and they must not need to know about accounting.

`contextvars` rather than a plain global because `/chat` turns run concurrently on a `ThreadPoolExecutor` — each turn needs its own counters even when sessions overlap. See the `_invoke_graph` note under `src/api/routes.py` for the propagation trap this creates.

## `src/tracing.py` (M9)

LangSmith wiring. We call the OpenAI SDK directly rather than through LangChain, so nothing is traced implicitly — the decorator on `OpenAILLMClient.structured` is what creates the Analyze/Respond spans and `trace_turn()` is what groups them.

- `tracing_enabled()` — requires **both** `LANGSMITH_TRACING` and a non-empty `LANGSMITH_API_KEY`.
- `configure_langsmith()` — idempotent (guarded by a module `_configured` flag); exports the `LANGSMITH_*` settings onto the env vars the SDK reads. When tracing is off it **forces `LANGSMITH_TRACING=false` into the environment**, because a stale ambient `true` would otherwise make the SDK attempt exports with no key. Returns the resulting enabled state; called from `run_startup()` before the graph compiles.
- `trace_turn(session_id, **metadata)` — context manager wrapping `langsmith.run_helpers.tracing_context`; tags the run `session:<id>` and attaches `session_id` plus any non-`None` metadata. A no-op (and imports nothing) when disabled.
- `add_turn_metadata(**metadata)` — updates `get_current_run_tree().extra["metadata"]`. Called from the analyze node because intent/category do not exist when `/chat` opens the run. No-ops without an active run.
- `traceable_or_passthrough(name)` — the decorator applied to `structured()`. It is installed at **import** time, long before settings load, so it cannot decide then. Instead the returned wrapper checks `tracing_enabled()` **per call**: off → call the raw function; on → build the `traceable`-wrapped variant once, cache it, and delegate. This is what keeps pytest offline: the developer `.env` normally has `LANGSMITH_TRACING=true`, and without the runtime gate simply *calling* a decorated function shipped a real trace (this actually happened during M9 development and hit the LangSmith monthly-trace rate limit).

Settings are read as `config.settings.X` rather than via `from src.config import settings`, so a test can swap the frozen `Settings` object wholesale and have this module see it.

## `src/turn_log.py` (M9)

Structured per-turn JSON logs, and the cost report's input format.

`log_turn(...)` builds one flat record — `event: "chat_turn"`, `session_id`, `turn_id`, `intent`, `category`, `latency_ms` (rounded to 2dp), `tools_fired`, `emails_sent`, `llm_calls` (the `TurnUsage.as_dict()`, or `None`), and `error` only when the turn failed — logs it on the `trailerplace.turn` logger via `extra={"turn": record}`, and returns it.

`tools_fired(turn_outcome)` maps `search_ran` → `"search"`, `inventory_lookup_ran` → `"inventory_lookup"`, and a non-empty `emails_sent` → `"email"`. The cost audit keys off `"search"` to decide whether one embedding is expected.

When `TURN_LOG_PATH` is set, `_ensure_file_handler()` (lazy, once) attaches a `FileHandler` using `_JsonLineFormatter`, which writes **only** `json.dumps(record.turn)` — bare JSON, no level/timestamp prefix — because `cost_report.load_turns` does `json.loads` per line. Records still propagate to the root logger for humans. `reset_for_tests()` removes and closes the handler so a test can repoint the path.

## `src/log_setup.py` (M9 additions)

Adds `JsonFormatter`, selected by `LOG_FORMAT=json`, which emits `ts`/`level`/`logger`/`message` and **inlines a record's `turn` payload** so ops queries like `.intent == "faq"` work on the main stream, not just the `TURN_LOG_PATH` file. Plain text remains the default for local development. `configure_trailerplace_logging()` now constructs its own `StreamHandler` rather than using `basicConfig`, so it can choose the formatter.

## `src/api/routes.py` (M9 additions)

`chat()` now opens `llm_usage.usage_scope()` and `tracing.trace_turn(session_id, turn_id=…)` around the whole turn, times it with `perf_counter`, and calls `_log_turn(...)` on both the success path and the contained `GraphFailure` path (where the record carries `error`). `_log_turn` reads intent from `state["turn"].intent` and category from the state, and swallows its own exceptions — observability must never break a served response.

**`_invoke_graph` runs the graph inside `contextvars.copy_context().run(graph.invoke, state)`.** `concurrent.futures.ThreadPoolExecutor` does *not* propagate contextvars into its worker threads: without this the nodes' `record_completion()` calls landed in an empty context, every turn logged `chat_completions: 0`, and `cost_report` — which treats a zero-completion turn as a receipt replay — silently audited nothing. The copied context shares the `TurnUsage` *object*, so the worker's mutations are visible to the request thread after `future.result()`. `tests/integration/test_observability.py::test_usage_counters_survive_the_graph_executor_thread` locks this in.

## `src/llm/client.py` (M9 additions)

`OpenAILLMClient.structured` is decorated with `@traceable_or_passthrough("llm.structured")` and, after parsing, calls `usage.record_completion(self.model, prompt_tokens, completion_tokens)` off `parsed.usage` (defensively, via `getattr`, since a stubbed client may not populate it).

## `src/search/pinecone_search.py` (M9 addition)

`_embed()` calls `usage.record_embedding(EMBEDDING_MODEL, prompt_tokens=…)` — the single embedding a search turn is permitted. This is the only M9 change to the ported module.

## `src/graph/nodes/analyze.py` (M9 addition)

After `analyze_turn` returns, calls `add_turn_metadata(intent=analysis.intent, category=analysis.category_mentioned or state["category"])`, annotating the run that `/chat` opened.

## `scripts/cost_report.py` (M9)

The cost audit. Network-free; reads the `TURN_LOG_PATH` JSONL.

Pure functions: `load_turns(path)` (skips blank lines and non-`chat_turn` events, raises `ValueError` with a line number on malformed JSON); `summarize(record) -> TurnCost`; `violations(turn, max_completions=2)`; `audit(records)` which aggregates totals and collects `(turn, problems)` pairs.

The budget, straight from the Locked Decisions:
- at most **2 chat completions** per turn (Analyze + Respond — haul classification and inventory-lookup detection are folded into Analyze);
- a search turn makes **exactly one** embedding; any other turn makes **zero**;
- an inventory-lookup turn adds nothing — the Excel/rapidfuzz matcher uses neither an LLM nor an embedding.

`TurnCost.is_replay` is `chat_completions == 0`: a receipt replay runs no graph, so zero calls is correct and the turn is skipped rather than failed. The CLI prints a per-turn table plus totals and **exits 1 on any violation**, so it can gate a release. `--max-completions` overrides the budget.

## `scripts/convo_runner.py` (M9 additions)

Every scenario YAML now declares a `tags:` list; `load_scenario` defaults it to `[]`. New pure helpers `filter_by_tag(scenarios, tag)` (returns matches sorted by name) and `load_suite(tag, dir)`.

CLI: `target` becomes optional, and `--suite <tag>` selects by tag instead (`parser.error` if neither is given). `--list` prints the selection with its tags and returns 0 **without making any network call** — the cheap way to confirm what a suite would run. `--repeat N` runs the selection N times and fails if any pass fails; the M9 gate is `--suite regression --repeat 2`.

`REGRESSION_SUITE = "regression"` is the release gate: all 47 scenarios carry it (20 M5 + 13 M6 + 9 M7 + 5 M9 adversarial). `tests/unit/test_convo_runner_parsing.py` asserts the regression suite covers *every* scenario file, so an untagged scenario fails CI rather than silently dropping out of the gate.

## `scripts/scenarios/*.yaml` (M9)

All 42 pre-existing scenarios gained `tags: [regression, m5|m6|m7]`. Five adversarial scenarios were added, each tagged `[regression, m9, adversarial]`:

- `adversarial-contradictory-sizes` — the cargo length is stated, contradicted, then contradicted again. The **last value wins** (requirement_change semantics); the bot must not average, keep both, or lose the category.
- `adversarial-category-change-twice` — two explicit switches. Asserts the pending-then-resolve shape precisely: on the switch turn `category` is **still the old one** (`_apply_category` builds `pending_category_change` and returns), and only the `keep_fields_answer` turn applies the new category. The second switch must not be confused by the first one's resolution.
- `adversarial-paragraph-requirements` — a pasted paragraph with name, email, phone, category, cargo, weight ("two tons" → 4000), length, width (`7'6"` → 7.5), hitch, and two non-searchable extras. Asserts every field lands in one turn and nothing already known is re-asked. Note `slots["hitch_type"]` is a **list** (`["Gooseneck"]`), matching `ExtractedFields.hitch_type`.
- `adversarial-faq-midqual-then-answer` — an FAQ interrupts qualification: the canned text and its gated email fire, *and* `pending_question_repeats` goes to 1. The user then answers, so the slot fills and `skipped_slots` stays empty — the repeat-once rule must not auto-skip a question that got answered.
- `adversarial-gibberish-input` — unparseable input twice: no crash, `category` stays `null`, no listings, and the digits in the noise are **not** matched as a stock number (the deliberately-deleted `_extract_stock` heuristic). Recovers on the next real message.

## `README.md` (M9)

The release run-book. Architecture summary (the two-LLM-call budget, the atomic durable turn), install → configure → `alembic upgrade head` → `python -m src.search.ingest` → `python main.py` → `streamlit run app.py`, a test-command table, the live scenario-suite and cost-audit commands, the full environment-variable table (every key `config.py` reads, with defaults and the `EMAIL_TO`-vs-`RECIPIENT_EMAIL` trap called out), a Security section, and an Operations section documenting the per-turn record plus the at-least-once email / never-downgraded-lead / text-only-restore invariants.

## `.env.example` and `.gitignore` (M9)

`.env.example` carries a placeholder for every key, grouped by concern. `.gitignore` excludes `.env`, `*.jsonl` turn logs, virtualenvs, and caches, while explicitly re-including `.env.example`.

**Credential status (the M9 §4 flag):** `.env` holds live OpenAI, Pinecone, Microsoft Graph client-secret, and Azure Postgres credentials. The repo's `.git` directory is **empty** — this is not an initialized repository — so those secrets are not in any git history and there is nothing to purge. They must still be **rotated before the repo is shared or deployed**, and production values belong in a secret manager. Separately, `DEBUG_STATE_ENDPOINT=1` exposes full raw session state (customer contact details included) on an unauthenticated endpoint; it exists only for the scenario runner.

## `tests/conftest.py` (M9 additions)

Two autouse safety fixtures and one helper:

- **`_never_touch_a_real_database`** (autouse, per-test) monkeypatches `db.database_enabled()` to `False` for every test **not** marked `db`. This fixes the issue flagged at the end of the M8 notes, and it is more serious than it read there: `persistence_enabled()` is `TRAILERPLACE_PERSIST_CHATS and db.database_enabled()`, both of which a developer `.env` satisfies, so `apply_analysis`'s `update_lead_contact` / `update_lead_item_of_interest` calls were invoking `ensure_persistence_schema()` → `db.ensure_schema()` → `Base.metadata.create_all()` **against the production Azure Postgres** on nearly every unit test. Wall-clock proof: the suite dropped from ~110 s to ~22 s once the guard landed. `db`-marked tests opt back in — they own a throwaway database via `TEST_DATABASE_URL` and skip when it is unset.
- **`_disable_langsmith_tracing`** (autouse, session) clears `LANGSMITH_TRACING` in the environment and swaps `config.settings` for one with tracing off, so no test can export a trace.
- **`replace_settings(monkeypatch, **values)`** — `Settings` is a frozen dataclass, so a test must swap the whole object (`dataclasses.replace`) rather than set a field. Modules that need to be patchable read `config.settings.X` instead of binding `settings` at import.

## `tests/unit/test_tracing.py` (M9)

Enablement needs both the flag and the key; `configure_langsmith` exports settings to env, forces a stale ambient `true` back to `false` when disabled, and is idempotent (a second call does not re-export changed settings). `trace_turn` tags `session:<id>` and drops `None` metadata (asserted against a fake `tracing_context`). `add_turn_metadata` updates a fake run tree and no-ops when there is none. Two decorator tests carry the real weight: **disabled → `langsmith.traceable` is never called**, and **enabled → the traced variant is built exactly once** and reused across calls.

## `tests/unit/test_turn_log.py` (M9)

Usage counters: recording outside a scope is a no-op; a normal turn is 2 completions / 0 embeddings; a search turn is 2 / 1; nested scopes do not leak into each other; `as_dict()` includes `total_tokens`. `tools_fired` is parametrized over the outcome shapes. `log_turn` is checked for its exact record shape (2dp latency rounding, the `error` key only when failing, `llm_calls: None` without usage), for writing bare JSON lines to `TURN_LOG_PATH`, for emitting on the `trailerplace.turn` logger, and `JsonFormatter` for inlining the payload.

## `tests/unit/test_cost_report.py` (M9)

JSONL loading (blank lines, foreign `event` types, a line-numbered error on bad JSON). Then one test per budget rule: 2 completions passes, 3 fails; a search turn needs exactly one embedding (0 fails); an embedding on a non-search turn fails; an inventory-lookup turn that embeds is flagged by the lookup-specific rule; a 0-completion receipt replay is **skipped, not failed**; `--max-completions` overrides. `test_audit_of_a_clean_ten_turn_conversation` is the milestone's "cost assertion holds on a 10-turn conversation" acceptance test at the unit level. CLI exit codes are covered, plus a round trip that writes through `turn_log` and reads through `cost_report` — that test is what guards the two modules' shared format contract.

## `tests/integration/test_observability.py` (M9)

Drives the real `/chat` route with a faked graph and persistence off. `_RecordingGraph` records completions/embeddings exactly the way the real nodes do — on the executor's worker thread, which is the point.

- `test_usage_counters_survive_the_graph_executor_thread` — the contextvar regression guard.
- `test_turn_log_captures_intent_category_tools_and_latency` — the full record through the real route.
- `test_each_turn_gets_its_own_usage_scope` — counters do not accumulate across turns.
- `test_a_failed_turn_is_logged_with_its_error` — a raising graph still returns 200 and the record carries `error`.
- `test_cost_report_audits_a_real_ten_turn_conversation` — the M9 acceptance test end to end: 10 turns, 20 completions, embeddings only on the 2 search turns, `failures == []`, `replays == 0`.
- `test_cost_report_catches_a_third_llm_call` — proves the audit is not vacuous.

## `tests/unit/test_convo_runner_parsing.py` (M9 additions)

The expected-scenario set grows by the 5 adversarial names. New suite tests: every scenario declares tags; the `regression` suite covers **all 47** scenario files (so an untagged scenario fails CI rather than silently dropping out of the release gate); the `adversarial` suite is exactly the 5 new ones; `filter_by_tag` sorts by name and excludes untagged/other-tagged files; and the CLI's `--list` selects a suite with no network, while a missing target `SystemExit`s and an unknown suite exits 1.

## Milestone 9 Status

Offline gate: **`pytest` green three consecutive runs — 228 passed, 10 skipped** (the skips are the `db`-marked tests, deliberately not run: `TEST_DATABASE_URL` in `.env` points at the *real* Azure Postgres, not a throwaway container). The cost audit was verified end to end against the real `/chat` route: 10 turns, 20 completions, 1 embedding per search turn, `cost_report` exit 0.

Not run here, by owner instruction — to be run by the owner:
- the live regression suite (`python scripts/convo_runner.py --suite regression --repeat 2`), which needs a backend booted with `DEBUG_STATE_ENDPOINT=1` and spends real OpenAI/Pinecone tokens;
- the `db`-marked tests, which must be pointed at a throwaway Postgres (`docker run -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16`) and **never** at the production database;
- the manual LangSmith trace check (traces appear in the `TrailerPlace` project, tagged `session:<id>`);
- the one manual live email test against the real Graph API to `RECIPIENT_EMAIL`.

## Post-M9 Follow-up: Non-Blocking Outbox + Full Conversation/Reasoning Logging

Two changes prompted by a real production log the owner shared: a `[Team Request]` escalation email hit a TLS handshake reset talking to `login.microsoftonline.com` (`ConnectionResetError [WinError 10054]`), and the `/chat` response the user was waiting on was visibly slow because of it.

### `src/conversation_store.py`: `deliver_pending_outbox_async`

`/chat` used to call `conversation_store.deliver_pending_outbox()` **synchronously**, after the durable-turn commit and before the response returned. Both Graph HTTP calls (`_send_via_graph`'s token request and `sendMail`) carry a `timeout=30`, so a slow or failing network path added up to ~60s of pure waiting onto a chat reply that has nothing to do with email delivery — the user stares at a spinner for a system whose own design (`chatbot_outbox`, at-least-once, `status in (pending, failed)` retried by the next drain) already tolerates that delay just fine.

`deliver_pending_outbox_async(limit=10)` submits `deliver_pending_outbox` to the existing `_pool` (a `ThreadPoolExecutor(max_workers=2)`, already used by `enqueue_save_user_feedback`) instead of calling it inline, and no-ops when persistence is off (same guard as the synchronous version). `src/api/routes.py`'s `_run_turn` now calls the async version as its last step before returning `ChatResponse(**body)`.

**Test impact:** two DB-marked tests in `tests/integration/test_email_outbox.py` asserted on outbox row status *immediately* after the `/chat` POST returned, which raced the new background drain. Fixed with a `_drain_outbox_pool()` helper — submit two no-ops to the two-worker pool and wait on both futures, the identical pattern `test_db.py`'s `_drain_feedback_pool()` already uses for the same reason. `test_duplicate_turn_replay_does_not_enqueue_second_event` only counts rows (not status), so it needed no change.

### `src/conversation_log.py`: human-readable reasoning + state log

A new logger, `trailerplace.conversation`, distinct from `src/turn_log.py`'s compact JSONL machine feed (which stays as the cost-audit's input — nothing about it changed). This one exists purely for an operator to read: one multi-line block per turn containing

- the session id and turn id,
- the raw `USER:` message,
- the Analyze LLM's **entire structured output** under `REASONING (Analyze):` — intent, category (+ info-only flag), every `ExtractedFields` value, slot answers, the haul-classification decision, inventory-lookup identifiers (only when `is_lookup`), email triggers, an in-message contact block (only when present), a listing reference, an interruption line (only when the pending question wasn't answered), and a requirement-change line (only when dropped/kept fields are present) — so a developer can see *why* the bot did what it did, not just *what* it did,
- `TOOLS FIRED:` — a one-line summary of search/inventory-lookup/email activity for the turn, plus the email gate's `EMAIL STATUS:` line when set,
- the `ASSISTANT:` reply, its `cited_listing_urls`, and how many listings were returned — or, on a contained `GraphFailure`, `ERROR: ...` in place of all three,
- `STATE AFTER TURN:` — category, clarification key, every collected slot, skipped slots, qualification-complete flag, the pending question and its repeat count, brand preference, non-metadata features, the full contact block (name/email/phone/declined), any pending category change, the stashed-email-actions count and contact-followup-pending flag, and shown-listing/shown-url counts.

`log_conversation_turn(...)` builds this from primitives already on hand in `routes.py` — `state["turn"]` (the `TurnAnalysis`), the returned `ChatResponse`, and the in-memory session `state` dict itself — so it costs nothing extra to compute. It wraps its own formatting in `try/except` and logs the exception instead of raising, mirroring `turn_log.log_turn`'s "observability must never break a served response" contract; `test_never_raises_on_a_malformed_analysis_object` locks that in with a property that raises on access.

**Wiring** (`src/api/routes.py`): `_log_turn` (already the M9 site for the JSONL record) now also calls `conversation_log.log_conversation_turn`, on both the success path (passing the real `ChatResponse`) and the `GraphFailure` path (passing `error=str(exc)`, `response=None`).

### `src/log_setup.py`: `DailyFileHandler` — file logging named by date

The ask was literal: "log files ... named after the current date such as 2026-07-11.log". A stdlib `TimedRotatingFileHandler` doesn't do this directly — it keeps writing to one fixed filename and only appends a date suffix to the file it rotates *away from* at midnight, so the file you're actually tailing right now never has today's date in its name.

`DailyFileHandler` is a small custom `logging.Handler`: on every `emit()` it computes today's date (`datetime.now().strftime("%Y-%m-%d")`), and if that differs from the date its inner `FileHandler` was opened for, it closes the old one and opens `<directory>/<today>.log`. The check happens per record rather than on a timer, so a process that's been idle across midnight still rolls over correctly on its next log line rather than needing a restart. `configure_trailerplace_logging()` attaches one of these (directory from `LOG_DIR`, default `logs/`) alongside the existing console `StreamHandler`, sharing the same formatter — so `LOG_FORMAT=json` affects both identically, and every line (including the multi-line conversation blocks, preserved verbatim — `test_multiline_record_preserved_verbatim`) lands in both places.

### Tool-level logging

Each place that actually calls an external system or a significant internal tool now logs its own line at the point it runs, independent of the per-turn summaries above (useful when grepping across many turns for one kind of activity):

- `src/graph/nodes/search.py` — `TOOL search: session=... category=... filters=... already_shown=N` before the call, a second line if the zero-result brand-relaxation retry fires, and `TOOL search: session=... results=N brand_relaxed=... urls=[...]` after.
- `src/graph/nodes/inventory_lookup.py` — the requested identifiers before `lookup_inventory` runs, then `status=... matches=N requested=...` after.
- `src/graph/nodes/email_actions.py` — for every resolved email trigger, whether it was queued to the outbox or is being sent directly (persistence off), and the direct-send outcome (`sent` / `FAILED`).
- `src/tools/email_sender.py` — `send_email` now logs `email_sent | backend=... subject=... to=...` **on success** (previously only `email_send_failed` on the exception path was logged, which is why the incident log the owner shared showed the failure but nothing about the successful retry that followed it); the failure log now also carries the backend.
- `src/conversation_store.py` — `deliver_pending_outbox`'s per-row loop now logs `TOOL outbox: event_id=... event_type=... session=... -> sent (attempt N)` on success, and the existing `outbox_delivery_failed` exception log gained `event_type`/`session_id`/`attempt` fields.

### Tests added

- `tests/unit/test_conversation_log.py` (8 tests) — every section of the block present; the "no analysis available" placeholder on a replay; the error line replacing the reply; haul-classification/inventory-lookup/email-trigger/interruption/requirement-change lines appearing only when the underlying data is present; tools-fired rendering for search/lookup/email; a malformed analysis object logged, not raised; `None`/empty state fields render as `-`.
- `tests/unit/test_log_setup.py` (7 tests) — writes to today's date-named file; creates the directory if missing; rolls to a new file when the (mocked) date changes; a formatter set after the first emit propagates to the already-open file; a broken record doesn't raise (`logging.Handler.handleError`); a multi-line record is preserved verbatim; `JsonFormatter` composes with the handler.
- `tests/unit/test_conversation_store_async_outbox.py` (3 tests) — no-op when persistence is off; submits `deliver_pending_outbox` with the given `limit` to `_pool` when persistence is on; the calling thread never invokes the drain function directly, only hands it to the pool.
- `tests/integration/test_observability.py` (+3 tests) — `_RecordingGraph` gained optional `turn=`/`reply=` params so a fake graph can populate `state["turn"]` with a real `TurnAnalysis`; new tests drive the actual `/chat` route and assert on the real `trailerplace.conversation` log records: a full block with the right user message/reply/intent/state, the real session and turn ids from the request, and an `ERROR: ...` block on a contained `GraphFailure`.

### Verified

`pytest` green twice: **252 passed, 10 skipped** (up from 231 — 21 new tests; the 10 skips are still the deliberately-unrun `db`-marked tests). End-to-end offline check: booted the real `/chat` route through `TestClient` with `configure_trailerplace_logging()` actually called (not mocked), `LOG_DIR` pointed at a scratch directory, and confirmed a file literally named `2026-07-11.log` was created containing the full block shown in the README's Operations section, with the same content also printed to the console in the same process.

Not independently re-verified against the live network path from the original incident (a real Graph OAuth timeout) — that requires reproducing the network condition, which isn't controllable from here. What's fixed is structural: the drain that was blocking the response no longer runs on the request's thread, which removes the *category* of problem regardless of which external call is slow.

## Immediate Follow-up Bug: Logging Was Never Wired Up

After the above landed, the owner ran the real backend (`uv run .\main.py`) and saw only uvicorn's own access-log lines (`GET /health`, `POST /chat`) — none of the new `trailerplace.turn` / `trailerplace.conversation` / `TOOL ...` lines, on the console or in `logs/`.

**Root cause:** `configure_trailerplace_logging()` (`src/log_setup.py`) has existed since Milestone 0 and was correctly implemented — but **nothing in the codebase ever called it**. `main.py` builds the app and hands it straight to `uvicorn.run()`; `src/api/app.py`'s `create_app()`/`run_startup()` never invoked it either. The root logger therefore had zero handlers and its default effective level (`WARNING`), so every `logger.info(...)` call anywhere in the app — the turn_log JSONL record, the conversation_log reasoning block, every `TOOL search/inventory_lookup/email/outbox` line added in the fix above — was silently discarded before it ever reached a handler. Only `uvicorn`/`uvicorn.access`, which uvicorn configures on its own namespaced loggers independent of root, ever printed anything. `tests/unit/test_scaffold.py`'s existing check only asserted the function *exists* (`getattr` on the module), never that it runs, so this shipped undetected through the whole M9 build.

**Fix:** `src/api/app.py::create_app()` now calls `configure_trailerplace_logging()` as its first line, before the `FastAPI(...)` object is constructed. This covers every real entry point: `main.py` calls `create_app()` at module scope before `uvicorn.run()` even starts (and uvicorn's default `LOGGING_CONFIG` has no `"root"` key, so its own `dictConfig` call does not touch or reset the handlers we just attached), and `TestClient(create_app())` in tests gets it too.

**Why this doesn't create `logs/` directories during `pytest`:** `configure_trailerplace_logging()` is guarded by `if not root.handlers`. Pytest's own `logging` plugin installs a handler on the root logger for the session before any test module runs (that's what backs `caplog`), so the guard is always already-true under pytest and the function no-ops — verified empirically by running the full suite and confirming no `logs/` directory or `turns.jsonl` appears in the repo afterward.

**Regression test:** `tests/integration/test_observability.py::test_create_app_wires_up_logging` monkeypatches `src.api.app.configure_trailerplace_logging` and asserts `create_app()` calls it — this is the test that would have caught the original gap.

**Verified for real** (not just via `TestClient` inside pytest): a standalone script outside pytest, mirroring `main.py`'s exact `create_app()` call, with `LOG_DIR` pointed at a scratch directory and no mocking of the logging setup. Confirmed both effects: the full conversation-reasoning block printed to console, and the identical content landed in a file literally named `2026-07-11.log`. `pytest` green twice after the fix: **253 passed, 10 skipped** (+1 test).

## Haul Classification Corrected: Lightweight Skips WEIGHT, Not Width

**Owner correction (supersedes milestone.md's Locked Decision wording).** milestone.md described the haul-classification flags as: `is_lightweight_utility_load` → "skip **width** question", `needs_width_question` → "ask width question". The first half was wrong and conflated the two flags. The intended, and now-implemented, behavior is:

- **`is_lightweight_utility_load` is a WEIGHT judgment** — the reference `mini_llm_classifier.py` sets it when "the haul item is likely 1500 lbs or less". If we already know the Utility load is light (golf cart, ATV, mower, kayak, camping gear, …), asking "what's the rough total weight of your load?" is pointless, so the code **skips the weight question** (`haul_weight_lbs`).
- **`needs_width_question` is a SIZE judgment** — set for large/wide/heavy-duty/vehicle cargo. The trailer must be wide enough, so the code **asks the width question** (injects `item_or_trailer_width_ft`). This half was already correct and is unchanged.

The two are independent: a light load sets only the weight flag, a big load sets only the width flag.

### Implementation (`src/graph/apply_analysis.py`)

- `_inject_width_question` no longer early-returns on `is_lightweight_utility_load` (that flag has nothing to do with width). It now depends solely on `needs_width_question` plus the category exclusions — a lightweight load simply never sets `needs_width_question`, so no special-case is needed.
- New `_skip_weight_for_lightweight(state, haul)`: when `category == "Utility"` and `is_lightweight_utility_load` is true and the user hasn't already given a weight, mark `haul_weight_lbs` skipped via `_mark_skipped`. Called from `apply_analysis_to_state` right after `_inject_width_question`. "Don't ask" is not "discard": if the user volunteers a weight in the same message, the extraction path fills the slot first and the `UTILITY_WEIGHT_SLOT not in slots` guard leaves it untouched; search then applies the payload filter as normal. When skipped, no payload filter is applied — correct, since any utility trailer handles a light load, so weight isn't a distinguishing search criterion. The invariant (`enforce_haul_classification_invariant`) already clears the flag when no cargo item is grounded, so a bare "I want a utility trailer" never triggers the skip.

### Prompt / schema

- `src/llm/schemas.py`: the two `HaulClassification` field descriptions now say explicitly which question each governs (weight vs width) and that they're a WEIGHT vs SIZE judgment.
- `src/llm/analyze.py`: the `=== HAUL CLASSIFICATION ===` block is retitled "judge the cargo's WEIGHT and SIZE independently", each flag's rule states the question it governs, and a closing line stresses the flags are independent and not both set.

### Tests

- `tests/unit/test_apply_analysis.py`: `test_no_preference_and_width_injection_cases` clarified (width is a size question, a lightweight load neither injects nor skips width). Three new tests: `test_lightweight_utility_load_skips_the_weight_question` (golf cart → `haul_weight_lbs` in `skipped_slots`, width untouched), `test_lightweight_does_not_skip_weight_the_user_actually_gave` (a volunteered "900 lbs" is kept, not skipped — raw under the slot, parsed under `payload_lbs`), `test_lightweight_flag_only_skips_weight_for_utility` (a non-Utility category is never weight-skipped by this path).
- `scripts/scenarios/lightweight-utility-skips-weight.yaml` (tagged `[regression, m5]`): a live scenario asserting a golf-cart Utility request lands `haul_weight_lbs` in `skipped_slots` and the reply never asks for the load weight. The M5 count in `tests/unit/test_convo_runner_parsing.py` goes 20 → 21 and the regression suite 47 → 48.

### Verified

`pytest` green twice: **257 passed, 10 skipped** (+4 from the correction). Runtime check with a `FakeLLM`-shaped analysis driving `apply_analysis_to_state` → `qualification_node`: a lightweight golf-cart Utility load skips `haul_weight_lbs` and completes qualification with `next_question=None` (straight to search, weight never asked), while a non-lightweight Utility load ("industrial generator") still surfaces "What's the rough total weight of your load?" as the next question.

**One assumption to confirm:** "skip the weight question" is implemented as *mark it not-asked, apply no payload filter* — not as *auto-fill a nominal light weight*. That matches the plain reading of "we should not ask the weight question" and avoids injecting a possibly-wrong payload constraint into search. If instead you want lightweight loads to auto-fill a nominal weight (e.g. so search still applies a small payload floor), that's a one-line change to `_skip_weight_for_lightweight` — say the word.

**milestone.md itself is left unedited** (it is the historical build spec); this note plus the overview.md entry are the authoritative record that the lightweight flag governs the weight question, not the width question.
