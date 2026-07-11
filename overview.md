# Milestone 0-9 Overview

Implemented scaffold, domain data layer, durable database layer, LLM structured-output layer, minimal LangGraph chat runtime, the M5 live-scenario harness, (M6) real Pinecone search with fit/brand reranking plus a deterministic Excel inventory-lookup matcher, (M7) the email tools, contact-info gate, multi-trigger/system-alert email actions, transactional outbox delivery, and hard-lead upgrades, (M8) the audited `/chat` response contract, boot-time readiness gating, error/timeout/size containment, and the feedback round trip, and (M9) the tagged regression suite, LangSmith tracing, per-turn structured logs, the ≤2-completions cost audit, and the release run-book:

```text
.env.example
.gitignore
README.md
alembic.ini
alembic/
  env.py
  versions/
    a001_initial.py
    20260525_0001_optional_contact_leads.py
    20260703_0002_durable_chat_state.py
main.py
docs/
  e2e_checklist.md
scripts/
  smoke_llm.py
  convo_runner.py
  cost_report.py
  scenarios/
    contact-full.yaml
    contact-partial-name.yaml
    contact-decline.yaml
    contact-ignore-asks-trailer.yaml
    category-direct.yaml
    category-info-vs-select.yaml
    category-exploration.yaml
    feature-request-no-category.yaml
    prefill-from-first-message.yaml
    interruption-repeat-once.yaml
    explicit-skip.yaml
    skip-all-show-results.yaml
    haul-item-lock.yaml
    clarification-term.yaml
    brand-only.yaml
    brand-plus-category.yaml
    brand-typo.yaml
    gooseneck-not-category.yaml
    defaults-applied.yaml
    loose-answers.yaml
    lightweight-utility-skips-weight.yaml
    search-happy-path.yaml
    refine-length-change.yaml
    more-options-dedupe.yaml
    category-change-keep-some.yaml
    reference-second-listing.yaml
    brand-filter-search.yaml
    brand-filter-no-match.yaml
    inventory-stock-lookup.yaml
    inventory-first-message-contact-deferred.yaml
    inventory-weight-not-stock.yaml
    inventory-model-typo.yaml
    inventory-ambiguous-clarify.yaml
    inventory-mid-qna.yaml
    faq-financing-with-contact.yaml
    escalate-quote-no-contact-then-provide.yaml
    escalate-decline-contact.yaml
    listing-interest-second-one.yaml
    faq-plus-escalate-one-message.yaml
    inventory-lookup-then-interest.yaml
    results-shown-alert.yaml
    unanswered-question-alert.yaml
    results-alert-stashed-then-sent.yaml
    adversarial-contradictory-sizes.yaml
    adversarial-category-change-twice.yaml
    adversarial-paragraph-requirements.yaml
    adversarial-faq-midqual-then-answer.yaml
    adversarial-gibberish-input.yaml
src/
  __init__.py
  config.py
  db.py
  db_models.py
  log_setup.py
  models.py
  thinking_agent.py
  tracing.py
  turn_log.py
  conversation_log.py
  conversation_store.py
  graph/
    __init__.py
    apply_analysis.py
    build.py
    state.py
    nodes/
      __init__.py
      analyze.py
      apply_analysis.py
      email_actions.py
      inventory_lookup.py
      qualification.py
      respond.py
      search.py
  llm/
    __init__.py
    analyze.py
    client.py
    respond.py
    schemas.py
    usage.py
  shown_listings_store.py
  api/
    __init__.py
    app.py
    readiness.py
    routes.py
    schemas.py
  domain/
    __init__.py
    brands.py
    canned_responses.py
    categories.py
    defaults.py
    make_aliases.py
    normalizer.py
    slot_map.py
    trailer_fields.py
    units.py
  search/
    __init__.py
    pinecone_search.py
    ingest.py
    inventory_matcher.py
  tools/
    __init__.py
    email_sender.py
tests/
  __init__.py
  fixtures/
    chat_response_contract.json
  integration/
    test_api_contract.py
    test_chat_roundtrip.py
    test_db.py
    test_email_outbox.py
    test_observability.py
  unit/
    __init__.py
    llm_helpers.py
    test_restore_feedback.py
    test_domain_categories.py
    test_domain_defaults_canned.py
    test_domain_normalizer_brands.py
    test_domain_slot_units.py
    test_apply_analysis.py
    test_qualification.py
    test_llm_analyze.py
    test_llm_apply_analysis.py
    test_llm_respond.py
    test_llm_schemas.py
    test_scaffold.py
    test_convo_runner_parsing.py
    test_search_node.py
    test_inventory_matcher.py
    test_inventory_lookup_node.py
    test_email_sender.py
    test_email_actions.py
    test_tracing.py
    test_turn_log.py
    test_cost_report.py
    test_conversation_log.py
    test_log_setup.py
    test_conversation_store_async_outbox.py
```

- `alembic.ini`: Alembic configuration pointing at the existing `alembic/` migration directory.
- `alembic/env.py`: Alembic runtime hook that uses `src.db.get_engine()` and `src.db_models.Base.metadata`.
- `alembic/versions/*`: migration history for leads, conversations, durable turn receipts, and outbox tables.
- `main.py`: uvicorn entrypoint for the FastAPI app.
- `docs/e2e_checklist.md`: M8 manual end-to-end checklist (login → converse → refresh → backend restart → feedback → New Conversation → logout, plus robustness spot-checks).
- `scripts/smoke_llm.py`: manual real-OpenAI smoke script for Analyze structured output.
- `src/config.py`: central typed settings loader for `.env` values used by planned milestones. M8 adds `chat_timeout_seconds` (`CHAT_TIMEOUT_SECONDS`, default 150 — must stay under app.py's 180 s client timeout) and `chat_max_message_chars` (`CHAT_MAX_MESSAGE_CHARS`, default 4000).
- `src/db.py`: Postgres engine/session helpers, database enablement check, and `create_all` schema guard. M8 adds `run_migrations()` (Alembic `upgrade head`, used at boot when `DB_AUTO_CREATE=1`) and `ping()` (a `SELECT 1` readiness probe that never raises).
- `src/api/readiness.py`: M8 process-readiness flags behind `/health` — `mark_graph_ready`/`mark_db_ready`/`mark_failed`/`reset`/`is_ready`/`readiness_status()`.
- `src/db_models.py`: SQLAlchemy models for leads, conversations, turn receipts, and email outbox.
- `src/log_setup.py`: idempotent stdlib logging setup used by `app.py`.
- `src/models.py`: lenient `TrailerListing` dataclass for Streamlit cards.
- `src/thinking_agent.py`: disabled thinking-agent shim for frontend compatibility.
- `src/conversation_store.py`: durable persistence helpers for leads, conversations, receipts, restore/close, feedback, and outbox. M8 adds `_apply_feedback_to_messages()` — `restore_session` now maps `conversation[i]["feedback"]` back onto the `2i+1`-th assistant message as `user_feedback` (and guarantees every restored message carries `listings`/`user_feedback` keys), so notes saved before a refresh reappear; `close_session` also calls `ensure_persistence_schema()` like its siblings. M7 adds an optional in-transaction `session` param to `promote_lead_to_hard`/`update_lead_item_of_interest` (so lead upgrades commit atomically with the durable turn) and `register_default_outbox_handlers()`, which maps all six outbox `event_type`s to `email_sender.send_email`.
- `src/tools/email_sender.py`: M7 email sender. `send_email(subject, body)` dispatches on `EMAIL_BACKEND` — `graph` (raw `requests` client-credentials token + `sendMail`, no msal) or `smtp` (`smtplib` STARTTLS to `EMAIL_TO`); any failure logs and returns False. Pure `render_email_body`/`render_subject` produce the exact spec body/subject format.
- `src/graph/state.py`: process-local session store, `SessionState`, and snapshot serialization helpers. M8 adds `session_lock(session_id)` — a per-session `threading.Lock` that gives the persistence-off path the same same-session serialization that `durable_turn`'s advisory lock gives the persistence-on path.
- `src/graph/build.py`: single compiled LangGraph skeleton for Analyze, Apply, Qualification, stubs, and Respond.
- `src/graph/apply_analysis.py`: deterministic state mutation, contact/category/slot handling, category-clarification wiring (ported resolver), slot-answer→metadata-target mapping (numbers stored under target keys, never re-encoded strings; vague answers still marked answered), skips, repeats, and the two independent haul-classification actions: `needs_width_question` (a SIZE judgment) → inject the width question (`_inject_width_question`); `is_lightweight_utility_load` (a WEIGHT judgment) → skip the Utility weight question `haul_weight_lbs` (`_skip_weight_for_lightweight`, only when the user hasn't volunteered a weight). See the "haul classification corrected" note in Details.md — this supersedes milestone.md's original wording that tied the lightweight flag to the width question.
- `src/graph/nodes/*`: LangGraph node wrappers for analyze, apply-analysis, qualification, respond, search, inventory-lookup, and email-actions. M6 made search/inventory real — `search.py` builds Pinecone metadata filters from answered slots and calls `search_pinecone_listings`, with a zero-result brand fallback re-run; `inventory_lookup.py` calls `lookup_inventory` and asserts the side-query invariant (qualification state untouched). Both emit `results_shown` system-alert triggers into `turn_outcome["system_email_triggers"]` and preserve the `inventory_lookup_ran`/`contact_invite_suppressed` contract respond.py reads. M7 makes `email_actions.py` real: it resolves every customer trigger (`turn.email_triggers`) + system alert (`system_email_triggers`) + stashed `pending_email_actions` into email events under the contact gate (Name + Email|Phone), delivers FAQ canned text immediately, stashes/asks-once/drops-on-decline, records `canned_keys`/`emails_sent`/`email_status`/`outbox_events`, and is idempotent across the two graph passes (consumes processed triggers). Persistence-off it sends directly via `email_sender`; persistence-on it emits `outbox_events` the route queues in-transaction.
- `src/llm/schemas.py`: strict pydantic models for Analyze and Respond structured outputs.
- `src/llm/client.py`: injectable LLM protocol and OpenAI structured-output implementation.
- `src/llm/analyze.py`: Analyze prompt builder/caller plus safety-net normalization; reads the real `SessionState` keys (pending question slot, flat contact fields, slots/sources).
- `src/llm/respond.py`: Respond prompt builder/caller for customer-facing replies; reads decision context from real state/analysis (repeat count, pending category change, interruption).
- `src/shown_listings_store.py`: in-process shown URL helpers used by the frontend.
- `src/api/app.py`: FastAPI app factory. M7 wires the transactional-outbox drain to the email sender via `register_default_outbox_handlers()`. M8 adds a `lifespan` handler running `run_startup()`: Alembic `upgrade head` when `DB_AUTO_CREATE=1`, compile the graph once, probe the DB, and record the result on `readiness` (failures are logged and surfaced through `/health`, never raised).
- `src/api/routes.py`: health endpoint plus M4 `/chat`, restore, and reset routes wired to the graph and durable store; unions already-shown URLs in both persistence modes, guards missing lead ids, and timestamps messages. M5 adds `GET /session/{id}/state`, a raw-snapshot debug route gated by `DEBUG_STATE_ENDPOINT` for the scenario runner. M6 filters the `/chat` response's `listings` down to `ReplyOutput.cited_listing_urls` when `SHOW_ONLY_LLM_MENTIONED_CARDS` is set. M7 queues the turn's gate-approved `outbox_events` inside `durable_turn`, promotes the lead to hard and updates item-of-interest in the same transaction, then drains the outbox after commit; the debug `/state` route also surfaces the last turn's `emails_sent` for scenario assertions. M8 hardens the surface: `/health` reports `ok` only once `readiness.is_ready()` (503 otherwise); `/chat` validates `session_id`/`turn_id` as UUIDs up front (422, so 409 stays reserved for genuine turn-id reuse), truncates oversized messages, runs the graph in a worker thread under `chat_timeout_seconds`, holds the per-session lock, and contains every graph failure or timeout as `GraphFailure` → a 200 full-contract apology (`ERROR_ASSISTANT_TEXT`) with no turn receipt written, so the frontend's retry replays cleanly; `/session/reset` is always 200 even for unknown or malformed ids.
- `src/api/schemas.py`: pydantic request/response schemas for scaffold routes; M5 adds `DebugStateResponse`.
- `src/domain/categories.py`: canonical trailer categories, synonym resolution, clarification prompt data, and make prompt compatibility wrapper.
- `src/domain/trailer_fields.py`: per-category qualification slots, questions, guidance, and field-spec accessors.
- `src/domain/brands.py`: make inventory loader, make prompt block, categories by make, and Pinecone make filter values.
- `src/domain/make_aliases.py`: canonical make alias map shared by domain/search logic.
- `src/domain/normalizer.py`: listing-catalog cleanup and canonical category/make/hitch/color/condition helpers.
- `src/domain/units.py`: listing-catalog length and weight parsers.
- `src/domain/defaults.py`: plug-and-play category defaults map and accessor; M5 adds an optional `CATEGORY_DEFAULTS_JSON` env-var merge so the `defaults-applied` scenario can seed a default without editing the file.
- `src/domain/slot_map.py`: slot-to-metadata map plus safety-net value normalization.
- `src/domain/canned_responses.py`: exact FAQ and non-FAQ canned response strings from the spec.
- `tests/unit/test_scaffold.py`: Milestone 0 scaffold acceptance tests.
- `tests/unit/test_domain_*.py`: Milestone 1 domain data acceptance tests.
- `tests/integration/test_db.py`: Milestone 2 Postgres/Alembic/durable persistence acceptance tests, skipped unless `TEST_DATABASE_URL` is set.
- `tests/unit/llm_helpers.py`: shared sample `TurnAnalysis` builder for LLM tests.
- `tests/unit/test_llm_*.py`: Milestone 3 strict-schema, prompt-construction, FakeLLM plumbing, normalization, and haul-invariant tests.
- `tests/unit/test_apply_analysis.py`: Milestone 4 deterministic state-merge, skip, category, contact, and width-injection tests.
- `tests/unit/test_qualification.py`: Milestone 4 qualification order/completion/injected-width tests.
- `tests/integration/test_chat_roundtrip.py`: Milestone 4 FakeLLM `/chat`, restore/reset, and width-question round-trip tests.
- `scripts/convo_runner.py`: Milestone 5 live scenario runner. Pure YAML-loading and assertion-evaluation functions (`load_scenario`, `load_scenarios`, `evaluate_assertions`) plus a network-touching `run_scenario`/CLI that POSTs each turn to a running `/chat` and reads `GET /session/{id}/state` for state assertions. Not run by pytest; requires a live backend and real `OPENAI_API_KEY`.
- `scripts/scenarios/*.yaml`: the 20 Milestone 5 scenarios named in the milestone doc (contact collection, category mapping/info-vs-select/exploration, feature-first prefill, interruptions/skips, haul-item lock, category clarification, brand handling incl. typo correction, the gooseneck-never-a-category guard, category defaults, and loose/range answers).
- `tests/unit/test_convo_runner_parsing.py`: Milestone 5 acceptance tests — network-free. Covers scenario loading/validation, every `evaluate_assertions` kind (`expect_state` incl. `_contains` and `slot:` prefixes, `expect_reply_contains_any/not_contains`, `expect_listings`, `expect_emails_sent`), and that every real file under `scripts/scenarios/` parses and matches the milestone's named set exactly. M6 extends the expected-set assertion with the 13 new M6 scenario names.
- `src/search/pinecone_search.py`: Milestone 6 port of the reference Pinecone search module (logic unchanged, imports updated to `src.domain.*`). Public `search_pinecone_listing_result`/`search_pinecone_listings`; builds hard metadata filters (category/make via `make_filter_values`/hitch_type/Aluminum subcategory/min length), runs the fit-rerank pass (`_rerank_listings_by_fit`, length-first strict no-under-length policy weighted by `RERANK_*` env knobs), then the category/make-priority quota reorder (`_apply_category_make_preference`, `CATEGORY_MAKE_PREFERENCES`).
- `src/search/ingest.py`: Milestone 6 port of the reference ingestion CLI (logic unchanged, imports updated). Runnable via `python -m src.search.ingest [--force] [--no-wipe]`; reads `listings_final_v5.xlsx`, builds/embeds/upserts vectors into the Pinecone index in batches.
- `src/search/inventory_matcher.py`: Milestone 6 reduced port of the reference Excel inventory matcher. The reference file's three embedded mini-LLMs (extractor/reply/feature-framing) and its entire regex-fallback extraction stack (`_fallback_extraction`, `_extract_year`, `_extract_stock`, `_wants_*`, `search_trailers`, etc.) are deleted — that job now belongs to the Analyze LLM (`TurnAnalysis.inventory_lookup`, M3) and the Respond LLM. Kept: `normalize_text`, `extract_model_code`, `prepare_inventory`/`prepared_inventory` (`lru_cache`, locates `listings_final_v5.xlsx` the same way `src/domain/brands.py` does), `_score_candidate` (rapidfuzz weighted: 0.30 make / 0.35 model-code / 0.15 model-text / 0.10 title / 0.10 search), `match_inventory` (stock/year+make/model-row/possible-model threshold logic, adapted to a lightweight `_Identifiers` dataclass replacing the deleted `TrailerQueryExtraction`), `_no_exact_alternative_rows`. `_row_to_listing` is extended with `color`/`axles`/`material`/`floor`/`relevance_score` for card-shape parity with Pinecone results. New public entry `lookup_inventory(*, year, make, model_text, stock_number, limit)` — a pure function of identifiers returning `{match_status, matches, requested_label}`.
- `src/graph/nodes/search.py` / `src/graph/nodes/inventory_lookup.py`: see the `src/graph/nodes/*` entry above.
- `tests/unit/test_search_node.py`: Milestone 6 unit tests for the search node using a monkeypatched `search_pinecone_listings` in place of a real Pinecone call — metadata-filter building per slot type (incl. Roll Off bin-size), brand→make filter, zero-result brand fallback re-run, shown-URL dedupe pass-through, results-shown system trigger, and a result-shape check against app.py's `TrailerListing` construction.
- `tests/unit/test_inventory_matcher.py`: Milestone 6 unit tests for the Excel matcher against a small in-memory fixture DataFrame (not the real 200KB workbook) — stock exact match, year+make, make+model partial, no-exact year fallback to same-make-other-year alternatives, an ambiguous two-close-models case, card-shape parity, and `normalize_text` mojibake/dimension cleanup.
- `tests/unit/test_inventory_lookup_node.py`: Milestone 6 unit tests for the inventory-lookup node — gate refusal on low confidence/`is_lookup=False`, shown-URL dedupe, the side-query invariant (category/slots/brand/skips/qualification_complete untouched), the `inventory_lookup_ran`/`contact_invite_suppressed` contract, and the results-shown system trigger.
- `tests/integration/test_chat_roundtrip.py`: M6 adds a first-turn inventory-lookup test (contact invite deferred to the next turn) and a `SHOW_ONLY_LLM_MENTIONED_CARDS` filtering test, both with `lookup_inventory` monkeypatched for determinism.
- `scripts/scenarios/*.yaml`: M6 adds 13 scenarios (`search-happy-path`, `refine-length-change`, `more-options-dedupe`, `category-change-keep-some`, `reference-second-listing`, `brand-filter-search`, `brand-filter-no-match`, `inventory-stock-lookup`, `inventory-first-message-contact-deferred`, `inventory-weight-not-stock`, `inventory-model-typo`, `inventory-ambiguous-clarify`, `inventory-mid-qna`) covering real search/refinement/brand-filter flows and the inventory-lookup gate, contact-deferral, and ambiguity-clarification behaviors named in the milestone doc.
- `src/tools/email_sender.py`: Milestone 7 email sender — `send_email` (graph/smtp backends, no msal, failures return False) plus exact `render_email_body`/`render_subject` formatters.
- `src/graph/nodes/email_actions.py`: Milestone 7 real email-actions node — contact-gated, multi-trigger, system-alert-aware, two-pass-idempotent; produces `outbox_events` (persistence-on) or sends directly (persistence-off).
- `tests/unit/test_email_sender.py`: Milestone 7 unit tests — exact body/subject format, graph-vs-smtp branch selection with mocked `requests`/`smtplib`, graph token+sendMail payload/recipient, smtp `EMAIL_TO` recipient, and failure→False-without-raise.
- `tests/unit/test_email_actions.py`: Milestone 7 unit tests (FakeEmailSender, persistence-off) — the full contact-gate matrix (complete→send, partial→stash+ask, provided→send, declined→drop+no-reask), FAQ canned reply present even when deferred, listing-interest selected/unselected/fallback variants, multi-trigger ordering, system-alert send/stash-silently behavior, and double-pass idempotency.
- `tests/integration/test_email_outbox.py`: Milestone 7 integration tests — persistence-off `/chat` escalation sends directly; DB-marked (`@pytest.mark.db`, skipped without `TEST_DATABASE_URL`) outbox-in-transaction creation + post-commit drain + hard-lead upgrade, failed-handler retry, and duplicate-turn-replay dedupe.
- `tests/unit/test_convo_runner_parsing.py`: M7 extends the expected-scenario-set assertion with the 9 new M7 scenario names.
- `tests/fixtures/chat_response_contract.json`: Milestone 8 golden `/chat` response — the exact key set app.py's `_process_assistant_reply` reads.
- `tests/integration/test_api_contract.py`: Milestone 8 acceptance tests (network-free, faked graph, persistence off) — golden key set incl. the always-present `customer_email`, a verbatim copy of app.py's `TrailerListing(...)` parser run over a returned listing, graph error and timeout both containing to a 200 apology, oversized-message truncation, malformed `session_id`/`turn_id` → 422, unknown-session reset → 200, closed-session shape, and `/health` 503-until-lifespan (plus 503 `error` when the DB is unreachable).
- `tests/unit/test_restore_feedback.py`: Milestone 8 unit tests for `_apply_feedback_to_messages` — turn N maps to assistant message 2N+1, restored messages always carry `listings`/`user_feedback`, empty/out-of-range feedback ignored, inputs not mutated.
- `tests/integration/test_db.py`: M8 adds three DB-marked tests — restore after a simulated backend restart (snapshot reload, listing dicts still in the snapshot), feedback landing on turns 0 and 3 and surviving a later conversation rewrite then returning on the right assistant message, and a closed session reporting `closed: true`.
- `tests/unit/test_scaffold.py`: M8 updates `test_health` to enter `TestClient` as a context manager, since `/health` is now `ok` only after the lifespan compiles the graph and clears the DB probe.
- `scripts/scenarios/*.yaml`: M7 adds 9 scenarios (`faq-financing-with-contact`, `escalate-quote-no-contact-then-provide`, `escalate-decline-contact`, `listing-interest-second-one`, `faq-plus-escalate-one-message`, `inventory-lookup-then-interest`, `results-shown-alert`, `unanswered-question-alert`, `results-alert-stashed-then-sent`) covering FAQ/escalation/listing-interest emails, the contact gate (provide/decline), multi-trigger fan-out, and the silent system-alert emails (results-shown, unanswered-question) including stash-then-send.
- `README.md`: Milestone 9 run-book — architecture summary, install/configure/migrate/ingest/start steps for backend + Streamlit, the test-command table, live-scenario and cost-audit commands, the full environment-variable table, an operations section describing the per-turn log record and the at-least-once email/lead invariants, and a security section flagging the live credentials in `.env`.
- `.env.example`: Milestone 9 environment template with placeholder secrets for every key `src/config.py` reads, grouped by concern (OpenAI, Pinecone, rerank knobs, frontend gate, database, email backends, API, observability), with inline notes on the two easily-confused recipients (`EMAIL_TO` for SMTP vs `RECIPIENT_EMAIL` for Graph) and the prod-unsafe `DEBUG_STATE_ENDPOINT`.
- `.gitignore`: Milestone 9 — excludes `.env` (which holds live credentials), the `TURN_LOG_PATH` JSONL output, virtualenvs, and Python/editor caches. `.env.example` is explicitly re-included.
- `src/tracing.py`: Milestone 9 LangSmith wiring. `configure_langsmith()` (idempotent; maps `LANGSMITH_*` settings onto the SDK's env vars, and force-disables a stale ambient `LANGSMITH_TRACING=true` when no key is configured), `trace_turn(session_id, **metadata)` (tags every span in the turn with `session:<id>`, dropping `None` metadata), `add_turn_metadata(**meta)` (attaches intent/category to the in-flight run after Analyze produces them), and `traceable_or_passthrough(name)` (a decorator that gates on `tracing_enabled()` **at call time**, so the pytest suite never ships a trace even when the developer's `.env` has tracing on; the traced variant is built once and cached). Reads settings via `config.settings` rather than a `from ... import settings` binding, so tests can swap the frozen `Settings` object.
- `src/turn_log.py`: Milestone 9 structured per-turn JSON logging. Emits one flat `chat_turn` record per `/chat` turn (`session_id`, `turn_id`, `intent`, `category`, `latency_ms`, `tools_fired`, `emails_sent`, `llm_calls`) on the `trailerplace.turn` logger, and additionally to an append-only JSONL file when `TURN_LOG_PATH` is set — that file is `scripts/cost_report.py`'s input, so the file formatter writes bare JSON with no level/timestamp prefix. `tools_fired()` derives `search` / `inventory_lookup` / `email` from `turn_outcome`; `reset_for_tests()` drops the file handler.
- `src/llm/usage.py`: Milestone 9 per-turn LLM call/token accounting. A `contextvars`-scoped `TurnUsage` (chat completions, embeddings, prompt/completion tokens, models) installed by `usage_scope()`; `record_completion()` / `record_embedding()` no-op outside a scope, so library code, scripts, and tests stay safe to call. contextvars rather than a global because `/chat` turns run concurrently on a `ThreadPoolExecutor`.
- `scripts/cost_report.py`: Milestone 9 cost audit. Network-free; parses the `TURN_LOG_PATH` JSONL and asserts the Locked Decision budget — at most 2 chat completions per turn (Analyze + Respond), exactly one embedding on search turns and none on any other, and zero extra calls on inventory-lookup turns (the Excel matcher uses neither an LLM nor an embedding). Receipt replays (0 completions) are skipped rather than failed. Pure `load_turns`/`summarize`/`violations`/`audit` functions plus a CLI that prints a per-turn table and exits nonzero on any violation, so it can gate a release.
- `scripts/convo_runner.py`: M9 consolidates the scenarios into a tagged suite. Every scenario YAML now declares `tags:`; new pure helpers `filter_by_tag()` / `load_suite()` select by tag, and the CLI gains `--suite <tag>` (target becomes optional), `--list` (prints the selection, makes no network calls), and `--repeat N` (all passes must pass — the M9 gate uses 2). `python scripts/convo_runner.py --suite regression` runs all 47 scenarios.
- `scripts/scenarios/*.yaml`: M9 tags all 42 existing scenarios `[regression, m5|m6|m7]` and adds the 5 adversarial ones (`adversarial-contradictory-sizes` — last value wins across two contradictions; `adversarial-category-change-twice` — each switch stays pending until the keep/drop answer; `adversarial-paragraph-requirements` — a pasted paragraph extracts every field, converts the quote-notation width and "two tons", and re-asks nothing; `adversarial-faq-midqual-then-answer` — FAQ email fires *and* the interruption counts as a non-answer, then the user answers so the slot is never skipped; `adversarial-gibberish-input` — no crash, no invented category, random digits are not a stock number, recovers on the next real message).
- `src/api/routes.py` (M9): `/chat` now wraps each turn in `llm_usage.usage_scope()` + `tracing.trace_turn()` and emits the structured record via `_log_turn()` (including on a contained `GraphFailure`). `_invoke_graph` runs the graph inside `contextvars.copy_context().run(...)` — **`ThreadPoolExecutor` does not propagate contextvars into its workers**, so without this the usage counters recorded by the nodes were invisible and every turn logged zero calls, which made `cost_report` classify real turns as receipt replays and audit nothing.
- `src/llm/client.py` (M9): `OpenAILLMClient.structured` is decorated with `traceable_or_passthrough("llm.structured")` and records each completion's token usage into the active `TurnUsage`.
- `src/search/pinecone_search.py` (M9): `_embed` records the one embedding a search turn is allowed to make.
- `src/graph/nodes/analyze.py` (M9): calls `add_turn_metadata(intent=..., category=...)` after Analyze returns, annotating the trace opened by `/chat`.
- `src/api/app.py` (M9): `run_startup()` calls `tracing.configure_langsmith()` before the graph compiles, so the first turn's spans are exported.
- `src/log_setup.py` (M9): adds `JsonFormatter` and `LOG_FORMAT=json` for machine-readable ops logs (a record's `turn` payload is inlined into the JSON object); plain text stays the default for local development.
- `src/config.py` (M9): adds `turn_log_path` (`TURN_LOG_PATH`).
- `tests/conftest.py` (M9): two autouse safety fixtures. `_never_touch_a_real_database` monkeypatches `db.database_enabled()` to False for every non-`db`-marked test — `persistence_enabled()` is satisfied by a developer `.env`, so `apply_analysis`'s lead updates were running `ensure_persistence_schema()` → `create_all()` against a **production** Postgres on nearly every unit test (this is what made the suite take ~110 s; it now takes ~22 s). `_disable_langsmith_tracing` clears the tracing settings and env var so no test can ship a trace. Also adds `replace_settings(monkeypatch, **values)`, since `Settings` is a frozen dataclass and must be swapped wholesale.
- `tests/unit/test_tracing.py`: Milestone 9 tracing tests — enablement requires both the flag and a key; `configure_langsmith` exports settings to env, force-disables a stale ambient flag, and is idempotent; `trace_turn` tags `session:<id>` and drops `None` metadata; `add_turn_metadata` updates the current run tree and no-ops without one; the decorator passes through untraced when disabled and builds its traced variant exactly once when enabled.
- `tests/unit/test_turn_log.py`: Milestone 9 usage-counter and turn-log tests — recording outside a scope is a no-op, scopes nest without leaking, `tools_fired` mapping, the record shape (incl. 2-dp latency rounding and the `error` key), bare-JSON JSONL output, emission on the `trailerplace.turn` logger, and `JsonFormatter` inlining the payload.
- `tests/unit/test_cost_report.py`: Milestone 9 cost-audit tests — JSONL loading (blank lines, foreign events, malformed JSON), the budget rules one at a time (3 completions fails; search turn needs exactly one embedding; an embedding on a non-search turn fails; an inventory-lookup turn that embeds fails; a 0-completion receipt replay is skipped, not failed), `--max-completions` override, audit aggregation over a clean 10-turn conversation, the CLI exit codes, and a round trip that writes with `turn_log` and reads with `cost_report` to guard the format contract.
- `tests/integration/test_observability.py`: Milestone 9 integration tests driving the real `/chat` route with a faked graph, persistence off — the contextvar regression (usage must survive the executor thread), the full turn-log record, per-turn scope isolation, a failed turn logging its error while still returning 200, the M9 acceptance test (`cost_report.audit` over a real 10-turn conversation: 20 completions, embeddings only on the 2 search turns, zero failures), and proof the audit actually fails on a third LLM call.
- `tests/unit/test_convo_runner_parsing.py` (M9): extends the expected-scenario set with the 5 adversarial names and adds suite tests — every scenario declares tags, the `regression` suite covers all 47 scenarios (a scenario left untagged would be a silent blind spot), the `adversarial` suite is exactly the 5 new ones, `filter_by_tag` sorts and excludes untagged files, and the CLI's `--list` selects a suite without network while a missing target or unknown suite exits nonzero.
- `src/conversation_log.py`: post-M9 follow-up — human-readable conversation reasoning + state log on the `trailerplace.conversation` logger, distinct from `turn_log.py`'s compact JSONL machine feed. `log_conversation_turn(...)` renders one block per turn: session/turn id, the raw user message, the Analyze LLM's full structured output (intent, extraction, haul classification, inventory-lookup identifiers, email triggers, interruptions, requirement changes), which tools fired and their results, the assistant's reply, and the full session state as it stood right after the turn (category, slots, skipped/pending questions, contact, brand preference, shown-listing counts). Never raises — wraps its own formatting in try/except so a malformed field can't break a served turn.
- `src/log_setup.py` (follow-up addition): `DailyFileHandler` — a custom `logging.Handler` that writes to `<LOG_DIR>/<YYYY-MM-DD>.log`, switching files when the date changes (checked per record, so an idle process still rolls over at its next log line — no restart needed). Unlike `TimedRotatingFileHandler`, the *live* file itself is always named after today's date, not a fixed name with the date appended to yesterday's rollover. `configure_trailerplace_logging()` now attaches both a console `StreamHandler` and a `DailyFileHandler` (dir from `LOG_DIR`, default `logs/`), sharing one formatter (`LOG_FORMAT=text|json`) — every log line, including the conversation-reasoning blocks, lands in both places.
- `src/conversation_store.py` (follow-up fix): `deliver_pending_outbox_async(limit=10)` — submits `deliver_pending_outbox` to the existing `_pool` background `ThreadPoolExecutor` instead of running it inline. Fixes a real incident: `/chat` used to call the outbox drain synchronously after the durable-turn commit, so a slow or failing Graph/SMTP send (observed: a TLS handshake reset talking to `login.microsoftonline.com`) added its own latency — up to the 30s+30s request timeouts — directly onto the user-facing chat response. The outbox is already designed for exactly this (a failed row stays retryable and the next drain, from any session's next turn, picks it up), so running the drain off the request thread makes that the *only* effect a slow send has.
- `src/api/routes.py` (follow-up additions): `/chat` calls `conversation_store.deliver_pending_outbox_async()` instead of the blocking `deliver_pending_outbox()`. `_log_turn` now also builds the full conversation-reasoning block via `conversation_log.log_conversation_turn(...)` — passing the request's raw `user_message`, the state's `TurnAnalysis` (`state["turn"]`), the returned `ChatResponse` (assistant text + listing count), the reply's `cited_listing_urls`, and the full in-memory session state — on both the success path and the contained `GraphFailure` path (where the block shows `ERROR: ...` instead of the reply).
- `src/graph/nodes/search.py` / `inventory_lookup.py` / `email_actions.py`, `src/tools/email_sender.py`, `src/conversation_store.py` (follow-up: tool-level logging): each tool now logs its own `TOOL search: ...` / `TOOL inventory_lookup: ...` / `TOOL email: ...` / `TOOL outbox: ...` info lines at the point it runs — search logs the filters used and result count (plus a second line when the zero-result brand-relaxation retry fires), inventory lookup logs the identifiers and match status/count, email actions logs whether an event was queued to the outbox or sent directly, `email_sender.send_email` now logs `email_sent` on success (previously only failures were logged) with the backend and recipient, and `deliver_pending_outbox` logs each event's outcome with its event id/type/attempt count.
- `tests/unit/test_conversation_log.py`: follow-up tests for the reasoning-block formatter — every required section present (session/turn id, user message, reasoning, tools, reply, state), the "no analysis available" placeholder on a replay/failed turn, the error line replacing the reply, haul-classification/inventory-lookup/email-trigger/interruption lines appearing only when relevant, tools-fired rendering for search/lookup/email, and that a malformed analysis object is swallowed rather than raised.
- `tests/unit/test_log_setup.py`: follow-up tests for `DailyFileHandler` — writes to today's date-named file, creates the directory if missing, rolls to a new file when the date changes (mocked), formatter changes propagate to an already-open file, a broken record doesn't raise, multi-line records (the conversation blocks) are preserved verbatim, and `JsonFormatter` composes with it.
- `tests/unit/test_conversation_store_async_outbox.py`: follow-up tests for `deliver_pending_outbox_async` — no-op when persistence is off, submits `deliver_pending_outbox` (with the given limit) to the background pool when persistence is on, and the calling thread never invokes the drain directly.
- `tests/integration/test_observability.py` (follow-up additions): `_RecordingGraph` gained `turn=`/`reply=` params so a fake graph run can populate `state["turn"]` with a real `TurnAnalysis`. New tests drive the real `/chat` route and assert on the actual `trailerplace.conversation` log records: a full reasoning block is emitted with the right user message/reply/intent/state, the block carries the real session and turn ids from the request, and a contained `GraphFailure` produces an `ERROR: ...` block instead of a reply.
- `tests/integration/test_email_outbox.py` (follow-up fix): the two DB-marked tests that assert on outbox row status right after a `/chat` POST (`test_outbox_created_in_transaction_drained_and_lead_hard`, `test_failed_handler_leaves_row_retryable`) now call a new `_drain_outbox_pool()` helper first — the same two-no-op-per-worker wait pattern `test_db.py` already uses for `enqueue_save_user_feedback` — since the drain they're asserting on now runs on the background pool rather than inline.
- `.env.example` / `.gitignore` / `README.md` (follow-up): `.env.example` documents `LOG_DIR` (default `logs`); `.gitignore` excludes the `logs/` directory the daily file handler creates; `README.md`'s Operations section documents both log records (the compact `trailerplace.turn` JSON and the human-readable `trailerplace.conversation` block) with a sample block, and its Emails section explains the background outbox drain and why it exists.
