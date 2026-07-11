# TrailerPlace Chatbot — Production Milestones

Build plan for the FastAPI + LangGraph backend behind the existing Streamlit frontend (`app.py`).
Each milestone is self-contained: an implementer should read **only the milestone section + the referenced spec sections in `prompt_structured.md`** and be able to complete it. Do not skip tests — they are the acceptance gate for every milestone.

## Locked Decisions (do not re-decide these)

| Decision | Value |
|---|---|
| Backend | FastAPI (`main.py`, port `CHATBOT_API_PORT`, default 8000) + LangGraph, Python 3.11+ |
| Graph design | **One single LangGraph** handles everything (contact collection included). No separate onboarding phase — backend always returns `sales_phase: "main"` and `onboarding_api_messages: []` to keep the frontend contract intact. |
| LLM | `gpt-4o-mini` (env `OPENAI_MODEL`) with OpenAI **structured outputs** (pydantic). **Exactly 2 LLM calls per normal turn**: (1) Analyze, (2) Respond. Embedding call only when searching. |
| Unit conversion | **The Analyze LLM does all normalization of user input**: lengths/widths/heights → feet, weights → lbs, `7'6"` → 7.5 ft, `AxB` splitting, ranges → smallest value, brand typos → canonical make. The code-side parsers (`units.py`, `normalizer.py`) exist **only** to parse listing-catalog strings inside search/ingest — they never touch user text. |
| Reasoning style | LLM decides intent/extraction/skips/haul-classification; code only dispatches on the LLM's structured output. Only deterministic behaviors allowed: Pinecone invocation gate, haul-item category lock, contact-info email gate, width-question injection guard (all code-side). |
| Haul classification | The Analyze LLM classifies the cargo's weight/size in one structured field (part of TurnAnalysis, not a separate call). Determines: lightweight utility (skip width question), heavy/wide/vehicle (ask width question), matched cargo item (grounded in user's words). Code enforces invariant: flag without cargo = flag cleared. |
| Dependencies | The existing pinned **`requirements.txt` is the dependency source of truth**. Notably absent on purpose: **no `msal`** (Graph email uses `requests` directly) and **no `pytest-mock`** (use pytest's built-in `monkeypatch`). Do not add packages without a demonstrated need. |
| Persistence | Two layers, per the reference `conversation_store.py` + `db_models.py` in this folder. (1) **In-memory working store**: a process-local `_sessions: dict[str, dict]` holds the live session state; it alone runs the bot when `TRAILERPLACE_PERSIST_CHATS` is off. (2) **Durable layer** (Postgres, 4 tables: `chatbot_leads`, `chatbot_conversations`, `chatbot_turns`, `chatbot_outbox`): each `/chat` request is an **atomic durable turn** — take a `pg_advisory_xact_lock(hashtext(session_id))`, check `chatbot_turns` for a receipt (same `turn_id` → return the stored response; same `turn_id` + different message → error), load `state_snapshot` into `_sessions` if not cached, run the graph in memory, deep-copy the final session into `state_snapshot`, write the simplified `conversation` list, increment `state_version`, write the turn receipt, queue email outbox events, commit — then drain the outbox. **Emails go through the transactional outbox** (at-least-once; failed rows stay retryable). No LangGraph checkpointer. Schema managed with Alembic; `ensure_schema()`/`create_all` as the best-effort runtime guard + test fixture. |
| Lead hardness | Lead row created on first turn as `lead_type='soft'`. Upgraded to `'hard'` **whenever ANY email actually sends** — customer-initiated (escalation, FAQ email, non-FAQ, log-interest) **and** system alerts (Unanswered Question, Results Shown to User), including repeat sends in the same session. Never downgraded. |
| System alert emails | Two **code-generated** (not LLM-detected) internal alerts, standard body format: **[Unanswered Question]** — fires when a qualification question is marked skipped after two non-answers (description = the question text + what the user said instead); **[Results Shown to User]** — fires on **every** turn that shows results (Pinecone search or inventory lookup, refinements and "more options" included; description = source + result count + category/identifiers). Both are **gated + stashed** exactly like other emails (contact complete → send; else stash in `pending_email_actions`; contact arrives → send all; declined → drop all). Both are **silent**: no canned text, never mentioned in the assistant reply, and they never generate their own contact ask — they wait passively on contact the normal flow collects. Every successful send upgrades the lead to hard. |
| FAQ behavior | FAQ intents reply with the canned response **and** send an internal email — but the email is subject to the contact-info gate (Name + Email\|Phone; ask once; skip email silently if declined). |
| Multi-trigger emails | A single message can trigger **several** email actions at once (e.g. an FAQ + an escalation). The Analyze LLM lists every one in `TurnAnalysis.email_triggers` and **all of them fire** — one email per trigger, in message order. The contact gate applies to the whole batch: contact incomplete → stash all + ask once; contact arrives → send all; declined/ignored → drop all silently. |
| Brand filtering | User brand preference **is** passed to Pinecone as a **hard `make` metadata filter** via `make_filter_values()` (owner decision 2026-07-07 — supersedes spec §Brand Handling's "not a searchable filter for now"). The Analyze LLM canonicalizes brand mentions itself; code never regex-matches user text for brands. |
| Inventory lookup | Deterministic **Excel fuzzy-match tool** (ported `inventory_matcher.py`, rapidfuzz — already pinned). Fires on **any** turn, including the very first message, when the Analyze LLM detects specific-inventory identifiers: make+year, make+model (partial/typo tolerated), or an explicit stock number — **identifiers alone suffice**, no price/availability wording required. Make-only stays in the brand-only flow. Bypasses qualification and the Pinecone gate; results returned as the **same listing cards as Pinecone results** and appended to `shown_listings` (so "the second one" / log-interest / dedupe work). A first-turn lookup **defers the initial contact invite to the next user turn**. Detection lives in Analyze, matching in pure code, reply in Respond — the reference file's three embedded mini-LLMs are **not** ported; still exactly 2 LLM calls and zero embeddings on a lookup turn. Ambiguous model matches → ask a clarification ("did you mean the 7210S-BT?"). |
| Session restore | `GET /session/{id}` returns message history with `listings: null` — **trailer cards do not survive a browser refresh or backend restart** (owner decision: app.py's `render_card` needs `TrailerListing` objects, crashes on JSON dicts, and must not be modified). Listing dicts ARE still persisted inside the conversation JSONB (feedback indexing, "the second one" references). Search dedupe therefore relies on the backend's persisted `shown_urls` state, never on the frontend's `already_shown_listing_urls` payload alone. |
| Thinking agent | **Out of scope.** Backend returns `thinking_context: null`; `src/thinking_agent.py` is a stub with `thinking_agent_enabled() -> False`. |
| Reference code | `categories.py`, `trailer_fields.py`, `pinecone_search.py`, `ingest.py`, `normalizer.py`, `units.py`, `make_inventory.py`, `make_aliases.py`, `inventory_matcher.py`, `conversation_store.py`, `db_models.py` in this folder are **reference logic** — port them into the new modular structure (logic preserved, adapted to the layout below). `db_models.py` and `conversation_store.py` port near-verbatim (they define the persistence contract). Reference files import `src.normalizer` / `src.chatbot.*` — those imports change to `src.domain.*` when porting. **Exception:** `inventory_matcher.py`'s three embedded mini-LLMs and its regex fallback extraction are deliberately NOT ported (see M6 — Analyze/Respond and pure code replace them). |
| Tests | `pytest` unit/integration with mocked LLM/Pinecone/SMTP everywhere; plus a live scripted-conversation runner (real API + real LLM) for flow-level milestones (M5–M9). |
| Spec | `prompt_structured.md` is the behavioral spec. When this doc says "per spec §X", open that section and implement it exactly. **Where this doc's Locked Decisions and the spec conflict, this doc wins** (it records later owner decisions — e.g. brand filtering). |

## Repository Layout (target)

`app.py` imports `src.log_setup`, `src.conversation_store`, `src.shown_listings_store`, `src.models`, `src.thinking_agent` — those five modules MUST live at `src/` top level with exactly the names/functions app.py uses.

```
main.py                       # uvicorn entrypoint: uvicorn app on CHATBOT_API_PORT
app.py                        # existing Streamlit frontend — DO NOT MODIFY
.env                          # already present
requirements.txt              # already present — dependency source of truth
alembic/ + alembic.ini        # schema migrations (M2)
src/
  config.py                   # all env parsing in one place (typed accessors)
  log_setup.py                # configure_trailerplace_logging()
  models.py                   # TrailerListing dataclass/pydantic (fields per app.py render_card; LENIENT types)
  thinking_agent.py           # STUB: enabled()->False, background()->False, generate/log no-ops
  conversation_store.py       # ported reference (M2): persistence_enabled(), leads, durable_turn,
                              #   restore_session, close_session, outbox, enqueue_save_user_feedback
  shown_listings_store.py     # add_shown_urls, add_shown_keys_and_urls, accumulate_shown_urls_from_chat_messages
  db.py                       # database_enabled(), engine + session factory, ensure_schema()
  db_models.py                # ported reference verbatim: ChatbotLead, ChatbotConversation,
                              #   ChatbotTurn (receipts), ChatbotOutbox (email outbox)
  api/
    app.py                    # FastAPI factory
    schemas.py                # ChatRequest/ChatResponse/SessionResponse pydantic
    routes.py                 # /health, /chat, /session/{id}, /session/reset
  domain/
    categories.py             # ported reference: CANONICAL_CATEGORIES, two-tier synonyms, clarification rules,
                              #   category_prompt_block(), resolve_category_from_text(), CategoryResolution
    trailer_fields.py         # ported reference: TrailerFieldSpec per category (required/optional/questions/guidance)
    brands.py                 # ported make_inventory.py: MakeInventory, load_make_inventory(), known_makes(),
                              #   make_prompt_block(), categories_for_make(), make_filter_values()
    make_aliases.py           # ported verbatim: MAKE_ALIASES alias -> canonical-make map
    normalizer.py             # ported verbatim: normalize_category/make/hitch/subcategory/color/condition,
                              #   clean_dealer_notes(), build_embedding_text()  (listing-data cleanup layer)
    units.py                  # ported verbatim: parse_length_ft(), parse_weight_lbs()  (listing-catalog parsers)
    defaults.py               # CATEGORY_DEFAULTS: dict[category -> {slot: value}] (plug-and-play, ships empty/example)
    slot_map.py               # _SLOT_METADATA_FILTER_MAP + Roll Off bin-size special case (safety-net parses only)
    canned_responses.py       # all FAQ + non-FAQ canned strings from spec §Tools, keyed by scenario id
  llm/
    client.py                 # thin OpenAI wrapper; injectable for tests (protocol + FakeLLM)
    schemas.py                # TurnAnalysis, ReplyOutput pydantic models (strict structured outputs — see M3)
    analyze.py                # LLM call #1: prompt builder + call
    respond.py                # LLM call #2: prompt builder + call
  graph/
    state.py                  # SessionState schema + (de)serialization to/from JSONB snapshot
    build.py                  # build_graph(): nodes + edges, compiled once at startup
    nodes/
      analyze.py              # runs llm.analyze -> writes TurnAnalysis into state
      apply_analysis.py       # deterministic: merge extraction into slots, contact, skips, locks, defaults
      qualification.py        # deterministic: pick next unanswered question / detect flow complete
      search.py               # Pinecone gate + call + dedupe + store shown listings
      inventory_lookup.py     # deterministic Excel lookup: gate + lookup_inventory + cards + shown listings
      email_actions.py        # contact gate + send escalation/faq/non-faq/log-interest email + lead upgrade
      respond.py              # runs llm.respond -> assistant_text (+ cited listings)
  search/
    pinecone_search.py        # ported reference module (public: search_pinecone_listings, search_pinecone_listing_result)
    inventory_matcher.py      # ported reference matcher, LLMs + regex fallback stripped (public: lookup_inventory)
    ingest.py                 # ported reference ingestion script (CLI: python -m src.search.ingest)
  tools/
    email_sender.py           # EMAIL_BACKEND selects smtp (default) | graph (requests-based); send(subject, body) -> bool
scripts/
  convo_runner.py             # live scripted conversation harness (M9, first used M5)
  scenarios/                  # YAML scenario files
tests/
  conftest.py                 # FakeLLM, FakePinecone, FakeEmailSender, db fixture, TestClient
  unit/ ...
  integration/ ...
```

---

## Milestone 0 — Scaffold, Config, Frontend Shims

**Goal:** repo skeleton; `streamlit run app.py` boots with zero import errors; `GET /health` returns `{"status":"ok"}`.

### Implementation
1. Dependencies: use the existing **`requirements.txt`** as-is (it already pins fastapi, uvicorn, langgraph, langchain-core, langchain-openai, langsmith, openai, pinecone, pydantic v2, sqlalchemy 2, alembic, psycopg + psycopg-binary, pandas, openpyxl, python-dotenv, requests, pytest, httpx, pyyaml, streamlit). Do **not** add `msal` or `pytest-mock` (see Locked Decisions).
2. `src/config.py`: one `Settings` object reading every `.env` key used anywhere. Full list (defaults in parentheses where `.env` or reference code defines one): `OPENAI_API_KEY`, `OPENAI_MODEL` (`gpt-4o-mini`), `OPENAI_EMBEDDING_MODEL` (`text-embedding-3-small`), `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (`trailerplace-listings`), `SEARCH_TOP_K` (50), `SEARCH_MAX_RECOMMENDATIONS` (5; `.env` sets 6), `RERANK_ENABLED` (on), `RERANK_WARN_RATIO`, `RERANK_EXTREME_RATIO`, `RERANK_LENGTH_WEIGHT`, `RERANK_MISSING_DIM_PENALTY`, `RERANK_VERBOSE_LOGS`, `MAKE_RERANK_VERBOSE_LOGS`, `SHOW_ONLY_LLM_MENTIONED_CARDS` (true), `TRAILERPLACE_WEBSITE`, DB `HOST`/`PGUSER`/`PASSWORD`/`DATABASE`/`PORT`, `TRAILERPLACE_PERSIST_CHATS`, `EMAIL_BACKEND` (`smtp`; `.env` sets `graph`), `TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET`, `SENDER_EMAIL`, `RECIPIENT_EMAIL` (Graph recipient), `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `EMAIL_TO` (SMTP recipient — distinct from `RECIPIENT_EMAIL`), `CHATBOT_API_PORT` (8000), `LANGSMITH_TRACING`/`LANGSMITH_ENDPOINT`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT`, `DEBUG_STATE_ENDPOINT` (off; M5), `DB_AUTO_CREATE` (off; M8), `INVENTORY_LOOKUP_LIMIT` (5; M6). No other module reads `os.getenv` directly.
3. `src/log_setup.py`: `configure_trailerplace_logging()` — stdlib logging, level from env, idempotent.
4. Frontend shims (exact signatures app.py calls):
   - `src/thinking_agent.py`: `thinking_agent_enabled()->False`, `thinking_agent_background()->False`, `generate_thinking_flow(payload)->{"status":"ok","thinking_markdown":""}`, `log_thinking_flow(sid,payload,result)->None`.
   - `src/conversation_store.py`: `persistence_enabled()->bool` (from `TRAILERPLACE_PERSIST_CHATS`), `enqueue_save_user_feedback(session_id, turn_idx, text, iso)` — no-op stub for now (real impl M2/M8).
   - `src/shown_listings_store.py`: in-process `dict[str,set[str]]` keyed by session id — `add_shown_urls(sid, urls)`, `add_shown_keys_and_urls(sid, keys, urls)`, `accumulate_shown_urls_from_chat_messages(messages)->list[str]` (walks messages, collects each listing's URL). Read the URL defensively: in-session the entries are `TrailerListing` objects, but be robust to plain dicts and `listings=None` — `getattr(l, "url", None)` first, then `l.get("url")` if it's a dict.
   - `src/models.py`: `TrailerListing` with the exact fields app.py constructs: `listing_id, title, condition, price, price_display, payments_from, category_subcategory, make, color, hitch_type, year, length, width, axles, gvwr, payload_capacity, trailer_material, floor, url, score` (all optional except title/url). **Types must be lenient** (`str | float | None` style, or a plain dataclass with no validation): app.py wraps `TrailerListing(...)` in a bare `try/except` and **silently drops** any listing whose construction raises — and `price` legitimately arrives as a float, a `"$8,400"`-style string, the literal string `"Call for price"`, or `None`; `year/length/width/axles/gvwr/payload_capacity` arrive as strings or numbers. A strict pydantic model here makes cards vanish with no error.
5. `src/api/app.py` + `routes.py`: `GET /health -> {"status":"ok"}`. Stub `POST /chat`, `GET /session/{id}`, `POST /session/reset` returning fixed shapes (filled in M4/M8). `main.py` runs uvicorn on `CHATBOT_API_PORT`.

### Tests (`tests/unit/test_scaffold.py`)
- `test_health` — TestClient GET /health == 200, `{"status":"ok"}`.
- `test_app_py_imports` — assert the five `src.*` shim modules import and expose the exact names app.py uses, via `getattr` checks (importing app.py itself triggers Streamlit side effects).
- `test_settings_loads` — Settings parses a synthetic env without error; missing optional keys get defaults.
- `test_trailer_listing_lenient` — `TrailerListing(title="x", url="y", price="Call for price")` and `price=8400.0` both construct without raising.
- `test_shown_listings_store` — add/accumulate round-trip; `accumulate_shown_urls_from_chat_messages` handles messages with `listings=None` and with dict listings.

**Done when:** pytest green; `python main.py` serves /health; `streamlit run app.py` reaches the login screen and, after login, shows "Chatbot initializing…" then the empty chat (backend /health ok).

---

## Milestone 1 — Domain Data Layer

**Goal:** all dealership data (categories, questions, brands, aliases, normalizers, unit parsers, defaults, slot mapping, canned responses) available as importable, tested modules. Spec: §Data, §Field Extraction, §Pinecone module.

### Implementation
1. Port `categories.py` → `src/domain/categories.py` unchanged in logic. **Preserve the two-tier synonym split** (`_NAMING_TERMS` vs `_CARGO_TERMS`) and `CategoryResolution.match_tier` — this salience ranking is what makes "tilt trailer to haul a tractor" resolve to Tilt instead of a tractor-mapped category. Public surface to keep: `CANONICAL_CATEGORIES`, `category_prompt_block()`, `resolve_category_from_text()`, `resolve_categories_from_text()`, `resolve_category_clarification_answer()`, `category_clarification_question()`, `advertised_categories_line()` (used in the respond prompt's "what we carry" line), and the Gooseneck/Bumper-Pull "never a category" guard inside `category_prompt_block()`.
2. Port `trailer_fields.py` → `src/domain/trailer_fields.py` unchanged. Keep the real public names: `get_trailer_fields(trailer_type) -> TrailerFieldSpec` and `get_trailer_fields_as_dict(trailer_type)` — the dict form **merges each slot's `answer_guidance` with `_loose_answer_guidance()`** (range→smallest, no-preference→skip rules per slot type); the analyze prompt consumes that merged text, so keep the merge intact. Also keep: `_DEFAULT_SPEC` fallback for unknown categories, per-spec `notes` (e.g. Livestock: "Do NOT ask about animal type or count"), `list_all_categories()`, and two documented caveats — the `"Welding"` spec is non-canonical/unreachable (kept for the ingest normalizer only), and `item_or_trailer_width_ft` is a **runtime-injected** width slot that belongs to no category spec.
3. Port `make_inventory.py` → `src/domain/brands.py`, logic unchanged: frozen `MakeInventory` dataclass (`canonical_makes`, `categories_by_make`, `filter_values_by_make`); `load_make_inventory()` (`lru_cache`; reads `listings_final_v5.xlsx` `make`+`category` columns; `_CATEGORY_ALIASES` folds Atv Trailer→Utility, Concession→Enclosed, Landscape→Utility, Tank→Diesel Tank; skips rows whose category normalizes to Unknown); `known_makes()`; `make_prompt_block()` (each make with its categories; excludes Gooseneck/Bumper Pull and includes the "never makes/brands" guard sentence); `categories_for_make()` (powers the brand-only flow); `make_filter_values()` (canonical make → all raw workbook variants — feeds the Pinecone `make` filter in M6).
4. Port verbatim into `src/domain/`: `make_aliases.py` (the `MAKE_ALIASES` alias→canonical map — consumed by `normalizer`, `brands`, and pinecone rerank display; it is **not** used to regex-match user text — the Analyze LLM canonicalizes user brand mentions itself); `normalizer.py` (`normalize_category`, `normalize_subcategory`, `normalize_make`, `normalize_color`, `normalize_hitch`, `normalize_condition`, `build_category_subcategory`, `clean_dealer_notes`, `build_embedding_text` — the listing-data cleanup layer for ingest/search); `units.py` (`parse_length_ft` — ft/in/yd/m/cm/mm including `7'6"` quote notation; `parse_weight_lbs` — lbs/kg/tons/k-suffix). `units`/`normalizer` parse **listing catalog strings** so query filters and rerank read the index exactly the way ingest wrote it; they are not user-input extractors (Locked Decision: the LLM converts user input).
5. `src/domain/defaults.py`: `CATEGORY_DEFAULTS: dict[str, dict[str, Any]] = {}` plus a documented example entry (commented) — e.g. `"Utility": {"item_or_trailer_width_ft": 6.92, "hitch_type": "Bumper Pull"}`. Accessor `defaults_for(category)`.
6. `src/domain/slot_map.py`: `_SLOT_METADATA_FILTER_MAP` exactly as in spec §Pinecone module. `normalize_slot_value(category, key, value)`: for Roll Off + a length target, route through `_normalize_roll_off_bin_size_as_length` (bin "15 yd" → length 15 ft — the numeric value maps directly, per the `bin_size` answer_guidance in trailer_fields; range → smallest); all other length/width/height targets → `parse_length_ft`; payload targets → `parse_weight_lbs`. These are **safety-net parses of values the Analyze LLM already normalized** — numeric inputs pass through unchanged; if a stray string slips through, the parser rescues it. Do not add user-text keyword extraction here.
7. `src/domain/canned_responses.py`: dict of every canned string from spec §Tools (FAQ: `contact_human, financing, trade_in, service_parts, store_info`; non-FAQ: `generic_team_request, escalation, listing_interest_selected, listing_interest_unselected, listing_interest_fallback`). Exact text, no paraphrasing.

### Tests (`tests/unit/test_domain_*.py`)
- Categories: "I need a dump trailer" resolves Dump; **"tilt trailer to haul a tractor" resolves Tilt with `match_tier="naming"`** (naming tier beats cargo tier); an office-trailer trigger term yields the clarification question; "gooseneck" never resolves to a category; every canonical category has a field spec (Welding is the known non-canonical extra — assert the exact expected set difference).
- Slot map: every key in `_SLOT_METADATA_FILTER_MAP` exists in some category's required/optional slots **except `item_or_trailer_width_ft`** (the documented runtime-injected slot — assert it is exactly the one exemption); `normalize_slot_value("Roll Off", …, "15 yd")` → 15.0 length; numeric passthrough (`normalize_slot_value(..., 14.0)` == 14.0).
- Units (ported parsers, listing-string behavior): `parse_length_ft("83 inches")≈6.92`, `parse_length_ft("7'6\"")==7.5`, `parse_length_ft("20 ft")==20`, `parse_weight_lbs("2 tons")==4000`, `parse_weight_lbs("5k")==5000`, `parse_weight_lbs("7,000 lbs")==7000`. (`AxB`/"16 by 8" splitting is an **LLM extraction rule** — tested via M3 prompt content and M5 live scenarios, not here.)
- Normalizer: `normalize_hitch("goose neck")=="Gooseneck"`, `normalize_category("enclose")=="Enclosed"`, `normalize_make("dimond c")=="Diamond C"`, `clean_dealer_notes` strips phone numbers/URLs/dealer boilerplate.
- Brands: with a small fixture Excel, `make_prompt_block()` lists makes with their categories and excludes Gooseneck/Bumper Pull; `make_filter_values("Diamond C")` returns the raw workbook variants; `categories_for_make` round-trips.
- Defaults: `defaults_for("Nonexistent") == {}`; example entry (temporarily enabled in the test via monkeypatch) round-trips.
- Canned responses: keys exist; strings contain `979-532-1486` where the spec says so.

**Done when:** pytest green; `python -c "from src.domain.categories import category_prompt_block; print(category_prompt_block())"` and the same for `make_prompt_block` print their blocks.

---

## Milestone 2 — Database Layer (durable turns, receipts, outbox)

**Goal:** the persistence design of the reference `conversation_store.py` + `db_models.py` (both in this folder) live in Postgres via Alembic: in-memory `_sessions` working store, atomic durable turns under an advisory lock, turn receipts for idempotency, and a transactional email outbox.

### Implementation
1. `src/db_models.py` — from the reference `db_models.py`, define **only the models this system actually uses** (do not blanket-copy; as of now that is all four, each with a purpose — drop any model that loses its consumer):
   - `ChatbotLead` — lead row (`psid` = session id is the lookup key, `lead_type` soft|hard check-constrained, `contact_status`, `item_of_interest` NOT NULL).
   - `ChatbotConversation` — `conversation` JSONB (simplified turn list, NOT NULL) + `state_snapshot` JSONB (full session dict) + `state_schema_version` + `state_version` + `closed_at` + timestamps.
   - `ChatbotTurn` — idempotency receipts: composite PK (session_id, turn_id), `request_message`, `response` JSONB (the exact /chat body to replay on retries).
   - `ChatbotOutbox` — transactional email outbox: unique (session_id, turn_id, event_key), `event_type`, `payload` JSONB, `status` pending|processing|sent|failed, `attempt_count`, `last_error`.
   Field definitions exactly as in the reference file — copy the needed classes, not the file. `session_id` columns are UUID (frontend sends uuid4 strings — parse them).
2. `src/db.py`: `database_enabled()` (True only when HOST/PGUSER/PASSWORD/DATABASE/PORT are all set), engine URL `postgresql+psycopg://PGUSER:PASSWORD@HOST:PORT/DATABASE` (URL-quote the password — it contains `?/+{`), `sslmode=require` for Azure, `get_session_factory()`, `ensure_schema()` (create_all — called best-effort by `ensure_persistence_schema()` at runtime and by test fixtures). **Real migrations via Alembic**: `alembic init`, point `env.py` at `Base.metadata`, one autogenerate revision, apply with `alembic upgrade head`.
3. `src/conversation_store.py` — port the reference file (this folder) **near-verbatim**, replacing the M0 stub (app.py imports this module directly). Keep these functions with their exact reference behavior:
   - `persistence_enabled()` = `TRAILERPLACE_PERSIST_CHATS` not disabled **AND** `db.database_enabled()`.
   - Leads (always looked up by `psid == session_id`): `create_or_get_soft_lead(...)` (first turn: `lead_type='soft'`, `item_of_interest="Trailer inquiry"`), `update_lead_contact(...)`, `update_lead_item_of_interest(...)`, `promote_lead_to_hard(...)`. `contact_status` column values per the reference: `"contact_available"` (phone or email present) | `"missing_contact"`. **The email gate's completeness check (Name + Email|Phone) is computed in code from the actual lead/state fields — never from this column.**
   - `durable_turn(session_id, turn_id, request_message)` context manager: `pg_advisory_xact_lock(hashtext(session_id))` (serializes concurrent turns for one session — works even before the row exists), load conversation row + `ChatbotTurn` receipt, **same turn_id with a different message → `ValueError`**, yield `(session, row, receipt)`, commit on success / rollback on error.
   - Dual conversation format: `conversation` JSONB = simplified turn list from `_messages_to_conversation`: `[{user, chatbot, feedback?, trailer_category?, metadata_filters_collected?}]` — one entry per user/assistant pair, so **`turn_idx` from the frontend indexes it directly**; `state_snapshot` = the full session dict including `messages` in the frontend shape (`{role, content, listings, user_feedback, ts}` — listing dicts persisted). `_merge_existing_feedback` preserves saved feedback whenever `conversation` is rebuilt.
   - `restore_session(sid)` → `{exists, closed, state_version, messages, sales_phase, customer_full_name, customer_email, customer_phone, contact_status}`; prefers `state_snapshot["messages"]`, falls back to rebuilding messages from the simplified `conversation` list.
   - `close_session(sid)`: set `closed_at=now()`; never delete rows.
   - `deliver_pending_outbox(limit=10)`: claim pending/failed rows with `FOR UPDATE SKIP LOCKED` → mark processing + `attempt_count += 1` → commit → invoke the handler for each `event_type` → mark sent, or failed with `last_error` (row stays retryable). Handler map (wired in M7): `interested_listing`, `non_sales_faq`, `escalation_alert`, `team_request`, `results_shown`, `unanswered_question` → `email_sender` functions. At-least-once delivery is accepted.
   - `enqueue_upsert_conversation`, `enqueue_save_user_feedback(sid, turn_idx, text, iso)` — background `ThreadPoolExecutor`; feedback sets `conversation[turn_idx]["feedback"]` (**turn-level index — matches app.py's `turn_idx = i // 2` directly; there is no `turn_idx*2+1` message math**). Empty text clears feedback.
4. Persistence-off mode: the in-memory `_sessions` dict (M4) is the only store; every `conversation_store` function no-ops via its `persistence_enabled()` guard, exactly as in the reference.

### Tests (`tests/integration/test_db.py`, marked `@pytest.mark.db`, skipped unless `TEST_DATABASE_URL` set; run against local Postgres, e.g. `docker run -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16`)
- Alembic `upgrade head` runs clean on an empty DB and is idempotent; all four tables exist.
- `durable_turn`: two concurrent turns for the same session serialize (thread test — second waits on the advisory lock); receipt replay: same turn_id + same message → receipt row available to short-circuit; same turn_id + different message → `ValueError`.
- Leads: psid lookup; soft→hard via `promote_lead_to_hard`; `contact_status` flips `missing_contact` ↔ `contact_available` on phone/email changes; item_of_interest updates.
- Conversation: `_messages_to_conversation` pairs user/assistant correctly (odd trailing user message too); `_merge_existing_feedback` keeps feedback across rebuilds; `state_snapshot` round-trips nested dicts/lists/None; `state_version` increments per durable turn.
- `enqueue_save_user_feedback` lands on `conversation[turn_idx]` for turn 0 and turn 3; `restore_session` falls back to rebuilding from `conversation` when the snapshot has no messages; `close_session` sets closed_at; unknown session → `{exists: False}`.
- Outbox: queued event delivered through a fake handler; handler raise → status failed + retried on next drain; unique (session_id, turn_id, event_key) rejects duplicate enqueues.
- Unit (no DB): persist-off mode — all functions no-op, bot runs purely in-memory.

**Done when:** pytest (incl. `-m db` against local Postgres) green.

---

## Milestone 3 — LLM Layer (Structured Outputs)

**Goal:** the two LLM calls fully specified — prompts, pydantic output schemas, injectable client. No graph yet.

### Implementation
1. `src/llm/client.py`: `class LLMClient(Protocol): def structured(self, *, system: str, messages: list[dict], schema: type[BaseModel]) -> BaseModel`. Real impl uses OpenAI structured outputs (`chat.completions` with `response_format` from the pydantic model, or `responses.parse`) with `OPENAI_MODEL`. `FakeLLM` (in tests/conftest.py) returns queued schema instances.
2. `src/llm/schemas.py`. **Hard constraints of OpenAI strict structured outputs — violating any of these fails at schema registration, not at runtime:** (a) no open-ended dicts — `dict[str, str]` compiles to `additionalProperties` and is rejected, use a list of pair objects instead; (b) every field must be required — express optionality as `| None`, never with defaults the schema relies on; (c) `Literal`/enums and nested models are fine. Give **every field a `Field(description=...)`** — the model reads them.
   ```python
   class ContactInfo(BaseModel):
       name: str | None
       email: str | None
       phone: str | None

   class SlotAnswer(BaseModel):
       slot_name: str    # one of the current category's slot names, exactly as listed in the prompt
       raw_answer: str   # what the user said for that slot, as given

   class EmailTrigger(BaseModel):
       kind: Literal["faq", "escalation", "team_request", "listing_interest"]
       faq_key: Literal["contact_human","financing","trade_in","service_parts","store_info"] | None
                                         # required when kind == "faq", else None
       listing_reference: int | None     # 1-based shown-listing index, when kind == "listing_interest"
       description: str                  # one-line summary of the request, for the email body

   class ExtractedFields(BaseModel):
       trailer_length_ft: float | None   # ALREADY converted to feet by the model
       trailer_width_ft: float | None
       trailer_height_ft: float | None
       payload_lbs: float | None         # ALREADY converted to lbs by the model
       hitch_type: list[Literal["Bumper Pull", "Gooseneck"]] | None
                                         # [one] = clear preference; [both] = either acceptable;
                                         # None = not mentioned; no-preference -> numeric_no_preference
       haul_item: str | None
       brand_preference: str | None      # canonical make from the known-makes block; verbatim if unknown
       non_metadata_features: list[str]  # ramp, winch, color…
       numeric_no_preference: list[str]  # slot names the user said "no preference" for

   class HaulClassification(BaseModel):
       is_lightweight_utility_load: bool  # Utility + cargo ≤1500 lbs (golf carts, ATVs, kayaks,
                                          #   lawn mowers, camping gear) → skip width question
       needs_width_question: bool         # heavy/wide/vehicle (tractors, skid steers, excavators)
                                          #   in non-excluded categories → ask item_or_trailer_width_ft
       haul_item_matched: str | None      # the specific cargo the user said (grounded in their words);
                                          #   null if no explicit cargo mentioned ("I want an equipment trailer")

   class InventoryLookup(BaseModel):
       is_lookup: bool                    # the message references specific inventory by identifier(s)
       year: int | None                   # model year mentioned (e.g. 2026) — NOT a stock number
       make: str | None                   # canonical make from the known-makes block (typos corrected:
                                          #   "iron bul" -> "Iron Bull Trailers")
       model_text: str | None             # model code/phrase AS THE USER TYPED IT ("FMAX210", "fhg 24k",
                                          #   "7210 bt") — the fuzzy matcher handles partial/typo matching
       stock_number: str | None           # ONLY when explicitly framed as stock/unit/#/id — NEVER a
                                          #   weight, length, price, year, or phone number
       wants: Literal["price","availability","details","general"] | None
       confidence: Literal["low","medium","high"]  # low never triggers the lookup (fail closed)

   class TurnAnalysis(BaseModel):
       intent: Literal[
         "general_question","category_exploration","category_selection",
         "feature_request_no_category","recommendation_request",
         "qualification_answer","skip_current","skip_all_show_results",
         "requirement_change","drop_requirements","category_change",
         "listing_interest","faq","team_request_escalation",
         "inventory_lookup",
         "contact_info_provided","contact_declined","smalltalk_other"]
                                         # the DOMINANT intent — drives routing only
       email_triggers: list[EmailTrigger]  # EVERY email-worthy request in this message, not just
                                           # the dominant one; empty list when none
       haul_classification: HaulClassification  # cargo weight/size decision — determines
                                           # whether to ask for trailer width
       inventory_lookup: InventoryLookup   # direct-inventory identifiers (make+year / make+model / stock)
       category_mentioned: str | None    # canonical name or None
       is_category_info_only: bool       # "what is a utility trailer?" vs "I want one"
       extracted: ExtractedFields
       slot_answers: list[SlotAnswer]    # NOT a dict — strict mode rejects open dicts
       contact: ContactInfo
       listing_reference: int | None     # 1-based index into shown listings ("the second one")
       dropped_fields: list[str]         # for drop_requirements
       keep_fields_answer: Literal["all","none","some"] | None
       kept_fields: list[str]
       answered_current_question: bool
       user_question_to_answer: str | None  # the interruption question, verbatim
   ```
   **ReplyOutput** (call #2): `assistant_text: str`, `cited_listing_urls: list[str]` (for `SHOW_ONLY_LLM_MENTIONED_CARDS`).
3. `src/llm/analyze.py`: `build_analyze_prompt(state) -> (system, messages)`. Messages: last N turns (N=10, configurable) + current user message. System prompt template — build exactly this structure (rule texts copied near-verbatim from `prompt_structured.md` where cited):
   ```
   You are the turn-analysis module for the TrailerPlace trailer-dealership chatbot (Wharton, TX).
   Analyze the LATEST USER MESSAGE in the context of the conversation and fill the TurnAnalysis
   schema. You never write customer-facing text; you only classify and extract.

   === TRAILER CATEGORIES ===
   {category_prompt_block()}

   === KNOWN MAKES/BRANDS ===
   {make_prompt_block()}

   === CURRENT STATE ===
   Selected category: {category or "none"}
   Active clarification question: {clarification question or "none"}
   Qualification questions for this category, in order: {slot: question, ...}
   Answer guidance per slot: {get_trailer_fields_as_dict(category)["answer_guidance"]}
   Category notes: {spec.notes}
   Already collected (NEVER re-extract unless the user changes them): {slot: value (source), ...}
   Skipped slots: {skipped_slots}    No-preference slots (stored null): {...}
   Pending question: "{question text}" (slot={slot}, already re-asked {n} time(s))
   Pending category change awaiting keep/drop answer: {new_category + transferable | "none"}
   Contact: name={...} email={...} phone={...} declined={bool}
   Listings shown so far (1-based): 1. {title} / 2. {title} / ...

   === INTENT RULES ===
   Interpret the message by intent; do NOT assume it answers the pending question.
   [one line per intent literal, with criteria — copy the descriptions from spec §Step 1
    and §Global Behavior; include explicitly:]
   - "what is a utility trailer?" is an information request, not a selection:
     category_mentioned="Utility", is_category_info_only=true.
   - category_change ONLY on explicit switch requests ("show me dump trailers instead",
     "switch to…", "I don't want tilt anymore").   [spec §Mapping Priority]
   - HAUL-ITEM LOCK: a category is already selected ⇒ cargo/haul-item words must NOT set
     category_mentioned; keep it null unless the user explicitly names a different trailer
     type with switch intent.   [spec §Mapping Priority]
   - Brands are never categories; a brand alone ⇒ brand_preference only, category null.
     Gooseneck / bumper pull are hitch types — never categories, never brands.   [spec §Brand Handling]
   - skip_current: "skip", "next", "I don't know", "I'd rather not answer".
   - skip_all_show_results: "just show me what you have", "no more questions",
     "give me recommendations".
   - faq: the message asks one of contact_human / financing / trade_in / service_parts / store_info.
   - team_request_escalation: call/meeting scheduling, quote requests, "email me",
     anything needing a human.
   - listing_interest: references a shown listing ("the second one", "that Iron Bull")
     → set listing_reference to its 1-based index.
   - email_triggers: list EVERY email-worthy request in the message — faq (with its faq_key),
     escalation, team_request, listing_interest (with its listing reference). One message may
     contain SEVERAL; intent carries only the dominant one, email_triggers carries them ALL.
     If intent is faq/team_request_escalation/listing_interest, that request must also appear
     as an entry in email_triggers.
   - If mid-qualification and the message is an interruption: answered_current_question=false
     and put the interruption verbatim in user_question_to_answer.

   === EXTRACTION RULES (apply to EVERY message, even unasked fields) ===
   [copy near-verbatim from spec §Field Extraction + §Extraction principles:]
   - AxB = width x length. AxBxC = width x length x height. "16 by 8" = 16 ft length, 8 ft width.
   - Convert ALL lengths/widths/heights to feet and ALL weights to lbs YOURSELF:
     "83 inches" -> 6.92, 7'6" -> 7.5, "2 tons" -> 4000, "5k lbs" -> 5000.
   - Numeric range -> the smallest value ("15-18 ft" -> 15).
   - Loose numeric no-preference ("no preference", "flexible", "not sure") -> null value
     + add the slot name to numeric_no_preference.
   - Hitch: only Bumper Pull / Gooseneck; "either" -> both in the list; no-preference ->
     add "hitch_type" to numeric_no_preference.
   - haul_item: store as the user said it; never over-normalize or discard vague descriptions.
   - Brand: map typos/variants to a canonical known make ("dimond c" -> "Diamond C");
     unknown brands verbatim.
   - Any non-searchable preference (ramp, winch, LED lights, color, …) -> non_metadata_features.
   - slot_answers: one {slot_name, raw_answer} pair for each current-category slot this
     message answers.

   === HAUL CLASSIFICATION (determine cargo weight/size to guide width-question logic) ===
   When the user mentions what they plan to haul:
   - is_lightweight_utility_load = true ONLY for Utility category + cargo ≤ 1500 lbs:
     golf carts, ATVs, UTVs, dirt bikes, motorcycles, lawn mowers, zero-turn mowers,
     gardening/landscaping tools, small generators, canoes, kayaks, bicycles, e-bikes,
     small furniture, camping gear, hobby equipment.
   - needs_width_question = true for heavy/wide/vehicle cargo in categories NOT
     {Roll Off, Enclosed, Fiber, Race Trailer, Diesel Tank} (list may vary per category):
     excavators, mini excavators, bulldozers, backhoes, skid steers, telehandlers,
     forklifts, loaders, tractors, combines, harvesters, rollers, compactors,
     scissor lifts, boom lifts, oversize/wide loads, vehicles being hauled.
   - haul_item_matched = the SPECIFIC CARGO the user said, grounded in their exact words:
     "tractor", "dirt bike", "lawn equipment", "camping gear", "small furniture".
     Do NOT return a trailer category ("equipment trailer"), hitch type ("gooseneck"),
     or a feature ("with a ramp") as matched_item. Return null if the message only names
     a trailer type ("I want an equipment trailer") or is a feature request without cargo.
   - INVARIANT: if is_lightweight_utility_load OR needs_width_question is true,
     haul_item_matched MUST be non-null. If you set a flag without a cargo item, that is
     an error — the code will clear both flags. Only set flags when a specific item is
     explicitly stated in the latest message or recent context.

   === INVENTORY LOOKUP RULES (direct identifier lookups against our stock list) ===
   Set inventory_lookup.is_lookup=true AND intent="inventory_lookup" when the message
   references specific inventory by identifier — on ANY turn, including the very first message:
   - (make + year): "what 2026 Aluma trailers do you have?"
   - (make + model code/phrase, partial or typo'd): "how much is the iron bull fhg 24k",
     "do you have the 7210 BT?"
   - explicit stock number: "stock #12914", "unit 70942", "is #05256 still available?"
   Field rules:
   - stock_number: a 4-6 digit number is a stock number ONLY when framed as one (stock/unit/
     #/id/listing wording) or the message is unmistakably an inventory reference. NEVER treat
     weights ("7000 lbs"), lengths, prices, years, or phone digits as stock numbers.
   - make: correct typos to a canonical known make ("iron bul" -> "Iron Bull Trailers").
   - model_text: keep exactly as the user typed it — code handles partial/typo model matching.
   - confidence: high = identifiers explicit and unambiguous; medium = probable; low = doubtful.
     Low never triggers the lookup. If you are unsure WHICH model they mean, still set
     is_lookup=true (the matcher will surface candidates for clarification); if you are unsure
     whether it is an inventory reference AT ALL, use low.
   NOT lookups: a make alone ("do you carry Diamond C?" -> brand-only flow); category shopping;
   feature requests; requirement/filter updates; references to listings already shown in this
   chat (use listing_reference instead).
   Coexistence rules:
   - A lookup NEVER changes the selected category, brand_preference, or any collected slot —
     do not set category_mentioned or extracted fields from the lookup identifiers themselves
     (a lookup is a side-question about specific stock, not a qualification answer).
   - Mid-qualification, a lookup is an interruption: set answered_current_question=false
     (the pending question is re-asked after the results, per the repeat-once rule).
   ```
   Call `client.structured(..., TurnAnalysis)`.
4. `src/llm/respond.py`: `build_respond_prompt(state, analysis, turn_outcome)` — system prompt template:
   ```
   You are the TrailerPlace sales assistant — a friendly trailer lead specialist for a dealership
   in Wharton, TX (979-532-1486, https://trailerplace.com; financing and delivery available).
   Write the next assistant reply. Warm, concise, conversational; never pushy or repetitive.

   === WHAT THE SYSTEM ALREADY DECIDED THIS TURN (do not contradict) ===
   [include only the lines that apply, from turn_outcome:]
   - Clarification question to ask: "{clarification_question}"
   - Next qualification question to ask: "{next_question}"
     [if pending_question_repeats == 1: "The user did not answer it last time — acknowledge
      their message first, then re-ask casually, once."]
   - Category change pending: ask which of these previously collected requirements to keep
     for {new_category}: {transferable list}.
   - Search ran: {k} matching listings, summarized below. Present the best matches
     conversationally; do not list every spec.
   - Inventory lookup result: {match_status}.
     exact -> present the match(es) warmly, then ask whether they're interested in any of
       the models shown (that interest flows into log-interest next turn).
     no_exact -> say we do not currently show {requested_label} in our inventory, then
       present the closest alternatives below; any price/availability you state applies
       ONLY to a listing shown below — never claim the requested exact trailer is available
       and never invent specs.
     ambiguous -> ask which model they mean, naming the candidates below ("did you mean the
       7210S-BT or the TSB 7K?"); do not state prices yet.
     [if a qualification question is pending: after presenting, re-ask the pending question.]
   - Contact invite suppressed this turn (inventory lookup fired) — do NOT ask for
     name/email/phone in this reply.
   - Canned text(s) — EACH must appear verbatim or near-verbatim in your reply, one per email
     trigger this turn: {canned_responses[key] for each key in turn_outcome.canned_keys}
   - Email status: {sent: [reasons] | deferred — ask once for the missing contact piece(s):
     {name | email or phone} | skipped (user declined)}.
   - First-turn contact invite: greet, then politely invite (once, zero pressure) their name
     and email or phone; make clear it's optional.
   - Interruption to answer first: "{user_question_to_answer}"

   === RULES ===
   - Answer the user's question FIRST, then re-ask the pending qualification question once.
   - Never re-ask anything already collected, skipped, or marked no-preference.
   - When presenting listings: mention only listings from the block below; for EVERY listing
     you mention, put its exact URL in cited_listing_urls.
   - Never invent inventory, prices, or policies. Store facts: Wharton TX, 979-532-1486,
     financing available, delivery available, {TRAILERPLACE_WEBSITE}.
   - We carry: {advertised_categories_line()}.
   - 2–6 sentences unless presenting listings.

   === LISTINGS AVAILABLE THIS TURN ===
   {i}. {title} — {price} — {length} x {width} — {hitch_type} — {make} — {url}

   === CONTEXT ===
   Category: {category}; collected: {slots}; customer: {name or "unknown"} (use their first
   name naturally when known).
   ```
   Messages: last N turns + current user message. Output `ReplyOutput`.
5. Post-processing in `analyze.py` — **safety net only**: pass each `extracted` numeric through `parse_length_ft`/`parse_weight_lbs` (floats pass through unchanged; a stray string gets rescued) and route Roll Off `bin_size` slot answers through `normalize_slot_value`. Do not re-extract from user text.

### Tests (`tests/unit/test_llm_*.py`, FakeLLM — prompt-construction and plumbing tests, not model-quality tests)
- Analyze prompt contains: category block, make block, current pending question + repeat count, already-collected slots, category notes, shown listing titles when present, contact status line, haul classification rules, and the AxB/unit-conversion rule text.
- TurnAnalysis round-trips from JSON; invalid intent literal rejected; `slot_answers` is a list of pairs; `haul_classification` and `inventory_lookup` are present and valid.
- Analyze prompt contains the inventory-lookup rules incl. the stock-number guard text ("NEVER treat weights") and the category-coexistence rule; Respond prompt embeds the inventory result block for each match_status (exact/no_exact/ambiguous) and the contact-suppression line when the lookup fired on a first turn.
- **Strict-schema compile test**: generate the OpenAI JSON schema for TurnAnalysis and ReplyOutput and assert every object node has `additionalProperties: false` and no free-form dict leaked (this is what OpenAI validates at call time).
- **Haul classification invariant**: test that `apply_analysis` clears both flags if one is set without `haul_item_matched` (code-side guard). Test cases: (a) lightweight=true, matched="golf cart" → kept; (b) lightweight=true, matched=null → both cleared; (c) width=true, matched="tractor" → kept; (d) width=true, matched=null → both cleared.
- Safety-net normalization: a fake TurnAnalysis with `trailer_width_ft=6.92` passes through unchanged; a stray `"83 inches"` string routed through the helper yields ≈6.92.
- Respond prompt embeds **every** canned string in `turn_outcome.canned_keys` (test with two at once); embeds listing summaries when results present; embeds the deferred-contact ask when emails are gated.

**Done when:** pytest green; a manual smoke script (`scripts/smoke_llm.py`, real API) prints a valid TurnAnalysis for "I want a 7x14 dump trailer for hauling dirt, I'm John, 555-1234" (expect: category Dump, width 7, length 14, haul dirt, name John, phone captured).

---

## Milestone 4 — Graph Skeleton + Minimal /chat Round Trip

**Goal:** single LangGraph compiled and wired to `/chat`; state loads/saves through the DB; a basic conversation works end-to-end in Streamlit (no search/emails yet).

### Implementation
1. `src/graph/state.py` — `SessionState` (TypedDict or pydantic), the single source of truth (spec §Global Behavior, §Extraction principles). Fields:
   ```python
   session_id: str; lead_id: str | None
   customer_name: str|None; customer_email: str|None; customer_phone: str|None
   contact_prompted_initial: bool          # asked once at start
   contact_followup_pending: str|None      # "name" | "contact_method" | None
   contact_declined: bool
   pending_email_actions: list[dict]       # stashed EmailTriggers awaiting the contact gate
                                           #   (all-or-nothing batch: send all / drop all)
   category: str|None
   clarification_key: str|None             # active category-clarification question
   slots: dict[str, Any]                   # answered (null = no preference)
   slot_sources: dict[str, str]            # "user" | "default"
   skipped_slots: list[str]
   non_metadata_features: list[str]
   brand_preference: str|None
   pending_question_slot: str|None
   pending_question_repeats: int           # 0 or 1 (re-ask once rule)
   qualification_complete: bool
   pending_category_change: dict|None      # {new_category, transferable: {...}} awaiting keep/drop answer
   shown_listings: list[dict]              # ordered listing dicts incl. url/title (for "the second one")
   shown_urls: list[str]
   last_search_filters: dict|None
   turn: TurnAnalysis|None                 # transient, current turn
   turn_outcome: dict                      # transient: next_question, results, canned_keys (list),
                                           #   emails_sent (list of Reason strings),
                                           #   system_email_triggers (code-generated alert dicts:
                                           #   {kind: "unanswered_question"|"results_shown", description})…
   messages: list[dict]                    # full transcript (persisted shape from M2)
   ```
   Plus `to_snapshot(state) -> dict` / `from_snapshot(dict)` (drop transients; `state_schema_version=1` — also written to the DB column). The live state lives in the process-local **`_sessions: dict[str, dict]`** working store (reference `service.py` pattern): `_get_session(session_id)` creates-or-returns; the snapshot is what `durable_turn` deep-copies into `chatbot_conversations.state_snapshot`. Turn idempotency comes from `ChatbotTurn` receipts, NOT from state fields.
2. `src/graph/build.py` — nodes/edges:
   ```
   analyze → apply_analysis
     → email_actions   (conditional pass-through: runs whenever turn.email_triggers,
                        turn_outcome.system_email_triggers, or pending_email_actions is
                        non-empty — sends / stashes / drops per M7 — then routing continues;
                        it no longer terminates the turn)
     → route:
       inventory_lookup approved  → inventory_lookup → email_actions² → respond
           (gate: intent=="inventory_lookup" AND turn.inventory_lookup.is_lookup AND
            confidence != "low" — fail closed: low confidence falls through to normal routing)
       needs_clarification        → respond
       category_change_pending    → respond          (ask keep/drop)
       search_ready (qualification_complete or skip_all) → search → email_actions² → respond
       otherwise                  → qualification → respond
   ```
   `email_actions²` = a second pass through the same node: `search`/`inventory_lookup` generate the
   Results-Shown system trigger *after* the first email pass, so the route runs email_actions again
   post-results (the node is idempotent — it consumes triggers it processes).
   Routing reads **only** `state.turn.intent` + deterministic state flags — the LLM made the decision, code dispatches.
3. `nodes/apply_analysis.py` (pure code, heavily commented) — in order:
   - Merge `analysis.contact` into customer fields → `conversation_store.update_lead_contact`. Handle contact-gate follow-ups (`contact_followup_pending`), `contact_declined` (never ask again — spec §Contact Collection).
   - **Haul classification invariant check**: if `is_lightweight_utility_load` or `needs_width_question` is set but `haul_item_matched` is null/empty, clear both flags (code-side enforcement; the LLM must not set a flag without grounding cargo).
   - Category set (respect `is_category_info_only`): apply `defaults_for(category)` (source="default"), load field spec, pre-fill slots from `analysis.slot_answers` + `analysis.extracted` (spec §Step 4). **Haul-item lock**: if `state.category` already set, `category_mentioned` from analysis is ignored unless `intent=="category_change"` (spec §Mapping Priority).
   - **Width question injection**: if `needs_width_question=True` and category is NOT in `{Roll Off, Enclosed, Fiber, Race Trailer, Diesel Tank}` and the user's category is not "Utility" (lightweight utility uses no width question), inject `item_or_trailer_width_ft` as a runtime-required slot **after** the currently-pending question. If `is_lightweight_utility_load=True` and category is Utility, do NOT inject (width not needed for lightweight haul).
   - `category_change` intent → build `pending_category_change` with transferable slots; do NOT auto-apply (spec §After Results 4). If `keep_fields_answer` present, resolve it (all/some/none), remap kept values through the new category's slot names, clear pending.
   - Skips: `skip_current` → mark `pending_question_slot` skipped; `skip_all_show_results` → mark all remaining required slots skipped, set `qualification_complete=True`.
   - Repeat-once rule: if `pending_question_slot` set and `answered_current_question` is False and intent was an interruption → increment `pending_question_repeats`; at 2 → mark skipped, move on (next question starts at count 0; if none remain, `qualification_complete=True` so Pinecone fires) **and enqueue an `unanswered_question` system trigger** into `turn_outcome.system_email_triggers` (Locked Decision: System alert emails) (spec §Step 6).
   - `requirement_change`/`drop_requirements`: update/remove named slots only.
   - Extraction bookkeeping: `numeric_no_preference` slots stored as `None` and counted answered; non-metadata features appended (deduped); brand stored (canonical form from the LLM).
   - Update `conversation_store.update_lead_item_of_interest` when category changes.
4. `nodes/qualification.py`: next unanswered required slot (spec order) → `turn_outcome.next_question` = spec question text. **Special handling for injected width question**: `item_or_trailer_width_ft` is not part of any category's field spec; it's injected by `apply_analysis` when `haul_classification.needs_width_question=True`. When picking the next question, check if it was injected; if so, ask it in sequence (as if it were a required slot of the selected category). If none remain → `qualification_complete=True` (search fires next route pass — implement as a conditional edge back through route so search happens in the same turn).
5. `nodes/respond.py`: call `llm.respond`; append assistant message to `state.messages`.
6. Stub `nodes/search.py`, `nodes/inventory_lookup.py`, and `nodes/email_actions.py` (set `turn_outcome` markers only — real impls M6/M6/M7).
7. `/chat` handler (`src/api/routes.py`) — mirrors the reference `handle_chat()` flow exactly:
   - **Persistence off** → `_handle_chat_in_memory()` only: `_get_session(session_id)` from `_sessions` (create if new), run the graph, return the response.
   - **Persistence on** → wrap the whole turn in `conversation_store.durable_turn(session_id, turn_id, message)`:
     1. Advisory lock is held for the session; if a `ChatbotTurn` receipt exists for this `turn_id` → return `receipt.response` verbatim and skip all processing (idempotent frontend retries; same `turn_id` with a different message raises → 409).
     2. If `_sessions` has no entry for this session and `row.state_snapshot` exists → load the snapshot into `_sessions[session_id]` (this is what survives backend restarts).
     3. First turn: `create_or_get_soft_lead(session_id=...)`. Union `already_shown_listing_urls` from the payload into state `shown_urls` (backend state is authoritative; the payload is a supplement).
     4. Run `_handle_chat_in_memory()` (append user message → `graph.invoke` → build the response body).
     5. Deep-copy the final session into `row.state_snapshot`; `row.conversation = _messages_to_conversation(messages)` (with `_merge_existing_feedback`); `state_version += 1`; insert the `ChatbotTurn` receipt carrying the exact response body; enqueue this turn's gate-approved email events into `chatbot_outbox` (M7).
     6. The context manager commits — **then** call `deliver_pending_outbox()` after commit. Each `/chat` request is one atomic durable turn.
   - Response JSON (exact keys app.py reads — every key present on every response): `assistant_text, listings, sales_phase:"main", onboarding_api_messages:[], customer_full_name, customer_email, customer_phone, main_prior_messages: null, thinking_context: null`.
8. `GET /session/{id}` → `conversation_store.restore_session(sid)` (prefers `state_snapshot["messages"]`, falls back to rebuilding from the simplified `conversation`) — with each message's `listings` forced to `null` in the HTTP response (Locked Decision: text-only restore; the snapshot itself keeps the listing dicts). `POST /session/reset` → pop `_sessions[session_id]` + `conversation_store.close_session(sid)` (sets `closed_at`; never deletes the row).
9. First-turn behavior: if `not contact_prompted_initial`, `turn_outcome.contact_ask=True` (respond LLM greets + politely asks name/email/phone once), set the flag. **Inventory-lookup exception (Locked Decision):** when the turn routes to `inventory_lookup` and `contact_prompted_initial` is still False, do NOT set `contact_ask` and do NOT set the flag — the invite then fires on the next user turn instead, alongside handling whatever that message says (standard §Contact Collection rules from there: ask once, accept partials, never re-ask after decline/ignore).

### Tests
- Unit (`tests/unit/test_apply_analysis.py`) — the core of the whole bot; be exhaustive: contact merge + status transitions; decline → never re-asked; defaults applied then overridden by user value; haul-item lock (category stays Tilt when haul item "tractor" maps elsewhere); explicit category_change builds pending change; keep all/some/none resolution; skip current/all; repeat-once → auto-skip on second non-answer; drop_requirements removes only named fields; range/no-preference handling; **width question injection**: Utility + lightweight cargo (needs_width_question=false) → width NOT injected; Equipment + heavy cargo (needs_width_question=true, haul_item_matched="tractor") → width injected after current pending; Roll Off + any cargo → width never injected (excluded category); **invariant enforcement**: needs_width_question=true but haul_item_matched=null → both flags cleared before injection logic runs.
- Unit (`test_qualification.py`): question order follows spec; pre-filled/default/skipped slots never asked; completion flag; **injected width question**: if injected, appears in the next-question sequence as if it were a required slot; can be skipped or answered like any other question.
- Integration (`tests/integration/test_chat_roundtrip.py`, FakeLLM + persistence-off in-memory mode; plus one `@pytest.mark.db` variant exercising `durable_turn` end-to-end): POST /chat twice with scripted TurnAnalysis fixtures → assistant text returned, second turn sees first turn's slots; **receipt idempotency**: replaying the same turn_id returns the identical stored body without invoking the graph again; same turn_id + different message → error; restart simulation (clear `_sessions`, keep DB) → next turn reloads from snapshot; /session restore returns messages with `listings: null`; reset pops `_sessions` and closes; **width question round-trip**: Equipment + tractor haul → width question asked and answered on a second turn.

**Done when:** pytest green; live smoke: run backend + Streamlit, hold a real multi-turn qualification conversation (greeting→contact→category→questions) with search/email still stubbed.

---

## Milestone 5 — Full Qualification & Conversation Behavior (live-tested)

**Goal:** every behavior in spec §Conversation Flow, §Global Behavior, §Field Extraction, §Mapping Priority, §Brand Handling works against the real LLM. This milestone is mostly prompt-tuning + the scripted-conversation harness.

### Implementation
1. `scripts/convo_runner.py`: reads a YAML scenario, POSTs each user turn to `/chat` (fresh uuid session), then asserts. Add a **debug endpoint** `GET /session/{id}/state` (enabled only when env `DEBUG_STATE_ENDPOINT=1`) returning the raw snapshot so scenarios can assert on slots/skips/category. Assertion kinds:
   ```yaml
   name: interruption-repeat-once
   turns:
     - user: "I want a dump trailer"
       expect_state: {category: "Dump"}
     - user: "what's the difference between tandem and single axles?"
       expect_reply_contains_any: ["axle"]           # answered the interruption
       expect_state: {pending_question_repeats: 1}    # and re-asked
     - user: "cool thanks"                            # still not answering
       expect_state: {skipped_slots_contains: "haul_material"}
   ```
   Runner exits nonzero on failure; prints a transcript diff. Support `expect_state`, `expect_reply_contains_any`, `expect_reply_not_contains`, `expect_listings: bool`, `expect_emails_sent: [str]|none` (ordered list of Reason strings; M7).
2. Write scenarios (one file each) covering, at minimum:
   - `contact-full / contact-partial-name / contact-decline / contact-ignore-asks-trailer` (spec §Contact Collection cases 1–5 + declines)
   - `category-direct`, `category-info-vs-select` ("what is a utility trailer?" must NOT start qualification), `category-exploration`, `feature-request-no-category` (features stored, category asked)
   - `prefill-from-first-message` ("7x14 dump for dirt, about 3 tons" → zero redundant questions; state has width 7, length 14, payload 6000)
   - `interruption-repeat-once`, `explicit-skip`, `skip-all-show-results`
   - `haul-item-lock` (tilt + tractor), `clarification-term` (office/fiber rule), `brand-only` (categories_for_make listed, category asked), `brand-plus-category`, `brand-typo` ("dimond c" → state brand_preference "Diamond C"), `gooseneck-not-category`
   - `defaults-applied` (temporarily seed a default via env/monkey config)
   - `loose-answers` (range → smallest; "no preference" → null and question not re-asked; `7'6"` tall → 7.5 in state)
3. Iterate on `analyze.py`/`respond.py` prompts until scenarios pass consistently (run each scenario 2×; both must pass).

### Tests
- All scenarios above green via `python scripts/convo_runner.py scripts/scenarios/ --all` (documented cost warning: real tokens).
- pytest: `test_convo_runner_parsing.py` (YAML parsing, assertion evaluation against canned responses — no network).

**Done when:** scenario suite green twice consecutively; manual Streamlit session feels per spec.

---

## Milestone 6 — Pinecone Search, Results, Refinement

**Goal:** real recommendations; spec §Pinecone tool, §After Pinecone Results.

### Implementation
1. Port `pinecone_search.py` → `src/search/pinecone_search.py` (keep public functions `search_pinecone_listings` / `search_pinecone_listing_result`, fit rerank, category make-preference quotas, env knobs). Port `ingest.py` → `src/search/ingest.py` (runnable module; expects `listings_final_v5.xlsx` at repo root). **Update imports** in both: `src.normalizer` → `src.domain.normalizer`, `src.chatbot.units` → `src.domain.units`, `src.chatbot.make_inventory` → `src.domain.brands`, `src.chatbot.make_aliases` → `src.domain.make_aliases`.
2. `nodes/search.py` (real):
   - **Deterministic gate** (spec): run only when `qualification_complete` (all required slots answered or skipped) — the route already enforces this; assert it here too.
   - Build `metadata_filters`: for each answered slot, map via `slot_map._SLOT_METADATA_FILTER_MAP` + `normalize_slot_value` (Roll Off special case). Category passed separately.
   - **Know what's actually a hard filter vs rerank input** (per the ported `_metadata_filter` / rerank code): hard Pinecone filters are only category, `make`, `hitch_type`, `subcategory` (Aluminum only), and min length (`$gte` on `length_ft_num`); width / payload / height feed the **fit rerank only** — do not invent Pinecone filters for them.
   - **Brand (Locked Decision — hard filter):** `metadata_filters["make"] = state.brand_preference` (canonical form from the LLM); the ported `_metadata_filter` expands it via `make_filter_values()` into `$eq`/`$in` over the raw stored variants. If a brand-filtered search returns **zero** results, re-run once without the make filter and set `turn_outcome.brand_relaxed=True` so respond can say no {brand} matches were found and present the alternatives.
   - Call search with `top_k=SEARCH_TOP_K`, cap `SEARCH_MAX_RECOMMENDATIONS`, pass `already_shown_urls=state.shown_urls` (the actual parameter name; dedupes already-shown).
   - Store results into `state.shown_listings` / `shown_urls`; `turn_outcome.search_results` = listing dicts shaped exactly as app.py's `/chat` listing parser expects: `{title, condition, price, category, make, color, hitch_type, year, length, width, axles, gvwr, payload_capacity, material, floor, url, relevance_score}`.
   - When results are non-empty, enqueue a `results_shown` system trigger (`description = "Pinecone search — {n} results — {category} {key filters}"`) into `turn_outcome.system_email_triggers` — **every** results turn, refinements included (Locked Decision: System alert emails). The route's `email_actions²` pass gates/stashes/sends it.
3. `/chat` response `listings`: if `SHOW_ONLY_LLM_MENTIONED_CARDS`, filter to `ReplyOutput.cited_listing_urls`; else all results.
4. Refinement flows (mostly already in `apply_analysis`; verify against spec §After Results): requirement change → re-run search same turn (route: apply → search when `qualification_complete` and filters changed); drop-requirements; category change keep/drop then new flow; "more options" → re-search excluding shown.
5. Listing reference resolution: `analysis.listing_reference` (1-based into `shown_listings`) — used by M7 log-interest; "tell me more about the second one" answers from stored listing dict without a new search.
6. **Inventory lookup** (Locked Decision; reference `inventory_matcher.py`):
   - Port → `src/search/inventory_matcher.py`, **stripping all three embedded mini-LLMs** (the extractor LLM's job moves to the Analyze `inventory_lookup` block; the reply-intro and feature-framing LLMs' grounding rules move into the Respond prompt — M3 already contains them) and **deleting the regex fallback extraction entirely** (`_fallback_extraction`, `_extract_year`, `_extract_stock`, `_wants_price/availability/details`, `_fallback_requested_features`, `_ordinal_reference`, `answer_from_last_listings`, `should_attempt_chat_lookup`, `extract_trailer_query`, `generate_inventory_response` and all `_inventory_*_llm*` functions). Keyword/regex decisions on user text belong to Analyze (cross-milestone rule 3); ordinal references are already handled by `TurnAnalysis.listing_reference`. **Deleting `_extract_stock`'s "any 4-6 digit number" heuristic is what fixes the "7000 lbs parsed as a stock number" bug.**
   - Keep, logic unchanged: `normalize_text` (mojibake/`&`/x-dimension cleanup — the Excel titles contain `�` junk), `_clean_scalar`, `_stock_text`, `extract_model_code` (skips `_COMMON_MODEL_WORDS`), `prepare_inventory`/`prepared_inventory` (`lru_cache` over `listings_final_v5.xlsx`; builds `*_norm` columns + `model_code` + `search_text`), `_score_candidate` (rapidfuzz weights: 0.30 make / 0.35 model-code / 0.15 model-text / 0.10 title / 0.10 search), the `match_inventory` entity thresholds (stock = exact; year+make needs make_score ≥85; model rows need model_code ≥75 or model_text ≥78 plus overall ≥58; possible-model needs overall ≥70), and `_no_exact_alternative_rows` (same-make-other-year first, then safe same-year rows).
   - Extend `_row_to_listing` to the **full card shape** — the Excel also has `color`, `axles`, `floor`, `trailer_material` columns; include them plus `relevance_score` (= match score) so the dict is byte-for-byte the same shape as Pinecone results (Locked Decision: identical presentation).
   - New single public entry: `lookup_inventory(*, year, make, model_text, stock_number, limit=INVENTORY_LOOKUP_LIMIT) -> {"match_status": "exact"|"no_exact"|"ambiguous"|"none", "matches": [card dicts], "requested_label": str}` — a pure function of identifiers, no user-text parsing. `ambiguous` = multiple distinct models above threshold with no clear winner (top-2 overall within 5 points, or possible-model confidence < 0.9); its `matches` are the clarification candidates. `requested_label` = "{year} {make} {model_text}" of what the user asked for (for the "we do not currently show X" reply).
   - `nodes/inventory_lookup.py` (replaces the M4 stub):
     - Gate: `intent=="inventory_lookup"` + `is_lookup` + confidence `medium|high` (fail closed, mirroring the reference's `validated_direct_inventory_extraction`).
     - Call `lookup_inventory` with the identifiers from `turn.inventory_lookup`; write `turn_outcome.inventory_result`.
     - Append matches to `state.shown_listings`/`shown_urls` (dedupe by URL) — this is what makes "the second one", log-interest, and later Pinecone dedupe work on lookup results.
     - **Never touches** `state.category`, `slots`, `brand_preference`, `skipped_slots`, or `qualification_complete` — a lookup is a side-query; qualification state survives intact (guard this with an assertion in the node).
     - When matches are non-empty, enqueue a `results_shown` system trigger (`description = "Inventory lookup — {n} results — {requested_label}"`) — every lookup turn with results, same as search (Locked Decision: System alert emails).
     - **Mid-qualification**: the pending question stays pending; Analyze already set `answered_current_question=false`, so `apply_analysis`'s repeat-once machinery counts the lookup as a non-answer exactly like any other interruption (count 1 → re-ask after results; count 2 → mark skipped, next question resets to 0, or Pinecone fires if none remain). Respond re-asks the pending question after presenting the lookup results.
     - Results go out through the same `/chat` `listings` array as search results (same `SHOW_ONLY_LLM_MENTIONED_CARDS` filtering).

### Tests
- Unit (`test_search_node.py`, FakePinecone): filter building for each slot type incl. Roll Off; brand → `make` filter expansion via `make_filter_values`; zero-result brand fallback re-runs without make filter; dedupe excludes shown URLs; results shape matches the app.py parser (write a test that literally runs app.py's parsing logic copied into the test against a sample response).
- Unit: gate — search node raises/refuses if qualification incomplete.
- Unit (`test_inventory_matcher.py`, small fixture Excel): stock number → exact match; year+make → all matching rows; make+model partial ("iron bull fhg" → FHG rows); model typo ("7210 bt" → the 7210S-BT row); no-exact year → same-make-other-year alternatives with `match_status="no_exact"`; two close models → `"ambiguous"`; card dict shape passes the copied app.py parser; `normalize_text` cleans `�`/`&`/`8.5'X18'` forms.
- Unit (`test_inventory_lookup_node.py`): gate refuses `confidence="low"` and `is_lookup=False`; matches appended to `shown_listings` + deduped; category/slots/brand untouched after a lookup (assert full state equality except shown listings + outcome); pending question survives and repeat count incremented via the standard interruption path.
- Integration (FakeLLM): first-turn lookup → listings in HTTP response, `contact_ask` absent; second turn → contact invite fires alongside handling the message.
- Live scenarios: `search-happy-path`, `refine-length-change` (12ft after 20ft → new search, other slots kept), `more-options-dedupe` (no repeated URLs), `category-change-keep-some`, `reference-second-listing`, `brand-filter-search` (Diamond C + category → only Diamond C cards), `brand-filter-no-match` (obscure brand → fallback results + "no {brand} matches" wording), `inventory-stock-lookup` ("do you have stock #12914?" → expect_listings true), `inventory-first-message-contact-deferred` (first msg "how much is the 2026 iron bull fhg 24k" → cards, no contact ask; next msg → contact invite + resumes), `inventory-weight-not-stock` ("I need to haul 7000 lbs" → expect_listings false, qualification continues), `inventory-model-typo` ("do you have the 7210 bt" → 7210S-BT card), `inventory-ambiguous-clarify` (vague model → reply asks which model, no prices), `inventory-mid-qna` (mid-Dump-flow lookup → cards shown, pending question re-asked, category still Dump).

**Done when:** pytest green; live scenarios green; Streamlit renders real trailer cards.

---

## Milestone 7 — Email Tools, Contact Gate, Leads

**Goal:** spec §Tools 1–3 + §Contact-Info Gate; hard-lead upgrades.

### Implementation
1. `src/tools/email_sender.py`: `send_email(subject, body) -> bool`. Backend chosen by `EMAIL_BACKEND` (default `smtp`; `.env` currently sets `graph`). **No msal** —
   - `graph` (raw `requests`): POST `https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token` with form fields `grant_type=client_credentials`, `client_id`, `client_secret`, `scope=https://graph.microsoft.com/.default` → `access_token`; then POST `https://graph.microsoft.com/v1.0/users/{SENDER_EMAIL}/sendMail` with `Authorization: Bearer …` and JSON `{"message": {"subject": …, "body": {"contentType": "Text", "content": …}, "toRecipients": [{"emailAddress": {"address": RECIPIENT_EMAIL}}]}}`; HTTP 202 = success.
   - `smtp`: stdlib `smtplib` STARTTLS to `SMTP_HOST:SMTP_PORT`, login `SMTP_USER`/`SMTP_PASSWORD`, From `SMTP_FROM`, To **`EMAIL_TO`** (note: the SMTP recipient var is `EMAIL_TO`; `RECIPIENT_EMAIL` is Graph's).
   - Any failure: log + return False (bot never crashes on email failure).
2. Email body (spec, exact):
   ```
   Full Name: {name or "Not provided"}
   Email: {email or "Not provided"}
   Phone Number: {phone or "Not provided"}

   [{Reason}] {one-line description}
   ```
   Reasons: `Escalation`, `FAQ – {faq_key}`, `Team Request`, `Listing Interest`, `Unanswered Question`, `Results Shown to User`. Listing-interest description includes listing title + URL; Unanswered-Question description = the skipped question text + what the user said instead; Results-Shown description = source (Pinecone search | inventory lookup) + result count + category/identifiers. Subject: `TrailerPlace Lead — {Reason} — {name or session id}`.
3. `nodes/email_actions.py`:
   - Process **every** entry in `turn.email_triggers` **plus** `turn_outcome.system_email_triggers` (code-generated: `unanswered_question`, `results_shown`) **plus** any stashed `state.pending_email_actions` — **one email per trigger, in order** (Locked Decision: multi-trigger emails all fire): `faq` → canned reply for its `faq_key` + email; `escalation` → escalation email + canned `escalation`; `team_request` → team email + canned `generic_team_request`; `listing_interest` → resolve the listing via the trigger's `listing_reference` (selected vs unselected canned variants; fallback variant if resolution ambiguous) + email; `unanswered_question`/`results_shown` → email only.
   - **"Sending" = enqueueing an outbox event** (persistence on): each gate-approved trigger becomes one `chatbot_outbox` row inside the durable transaction — `event_type` per kind (`faq`→`non_sales_faq`, `escalation`→`escalation_alert`, `team_request`→`team_request`, `listing_interest`→`interested_listing`, plus `results_shown`, `unanswered_question`), `event_key = "{kind}:{i}"` (unique per turn — the DB constraint dedupes retried turns), `payload` = subject/body args. `deliver_pending_outbox()` runs after commit and invokes `email_sender` per event; failures stay retryable (at-least-once). Persistence off → call `email_sender.send_email` directly.
   - **Contact gate (ALL email kinds — FAQ and system alerts included) — applies to the batch:** if contact is complete (**Name + Email|Phone, computed from state/lead fields — never from the `contact_status` column**) → enqueue all now. Else stash the whole list in `state.pending_email_actions`; for **customer-initiated** triggers also set `contact_followup_pending` so respond asks **once** for the missing piece(s) — while still answering every FAQ canned text immediately (the user gets their answers; only the emails wait). **System alerts never set `contact_followup_pending`** — they stash silently and wait on contact the normal flow collects (initial invite, gate ask, or volunteered). Next turn (the `pending_email_actions` route from M4): contact arrives → enqueue all stashed; declined/ignored → drop all silently, set `contact_declined`, never re-ask (spec §Contact-Info Gate).
   - On **every** gate-approved enqueue of any kind: `conversation_store.promote_lead_to_hard(session_id)` in the same transaction (Locked Decision — system alerts included, repeats included); listing interest also updates `item_of_interest`.
   - `turn_outcome.canned_keys` (ordered list; **system alerts contribute none**) + `turn_outcome.emails_sent` (list of Reason strings, system alerts included) recorded for respond + tests. The respond prompt's email-status line must **exclude** system alerts — the user is never told about them.
   - **Idempotency for the double pass (`email_actions²`)**: the node consumes triggers it processes (clears `turn.email_triggers` / `system_email_triggers` entries it has sent or stashed) so the second pass only sees the newly generated `results_shown` trigger.
4. FAQ replies: respond prompt instructed to deliver the canned text (verbatim or lightly woven in) and then continue helping.

### Tests
- Unit (`test_email_sender.py`): body format exact-match; graph vs smtp branch selection; graph path with mocked `requests` asserts the token POST fields and the sendMail payload/recipient; smtp path sends to `EMAIL_TO`; failure → False, no raise.
- Unit (`test_email_actions.py`, FakeEmailSender): gate matrix — complete→send; partial→stash+ask; then provided→send; then declined→dropped + no re-ask ever; FAQ canned reply present even when email deferred; listing resolution → selected/unselected/fallback canned variants; hard-lead upgrade on each send kind; no upgrade when gate blocks; **multi-trigger turn** (FAQ financing + escalation in one TurnAnalysis) → two emails fired in order, both keys in `canned_keys`, exactly one hard-lead upgrade; multi-trigger with gate blocked → both stashed, contact next turn → both sent; **system alerts**: `results_shown` with contact complete → sent + lead hard + NOT in `canned_keys`; with contact missing → stashed silently and `contact_followup_pending` NOT set; `unanswered_question` enqueued by the repeat-once skip → same gate behavior; results alert fires on every results turn (two searches → two emails); double-pass idempotency (a trigger processed in pass 1 is not re-sent in pass 2).
- Integration: `/chat` turn triggering escalation with full contact → outbox row created in-transaction, drained post-commit, FakeEmailSender (wired as the outbox handler) captured payload correct; DB lead is hard; a crashed handler leaves the row `failed` and the next drain retries it; duplicate turn replay does not enqueue a second event (unique event_key).
- Live scenarios: `faq-financing-with-contact` (expect_emails_sent: ["FAQ – financing"]), `escalate-quote-no-contact-then-provide`, `escalate-decline-contact` (expect_emails_sent: none, conversation continues), `listing-interest-second-one`, `faq-plus-escalate-one-message` ("do you offer financing? also have someone call me about a quote" → expect_emails_sent: ["FAQ – financing", "Escalation"], reply contains both canned texts), `inventory-lookup-then-interest` (first msg lookup shows cards → "I'm interested in the first one" → contact flow → Listing Interest email, lead hard), `results-shown-alert` (contact given first, then qualification → search → expect_emails_sent includes "Results Shown to User"; refine → second alert), `unanswered-question-alert` (contact given, then two non-answers to the same question → expect_emails_sent includes "Unanswered Question", reply never mentions any email), `results-alert-stashed-then-sent` (anonymous lookup shows cards, no alert sent; user later gives name+phone → stashed "Results Shown to User" sends, lead hard).
- One manual live email test against real Graph API to `RECIPIENT_EMAIL` (checklist item, not automated).

**Done when:** pytest + scenarios green; a real escalation email lands in the inbox.

---

## Milestone 8 — Full API Contract, Persistence Hardening, Frontend E2E

**Goal:** everything app.py does works against the backend, cold restarts included.

### Implementation
1. Verify/finish the exact `/chat` response contract from M4 against app.py's parser (walk `_process_assistant_reply` line by line: every `data.get(...)` key must exist or be safely absent; `customer_email` is checked with `"customer_email" in data` — always include the key).
2. Session restore is **text-only** (Locked Decision): `GET /session/{id}` returns each message with `listings: null` — app.py's `render_card` requires `TrailerListing` objects and crashes on JSON dicts, and app.py must not be modified, so cards do not reappear after a refresh/restart. Listing dicts remain persisted in the conversation JSONB (feedback indexing, "the second one" resolution, ops). Because the frontend's `already_shown_listing_urls` payload will be empty after a refresh, **dedupe must rely on the backend's persisted `state.shown_urls`** — never on the payload alone.
3. Feedback path end-to-end: Streamlit "Save feedback" → `enqueue_save_user_feedback` → `conversation[turn_idx]["feedback"]` updated (the simplified turn list has one entry per user/assistant pair, so the frontend's `turn_idx = i // 2` indexes it **directly** — no message-index math); `_merge_existing_feedback` keeps it across subsequent turn writes.
4. Robustness: `/chat` wraps graph errors → 200 with apologetic `assistant_text` + error logged (frontend shows errors poorly otherwise); request timeout budget < frontend's 180 s; concurrent same-session POSTs serialized by the **per-session advisory lock** inside `durable_turn`; oversized messages truncated; unknown session on `/session/reset` is a no-op 200.
5. `closed_at` honored: `/session/{id}` for a closed session → `{exists: true, closed: true}` → frontend starts fresh (per its logic).
6. Startup: run `alembic upgrade head` on boot when `DB_AUTO_CREATE=1` (otherwise document running it manually before first boot); `/health` returns ok only after graph compiled + DB reachable (or persistence off).

### Tests
- Integration (`test_api_contract.py`): golden-file JSON of a full /chat response; assert against a fixture copied from app.py expectations; restore-after-restart (new engine, same DB) returns messages (text restored, `listings: null`); closed session shape; idempotent turn replay; feedback lands correctly for turn 0 and turn 3.
- Live E2E checklist (manual, documented in the milestone): login → converse → cards render → refresh browser mid-conversation (chat text restores; cards do not — expected per Locked Decision) → restart backend (same) → New Conversation (old row closed) → feedback saved (check DB) → log out.

**Done when:** pytest green; the manual E2E checklist passes in one sitting.

---

## Milestone 9 — Regression Suite, Tracing, Release Hardening

**Goal:** confidence to hand to the dealership.

### Implementation
1. Consolidate all live scenarios into a tagged regression suite: `python scripts/convo_runner.py --suite regression` (M5–M7 scenarios + 5 new adversarial ones: user gives contradictory sizes; changes category twice; pastes a whole paragraph of requirements; asks FAQ mid-qualification then answers the pending question; gibberish input).
2. LangSmith: wire tracing via env (`LANGSMITH_*` already set) — tag traces with session_id + milestone-relevant metadata (intent, category); verify traces appear in the `TrailerPlace` project.
3. Cost audit script (`scripts/cost_report.py`): parse LangSmith or local logs for tokens/turn; assert ≤ 2 chat completions per normal turn (Analyze + Respond — haul classification and inventory-lookup detection are integrated into Analyze; +1 embedding on Pinecone search turns; inventory-lookup turns add **zero** extra calls — the Excel matcher uses no LLM and no embedding).
4. Ops polish: structured JSON logs per turn (session, intent, latency, tools fired); README run-book (start backend, start Streamlit, alembic + ingest commands, env table, test commands); `.env.example` with placeholder secrets. **Flag to the owner: `.env` currently holds real credentials in the repo — rotate and gitignore before any sharing/deploy.**
5. Final pass: re-run entire pytest suite + full regression scenario suite twice.

### Tests
- Everything green: `pytest`, `pytest -m db`, regression suite ×2.
- Cost assertion holds on a 10-turn scripted conversation.
- LangSmith shows traced runs (manual check).

**Done when:** all suites green; README complete; handoff demo (scripted 10-turn conversation touching search + escalation + interest) runs clean in Streamlit.

---

## Cross-Milestone Rules for Implementers

1. **Never edit `app.py`.** If the backend and frontend disagree, the backend is wrong (the one consequence we accept: text-only session restore — see Locked Decisions).
2. **Never paraphrase canned responses or the email body format** — copy from `src/domain/canned_responses.py` / this doc.
3. Every LLM behavior rule lives in the prompts (`src/llm/analyze.py` / `respond.py`), built from `prompt_structured.md` text. Every deterministic rule lives in `apply_analysis.py` / node gates with a comment citing the spec section. If you're writing an `if` on user text (keywords/regex), stop — that decision belongs to the Analyze LLM. **Unit conversion of user input is the Analyze LLM's job**; `units.py`/`normalizer.py` only ever parse listing-catalog strings (and act as pass-through safety nets on LLM output).
4. One module = one job; LLM calls, prompts, nodes, stores each in their own file (per the layout above). No module reads env directly except `src/config.py`.
5. New tests must not call real OpenAI/Pinecone/SMTP — use the conftest fakes. Live scenarios are the only exception and live under `scripts/`.
6. When a milestone is done, run **all previous milestones' tests too** — regressions block completion.
7. Where this doc's Locked Decisions and `prompt_structured.md` conflict (brand hard filter, FAQ email gate, text-only restore), **this doc wins** — it records later owner decisions.
