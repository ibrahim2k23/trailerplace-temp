# TrailerPlace Codebase Architecture

## Scope
This document covers first-party Python files in this repo (excluding `.venv` and third-party packages).

## Fast Navigation Index
- `A01` [app.py] - Streamlit frontend app and UI/session orchestration.
- `A02` [main.py] - FastAPI entrypoint exposing `/chat`, `/session/reset`, `/health`.
- `A03` [trailer_fields.py] - Trailer category slot schema + question definitions.
- `A04` [src/models.py] - Shared Pydantic API and listing models.
- `A05` [src/chatbot/service.py] - Session lifecycle + onboarding/main routing.
- `A06` [src/chatbot/graph.py] - LangGraph decision/action state machine.
- `A07` [src/chatbot/state.py] - TypedDict graph state contract.
- `A08` [src/chatbot/prompts.py] - System prompt for decision model.
- `A09` [src/chatbot/categories.py] - Category normalization/disambiguation logic.
- `A10` [src/chatbot/formatting.py] - Deterministic listing response formatter.
- `A11` [src/chatbot/tools/pinecone_search.py] - Inventory vector search tool.
- `A12` [src/chatbot/tools/email_tools.py] - Email tools used by graph actions.
- `A13` [src/email_sender.py] - SMTP backend and email dispatch helpers.
- `A14` [src/conversation_store.py] - Lead + conversation persistence service.
- `A15` [src/db.py] - SQLAlchemy engine/session/bootstrap utilities.
- `A16` [src/db_models.py] - ORM table models for leads/conversations.
- `A17` [src/shown_listings_store.py] - In-memory shown-listings store.
- `A18` [src/normalizer.py] - Canonicalization + embedding text prep.
- `A19` [src/ingest.py] - Excel -> embeddings -> Pinecone indexing pipeline.
- `A20` [src/log_setup.py] - Daily file + console logging setup.
- `A21` [src/thinking_agent.py] - Optional thinking-agent feature toggles/stub.
- `A22` [src/__init__.py] - Package marker.
- `A23` [src/chatbot/__init__.py] - Subpackage marker.

## Layered Architecture
1. Presentation Layer
- `A01` Streamlit UI: auth, session UX, rendering cards, calling backend API.

2. API Layer
- `A02` FastAPI surface: validates requests and forwards to service logic.
- `A04` shared request/response/listing schemas.

3. Conversation Orchestration Layer
- `A05` service coordinator: session state, onboarding gate, graph invocation.
- `A06` graph: action planning + tool routing.
- `A07/A08/A09/A10` supporting state, prompt, category, formatting modules.

4. Integration/Tooling Layer
- `A11` vector retrieval via OpenAI embeddings + Pinecone.
- `A12/A13` outbound business notifications.

5. Data/Persistence Layer
- `A14` persistence use-cases (lead upsert, convo upsert, feedback save).
- `A15/A16` DB connectivity and ORM definitions.
- `A17` session-memory de-dup of shown listings.

6. Data Engineering Layer
- `A18/A19` ingest normalization and indexing.

7. Cross-Cutting
- `A20` logging.
- `A21` optional thought-generation hook (currently disabled behavior).

## Runtime Flows
### Flow F1: User Chat Roundtrip
- `A01` posts chat payload to `A02 /chat`.
- `A02` forwards `ChatRequest` to `A05.handle_chat`.
- `A05` enforces contact gate and either:
  - returns onboarding response, or
  - invokes `A06` graph for main-phase decisions.
- `A06` may call `A11`, `A12` tool functions.
- `A05` persists + returns `ChatResponse`.
- `A01` renders text/cards and records shown listing URLs.

### Flow F2: Lead + Conversation Persistence
- `A05` creates soft lead via `A14.create_or_get_soft_lead` when name+phone exist.
- Conversation turns are upserted async via `A14.enqueue_upsert_conversation`.
- DB infra uses `A15` engine/session and `A16` tables.

### Flow F3: Search Index Build
- `A19` reads `listings_final_v5.xlsx`, normalizes using `A18`, embeds with OpenAI, upserts to Pinecone.

## Token-Efficient Referencing Convention
When reusing this document in future prompts:
- Reference by file ID only, e.g. `A06 + A11 interaction`.
- For a flow question, use `F1/F2/F3` labels first.
- Ask for "delta against Axx" instead of full file explanation.

## Refactor Additions (v4.4)
New shared modules introduced by the prompt/flow/code-health refactor:
- `src/chatbot/constants.py` — recent-window sizing + `compact_recent_messages`/`compact_listings`, `DYNAMIC_WIDTH_EXCLUDED_CATEGORIES`, dealership phone/URL.
- `src/chatbot/llm.py` — central LLM factory: `make_llm` / `safe_invoke` / `resolve_model`. All ChatOpenAI construction now flows through here.
- `src/chatbot/units.py` — single-source weight/length parsing (`parse_weight_lbs`, `parse_length_ft`) shared by `ingest` (index write) and `pinecone_search` (query/rerank), so both parse raw catalog strings identically.
- `src/chatbot/make_aliases.py` — single-source `MAKE_ALIASES` map consumed by `normalizer`, `make_resolver`, and `pinecone_search`; the index stores the canonical short display form (re-ingest applied).
- `src/chatbot/graph/` — `graph.py` is now a package; `graph/apply_mind/` holds `build_state_return`, the single state-assembly helper for `_apply_mind_node` exits.
- `tools/email_tools.py` — in-memory-path SMTP sends dispatched to a background pool (non-blocking); `EMAIL_SEND_SYNC=1` forces inline for tests.
- `tests/test_live_conversations.py` — RUN_LIVE conversation harness for verifying prompt/flow changes.
