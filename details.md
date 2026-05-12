# TrailerPlace Codebase Details

## How To Use This File Efficiently
- Ask by ID to reduce tokens: `Explain D06`, `Compare D05 vs D06`, `Trace F1 with D05+D06+D11`.
- IDs align with `architecture.md` (`Axx` <-> `Dxx`).

## D01 - app.py
Purpose: Streamlit chat frontend for authenticated users.
- Loads env + logging, configures app UI theme/CSS, and handles login via `TRAILERPLACE_APP_USERNAME/PASSWORD`.
- Manages client session state: chat history, sales phase, onboarding messages, customer identity, and optional thinking-agent lifecycle.
- Sends user input to backend `/chat`, maps API listing payloads into `TrailerListing`, renders trailer cards, and persists per-response feedback asynchronously.
- Keeps shown listing URLs for de-dup and resets backend session via `/session/reset` when new conversation/logout occurs.

## D02 - main.py
Purpose: API bootstrap.
- Configures FastAPI app and exposes `GET /health`, `POST /chat`, `POST /session/reset`.
- Delegates business logic to `src.chatbot.service`.
- Runs uvicorn locally using `CHATBOT_API_PORT` (default `8000`).

## D03 - trailer_fields.py
Purpose: Category -> qualification schema.
- Defines `TrailerFieldSpec` with required/optional slots, prompts, notes.
- `_SPECS` maps categories (Equipment, Dump, Enclosed, etc.) to slot definitions.
- Exposes `get_trailer_fields`, `get_trailer_fields_as_dict`, `list_all_categories`.
- Acts as source of truth for question queueing in graph logic.

## D04 - src/models.py
Purpose: Shared contracts.
- `TrailerListing` UI listing shape.
- `ChatRequest` inbound API payload including phase and shown URLs.
- `ChatResponse` outbound assistant payload plus optional thinking context.
- `ResetSessionRequest` for clearing server session state.

## D05 - src/chatbot/service.py
Purpose: Main orchestrator for chat behavior.
- Maintains in-memory session store keyed by `session_id` with lock safety.
- Extracts contacts using structured LLM + regex fallback.
- Enforces onboarding/contact gate (name+phone required for main flow).
- Creates soft lead in DB once contact requirement is met.
- Routes:
  - onboarding reply generator,
  - smalltalk reply generator,
  - full graph invocation for actionable trailer intents.
- Persists conversations asynchronously through `conversation_store`.

## D06 - src/chatbot/graph.py
Purpose: LangGraph decision + tool workflow.
- Defines `MindDecision` structured output schema for planner model.
- `mind` node decides next action (ask, search, email interest, faq email, respond).
- `apply_mind` updates slots/category/pending question queue and final action.
- Conditional routes to:
  - Pinecone search node,
  - interested-listing email node,
  - non-sales FAQ email node.
- Formats final assistant output and emits tool events for downstream persistence/debugging.

## D07 - src/chatbot/state.py
Purpose: Strongly-typed graph state.
- `ChatbotState` TypedDict captures all state fields exchanged across nodes.
- `QuestionItem` defines queued slot-question entries.

## D08 - src/chatbot/prompts.py
Purpose: Planner system prompt content.
- Builds `MIND_SYSTEM_PROMPT` including policy-like behavioral rules and tool-use conditions.
- Injects canonical category block from `categories.py` into prompt.

## D09 - src/chatbot/categories.py
Purpose: Category resolution + disambiguation.
- Maintains canonical categories and synonym map.
- Resolves category from user text via regex term detection.
- Handles office/cooldown disambiguation to Fiber vs Enclosed with clarification mode.

## D10 - src/chatbot/formatting.py
Purpose: Deterministic listing response renderer.
- Computes relevance-weighted bullet ordering from user intent + slots.
- Produces markdown structure for each listing: link/title, bullet specs, why-it-fits sentence(s).
- Ensures empty-result fallback text is returned when no matches exist.

## D11 - src/chatbot/tools/pinecone_search.py
Purpose: Inventory retrieval tool.
- Creates embedding query text from message + category + slots.
- Builds Pinecone metadata filter from hitch/color/price/length/gvwr-style constraints.
- Queries Pinecone index with embeddings, cleans match metadata into app-consumable listing dict.
- Excludes previously shown URLs and caps recommendations.

## D12 - src/chatbot/tools/email_tools.py
Purpose: Graph-callable email tool wrappers.
- `send_interested_listing_email`: sales notification + updates lead item of interest.
- `send_non_sales_faq_email`: category-normalized FAQ/contact notification.
- Defines `FAQ_CATEGORY_LABELS` canonical map.

## D13 - src/email_sender.py
Purpose: Email transport layer.
- Implements SMTP sender (`SmtpEmailSender`) with env-driven config.
- Provides Graph sender stub for future implementation.
- Exposes synchronous send helpers and async FAQ enqueue wrapper.

## D14 - src/conversation_store.py
Purpose: Persistence service abstraction.
- Feature-gates persistence by env flag + DB availability.
- Ensures schema, upserts conversations, creates/updates soft leads.
- Saves user feedback into conversation turns.
- Uses thread pool for non-blocking persistence calls.

## D15 - src/db.py
Purpose: DB infra setup.
- Detects whether DB env vars are complete.
- Chooses postgres driver (`psycopg2` or `psycopg`).
- Creates cached SQLAlchemy engine/session factory.
- Ensures ORM schema exists via metadata `create_all`.

## D16 - src/db_models.py
Purpose: ORM definitions.
- `ChatbotLead`: lead identity/contact and item of interest.
- `ChatbotConversation`: session->lead linked JSONB conversation payload with timestamps.
- Defines `Base` declarative root for schema operations.

## D17 - src/shown_listings_store.py
Purpose: In-memory dedup helpers.
- Tracks shown URLs/keys per session with thread-safe lock.
- Extracts shown URLs from message history.
- Supports add/get/reset operations.

## D18 - src/normalizer.py
Purpose: Canonicalization and embedding prep.
- Normalizes category/subcategory/make/color/hitch/condition.
- Cleans dealer notes (removes contact/site noise).
- Builds normalized embedding text payload while excluding fields like MSRP.

## D19 - src/ingest.py
Purpose: Offline indexing pipeline.
- Reads Excel inventory, parses structured fields and nested JSON specs.
- Builds per-row metadata + embedding text with normalization/parsing helpers.
- Creates/validates Pinecone index and upserts vectors in batches.
- Supports idempotent behavior and force-reindex flags.

## D20 - src/log_setup.py
Purpose: Logging configuration.
- Custom daily rotating file handler writing `logs/YYYY-MM-DD.log`.
- Adds console handler and respects `LOG_LEVEL`.

## D21 - src/thinking_agent.py
Purpose: Optional insight-generation feature stub.
- Env toggles indicate enabled/background behavior.
- Current generator/log functions return disabled/no-op payload.

## D22 - src/__init__.py
Purpose: Package marker for `src`.

## D23 - src/chatbot/__init__.py
Purpose: Package marker for chatbot module.

## Cross-File Dependency Map (Compact)
- `D01 -> D02,D04,D14,D17,D21`
- `D02 -> D04,D05`
- `D05 -> D06,D09,D14,D04`
- `D06 -> D03,D08,D09,D10,D11,D12,D07`
- `D12 -> D13,D14`
- `D14 -> D15,D16`
- `D19 -> D18`

## Suggested Prompt Shortcuts
- "Use `Axx/Dxx` references only."
- "Summarize only deltas in `D05,D06,D11` since last review."
- "Trace user message path through `F1` with function names only."
