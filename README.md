# TrailerPlace Chatbot

A FastAPI + LangGraph backend behind a Streamlit frontend for a trailer dealership in
Wharton, TX. The bot qualifies a lead through a short conversation, searches the live
listing index, answers direct inventory questions, and emails the sales team.

- **Behavioral spec:** `prompt_structured.md`
- **Build plan & locked decisions:** `milestone.md`
- **Code map:** `overview.md` (structure) and `Details.md` (per-file detail)

---

## Architecture in one paragraph

Every `/chat` turn makes **exactly two LLM calls**: *Analyze* (classify intent, extract
fields, normalize units, detect inventory lookups and email triggers — all as one strict
structured output) and *Respond* (write the reply). Code never regex-matches user text;
it only dispatches on the Analyze output. A Pinecone search adds one embedding call; an
inventory lookup adds none (it is a pure Excel + rapidfuzz match). `scripts/cost_report.py`
enforces that budget.

State lives in a process-local `_sessions` dict. When persistence is on, each request is
one **atomic durable turn**: an advisory lock, a receipt check for idempotency, the graph
run, a state snapshot, and email events written to a transactional outbox — committed
together, then the outbox is drained.

---

## Run book

### 1. Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` is the dependency source of truth. Do not add packages without need;
`msal` and `pytest-mock` are absent on purpose.

### 2. Configure

```bash
cp .env.example .env            # then fill in real values
```

See the [environment table](#environment) below.

### 3. Migrate the database (persistence on only)

```bash
alembic upgrade head
```

Set `DB_AUTO_CREATE=1` to have the backend do this on boot instead. Persistence is off
entirely unless all five of `HOST`/`PORT`/`DATABASE`/`PGUSER`/`PASSWORD` are set **and**
`TRAILERPLACE_PERSIST_CHATS` is truthy — the bot runs fully in memory otherwise.

### 4. Ingest listings into Pinecone (once, or after the workbook changes)

```bash
python -m src.search.ingest              # add --force to re-embed, --no-wipe to keep old vectors
```

Reads `listings_final_v5.xlsx` from the repo root.

### 5. Start the backend

```bash
python main.py                           # uvicorn on CHATBOT_API_PORT (default 8000)
```

`GET /health` returns `{"status":"ok"}` only once the graph has compiled and the database
is reachable (or persistence is off). It returns 503 with a reason otherwise.

### 6. Start the frontend

```bash
streamlit run app.py
```

`app.py` is **never modified** — the backend conforms to it, not the other way around.

---

## Tests

| Command | What it covers |
|---|---|
| `pytest` | The full offline suite. No network: LLM, Pinecone, and email are all faked. |
| `pytest -m db` | The Postgres-backed tests. Skipped unless `TEST_DATABASE_URL` is set. |
| `pytest -m "not db"` | Explicitly skip the database tests. |

A throwaway database for the `db` tests:

```bash
docker run -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16
```

### Live scenario suites (real OpenAI + Pinecone — costs tokens)

These need a running backend started with `DEBUG_STATE_ENDPOINT=1`, which exposes
`GET /session/{id}/state` for state assertions. Keep that flag off in production.

```bash
python scripts/convo_runner.py --suite regression --list      # no network: just list them
python scripts/convo_runner.py --suite regression             # the M9 release gate
python scripts/convo_runner.py --suite regression --repeat 2  # gate requires two clean passes
python scripts/convo_runner.py --suite adversarial            # the 5 hardening scenarios
python scripts/convo_runner.py scripts/scenarios/contact-full.yaml   # one scenario
```

Scenarios live in `scripts/scenarios/*.yaml` and select by their `tags:` list
(`regression`, `adversarial`, `m5`/`m6`/`m7`/`m9`).

### Cost audit

Run the backend with a turn log, hold a conversation, then audit it:

```bash
TURN_LOG_PATH=turns.jsonl python main.py
python scripts/cost_report.py turns.jsonl
```

Exits nonzero if any turn exceeded 2 chat completions, or made an embedding call on a
non-search turn.

---

## Environment

| Key | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Analyze + Respond calls, and search embeddings. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for the customer-facing Respond call. |
| `ANALYZE_MODEL` | `gpt-5-mini` | Model for the Analyze call. |
| `ANALYZE_REASONING_EFFORT` | `minimal` | Reasoning effort for Analyze. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Must match what `ingest` wrote. |
| `PINECONE_API_KEY` | — | Listing vector index. |
| `PINECONE_INDEX_NAME` | `trailerplace-listings` | Index name. |
| `SEARCH_TOP_K` | `50` | Candidates fetched per query before rerank. |
| `SEARCH_MAX_RECOMMENDATIONS` | `5` | Cards shown after rerank. |
| `FEATURE_LLM_RERANK_ENABLED` | `true` | Use evidence-grounded GPT reranking only when non-metadata features are present. |
| `FEATURE_RERANK_MODEL` | `gpt-5-nano-2025-08-07` | Pinned structured-output feature reranker. |
| `FEATURE_RERANK_REASONING_EFFORT` | `medium` | Feature-rerank reasoning effort; set `high` after latency evaluation if needed. |
| `FEATURE_RERANK_TIMEOUT_SECONDS` | `60` | Timeout for the single feature-rerank request. |
| `FEATURE_RERANK_WEIGHT` / `FEATURE_FIT_WEIGHT` | `0.85` / `0.15` | Semantic feature coverage and unchanged fit-order blend. |
| `RERANK_ENABLED` | `true` | Fit rerank (length-first, no under-length). |
| `RERANK_WARN_RATIO` / `RERANK_EXTREME_RATIO` | `1.35` / `1.9` | Oversize penalties. |
| `RERANK_LENGTH_WEIGHT` / `RERANK_MISSING_DIM_PENALTY` | `8.0` / `0.35` | Rerank weights. |
| `RERANK_VERBOSE_LOGS` / `MAKE_RERANK_VERBOSE_LOGS` | `false` | Rerank debug logs. |
| `SHOW_ONLY_LLM_MENTIONED_CARDS` | `true` | Show a card only if the reply cites its URL. |
| `TRAILERPLACE_WEBSITE` | — | Quoted in replies. |
| `HOST` `PORT` `DATABASE` `PGUSER` `PASSWORD` | — | Postgres. All five required to enable persistence. |
| `TRAILERPLACE_PERSIST_CHATS` | `1` | Set to `0` to force in-memory mode. |
| `DB_AUTO_CREATE` | `0` | Run `alembic upgrade head` on boot. |
| `TEST_DATABASE_URL` | — | Enables `pytest -m db`. |
| `EMAIL_BACKEND` | `smtp` | `graph` or `smtp`. |
| `TENANT_ID` `CLIENT_ID` `CLIENT_SECRET` `SENDER_EMAIL` `RECIPIENT_EMAIL` | — | Graph backend (`RECIPIENT_EMAIL` is where leads land). |
| `SMTP_HOST` `SMTP_PORT` `SMTP_USER` `SMTP_PASSWORD` `SMTP_FROM` `EMAIL_TO` | — | SMTP backend (`EMAIL_TO` is the recipient — *not* `RECIPIENT_EMAIL`). |
| `CHATBOT_API_PORT` | `8000` | uvicorn port. |
| `DEBUG_STATE_ENDPOINT` | `0` | Exposes `GET /session/{id}/state`. Scenario runner needs it. **Off in prod.** |
| `INVENTORY_LOOKUP_LIMIT` | `5` | Max inventory-lookup cards. |
| `LANGSMITH_TRACING` | `false` | Trace export. Needs `LANGSMITH_API_KEY`. |
| `LANGSMITH_ENDPOINT` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | — | Traces are tagged `session:<id>` with `intent`/`category` metadata. |
| `LOG_FORMAT` | `text` | `json` for structured ops logs. Applies to both the console and the daily log file. |
| `LOG_LEVEL` | `INFO` | Root log level. |
| `LOG_DIR` | `logs` | Every log line also lands in `<LOG_DIR>/<YYYY-MM-DD>.log`, alongside the terminal. |
| `TURN_LOG_PATH` | — | JSONL of per-turn records; the cost report's input. |
| `CHAT_TIMEOUT_SECONDS` | `150` | Server-side graph budget. Must stay under app.py's 180 s. |
| `CHAT_MAX_MESSAGE_CHARS` | `4000` | Oversized messages are truncated. |

---

## Security

> **`.env` currently holds live credentials** (OpenAI, Pinecone, Microsoft Graph client
> secret, and the Azure Postgres password). The repo now ships a `.gitignore` that
> excludes it, and there is no git history containing it. **Rotate every one of those
> secrets before sharing this repo or deploying**, and keep production values in a secret
> manager rather than a file on disk.

Also: `DEBUG_STATE_ENDPOINT=1` exposes the full raw session state, including customer
contact details, on an unauthenticated endpoint. It exists for the scenario runner. Leave
it off anywhere real.

---

## Operations

### Logs

Every backend run writes to the terminal **and** to `<LOG_DIR>/<YYYY-MM-DD>.log`
(default `logs/2026-07-11.log`, etc — one file per day, no restart or config needed to
roll over). Two kinds of records land there for every `/chat` turn:

**1. A compact machine record** on the `trailerplace.turn` logger (also mirrored to
`TURN_LOG_PATH` if set, for `scripts/cost_report.py`):

```json
{"event":"chat_turn","session_id":"…","turn_id":"…","intent":"qualification_answer",
 "category":"Dump","latency_ms":2140.5,"tools_fired":["search","email"],
 "emails_sent":["Results Shown to User"],
 "llm_calls":{"chat_completions":2,"embeddings":1,"prompt_tokens":3011,
              "completion_tokens":180,"total_tokens":3191,"models":["gpt-4o-mini"]}}
```

**2. A human-readable reasoning block** on the `trailerplace.conversation` logger — the
session id, the raw user message, everything the Analyze LLM extracted and decided
(intent, category, extracted fields, haul classification, inventory-lookup identifiers,
email triggers), which tools fired (search/inventory lookup/email, with their own
`TOOL search: …` / `TOOL email: …` lines logged where they run), the assistant's reply,
and the full session state as it stood right after the turn (category, slots,
skipped/pending questions, contact, brand preference, shown listings):

```
====================================================================================
TURN  session=55edd473-…  turn=5492f214-…
------------------------------------------------------------------------------------
USER: I need a dump trailer for gravel
------------------------------------------------------------------------------------
REASONING (Analyze):
  intent: qualification_answer
  category_mentioned: Dump  (info_only=False)
  extracted: length=14.0 width=7.0 height=- payload_lbs=- hitch=- brand=-
  ...
------------------------------------------------------------------------------------
TOOLS FIRED: search(results=3, brand_relaxed=False)
------------------------------------------------------------------------------------
ASSISTANT: Got it — here are a few dump trailers that fit.
  listings_returned: 3
------------------------------------------------------------------------------------
STATE AFTER TURN:
  category: Dump   clarification_key: -
  slots: haul_material=gravel, haul_weight_lbs=6000.0
  ...
====================================================================================
```

Set `LOG_FORMAT=json` to switch both the console and the daily file to structured JSON
lines instead (the conversation block's text becomes the `message` field).

### Emails and leads

- **Emails are at-least-once.** They go through the `chatbot_outbox` table inside the
  turn's transaction, and the drain runs **after** the `/chat` response is returned, on a
  background thread — a slow or failing send (an OAuth handshake reset, a blocked SMTP
  port) never adds latency to the reply the user is waiting on. A failed send stays
  retryable and is picked up by the next turn's drain, on any session.
- **Leads** start `soft` and upgrade to `hard` whenever any email actually sends —
  system alerts included. Never downgraded.
- **Session restore is text-only.** After a browser refresh or backend restart the chat
  text returns but trailer cards do not (`app.py`'s `render_card` needs `TrailerListing`
  objects and must not be modified). Listing dicts stay in the conversation JSONB, so
  "the second one" and search dedupe still work off the backend's `shown_urls`.
