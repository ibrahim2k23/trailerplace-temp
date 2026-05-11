# TrailerPlace Chatbot — Codex brief (tool development)

Dense reference for adding **OpenAI-style function tools** in this repo. Paths relative to repo root (`Chatbot/src`).

## Product

Streamlit UI + FastAPI (`main.py` → `POST /chat`). Pre-main: `contact_onboarding.py` collects contact via `submit_customer_contact`. Main chat: **`TrailerAgentLG`** in `src/agent_lg.py` (LangGraph + LangChain `ChatOpenAI.bind_tools`).

## Graph (agent_lg)

```
master_router_node → (trailer_type set?) → specialist_node → (has search_results for current category?) → recommendation_node → END
```

- **master_router**: only `set_trailer_type`. Sets `trailer_type`, seeds `required_slots` / `optional_slots` from `trailer_fields`.
- **specialist**: `fetch_trailer_fields`, `record_slot_answer`, `search_trailers`, `set_trailer_type` (category pivot). Tool loop; **early exit** to parent updates if `search_trailers` returns results (`recommendation_entry_due`).
- **recommendation**: `log_product_interest`, `search_trailers`, `set_trailer_type`, `record_slot_answer`.

## `SessionState` (`src/state.py`)

| Key | Role |
|-----|------|
| `messages` | LangChain `HumanMessage` / `AIMessage` / `ToolMessage` (append-only reducer) |
| `trailer_type` | Canonical category string |
| `required_slots` / `optional_slots` | From `get_trailer_fields_as_dict` |
| `slots_collected` | slot name → answer |
| `slots_asked` | order of `record_slot_answer` |
| `utility_lightweight_decided` | Utility-only; strips weight slots when true |
| `search_results` | Listing dicts for prompts |
| `search_results_for_category` | Must match `trailer_type` to route to recommendation |
| `api_listings_this_turn` | Cleared each `chat()`; returned to API/UI for cards |
| `last_search_args` | For show-more merge (`_LAST_SEARCH_MERGE_KEYS`) |
| `session_id`, `client_shown_urls` | Show-more URL exclusion |
| `recommendation_entry_due` | Specialist sets true when search just ran this invoke |

## Existing tools (where to mirror)

| Tool | Schema const | Executed in |
|------|----------------|-------------|
| `set_trailer_type` | `SET_TRAILER_TYPE_TOOL` | `_master_router_node`, `_execute_specialist_tool`, `_run_set_trailer_type_tool`, recommendation loop |
| `fetch_trailer_fields` | `FETCH_TRAILER_FIELDS_TOOL` | `_execute_specialist_tool` → `get_trailer_fields_as_dict` |
| `record_slot_answer` | `RECORD_SLOT_ANSWER_TOOL` | `_execute_specialist_tool` |
| `search_trailers` | `SEARCH_TRAILERS_TOOL` | `_execute_search_tool` |
| `log_product_interest` | `LOG_INTEREST_TOOL` | `_recommendation_node` → `_execute_log_interest` |
| `submit_customer_contact` | `SUBMIT_CUSTOMER_TOOL` | `contact_onboarding.py` (separate OpenAI client, not LangGraph) |

Tool defs: **~L301–455** `agent_lg.py`. Specialist binds tools ~**L1158**; recommendation ~**L1474**. Dispatch: `_execute_specialist_tool` ~**L1206**, recommendation tool loop ~**L1496**.

## Add a new tool — checklist

1. **Schema**: Add `{"type":"function","function":{...}}` next to peers in `agent_lg.py` (or separate module if large; still wire in `agent_lg`).
2. **Bind**: Append to `tools = [...]` in the node(s) that should see it (`_specialist_node` and/or `_recommendation_node`; rarely master).
3. **Execute**:
   - Specialist path: branch in `_execute_specialist_tool` **or** extend recommendation loop (same pattern as `log_product_interest`).
   - Return **`(json_string, extra_dict)`** where `extra_dict` keys are **`SessionState` updates** (merged into graph state).
4. **Prompts**: Update `MASTER_ROUTER_PROMPT`, `_SPECIALIST_PROMPT_TEMPLATE`, and/or `RECOMMENDATION_PROMPT` so the model knows when to call the tool.
5. **Side effects**: Log with `logger.info`; email/DB use existing helpers (`email_sender`, `conversation_store`).
6. **Tests**: `tests/` uses unittest; see `test_agent_lg_api_listings.py` for agent-related patterns.

## Search (do not duplicate blindly)

- `_execute_search_tool` builds `TrailerFilter` (`src/models.py`), calls `_run_search` (Pinecone + embed), then **`TrailerAgent._rerank_by_fit`** via a dummy instance (`agent.py`), `_apply_category_make_priority`, caps with `SEARCH_MAX_RECOMMENDATIONS`.
- **`required_payload_lbs`**: rerank only — **not** a Pinecone metadata filter (see `CLAUDE.md` / `TrailerFilter`).
- Show-more: `more_results=true` merges `last_search_args`; `merge_shown_urls_for_show_more(session_id, client_shown_urls)`.

## Qualification slots

- **`src/trailer_fields.py`**: `_SPECS` per category; `get_trailer_fields_as_dict(category)` → `required_slots`, `optional_slots`, `questions`, `notes`. Slot names must match `record_slot_answer` and `slots_collected` keys.

## Normalization

- **`src/normalizer.py`**: `normalize_category`, `normalize_make`, `normalize_color`, `normalize_hitch`, `normalize_subcategory`, `HITCH_MAP`. Hitch ≠ category (enforced in prompts + `_validate_set_trailer_type_arg`).

## HTTP / session

- **`src/api_service.py`**: `_agents[session_id] = TrailerAgentLG(customer=CustomerContact)`. `run_chat` passes `already_shown_listing_urls` into `agent.chat(..., client_shown_urls=...)`.
- **`POST /session/reset`**: drops agent for `session_id`.

## Key deps

OpenAI, LangGraph, LangChain OpenAI, Pinecone, Pydantic, FastAPI, Streamlit, Postgres/Supabase optional (`conversation_store`, Alembic).

## Files map (non-exhaustive)

| File | Purpose |
|------|---------|
| `src/agent_lg.py` | LangGraph agent, all main sales tools |
| `src/agent.py` | Pinecone filter, rerank, dedupe, metadata→listing (imported by LG) |
| `src/state.py` | `SessionState` |
| `src/models.py` | `TrailerFilter`, `TrailerListing`, `CustomerContact` |
| `src/trailer_fields.py` | Category slot definitions |
| `src/normalizer.py` | Canonical strings for filters |
| `src/api_service.py` | `run_chat`, session dict |
| `main.py` | FastAPI app |
| `app.py` | Streamlit |
| `src/contact_onboarding.py` | Pre-chat tool `submit_customer_contact` |
| `src/thinking_agent.py` | Debug sidebar (separate from tool loop) |

## Env (minimal)

`OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (default `trailerplace-listings`), optional `OPENAI_MODEL`, `CHATBOT_API_PORT`, `TRAILERPLACE_*`.

---

*Generated for low-token handoff to Codex; extend `CLAUDE.md` for full ops detail.*
