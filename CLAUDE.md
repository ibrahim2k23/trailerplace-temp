# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TrailerPlace chatbot: a LangGraph-driven sales/support assistant for trailer inventory, with a Streamlit frontend and FastAPI backend, Pinecone vector search over an Excel-sourced inventory catalog, Postgres (Azure) lead/conversation persistence, and email notifications (SMTP or Microsoft Graph).

Package manager is `uv` (see `uv.lock`, `pyproject.toml`, Python >=3.12).

## Commands

```bash
# install deps
uv sync

# run backend API (FastAPI, port from CHATBOT_API_PORT, default 8000)
uv run python main.py

# run frontend (Streamlit)
uv run streamlit run app.py

# run tests
uv run pytest
uv run pytest tests/test_categories.py            # single file
uv run pytest tests/test_categories.py::test_name -v   # single test

# rebuild the Pinecone inventory index from listings_final_v5.xlsx
uv run python -m src.ingest

# db migrations
uv run alembic upgrade head
uv run alembic revision -m "message"
```

Config is env-driven via `.env` (OpenAI/Pinecone keys, DB connection, email backend, `TRAILERPLACE_PERSIST_CHATS` feature flag, `SHOW_ONLY_LLM_MENTIONED_CARDS`, search tuning knobs like `SEARCH_TOP_K`/`SEARCH_MAX_RECOMMENDATIONS`). Persistence and email sending are feature-gated — code must tolerate them being disabled/absent.

## Architecture

This repo already has two detailed reference docs — **read these before making non-trivial changes**, and update them when architecture shifts:
- `architecture.md` — layered architecture + runtime flows, using file IDs (`A01`..`A23`) and flow IDs (`F1`..`F3`) for token-efficient referencing.
- `details.md` — per-file responsibilities (`D01`..`D23`, aligned with `architecture.md` IDs) plus a compact cross-file dependency map.

High-level flow (Flow F1 in architecture.md): Streamlit (`app.py`) → FastAPI `/chat` (`main.py`) → `src/chatbot/service.py` (session state, onboarding/contact gate) → `src/chatbot/graph.py` (LangGraph `mind` node picks an action: ask a qualifying question, search inventory, send an interest email, send a non-sales FAQ email, or respond) → tools in `src/chatbot/tools/` (Pinecone search, email) → response formatted by `src/chatbot/formatting.py` and persisted async via `src/conversation_store.py`.

Key state/schema contracts to check when touching graph logic:
- `src/chatbot/state.py` — `ChatbotState` TypedDict, the full state passed between graph nodes.
- `trailer_fields.py` — per-category required/optional slot schema driving what questions get asked.
- `src/chatbot/categories.py` — canonical category list + synonym/disambiguation resolution (e.g. office vs. cooldown → Fiber vs. Enclosed).
- `src/models.py` — API-facing Pydantic models (`ChatRequest`/`ChatResponse`/`TrailerListing`).

Data engineering pipeline (Flow F3): `src/ingest.py` reads `listings_final_v5.xlsx`, normalizes fields via `src/normalizer.py`, embeds with OpenAI, and upserts to Pinecone. `src/chatbot/tools/pinecone_search.py` is the read-side counterpart used at query time (embedding query + metadata filters from hitch/color/price/length/gvwr constraints).

When asking about this codebase or requesting changes, prefer referencing the `Axx`/`Dxx`/`Fx` IDs from those docs (e.g. "trace F1 through D05+D06+D11") to keep exchanges concise.

## Notes

- `logs/` contains daily rotating log files (`src/log_setup.py`); not something to edit by hand.
- `PLAN.md`, `PROMPT_AUDIT.md`, `PROMPT_AUDIT_VALIDATION.md`, `langgraph_rules_vs_excel.md`, `missing_inventory_data_qna_flows.md` are working notes on prompt/flow design — check them for context on in-progress prompt/flow changes before modifying `src/chatbot/prompts.py` or `graph.py`.
