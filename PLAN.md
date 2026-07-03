# Durable Serverless Conversation Persistence

## Summary

Make PostgreSQL the source of truth for chatbot sessions. Every `/chat` request will load the session by `session_id`, process one idempotent turn, and synchronously commit the complete workflow state before returning success. Streamlit will preserve the IDs in browser `sessionStorage` and fully reconstruct the UI after a restart.

## Backend and Database Changes

- Extend `chatbot_conversations` through Alembic with:

  - `state_snapshot JSONB`, containing the complete `_new_session()` state.
  - `state_schema_version INTEGER`.
  - `state_version INTEGER` for concurrency control.
  - `closed_at TIMESTAMPTZ`, nullable.

- Add a durable turn-receipt table keyed by `(session_id, turn_id)` containing the request message, processing status, serialized `ChatResponse`, timestamps, and failure metadata.
- Add an outbox table with a unique event ID, session/turn IDs, event type, JSON payload, delivery status, attempt count, claim time, and last error.
- Store every resumable field, including messages, contact/onboarding state, category decisions, collected/skipped/defaulted fields, non-metadata requirements, pending questions, active question tracking, shown listings, listing URLs, confusion counters, pending contact actions, notification flags, and category-cycle state.
- Keep transient runtime objects, futures, locks, and clients out of the snapshot.
- Replace `_sessions` as the authoritative store. A process-local cache may remain only if validated against `state_version`; correctness must never depend on it.
- Load or create the database session at the start of every turn and serialize concurrent requests for the same session using a PostgreSQL row lock.
- Commit the updated state, full transcript, turn receipt, and generated outbox events in one transaction. Return HTTP 503 on commit failure so an unsaved turn is never presented as successful.
- Move email calls out of graph/service execution. Graph nodes will generate outbox event descriptions instead of sending immediately.
- Deliver committed outbox events after the transaction. Retry pending or expired claims on subsequent requests/startup; favor delivery while documenting the unavoidable rare SMTP duplicate window.
- Retain old transcript rows without snapshots. They will display as text history, while workflow state initializes fresh with the current schema version.
- `/session/reset` will close the durable session and clear its resumable snapshot while retaining transcript history and completed receipts for audit.

## API and Streamlit Changes

- Add `turn_id: UUID` to `ChatRequest`. Reusing it returns the stored response without advancing state or creating duplicate outbox events.
- Keep `session_id` mandatory on every chat payload.
- Add `GET /session/{session_id}` returning whether the session exists/is closed plus restorable UI fields: messages with listing metadata, sales phase, contact values, and state version.
- Preserve response compatibility while ensuring `main_prior_messages` comes from the committed snapshot.
- Streamlit will create and retain:

  - `session_id` in browser `sessionStorage`, bridged into Python through the URL/query-parameter mechanism.
  - A stable `turn_id` for each submitted user message, retained until that request succeeds.

- On initial load/reconnection, Streamlit will fetch the saved session before rendering and rebuild transcript text, listing cards, feedback metadata, contact fields, phase, and shown-listing context.
- On a retryable backend failure, keep the pending user message and its `turn_id`; retrying must not duplicate the turn.
- “New Conversation” and logout will clear browser session storage and local Streamlit state. A new conversation creates a new session ID.

## Tests and Acceptance Criteria

- Simulate clearing `_sessions` or restarting FastAPI between turns; the next turn must retain collected features, missing questions, non-metadata requirements, listing context, contact flow, and active qualification state.
- Verify Streamlit restart/reload in the same tab restores the same session ID and complete rendered conversation.
- Verify closing the tab ends browser-side recovery because `sessionStorage` is used.
- Submit the same `turn_id` repeatedly and confirm identical responses, one state transition, one transcript turn, and one outbox event.
- Submit concurrent turns for one session and confirm deterministic serialization with no lost state.
- Force database failure and confirm `/chat` returns a retryable error without reporting success.
- Test snapshot serialization for every field in `_new_session()` and reject unsupported/non-JSON values.
- Test legacy transcript-only sessions, closed/reset sessions, missing sessions, malformed UUIDs, and snapshot schema-version mismatches.
- Test outbox success, failure, expired-claim recovery, and application-level event deduplication.
- Run the existing chatbot, contact-flow, inventory, feedback, and email tests to ensure behavior remains compatible.

## Assumptions

- PostgreSQL remains available to both Streamlit and FastAPI deployments.
- Full transcript and workflow state may be stored in JSONB under the project’s existing retention policy.
- Same-tab recovery is required; reopening a closed tab is not.
- Newly persisted sessions are exactly resumable. Legacy sessions display their transcript but begin with fresh workflow logic.
- Email delivery is at-least-once: delivery is favored over eliminating the extremely small SMTP crash-window duplicate risk.
