# Milestone 8 — Manual E2E Checklist

Run in one sitting, against a real backend and a real database. Tick every box.

## Setup

```bash
alembic upgrade head          # or boot with DB_AUTO_CREATE=1 to do it automatically
python main.py                # backend on CHATBOT_API_PORT (default 8000)
streamlit run app.py          # frontend
```

- [ ] `GET /health` returns **503** `{"status": "starting"}` for the first moment, then **200** `{"status": "ok"}`.
      The Streamlit "Chatbot initializing…" screen clears on its own once it flips.

## Conversation

- [ ] Log in. Greeting arrives with a single, low-pressure invite for name + email/phone.
- [ ] Hold a normal qualification conversation through to a search.
- [ ] Trailer cards render (this only works on a live turn — see restore below).

## Refresh mid-conversation

- [ ] Refresh the browser. **Chat text restores; cards do NOT reappear.**
      This is expected and locked: `render_card` needs `TrailerListing` objects and would crash
      on JSON dicts, and `app.py` must not be modified. `/session/{id}` returns `listings: null`.
- [ ] Continue the conversation. Ask for "more options" — no listing repeats from before the
      refresh. (Dedupe reads the backend's persisted `state.shown_urls`; the frontend's
      `already_shown_listing_urls` payload is empty after a refresh, so this proves the
      backend is the source of truth.)

## Restart the backend

- [ ] Stop `main.py`, start it again. Do **not** refresh the browser.
- [ ] Send another message. It is answered with full context — the session reloads from
      `chatbot_conversations.state_snapshot`.

## Feedback

- [ ] Open "Optional feedback" on an assistant turn, save a note.
- [ ] In the DB: `select conversation from chatbot_conversations where session_id = '<sid>';`
      → the matching turn object has `"feedback": "<your note>"`.
- [ ] Send another message, then re-check: the feedback is still there
      (`_merge_existing_feedback` preserves it when the conversation list is rewritten).
- [ ] Refresh the browser: the saved note reappears in the feedback box.

## New Conversation / logout

- [ ] Click "New Conversation". The old row gets `closed_at` set (never deleted):
      `select closed_at from chatbot_conversations where session_id = '<old sid>';`
- [ ] `GET /session/<old sid>` returns `{"exists": true, "closed": true}` and the frontend
      starts fresh.
- [ ] Click "New Conversation" twice in a row / reset an unknown session → still HTTP 200.
- [ ] Log out.

## Robustness spot-checks

- [ ] `curl -XPOST localhost:8000/chat -d '{"session_id":"nope","message":"hi"}' -H 'content-type: application/json'`
      → **422** (not 409).
- [ ] Replay a real `/chat` body with the same `turn_id` and same message → identical response,
      no second graph run. Same `turn_id` with a *different* message → **409**.
- [ ] Paste a 10,000-character message → answered normally (truncated server-side, logged).
- [ ] Kill Pinecone credentials and send a search turn → **200** with an apologetic reply,
      an exception in the backend log, and no crash in the UI. Retrying the same turn works.
