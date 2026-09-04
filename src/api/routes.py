from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from src import conversation_log, conversation_store, tracing, turn_log, turn_status
from src.api import readiness
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DebugStateResponse,
    ResetRequest,
    SessionResponse,
    TurnStatusResponse,
)
from src.config import settings
from src.db_models import ChatbotConversation, ChatbotTurn
from src.domain.reply_chunks import split_reply_into_chunks
from src.graph.build import build_graph
from src.graph.state import _get_session, _sessions, clear_session, from_snapshot, session_lock, to_snapshot
from src.llm import usage as llm_usage
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter()
_GRAPH = None

# Bounds every graph run below app.py's 180 s client timeout (M8 §4).
_GRAPH_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat_graph")
# Its own pool, never _GRAPH_EXECUTOR: a streamed turn occupies a worker for its whole run
# and then submits the graph run itself, so sharing one pool would let N streams deadlock by
# holding every worker while waiting for a worker.
_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat_stream")
# How often the streaming turn checks what the graph is doing, to forward the search line.
_STREAM_STATUS_POLL_SECONDS = 0.2

ERROR_ASSISTANT_TEXT = (
    "Sorry — something went wrong on our end and I couldn't finish that thought. "
    "Could you send that again?"
)


class GraphFailure(Exception):
    """A graph run raised or timed out. Contained into a 200 apology, never a 500."""


class TurnSuperseded(Exception):
    """The customer said more while we were answering, so this turn is thrown away.

    Raised INSIDE durable_turn, which is the whole point: everything the turn wrote -
    the state snapshot, the conversation, the turn receipt, and the queued alert emails -
    is in that one transaction, so the rollback erases all of it. The caller then reruns
    the turn with the new messages appended, and only THAT reply reaches the customer.
    """


# Set by a channel that can tell when a customer has said more mid-turn (the Messenger
# webhook). A contextvar rather than a request field because it is a callable, and
# because it must not appear on the public /chat contract.
_ABANDON_CHECK: contextvars.ContextVar = contextvars.ContextVar("turn_abandon_check", default=None)


@contextlib.contextmanager
def abandon_turn_if(predicate):
    """Run the turn inside this to have it discarded when `predicate()` becomes true.

    Pass None to disable - which is how the caller stops retrying and lets an answer
    through after the customer has interrupted too many times.
    """
    token = _ABANDON_CHECK.set(predicate)
    try:
        yield
    finally:
        _ABANDON_CHECK.reset(token)


def _turn_was_superseded() -> bool:
    check = _ABANDON_CHECK.get()
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:  # noqa: BLE001 - a failed check must not lose the customer's reply
        logger.exception("abandon check failed; keeping the reply")
        return False


def set_graph_client(client: LLMClient) -> None:
    global _GRAPH
    _GRAPH = build_graph(client)


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def ensure_graph() -> None:
    """Compile the graph at startup so the first /chat doesn't pay for it."""
    _get_graph()


def _require_uuid(value: str, field: str) -> uuid.UUID:
    """Reject malformed ids with 422 up front.

    Without this, uuid.UUID() raises deep inside the durable path and the ValueError
    is caught by the handler that means 'turn_id reused' — reporting 409 for a bad id.
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be a UUID") from exc


def _require_session_id(value: str) -> str:
    """Session ids are NOT required to be UUIDs — Messenger's are PSIDs.

    Streamlit sends a uuid4; Messenger sends "9876543210987654". Both are legal here
    because conversation_store.as_session_uuid maps whatever arrives onto the UUID the
    tables are keyed by, while chatbot_leads.psid keeps the raw value. The only real
    constraint is the length of that psid column.
    """
    text = str(value or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="session_id must not be empty")
    if len(text) > 255:
        raise HTTPException(status_code=422, detail="session_id must be at most 255 characters")
    return text


def _error_response(session_id: str) -> dict[str, Any]:
    """Full-contract body for a failed turn: every key app.py reads is present."""
    state = _sessions.get(session_id) or {}
    return {
        "assistant_text": ERROR_ASSISTANT_TEXT,
        "listings": [],
        "sales_phase": "main",
        "onboarding_api_messages": [],
        "customer_full_name": state.get("customer_name"),
        "customer_email": state.get("customer_email"),
        "customer_phone": state.get("customer_phone"),
        "main_prior_messages": None,
        "thinking_context": None,
    }


@router.get("/health")
def health(response: Response) -> dict[str, object]:
    """ok only once the graph is compiled and the DB is reachable (or persistence off).

    app.py polls this and retries on any non-ok body or non-2xx status, so 503 is safe.
    """
    status = readiness.readiness_status()
    if status["status"] != "ok":
        response.status_code = 503
    return status


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    _require_session_id(request.session_id)
    turn_id = request.turn_id or str(uuid.uuid4())
    _require_uuid(turn_id, "turn_id")

    if len(request.message) > settings.chat_max_message_chars:
        logger.warning(
            "Truncating oversized message for session %s (%d chars)", request.session_id, len(request.message)
        )
        request = request.model_copy(update={"message": request.message[: settings.chat_max_message_chars]})

    started = time.perf_counter()
    # Nothing from a previous turn may be visible to a poller for this one.
    turn_status.clear(request.session_id)
    # One usage scope + one trace per turn (M9 §2/§3). The scope must wrap the graph
    # run, so it sits outside the session lock's critical section but inside the request.
    with llm_usage.usage_scope() as usage, tracing.trace_turn(request.session_id, turn_id=turn_id):
        try:
            with session_lock(request.session_id):
                response = _run_turn(request, turn_id)
        except HTTPException:
            raise
        except ValueError as exc:
            # durable_turn's only ValueError: same turn_id replayed with a different message.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except GraphFailure as exc:
            # durable_turn rolled back, so no receipt was written: the frontend's retry with
            # the same turn_id replays cleanly rather than reading back this apology.
            logger.exception("Chat turn failed for session %s", request.session_id)
            _log_turn(request.session_id, turn_id, started, usage, request.message, error=str(exc))
            return ChatResponse(**_error_response(request.session_id))
        finally:
            # The turn is over either way: the line now travels on the response itself, and
            # leaving it here would both leak an entry per session and let the next turn's
            # first poll read this turn's line.
            turn_status.clear(request.session_id)
        _log_turn(request.session_id, turn_id, started, usage, request.message, response=response)
        return response


# ---------------------------------------------------------------------------
# Streaming turn (SSE)
# ---------------------------------------------------------------------------
# What this does and does NOT do, because the distinction matters:
#
# It does NOT stream tokens out of the model. The reply is a STRUCTURED output that the
# respond node validates and repairs before it is allowed out - a draft that cites a trailer
# we never showed, or drops one we did, is rejected and rewritten. Emitting tokens as they
# arrive would mean publishing drafts we are about to reject, and would throw away
# cited_listing_urls, which is what decides the cards.
#
# What it DOES stream is the finished reply: the search line the moment the graph publishes
# it (so the UI no longer polls for it), then the reply typed out message by message - the
# intro, then ONE MESSAGE PER TRAILER, then the closing question. See reply_chunks.py.
#
# Events: status | chunk_start | delta | chunk_end | done | error. `done` carries the exact
# body POST /chat returns, plus `chunks`, so a client can ignore the typing entirely.

_WORD_RE = re.compile(r"\S+\s*")


def _sse(event: str, data: dict[str, Any]) -> str:
    """One SSE frame: an event name, a single-line JSON payload, and the blank-line terminator."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _typing_deltas(chunk: str, words_per_delta: int) -> list[str]:
    """The chunk cut into keystroke-sized pieces, joinable back into it exactly."""
    words = _WORD_RE.findall(chunk)
    if not words:
        return [chunk] if chunk else []
    step = max(1, words_per_delta)
    return ["".join(words[i : i + step]) for i in range(0, len(words), step)]


def _chat_event_stream(request: ChatRequest):
    """Run the turn on a worker while forwarding progress, then type the reply out."""
    # ThreadPoolExecutor does not propagate contextvars, and the turn's usage scope and trace
    # are installed inside chat() on the worker - the copy keeps request-scoped state (M9 §3).
    context = contextvars.copy_context()
    future = _STREAM_EXECUTOR.submit(context.run, chat, request)
    last_note: str | None = None
    while not future.done():
        time.sleep(_STREAM_STATUS_POLL_SECONDS)
        note = turn_status.peek(request.session_id)
        if note and note != last_note:
            last_note = note
            yield _sse("status", {"search_status_message": note})
    try:
        response = future.result()
    except HTTPException as exc:
        # The response has already begun, so the status code is spent: the client reads the
        # failure off the event instead.
        yield _sse("error", {"status_code": exc.status_code, "detail": str(exc.detail)})
        return
    except Exception as exc:  # noqa: BLE001 - a stream must end with an event, never a traceback
        logger.exception("Streaming chat turn failed for session %s", request.session_id)
        yield _sse("error", {"status_code": 500, "detail": str(exc)})
        return

    body = response.model_dump()
    note = (body.get("search_status_message") or "").strip()
    # chat() clears the live status on its way out, so a search that finished between two
    # polls was never forwarded above. The response still carries it.
    if note and note != last_note:
        yield _sse("status", {"search_status_message": note})

    chunks = split_reply_into_chunks(body.get("assistant_text") or "")
    body["chunks"] = chunks
    delta_delay = max(0.0, settings.chat_stream_delta_seconds)
    chunk_pause = max(0.0, settings.chat_stream_chunk_pause_seconds)
    for index, chunk in enumerate(chunks):
        yield _sse("chunk_start", {"index": index, "total": len(chunks)})
        for delta in _typing_deltas(chunk, settings.chat_stream_words_per_delta):
            yield _sse("delta", {"index": index, "text": delta})
            if delta_delay:
                time.sleep(delta_delay)
        yield _sse("chunk_end", {"index": index, "text": chunk})
        if chunk_pause and index < len(chunks) - 1:
            time.sleep(chunk_pause)
    yield _sse("done", body)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """POST /chat, delivered as Server-Sent Events. Same request body, same turn."""
    if not settings.chat_stream_enabled:
        raise HTTPException(status_code=404, detail="Streaming is disabled")
    # Both ids are validated HERE rather than inside the generator: once the first byte is
    # written the status code can no longer be changed, so a malformed id must still 422.
    _require_session_id(request.session_id)
    turn_id = request.turn_id or str(uuid.uuid4())
    _require_uuid(turn_id, "turn_id")
    request = request.model_copy(update={"turn_id": turn_id})
    return StreamingResponse(
        _chat_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this an nginx in front of the API buffers the whole stream and the
            # customer sees nothing until the last event - the one failure mode that makes
            # streaming strictly worse than not streaming.
            "X-Accel-Buffering": "no",
        },
    )


def _log_turn(
    session_id: str,
    turn_id: str,
    started: float,
    usage: Any,
    user_message: str,
    response: ChatResponse | None = None,
    error: str | None = None,
) -> None:
    """Emit the compact JSONL record (M9 §4) and the human-readable reasoning
    block (session id, user message, LLM reasoning, tools fired, reply, and the
    full session state after the turn). Never fails a served turn."""
    state = _sessions.get(session_id) or {}
    turn = state.get("turn")
    turn_outcome = state.get("turn_outcome") or {}
    try:
        turn_log.log_turn(
            session_id=session_id,
            turn_id=turn_id,
            intent=getattr(turn, "intent", None),
            category=state.get("category"),
            latency_ms=(time.perf_counter() - started) * 1000,
            turn_outcome=turn_outcome,
            usage=usage,
            error=error,
        )
    except Exception:  # noqa: BLE001 - observability must not break the response
        logger.exception("turn logging failed for session %s", session_id)

    reply = turn_outcome.get("reply")
    conversation_log.log_conversation_turn(
        session_id=session_id,
        turn_id=turn_id,
        user_message=user_message,
        analysis=turn,
        assistant_text=response.assistant_text if response else None,
        cited_listing_urls=getattr(reply, "cited_listing_urls", None),
        listings_returned=len(response.listings) if response else 0,
        turn_outcome=turn_outcome,
        state=state,
        error=error,
    )


def _restore_session(session_id: str, pre_turn: dict | None) -> None:
    """Put the in-memory session back exactly as the discarded turn found it."""
    if pre_turn is None:
        _sessions.pop(session_id, None)
    else:
        _sessions[session_id] = pre_turn


def _run_turn(request: ChatRequest, turn_id: str) -> ChatResponse:
    if not conversation_store.persistence_enabled():
        # No transaction to roll back, so the in-memory state is restored by hand. The
        # copy is taken before the graph runs because the graph mutates it in place.
        pre_turn = copy.deepcopy(_sessions.get(request.session_id))
        body = _handle_chat_in_memory(request)
        if _turn_was_superseded():
            _restore_session(request.session_id, pre_turn)
            raise TurnSuperseded(request.session_id)
        return ChatResponse(**body)
    with conversation_store.durable_turn(request.session_id, turn_id, request.message) as (db_session, row, receipt):
        if receipt:
            return ChatResponse(**receipt.response)
        # Reload whenever the DATABASE is ahead of this process, not merely when this
        # process has never seen the session.
        #
        # The old check was `session_id not in _sessions`, which is only correct on one
        # instance. Run two, and turns alternate: instance A answers turn 1 and caches the
        # state, B answers turn 2 (cache empty, so it loads the snapshot correctly and
        # writes turn 2), then turn 3 lands back on A - which still HAS the session in
        # memory, skips the reload, and answers from state frozen before turn 2 existed.
        # The customer watches the bot forget an answer it already acknowledged.
        #
        # row.state_version is the fencing token: it is incremented on every committed
        # turn below, and stamped into the state so the two are comparable. row itself was
        # read AFTER durable_turn took the advisory lock, so it cannot be stale here.
        stored_version = int(getattr(row, "state_version", 0) or 0) if row else 0
        cached = _sessions.get(request.session_id)
        cached_version = int((cached or {}).get("persisted_state_version") or 0)
        if row and row.state_snapshot and (cached is None or cached_version < stored_version):
            if cached is not None:
                logger.info(
                    "session %s reloaded: this process was %d turn(s) behind",
                    request.session_id,
                    stored_version - cached_version,
                )
            _sessions[request.session_id] = from_snapshot(row.state_snapshot)
        lead_id = conversation_store.create_or_get_soft_lead(
            session_id=request.session_id,
            full_name=request.customer_full_name,
            email=request.customer_email,
            phone=request.customer_phone,
        )
        state = _get_session(request.session_id)
        if lead_id:
            state["lead_id"] = lead_id
        # Taken before the graph runs: the graph mutates the session dict in place, and
        # the database rollback below cannot undo that.
        pre_turn = copy.deepcopy(_sessions.get(request.session_id))
        body = _handle_chat_in_memory(request)
        if _turn_was_superseded():
            # Leaving durable_turn by exception rolls the transaction back, so this turn
            # leaves nothing behind: no snapshot, no receipt, and no queued email. The
            # outbox drain is called only after a successful commit, further down.
            _restore_session(request.session_id, pre_turn)
            logger.info("turn superseded by a newer message | session=%s", request.session_id)
            raise TurnSuperseded(request.session_id)
        # Persist snapshot + receipt only when a lead row backs the FK; without
        # one we cannot write a conversation/turn row, so degrade gracefully.
        lead_uuid = state.get("lead_id")
        if lead_uuid:
            sid = conversation_store.as_session_uuid(request.session_id)
            if row is None:
                row = ChatbotConversation(session_id=sid, lead_id=uuid.UUID(lead_uuid), conversation=[])
                db_session.add(row)
                # The turn and outbox rows below reference this session_id. Land the parent
                # first so their INSERTs can never race ahead of it inside one flush.
                db_session.flush()
            # Stamp the version this turn is about to become BEFORE snapshotting, so the
            # snapshot carries it and the comparison above works on the next turn -
            # whichever instance handles it.
            next_version = int(row.state_version or 0) + 1
            _get_session(request.session_id)["persisted_state_version"] = next_version
            snapshot = to_snapshot(_get_session(request.session_id))
            row.state_snapshot = copy.deepcopy(snapshot)
            row.state_schema_version = snapshot["state_schema_version"]
            row.conversation = conversation_store._merge_existing_feedback(
                row.conversation,
                conversation_store._messages_to_conversation(snapshot.get("messages", [])),
            )
            row.state_version = next_version
            db_session.add(ChatbotTurn(session_id=sid, turn_id=uuid.UUID(turn_id), request_message=request.message, response=body))
            # Queue gate-approved email events + upgrade the lead, all inside the
            # durable transaction (M7 step 3). The post-commit drain sends them.
            outcome = _get_session(request.session_id).get("turn_outcome", {}) or {}
            outbox_events = outcome.get("outbox_events") or []
            for event in outbox_events:
                conversation_store.enqueue_outbox_event(
                    db_session,
                    session_id=sid,
                    turn_id=uuid.UUID(turn_id),
                    event_key=event["event_key"],
                    event_type=event["event_type"],
                    payload=event["payload"],
                )
            if outbox_events:
                conversation_store.promote_lead_to_hard(request.session_id, session=db_session)
            if outcome.get("lead_item_of_interest"):
                conversation_store.update_lead_item_of_interest(
                    request.session_id, outcome["lead_item_of_interest"], session=db_session
                )
    # Background: a slow/failing Graph or SMTP send must not add latency to the
    # user-facing reply. The outbox is retryable by design (M9 fix — see
    # deliver_pending_outbox_async's docstring for the incident that prompted this).
    conversation_store.deliver_pending_outbox_async()
    return ChatResponse(**body)


@router.get("/session/{session_id}/turn-status", response_model=TurnStatusResponse)
def get_turn_status(session_id: str) -> TurnStatusResponse:
    """Live progress for the turn currently running, for the UI to poll while it waits.

    Deliberately does NOT take the session lock: /chat holds that lock for the whole turn, so
    waiting on it here would block until the very turn we are reporting on had finished. Reads
    a plain dict instead, which is why turn_status guards itself with its own lock.
    """
    return TurnStatusResponse(search_status_message=turn_status.peek(session_id))


@router.get("/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    restored = conversation_store.restore_session(session_id)
    messages = []
    for message in restored.get("messages", []) or []:
        item = dict(message)
        item["listings"] = None
        messages.append(item)
    return SessionResponse(
        exists=bool(restored.get("exists")),
        closed=bool(restored.get("closed", False)),
        messages=messages,
        sales_phase=restored.get("sales_phase", "main"),
        customer_full_name=restored.get("customer_full_name"),
        customer_email=restored.get("customer_email"),
        customer_phone=restored.get("customer_phone"),
    )


@router.get("/session/{session_id}/state", response_model=DebugStateResponse)
def get_session_debug_state(session_id: str) -> DebugStateResponse:
    """Raw in-memory session snapshot for the M5 scenario runner.

    Only registered behavior when DEBUG_STATE_ENDPOINT is enabled; otherwise 404s
    so this never leaks internal state shape in a normal deployment.
    """
    if not settings.debug_state_endpoint:
        raise HTTPException(status_code=404, detail="Not found")
    session = _sessions.get(session_id)
    if session is None:
        return DebugStateResponse(exists=False, state=None)
    snapshot = to_snapshot(session)
    # Surface the last turn's email decisions (transient turn_outcome, dropped from
    # the durable snapshot) so M7 scenarios can assert expect_emails_sent.
    outcome = session.get("turn_outcome") or {}
    snapshot["emails_sent"] = outcome.get("emails_sent", [])
    return DebugStateResponse(exists=True, state=snapshot)


@router.post("/session/reset")
def reset_session(request: ResetRequest) -> dict[str, str]:
    """Always 200. An unknown or malformed session id is a no-op (M8 §4).

    Only drops the in-memory copy so "New Conversation" starts clean. The durable row is
    never closed: a session id is a bookmark someone can put back in the URL bar at any
    time, and it must still load its full history whenever that happens.
    """
    clear_session(request.session_id)
    return {"status": "ok"}


def _invoke_graph(state: dict[str, Any]) -> dict[str, Any]:
    """Run the graph under a server-side time budget; contain every failure.

    Raised as GraphFailure so the caller can tell a broken turn apart from
    durable_turn's turn_id-conflict ValueError.

    The graph runs inside a *copy* of this request's context because
    ThreadPoolExecutor does not propagate contextvars into its workers. Without
    this, the LLM/embedding counters that `usage_scope()` installed here would be
    invisible to the nodes that do the recording, and every turn would log zero
    calls (M9 §3). The copied context shares the TurnUsage object, so the worker's
    mutations are visible to us after the run.
    """
    context = contextvars.copy_context()
    future = _GRAPH_EXECUTOR.submit(context.run, _get_graph().invoke, state)
    try:
        return future.result(timeout=settings.chat_timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise GraphFailure(f"graph run exceeded {settings.chat_timeout_seconds}s") from exc
    except Exception as exc:  # noqa: BLE001 - the frontend must never see a 500
        raise GraphFailure(str(exc)) from exc


def merge_frontend_shown_urls(state: dict[str, Any], urls: list[str] | None) -> None:
    """Union frontend-supplied shown URLs into backend state (dedupe source of truth).

    Runs on every turn in both persistence modes. shown_urls is per category: the frontend
    reports the WHOLE chat's URLs each time, so a URL already recorded under another
    category belongs to that bucket and must not be re-added here — after a category
    switch it was flooding the new category's freshly-swapped exclude list with the old
    category's history (seen live: a switch to Livestock inherited all 22 Dump URLs).
    """
    if not urls:
        return
    buckets = state.setdefault("shown_urls_by_category", {})
    category = state.get("category") or ""
    elsewhere = {url for cat, bucket in buckets.items() if cat != category for url in (bucket or [])}
    fresh = [url for url in urls if url and url not in elsewhere]
    if fresh:
        state["shown_urls"] = sorted(set(state.get("shown_urls", [])) | set(fresh))
        buckets[category] = sorted(set(buckets.get(category, [])) | set(fresh))


def _handle_chat_in_memory(request: ChatRequest) -> dict[str, Any]:
    state = _get_session(request.session_id)
    if request.customer_full_name and not state.get("customer_name"):
        state["customer_name"] = request.customer_full_name
    if request.customer_email and not state.get("customer_email"):
        state["customer_email"] = request.customer_email
    if request.customer_phone and not state.get("customer_phone"):
        state["customer_phone"] = request.customer_phone
    merge_frontend_shown_urls(state, request.already_shown_listing_urls)
    state.setdefault("messages", []).append(
        {
            "role": "user",
            "content": request.message,
            "listings": None,
            "user_feedback": None,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    result = _invoke_graph(state)
    _sessions[request.session_id] = result
    reply = result.get("turn_outcome", {}).get("reply")
    assistant_text = reply.assistant_text if reply else (result.get("messages") or [{}])[-1].get("content", "")
    listings = result.get("turn_outcome", {}).get("listings") or []
    if settings.show_only_llm_mentioned_cards and reply is not None:
        cited_urls = set(reply.cited_listing_urls or [])
        listings = [item for item in listings if item.get("url") in cited_urls]
    return {
        "assistant_text": assistant_text,
        # Present only on turns where the inventory search actually fired, so the UI can never
        # tell the customer we went to look on a turn that never searched.
        "search_status_message": result.get("turn_outcome", {}).get("search_status_message"),
        "listings": listings,
        "sales_phase": "main",
        "onboarding_api_messages": [],
        "customer_full_name": result.get("customer_name"),
        "customer_email": result.get("customer_email"),
        "customer_phone": result.get("customer_phone"),
        "main_prior_messages": None,
        "thinking_context": None,
    }
