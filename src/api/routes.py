from __future__ import annotations

import contextvars
import copy
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from src import conversation_log, conversation_store, tracing, turn_log
from src.api import readiness
from src.api.schemas import ChatRequest, ChatResponse, DebugStateResponse, ResetRequest, SessionResponse
from src.config import settings
from src.db_models import ChatbotConversation, ChatbotTurn
from src.graph.build import build_graph
from src.graph.state import _get_session, _sessions, clear_session, from_snapshot, session_lock, to_snapshot
from src.llm import usage as llm_usage
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter()
_GRAPH = None

# Bounds every graph run below app.py's 180 s client timeout (M8 §4).
_GRAPH_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="chat_graph")

ERROR_ASSISTANT_TEXT = (
    "Sorry — something went wrong on our end and I couldn't finish that thought. "
    "Could you send that again?"
)


class GraphFailure(Exception):
    """A graph run raised or timed out. Contained into a 200 apology, never a 500."""


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
    _require_uuid(request.session_id, "session_id")
    turn_id = request.turn_id or str(uuid.uuid4())
    _require_uuid(turn_id, "turn_id")

    if len(request.message) > settings.chat_max_message_chars:
        logger.warning(
            "Truncating oversized message for session %s (%d chars)", request.session_id, len(request.message)
        )
        request = request.model_copy(update={"message": request.message[: settings.chat_max_message_chars]})

    started = time.perf_counter()
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
        _log_turn(request.session_id, turn_id, started, usage, request.message, response=response)
        return response


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


def _run_turn(request: ChatRequest, turn_id: str) -> ChatResponse:
    if not conversation_store.persistence_enabled():
        return ChatResponse(**_handle_chat_in_memory(request))
    with conversation_store.durable_turn(request.session_id, turn_id, request.message) as (db_session, row, receipt):
        if receipt:
            return ChatResponse(**receipt.response)
        if request.session_id not in _sessions and row and row.state_snapshot:
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
        body = _handle_chat_in_memory(request)
        # Persist snapshot + receipt only when a lead row backs the FK; without
        # one we cannot write a conversation/turn row, so degrade gracefully.
        lead_uuid = state.get("lead_id")
        if lead_uuid:
            sid = uuid.UUID(request.session_id)
            if row is None:
                row = ChatbotConversation(session_id=sid, lead_id=uuid.UUID(lead_uuid), conversation=[])
                db_session.add(row)
                # The turn and outbox rows below reference this session_id. Land the parent
                # first so their INSERTs can never race ahead of it inside one flush.
                db_session.flush()
            snapshot = to_snapshot(_get_session(request.session_id))
            row.state_snapshot = copy.deepcopy(snapshot)
            row.state_schema_version = snapshot["state_schema_version"]
            row.conversation = conversation_store._merge_existing_feedback(
                row.conversation,
                conversation_store._messages_to_conversation(snapshot.get("messages", [])),
            )
            row.state_version = int(row.state_version or 0) + 1
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


def _handle_chat_in_memory(request: ChatRequest) -> dict[str, Any]:
    state = _get_session(request.session_id)
    if request.customer_full_name and not state.get("customer_name"):
        state["customer_name"] = request.customer_full_name
    if request.customer_email and not state.get("customer_email"):
        state["customer_email"] = request.customer_email
    if request.customer_phone and not state.get("customer_phone"):
        state["customer_phone"] = request.customer_phone
    # Union frontend-supplied shown URLs into backend state (dedupe source of
    # truth). Applies in BOTH persistence modes since this runs on every turn.
    # They land in the CURRENT category's bucket too — shown_urls is per category now.
    if request.already_shown_listing_urls:
        state["shown_urls"] = sorted(set(state.get("shown_urls", [])) | set(request.already_shown_listing_urls))
        buckets = state.setdefault("shown_urls_by_category", {})
        category = state.get("category") or ""
        buckets[category] = sorted(set(buckets.get(category, [])) | set(request.already_shown_listing_urls))
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
        "listings": listings,
        "sales_phase": "main",
        "onboarding_api_messages": [],
        "customer_full_name": result.get("customer_name"),
        "customer_email": result.get("customer_email"),
        "customer_phone": result.get("customer_phone"),
        "main_prior_messages": None,
        "thinking_context": None,
    }
