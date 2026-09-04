"""Facebook Messenger webhook.

Meta POSTs a customer's message here; we run the ordinary chatbot turn and push the
reply back through the Send API. Three things shape every decision in this file.

1. META RETRIES, SO NOTHING MAY RUN TWICE. Meta expects 200 within ~20 seconds and
   redelivers when it does not get one. A chatbot turn takes far longer than that
   (settings.chat_timeout_seconds is 150 s), so we acknowledge IMMEDIATELY and do the
   work in the background. That makes retries likely rather than rare, and every retry
   carries the same message id (`mid`), which the queue's unique constraint rejects.

2. A CUSTOMER SENDS ONE THOUGHT AS SEVERAL MESSAGES. "I need a trailer", then "20ft",
   then "for hay" - often while we are still answering the first. Answering each on its
   own produces replies that ask what the next message already said, so instead every
   pending message is answered as ONE combined turn, and a turn interrupted by a new
   message is DISCARDED and rerun with that message folded in. See _answer_pending_batch.

   That also settles ordering across instances, which a per-process queue cannot: behind
   a load balancer two deliveries can land on two instances, so the queue lives in
   chatbot_inbound_messages and only the instance holding that customer's advisory lock
   answers it, oldest first by the timestamp FACEBOOK stamped on the message.

3. THE PSID IS THE SESSION. It goes in unchanged: conversation_store.as_session_uuid
   maps it onto the UUID the chatbot_* tables are keyed by, and chatbot_leads.psid keeps
   the raw value so a lead can still be traced back to the person who sent it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import APIRouter, HTTPException, Request, Response

from src import conversation_store, turn_status
from src.api.schemas import ChatRequest
from src.config import settings
from src.domain.reply_chunks import split_reply_into_chunks

logger = logging.getLogger(__name__)

router = APIRouter()

_GRAPH_URL = "https://graph.facebook.com/{version}/me/messages"

# Messenger's own hard limit on one message. Longer text is split rather than rejected,
# because silently dropping the tail of a reply is worse than an extra bubble.
_MESSENGER_TEXT_LIMIT = 2000

# How many times a reply may be thrown away because the customer sent another message
# while it was being written. After this, the next answer is allowed out regardless -
# otherwise someone typing continuously would never be answered at all.
_MAX_REGENERATIONS = 3

# Messages sent in a burst are joined with a newline, so the stored conversation holds
# one customer message for the turn rather than a fragment per delivery.
_NEWLINE = "\n"


def messenger_enabled() -> bool:
    """Both routes 404 unless this is the Messenger deployment AND it is configured.

    A page token with no app secret would serve a bot that anyone could speak through,
    so the secret is part of "configured", not an optional extra.
    """
    return bool(
        settings.messenger_enabled
        and settings.messenger_app_secret
        and settings.messenger_page_access_token
    )


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------
# Meta's `mid` is unique per message and constant across retries, so it is the natural
# idempotency key. It is not a UUID, so it goes through the same uuid5 mapping the PSID
# does before it can be a turn_id.

_seen_mids: OrderedDict[str, float] = OrderedDict()
_seen_lock = threading.Lock()


def _claim_mid(mid: str) -> bool:
    """Record `mid` as being handled. False if some other delivery got there first.

    Claiming happens BEFORE the work starts, not after, so a retry that arrives while
    the first copy is still running is rejected too - that is the common case, since a
    retry is triggered by exactly the slowness that means we are still busy.
    """
    with _seen_lock:
        if mid in _seen_mids:
            return False
        _seen_mids[mid] = time.time()
        while len(_seen_mids) > max(1, settings.messenger_seen_mid_cache_size):
            _seen_mids.popitem(last=False)
        return True


def _turn_id_for(mid: str) -> str:
    """Meta's message id -> the turn_id the chatbot deduplicates on."""
    return str(uuid.uuid5(conversation_store.SESSION_ID_NAMESPACE, f"messenger:{mid}"))


# ---------------------------------------------------------------------------
# Ordered dispatch
# ---------------------------------------------------------------------------
# Ordering is SHARED, not per-process. Every message is recorded in
# chatbot_inbound_messages first; whichever instance wins that customer's advisory
# drain lock then answers their queue oldest first, by the timestamp Facebook stamped
# on the message. An instance that cannot get the lock does nothing and returns - the
# holder drains until the queue is empty, so it picks up what we just recorded.
#
# With persistence off there is no shared queue and no lock, so this degrades to the
# in-process deque below. That is correct for a single instance and is what the
# Streamlit-only and local-test deployments run.

_queues: dict[str, deque] = {}
_running: set[str] = set()
_dispatch_lock = threading.Lock()


def _enqueue(psid: str, text: str, mid: str, sent_at: datetime | None = None) -> None:
    """Record the message, then try to become this customer's drain worker."""
    if not conversation_store.persistence_enabled():
        _enqueue_in_process(psid, text, mid)
        return
    if not conversation_store.record_inbound_message(
        session_id=psid,
        external_id=mid,
        body=text,
        sent_at=sent_at or datetime.now(timezone.utc),
    ):
        # The unique constraint rejected it: a retry of something already queued or
        # answered. Durable, so it holds across instances and across restarts.
        logger.info("messenger duplicate suppressed at the queue | psid=%s mid=%s", psid, mid)
        return
    threading.Thread(
        target=_drain_shared, args=(psid,), daemon=True, name=f"messenger:{psid[:12]}"
    ).start()


def _drain_shared(psid: str) -> None:
    """Answer this customer's queued messages, if no other instance already is.

    The re-check after releasing the lock closes the one gap in the hand-off: another
    instance may have recorded a message and failed to take the lock while we were
    finishing, in the moment before we let go.
    """
    while True:
        worked = False
        with conversation_store.inbound_drain_lock(psid) as acquired:
            if not acquired:
                logger.debug("messenger drain already running elsewhere | psid=%s", psid)
                return
            while conversation_store.pending_inbound_batch(psid):
                worked = True
                _answer_pending_batch(psid)
        if not worked or not conversation_store.has_pending_inbound(psid):
            return


def _answer_pending_batch(psid: str) -> None:
    """Answer everything this customer has waiting, as ONE turn.

    A customer often sends their thought in pieces - "I need a trailer", then "20ft",
    then "for hauling hay". Answering each separately produces replies that ask what the
    next message already said. So the pending messages are concatenated and answered
    together, and if MORE arrive while that turn is running, the turn is thrown away and
    rerun with those included too.

    Discarding is cheap and, more importantly, complete: the turn runs inside one
    database transaction, so abandoning it erases the state it wrote, the receipt, and
    the alert emails it queued. The customer never sees the discarded reply and the team
    never gets an email about it.

    After _MAX_REGENERATIONS the abandon check is switched off and the next answer is
    allowed out, so someone typing continuously still gets a reply.
    """
    from src.api.routes import TurnSuperseded, abandon_turn_if

    for attempt in range(_MAX_REGENERATIONS + 1):
        batch = conversation_store.pending_inbound_batch(psid)
        if not batch:
            return
        known_ids = [message["message_id"] for message in batch]
        # Concatenated, so the conversation stored against this customer holds ONE
        # customer message for the turn rather than a fragment per delivery.
        text = _NEWLINE.join(message["body"] for message in batch).strip()
        turn_id = _turn_id_for("|".join(message["external_id"] for message in batch))
        last = attempt == _MAX_REGENERATIONS
        if last:
            logger.info(
                "messenger answering after %d interruptions | psid=%s", attempt, psid)

        def customer_said_more() -> bool:
            return conversation_store.has_inbound_beyond(psid, known_ids)

        error = None
        try:
            with abandon_turn_if(None if last else customer_said_more):
                _handle_message(psid, text, turn_id)
        except TurnSuperseded:
            logger.info(
                "messenger reply discarded, customer sent more | psid=%s attempt=%d",
                psid, attempt + 1)
            continue  # rerun with the new messages folded in
        except Exception as exc:  # noqa: BLE001 - must not stall this customer's queue
            logger.exception("messenger turn failed | psid=%s", psid)
            error = f"{type(exc).__name__}: {exc}"
        conversation_store.mark_inbound_answered(known_ids, turn_id=turn_id, error=error)
        return


def _enqueue_in_process(psid: str, text: str, mid: str) -> None:
    """Single-instance fallback: one queue, one worker per customer, in memory."""
    with _dispatch_lock:
        _queues.setdefault(psid, deque()).append((text, mid))
        if psid in _running:
            return
        _running.add(psid)
    threading.Thread(target=_drain, args=(psid,), daemon=True, name=f"messenger:{psid[:12]}").start()


def _drain(psid: str) -> None:
    while True:
        with _dispatch_lock:
            queue = _queues.get(psid)
            if not queue:
                _queues.pop(psid, None)
                _running.discard(psid)
                return
            text, mid = queue.popleft()
        if not _claim_mid(mid):
            continue  # a retry Meta redelivered to this same process
        try:
            _handle_message(psid, text, _turn_id_for(mid))
        except Exception:  # noqa: BLE001 - one bad message must not stall the queue
            logger.exception("messenger turn failed | psid=%s mid=%s", psid, mid)


# ---------------------------------------------------------------------------
# Keeping the customer company while the turn runs
# ---------------------------------------------------------------------------


# The search line already shown to each customer, so a turn that is discarded and rerun
# does not say the same thing twice. Cleared once the real reply has been delivered.
_status_lines: dict[str, str] = {}
_status_lock = threading.Lock()


def _claim_status_line(psid: str, line: str) -> bool:
    """True if this line has not already been shown to this customer."""
    with _status_lock:
        if _status_lines.get(psid) == line:
            return False
        _status_lines[psid] = line
        return True


def _clear_status_line(psid: str) -> None:
    with _status_lock:
        _status_lines.pop(psid, None)


class _TurnKeepAlive:
    """Everything the customer sees BETWEEN their message and the answer.

    A turn takes 25-30 seconds. Left alone that is a long silence, and it looks broken
    for two reasons: Messenger dismisses a typing indicator after roughly 20 seconds, so
    typing visibly stops; and until the answer lands there is nothing else at all.

    So a background thread does two things while the turn runs:

      * re-sends typing_on, so the indicator never lapses;
      * watches for the search line the graph publishes the moment it starts looking
        ("Let me see what we have on the lot for you.") and sends it straight away -
        the same line the web UI shows, from the same place, so the two channels say the
        same thing.

    Everything here is cosmetic and best effort. It runs on its own thread, its failures
    are swallowed, and it can neither delay nor fail the turn. Only turns that actually
    search publish a status line; on a turn that just asks a question there is nothing to
    send, and the typing refresh carries it alone.
    """

    def __init__(self, psid: str):
        self._psid = psid
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self.status_sent: str | None = None   # exposed for tests and logging

    def start(self) -> None:
        interval = max(0.0, settings.messenger_typing_refresh_seconds)
        if interval <= 0 and not settings.messenger_send_search_status:
            return  # both features off: no thread at all
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"messenger-alive:{self._psid[:12]}")
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._thread is not None:
            # Bounded: the loop only ever waits on the event, so it returns promptly.
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        interval = max(0.0, settings.messenger_typing_refresh_seconds)
        # The status line is polled far more often than typing is refreshed, because its
        # value is being early - a line delivered 10 seconds late has missed the point.
        # Capped by the refresh interval as well, or a shorter interval than the tick
        # could never come due.
        tick = min(0.5, interval) if interval > 0 else 0.5
        elapsed_since_typing = 0.0
        while not self._done.wait(tick):
            try:
                if self.status_sent is None and settings.messenger_send_search_status:
                    self._send_status_if_ready()
                elapsed_since_typing += tick
                if interval > 0 and elapsed_since_typing >= interval:
                    elapsed_since_typing = 0.0
                    _send_sender_action(self._psid, "typing_on")
            except Exception:  # noqa: BLE001 - cosmetic; never let it touch the turn
                logger.exception("messenger keep-alive failed | psid=%s", self._psid)

    def _send_status_if_ready(self) -> None:
        line = (turn_status.peek(self._psid) or "").strip()
        if not line:
            return
        self.status_sent = line
        if not _claim_status_line(self._psid, line):
            # An attempt that was later discarded already sent this exact line. The rerun
            # searches again and publishes it again, but the customer has read it once
            # and does not need it twice.
            return
        _send_text(self._psid, line)
        logger.info("messenger sent the search line early | psid=%s", self._psid)


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------


def _handle_message(psid: str, text: str, turn_id: str) -> None:
    """Run one turn and send the reply. The caller owns deduplication.

    Deliberately NOT deduplicating here: the shared queue does it durably via its unique
    constraint on the message id, and the in-process fallback does it with _claim_mid.
    Checking again on turn_id would be wrong for a regenerated turn, whose id changes
    every time the batch it covers changes.
    """
    if conversation_store.turn_already_handled(psid, turn_id):
        logger.info("messenger turn already answered, not resending | psid=%s", psid)
        return

    # Imported here, not at module scope: routes builds the graph and its executors at
    # import time, and a top-level import would make these two modules circular.
    from src.api.routes import chat

    _send_sender_action(psid, "mark_seen")
    _send_sender_action(psid, "typing_on")
    keep_alive = _TurnKeepAlive(psid)
    keep_alive.start()
    try:
        response = chat(ChatRequest(session_id=psid, turn_id=turn_id, message=text))
        reply = response.assistant_text
    except HTTPException as exc:
        # 409 means this turn_id was already answered with a different message, which can
        # only happen if Meta reused a mid. Replying again would double-send, so stop.
        if exc.status_code == 409:
            logger.warning("messenger turn_id conflict, not resending | psid=%s", psid)
            return
        raise
    finally:
        keep_alive.stop()
        _send_sender_action(psid, "typing_off")

    # One reply, several bubbles: the intro, one per trailer, then the closing question -
    # the same split the web UI streams. reply_chunks guarantees a non-empty list for
    # non-empty text, so the `or [reply]` only covers a reply that was empty anyway.
    chunks = split_reply_into_chunks(reply) or [reply]
    for index, chunk in enumerate(chunks):
        if index:
            time.sleep(max(0.0, settings.messenger_chunk_pause_seconds))
        for part in _split_for_messenger(chunk):
            _send_text(psid, part)
    # The answer has landed, so the next question starts with a clean slate.
    _clear_status_line(psid)


def _split_for_messenger(text: str) -> list[str]:
    """Keep every bubble inside Messenger's 2000-character limit, splitting on lines."""
    body = str(text or "").strip()
    if len(body) <= _MESSENGER_TEXT_LIMIT:
        return [body] if body else []
    parts: list[str] = []
    current = ""
    for line in body.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= _MESSENGER_TEXT_LIMIT:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        # A single line over the limit is rare (no listing card is), so a hard slice is
        # an acceptable last resort rather than something worth a word-wrapper.
        while len(line) > _MESSENGER_TEXT_LIMIT:
            parts.append(line[:_MESSENGER_TEXT_LIMIT])
            line = line[_MESSENGER_TEXT_LIMIT:]
        current = line
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# Send API
# ---------------------------------------------------------------------------


def _send(payload: dict) -> None:
    url = _GRAPH_URL.format(version=settings.messenger_graph_api_version)
    try:
        response = requests.post(
            url,
            params={"access_token": settings.messenger_page_access_token},
            json=payload,
            timeout=settings.messenger_send_timeout_seconds,
        )
    except requests.RequestException:
        logger.exception("messenger send failed (network)")
        return
    if response.status_code >= 400:
        # The token and the recipient are the usual culprits; log the body, never the token.
        logger.error("messenger send failed | status=%s body=%s", response.status_code, response.text[:500])


def _send_text(psid: str, text: str) -> None:
    if not text.strip():
        return
    _send({
        "recipient": {"id": psid},
        "messaging_type": "RESPONSE",
        "message": {"text": text},
    })


def _send_sender_action(psid: str, action: str) -> None:
    """mark_seen / typing_on / typing_off. Cosmetic, so a failure is never fatal."""
    _send({"recipient": {"id": psid}, "sender_action": action})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _signature_is_valid(body: bytes, header: str | None) -> bool:
    """Verify X-Hub-Signature-256 over the RAW body.

    Must be the raw bytes, not a re-serialised dict: any difference in key order or
    spacing changes the digest. compare_digest, not ==, so the check does not leak how
    much of a forged signature was correct through its timing.
    """
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.messenger_app_secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1].strip())


@router.get("/webhooks/messenger")
def verify_webhook(request: Request) -> Response:
    """Meta's one-time subscription handshake: echo hub.challenge if the token matches."""
    if not messenger_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == settings.messenger_verify_token:
        logger.info("messenger webhook verified")
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    logger.warning("messenger webhook verification rejected")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhooks/messenger")
async def receive_webhook(request: Request) -> Response:
    """Acknowledge in milliseconds, answer in the background.

    Returning 200 only after the turn finished would blow Meta's ~20 s budget on every
    single message and put us in a permanent retry storm, so the ack is unconditional
    once the signature checks out.
    """
    if not messenger_enabled():
        raise HTTPException(status_code=404, detail="Not found")

    body = await request.body()
    if not _signature_is_valid(body, request.headers.get("X-Hub-Signature-256")):
        logger.warning("messenger webhook rejected: bad signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        logger.warning("messenger webhook rejected: body was not JSON")
        return Response(status_code=200)

    for psid, text, mid, sent_at in _extract_messages(payload):
        _enqueue(psid, text, mid, sent_at)
    return Response(status_code=200)


def _sent_at(event: dict) -> datetime:
    """When the CUSTOMER pressed send, per Meta's millisecond timestamp.

    This is the ordering key, and it has to come from Facebook: our own clock only knows
    when a delivery reached us, which for a retry is long after the message it repeats.
    Falls back to now for an event carrying no timestamp, which keeps such a message at
    the back of the queue rather than dropping it.
    """
    raw = event.get("timestamp")
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.now(timezone.utc)


def _extract_messages(payload: dict) -> list:
    """Pull (psid, text, mid, sent_at) out of a webhook body, dropping what we cannot answer.

    Deliberately ignored: `is_echo` (our OWN outgoing message, echoed back - answering it
    would talk to ourselves forever), delivery/read receipts, and attachment-only messages
    with no text. `postback` is included because the Get Started button arrives as one.
    """
    found: list = []
    if payload.get("object") != "page":
        return found
    for entry in payload.get("entry") or []:
        for event in entry.get("messaging") or []:
            psid = ((event.get("sender") or {}).get("id") or "").strip()
            if not psid:
                continue
            message = event.get("message") or {}
            if message.get("is_echo"):
                continue
            text = (message.get("text") or "").strip()
            mid = (message.get("mid") or "").strip()
            if not text:
                postback = event.get("postback") or {}
                text = (postback.get("payload") or postback.get("title") or "").strip()
                # A postback carries no mid, so key the dedupe on what does identify it.
                mid = mid or f"postback:{psid}:{event.get('timestamp')}"
            if not text or not mid:
                continue
            found.append((psid, text, mid, _sent_at(event)))
    return found
