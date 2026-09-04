"""Messages sent in a burst become one answer, and an interrupted answer is thrown away.

A customer sends one thought as several messages - "I need a trailer", then "20ft", then
"for hay" - and often sends the next while we are still writing the reply to the last.
Answering each on its own gives them replies that ask what they already told us.

So: every pending message is answered as ONE combined turn, and if more arrive mid-turn
the reply is discarded and the turn rerun with those folded in, up to _MAX_REGENERATIONS.

The database is stood in for here. What is under test is the dispatch algorithm - what
gets combined, who takes the lock, when a turn is discarded and how often - and that is
the same logic whether the shared queue is Postgres or a dict. The real table only has to
supply what the fake does: a unique constraint on the message id, and one lock holder at
a time.
"""
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from src.api import messenger, routes
from src.config import Settings

BASE = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)


class SharedQueue:
    """Stands in for chatbot_inbound_messages plus its advisory drain lock."""

    def __init__(self):
        self.rows = []
        self.seen = set()
        # ONE LOCK PER CUSTOMER, mirroring the real advisory lock, whose key includes the
        # session id. A single shared lock would make one customer's turn block everyone
        # else's - and would quietly make the "different customers" test prove nothing.
        self.locks = {}
        self._guard = threading.Lock()
        self.answered = []          # the text of each turn that was actually run
        self.delivered = []         # the text of each reply that reached the customer

    def record(self, *, session_id, external_id, body, sent_at, channel="messenger"):
        with self._guard:
            if (channel, external_id) in self.seen:
                return False  # the unique constraint
            self.seen.add((channel, external_id))
            self.rows.append({
                "message_id": f"row-{len(self.rows)}",
                "session_id": session_id,
                "external_id": external_id,
                "body": body,
                "sent_at": sent_at,
                "status": "pending",
            })
            return True

    @contextmanager
    def drain_lock(self, session_id):
        with self._guard:
            lock = self.locks.setdefault(session_id, threading.Lock())
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    def _pending(self, session_id):
        return sorted(
            (r for r in self.rows if r["session_id"] == session_id and r["status"] == "pending"),
            key=lambda r: r["sent_at"],
        )

    def batch(self, session_id, channel="messenger"):
        with self._guard:
            return [dict(r) for r in self._pending(session_id)]

    def beyond(self, session_id, known_ids, channel="messenger"):
        with self._guard:
            return any(r["message_id"] not in set(known_ids) for r in self._pending(session_id))

    def mark_answered(self, message_ids, *, turn_id=None, error=None):
        with self._guard:
            wanted = set(message_ids)
            for row in self.rows:
                if row["message_id"] in wanted:
                    row["status"] = "done"
                    row["last_error"] = error

    def has_pending(self, session_id, channel="messenger"):
        with self._guard:
            return bool(self._pending(session_id))


@pytest.fixture
def shared(monkeypatch):
    queue = SharedQueue()
    store = messenger.conversation_store
    monkeypatch.setattr(store, "persistence_enabled", lambda: True)
    monkeypatch.setattr(store, "record_inbound_message", queue.record)
    monkeypatch.setattr(store, "inbound_drain_lock", queue.drain_lock)
    monkeypatch.setattr(store, "pending_inbound_batch", queue.batch)
    monkeypatch.setattr(store, "has_inbound_beyond", queue.beyond)
    monkeypatch.setattr(store, "mark_inbound_answered", queue.mark_answered)
    monkeypatch.setattr(store, "has_pending_inbound", queue.has_pending)
    monkeypatch.setattr(store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger, "settings", Settings(
        messenger_enabled=True, messenger_app_secret="s", messenger_page_access_token="t",
        messenger_chunk_pause_seconds=0.0))
    monkeypatch.setattr(messenger, "_send", lambda payload: None)
    messenger._seen_mids.clear()

    def fake_chat(request):
        """Stands in for a turn, including the check the real one makes before committing.

        Calling routes._turn_was_superseded() rather than faking the decision means the
        contextvar the webhook sets through abandon_turn_if is genuinely exercised.
        """
        queue.answered.append(request.message)
        if routes._turn_was_superseded():
            raise routes.TurnSuperseded(request.session_id)
        queue.delivered.append(request.message)
        return type("R", (), {"assistant_text": f"re: {request.message}"})()

    monkeypatch.setattr("src.api.routes.chat", fake_chat)
    yield queue
    messenger._seen_mids.clear()


def _deliver(psid, text, mid, seconds):
    """One webhook delivery, with the timestamp Facebook would have stamped on it."""
    messenger._enqueue(psid, text, mid, BASE + timedelta(seconds=seconds))


def _wait_until(condition, timeout=5.0):
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        if condition():
            return
        _time.sleep(0.01)
    raise AssertionError("condition not met within the timeout")


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------


def test_messages_waiting_together_are_answered_as_one_turn(shared):
    """Two messages, ONE reply that read both - not two replies talking past each other."""
    shared.record(session_id="P", external_id="m.1", body="I need a trailer", sent_at=BASE)
    shared.record(session_id="P", external_id="m.2", body="20ft", sent_at=BASE + timedelta(seconds=1))
    messenger._drain_shared("P")
    assert shared.delivered == ["I need a trailer\n20ft"]


def test_the_combined_message_is_ordered_by_when_the_customer_sent_it(shared):
    """Not by which delivery arrived first - a retry can arrive after a later message."""
    shared.record(session_id="P", external_id="m.2", body="second", sent_at=BASE + timedelta(seconds=5))
    shared.record(session_id="P", external_id="m.1", body="first", sent_at=BASE)
    messenger._drain_shared("P")
    assert shared.delivered == ["first\nsecond"]


def test_every_message_in_the_batch_is_marked_answered(shared):
    shared.record(session_id="P", external_id="m.1", body="a", sent_at=BASE)
    shared.record(session_id="P", external_id="m.2", body="b", sent_at=BASE + timedelta(seconds=1))
    messenger._drain_shared("P")
    assert all(row["status"] == "done" for row in shared.rows)


# ---------------------------------------------------------------------------
# Discard and regenerate
# ---------------------------------------------------------------------------


def test_a_reply_is_discarded_when_the_customer_sends_more_mid_turn(shared):
    """The heart of it. The first answer is never delivered; the rerun sees both messages."""
    shared.record(session_id="P", external_id="m.1", body="I need a trailer", sent_at=BASE)

    original = routes.chat

    def interrupting_chat(request):
        if len(shared.answered) == 0:
            # The customer types again while this turn is being prepared.
            shared.record(session_id="P", external_id="m.2", body="20ft",
                          sent_at=BASE + timedelta(seconds=1))
        return original(request)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.routes.chat", interrupting_chat)
        messenger._drain_shared("P")

    assert shared.answered == ["I need a trailer", "I need a trailer\n20ft"]
    assert shared.delivered == ["I need a trailer\n20ft"]  # the first reply never went out


def test_several_interruptions_are_all_absorbed(shared):
    """Each rerun folds in whatever arrived during the last one."""
    shared.record(session_id="P", external_id="m.1", body="one", sent_at=BASE)
    original = routes.chat

    def interrupting_chat(request):
        count = len(shared.answered)
        if count < 2:
            shared.record(session_id="P", external_id=f"m.{count + 2}", body=str(count + 2),
                          sent_at=BASE + timedelta(seconds=count + 1))
        return original(request)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.routes.chat", interrupting_chat)
        messenger._drain_shared("P")
    assert shared.delivered == ["one\n2\n3"]


def test_a_customer_who_never_stops_typing_still_gets_an_answer(shared):
    """After _MAX_REGENERATIONS the abandon check is switched off, so a reply gets out.

    Without the cap, someone typing continuously would be regenerated against forever and
    never hear back at all.
    """
    shared.record(session_id="P", external_id="m.1", body="one", sent_at=BASE)
    original = routes.chat

    def always_interrupting_chat(request):
        count = len(shared.answered)
        shared.record(session_id="P", external_id=f"m.{count + 2}", body="more",
                      sent_at=BASE + timedelta(seconds=count + 1))
        return original(request)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.routes.chat", always_interrupting_chat)
        messenger._answer_pending_batch("P")

    assert len(shared.answered) == messenger._MAX_REGENERATIONS + 1
    assert len(shared.delivered) == 1  # the last attempt is allowed through


# ---------------------------------------------------------------------------
# Locking and hand-off
# ---------------------------------------------------------------------------


def test_only_one_instance_drains_a_customer(shared):
    """Two instances, one lock: the loser returns instead of answering in parallel."""
    entered, release = threading.Event(), threading.Event()
    original = routes.chat

    def slow_chat(request):
        entered.set()
        release.wait(timeout=3)
        return original(request)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.api.routes.chat", slow_chat)
        _deliver("P", "first", "m.1", 0)
        entered.wait(timeout=3)
        _deliver("P", "second", "m.2", 1)   # a second instance receives a message
        assert shared.delivered == []       # nothing answered in parallel
        release.set()
        _wait_until(lambda: len(shared.delivered) >= 1)


def test_a_message_recorded_during_the_handoff_gap_is_not_orphaned(shared):
    """Instance B records a message and fails the lock in the instant before A lets go.

    The re-check after releasing is what stops that message sitting unanswered forever.
    """
    shared.record(session_id="P", external_id="m.1", body="first", sent_at=BASE)
    messenger._drain_shared("P")
    shared.record(session_id="P", external_id="m.2", body="orphan", sent_at=BASE + timedelta(seconds=1))
    messenger._drain_shared("P")
    assert shared.delivered == ["first", "orphan"]


def test_different_customers_are_not_serialised_against_each_other(shared):
    _deliver("P-A", "a1", "m.a1", 0)
    _deliver("P-B", "b1", "m.b1", 0)
    _wait_until(lambda: len(shared.delivered) == 2)
    assert sorted(shared.delivered) == ["a1", "b1"]


# ---------------------------------------------------------------------------
# Duplicates and failures
# ---------------------------------------------------------------------------


def test_a_retry_is_rejected_by_the_queue_before_any_work_happens(shared):
    _deliver("P", "I have a complaint", "m.1", 0)
    _wait_until(lambda: shared.delivered == ["I have a complaint"])
    _deliver("P", "I have a complaint", "m.1", 0)  # Meta redelivers
    assert shared.delivered == ["I have a complaint"]
    assert len([r for r in shared.rows if r["external_id"] == "m.1"]) == 1


def test_a_failing_batch_does_not_block_the_messages_behind_it(shared, monkeypatch):
    """A batch that cannot be answered is marked done WITH its error. Left pending it
    would sit at the head of the queue and silence the customer permanently."""
    def exploding_chat(request):
        raise RuntimeError("graph blew up")

    monkeypatch.setattr("src.api.routes.chat", exploding_chat)
    shared.record(session_id="P", external_id="m.1", body="boom", sent_at=BASE)
    messenger._answer_pending_batch("P")
    assert all(row["status"] == "done" for row in shared.rows)
    assert "graph blew up" in (shared.rows[0].get("last_error") or "")


# ---------------------------------------------------------------------------
# Timestamps and the single-instance fallback
# ---------------------------------------------------------------------------


def test_facebooks_timestamp_is_used_as_the_ordering_key():
    assert messenger._sent_at({"timestamp": 1788523200000}) == \
        datetime.fromtimestamp(1788523200, tz=timezone.utc)


def test_an_event_with_no_timestamp_falls_back_to_now_rather_than_being_dropped():
    before = datetime.now(timezone.utc)
    assert messenger._sent_at({}) >= before


def test_a_nonsense_timestamp_does_not_raise():
    assert isinstance(messenger._sent_at({"timestamp": "not-a-number"}), datetime)


def test_persistence_off_falls_back_to_the_in_process_queue(monkeypatch):
    """The Streamlit-only and local-test deployments have no shared queue."""
    answered = []
    monkeypatch.setattr(messenger.conversation_store, "persistence_enabled", lambda: False)
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger, "_send", lambda p: None)
    monkeypatch.setattr(messenger, "settings", Settings(
        messenger_enabled=True, messenger_app_secret="s", messenger_page_access_token="t",
        messenger_chunk_pause_seconds=0.0))
    monkeypatch.setattr("src.api.routes.chat",
                        lambda r: answered.append(r.message) or type("R", (), {"assistant_text": "ok"})())
    messenger._seen_mids.clear()
    messenger._enqueue("P", "hello", "m.1", BASE)
    _wait_until(lambda: answered == ["hello"])
    messenger._seen_mids.clear()
