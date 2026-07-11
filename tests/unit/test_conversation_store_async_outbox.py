"""deliver_pending_outbox_async — the non-blocking outbox drain (M9 latency fix).

A slow/failing Graph or SMTP send must not add latency to the /chat response; the
outbox is retryable by design, so the drain moves to a background thread instead
of running inline before the response returns.
"""
from __future__ import annotations

from src import conversation_store as store


def test_noop_when_persistence_disabled(monkeypatch):
    monkeypatch.setattr(store, "persistence_enabled", lambda: False)
    submitted = []
    monkeypatch.setattr(store._pool, "submit", lambda *a, **kw: submitted.append((a, kw)))

    store.deliver_pending_outbox_async()

    assert submitted == []  # nothing queued -- persistence off means no outbox to drain


def test_submits_deliver_pending_outbox_to_the_background_pool(monkeypatch):
    monkeypatch.setattr(store, "persistence_enabled", lambda: True)
    submitted = []
    monkeypatch.setattr(store._pool, "submit", lambda func, *a, **kw: submitted.append((func, a, kw)))

    store.deliver_pending_outbox_async(limit=7)

    assert len(submitted) == 1
    func, args, kwargs = submitted[0]
    assert func is store.deliver_pending_outbox
    assert args == (7,)


def test_does_not_call_deliver_pending_outbox_synchronously(monkeypatch):
    """The whole point: the calling thread must return before the drain runs."""
    monkeypatch.setattr(store, "persistence_enabled", lambda: True)
    called = []
    monkeypatch.setattr(store, "deliver_pending_outbox", lambda limit=10: called.append(limit))
    # Run the pool inline via a fake submit so we can assert ordering deterministically.
    monkeypatch.setattr(store._pool, "submit", lambda func, *a, **kw: called.append("submitted-not-called"))

    store.deliver_pending_outbox_async()

    # deliver_pending_outbox itself was never invoked directly by the async wrapper --
    # only handed to the pool.
    assert called == ["submitted-not-called"]
