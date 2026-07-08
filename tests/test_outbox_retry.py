"""F1 — outbox claim/retry policy (pure logic, no DB, no live send)."""
from datetime import datetime, timedelta, timezone

from src.conversation_store import (
    _OUTBOX_LEASE_SECONDS,
    _OUTBOX_MAX_ATTEMPTS,
    _outbox_backoff_seconds,
    _outbox_claim_decision,
)

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _ago(seconds: float) -> datetime:
    return NOW - timedelta(seconds=seconds)


def test_pending_always_claims():
    assert _outbox_claim_decision("pending", 0, None, NOW) == "claim"


def test_failed_null_claim_time_claims_immediately():
    # A row marked failed without a claimed_at stamp is eligible now.
    assert _outbox_claim_decision("failed", 1, None, NOW) == "claim"


def test_failed_within_backoff_skips():
    # attempt_count=1 → backoff 60s; 30s elapsed → not yet.
    assert _outbox_claim_decision("failed", 1, _ago(30), NOW) == "skip"


def test_failed_past_backoff_claims():
    assert _outbox_claim_decision("failed", 1, _ago(61), NOW) == "claim"


def test_backoff_is_exponential_and_capped():
    assert _outbox_backoff_seconds(1) == 60
    assert _outbox_backoff_seconds(2) == 120
    assert _outbox_backoff_seconds(3) == 240
    # capped
    assert _outbox_backoff_seconds(100) == 3600


def test_failed_second_attempt_uses_longer_window():
    # attempt_count=2 → backoff 120s; 90s elapsed → still skip.
    assert _outbox_claim_decision("failed", 2, _ago(90), NOW) == "skip"
    assert _outbox_claim_decision("failed", 2, _ago(121), NOW) == "claim"


def test_exhausted_failed_is_dead():
    assert _outbox_claim_decision("failed", _OUTBOX_MAX_ATTEMPTS, _ago(99999), NOW) == "dead"


def test_processing_within_lease_skips():
    # A live/in-flight processing row is left alone until its lease expires.
    assert _outbox_claim_decision("processing", 1, _ago(_OUTBOX_LEASE_SECONDS - 5), NOW) == "skip"


def test_processing_past_lease_reclaims():
    # Orphaned by a crashed drainer → reclaim.
    assert _outbox_claim_decision("processing", 1, _ago(_OUTBOX_LEASE_SECONDS + 5), NOW) == "claim"


def test_processing_null_claim_time_reclaims():
    assert _outbox_claim_decision("processing", 1, None, NOW) == "claim"


def test_exhausted_orphaned_processing_is_dead():
    assert _outbox_claim_decision(
        "processing", _OUTBOX_MAX_ATTEMPTS, _ago(_OUTBOX_LEASE_SECONDS + 5), NOW
    ) == "dead"
