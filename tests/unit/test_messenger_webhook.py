"""Messenger webhook: signature, payload parsing, and the duplicate-message guard.

The duplicate tests are the point of this file. Meta redelivers a message whenever it
does not get a 200 within ~20 s, and a chatbot turn takes far longer than that, so a
retry is the normal case rather than an edge case. Every retry carries the same `mid`.
"""
import hashlib
import hmac
import json
import uuid

import pytest

from src.api import messenger
from src.config import Settings


@pytest.fixture(autouse=True)
def _clean_dedupe_state():
    """The mid cache is module-level, so one test's claim would leak into the next."""
    messenger._seen_mids.clear()
    yield
    messenger._seen_mids.clear()


@pytest.fixture
def configured(monkeypatch):
    cfg = Settings(
        messenger_enabled=True,
        messenger_app_secret="app-secret",
        messenger_page_access_token="page-token",
        messenger_verify_token="verify-me",
    )
    monkeypatch.setattr(messenger, "settings", cfg)
    return cfg


def _webhook_body(psid="9876543210987654", text="hi", mid="m.abc123"):
    return {
        "object": "page",
        "entry": [{"messaging": [{"sender": {"id": psid}, "message": {"mid": mid, "text": text}}]}],
    }


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------


def test_the_same_mid_is_claimed_only_once(configured):
    assert messenger._claim_mid("m.abc") is True
    assert messenger._claim_mid("m.abc") is False


def test_different_mids_are_both_claimed(configured):
    assert messenger._claim_mid("m.one") is True
    assert messenger._claim_mid("m.two") is True


def test_the_mid_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(messenger, "settings", Settings(messenger_seen_mid_cache_size=3))
    for i in range(10):
        messenger._claim_mid(f"m.{i}")
    assert len(messenger._seen_mids) == 3
    # Oldest evicted, newest kept.
    assert "m.9" in messenger._seen_mids and "m.0" not in messenger._seen_mids


def test_a_mid_maps_to_a_stable_turn_id(configured):
    """A retry must produce the SAME turn_id, or durable_turn's receipt cannot match it."""
    first = messenger._turn_id_for("m.abc123")
    assert first == messenger._turn_id_for("m.abc123")
    assert first != messenger._turn_id_for("m.abc124")
    uuid.UUID(first)  # must be a real UUID: chatbot_turns.turn_id is a UUID column


def test_a_turn_already_committed_is_not_answered_again(configured, monkeypatch):
    """The cold-start case: the in-process cache is gone, the stored receipt is not."""
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: True)
    sent = []
    monkeypatch.setattr(messenger, "_send", lambda p: sent.append(p))
    messenger._handle_message("PSID", "hi", messenger._turn_id_for("m.abc"))
    assert sent == []


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------


def test_a_correct_signature_is_accepted(configured):
    body = json.dumps(_webhook_body()).encode()
    digest = hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()
    assert messenger._signature_is_valid(body, f"sha256={digest}") is True


def test_a_forged_signature_is_rejected(configured):
    body = json.dumps(_webhook_body()).encode()
    assert messenger._signature_is_valid(body, "sha256=" + "0" * 64) is False


def test_a_missing_signature_is_rejected(configured):
    assert messenger._signature_is_valid(b"{}", None) is False


def test_a_signature_over_different_bytes_is_rejected(configured):
    """Guards the raw-body requirement: re-serialising the JSON changes the digest."""
    signed = json.dumps(_webhook_body(), separators=(",", ":")).encode()
    received = json.dumps(_webhook_body()).encode()
    digest = hmac.new(b"app-secret", signed, hashlib.sha256).hexdigest()
    assert messenger._signature_is_valid(received, f"sha256={digest}") is False


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------


def test_enabled_requires_a_secret_as_well_as_a_token(monkeypatch):
    monkeypatch.setattr(messenger, "settings", Settings(
        messenger_enabled=True, messenger_page_access_token="page-token"))
    assert messenger.messenger_enabled() is False


def test_enabled_when_fully_configured(configured):
    assert messenger.messenger_enabled() is True


def test_disabled_by_default():
    assert messenger.messenger_enabled() is False


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def test_a_plain_text_message_is_extracted():
    assert [row[:3] for row in messenger._extract_messages(_webhook_body())] ==         [("9876543210987654", "hi", "m.abc123")]


def test_our_own_echo_is_ignored():
    """Answering our own outgoing message would loop forever."""
    payload = _webhook_body()
    payload["entry"][0]["messaging"][0]["message"]["is_echo"] = True
    assert messenger._extract_messages(payload) == []


def test_delivery_and_read_receipts_are_ignored():
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "P"}, "delivery": {"mids": ["m.1"]}},
        {"sender": {"id": "P"}, "read": {"watermark": 1}},
    ]}]}
    assert messenger._extract_messages(payload) == []


def test_an_attachment_with_no_text_is_ignored():
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "P"}, "message": {"mid": "m.1", "attachments": [{"type": "image"}]}},
    ]}]}
    assert messenger._extract_messages(payload) == []


def test_a_get_started_postback_is_treated_as_a_message():
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "P"}, "timestamp": 123, "postback": {"payload": "GET_STARTED"}},
    ]}]}
    assert [row[:3] for row in messenger._extract_messages(payload)] ==         [("P", "GET_STARTED", "postback:P:123")]


def test_a_non_page_payload_is_ignored():
    payload = _webhook_body()
    payload["object"] = "instagram"
    assert messenger._extract_messages(payload) == []


def test_several_messages_in_one_delivery_are_all_extracted():
    payload = {"object": "page", "entry": [{"messaging": [
        {"sender": {"id": "P"}, "message": {"mid": "m.1", "text": "one"}},
        {"sender": {"id": "P"}, "message": {"mid": "m.2", "text": "two"}},
    ]}]}
    assert len(messenger._extract_messages(payload)) == 2


# ---------------------------------------------------------------------------
# Bubble splitting
# ---------------------------------------------------------------------------


def test_a_short_reply_is_one_bubble():
    assert messenger._split_for_messenger("Sure, here are three options.") == \
        ["Sure, here are three options."]


def test_a_long_reply_is_split_under_the_limit():
    parts = messenger._split_for_messenger("\n".join(["a line of text"] * 400))
    assert len(parts) > 1
    assert all(len(part) <= messenger._MESSENGER_TEXT_LIMIT for part in parts)


def test_splitting_loses_no_text():
    body = "\n".join(f"line {i}" for i in range(400))
    assert "\n".join(messenger._split_for_messenger(body)) == body


def test_one_oversized_line_is_hard_sliced():
    parts = messenger._split_for_messenger("x" * 5000)
    assert all(len(part) <= messenger._MESSENGER_TEXT_LIMIT for part in parts)
    assert "".join(parts) == "x" * 5000


def test_an_empty_reply_produces_no_bubble():
    assert messenger._split_for_messenger("   ") == []


# ---------------------------------------------------------------------------
# End to end: does a retry actually stop a second reply reaching the customer?
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text):
        self.assistant_text = text


@pytest.fixture
def wired(configured, monkeypatch):
    """Replace the two edges of _handle_message: the chat turn, and the Send API."""
    sent, turns = [], []

    def fake_chat(request):
        turns.append((request.session_id, request.turn_id, request.message))
        return _FakeResponse("Sure - what are you hauling?")

    def fake_send(payload):
        if "message" in payload:
            sent.append(payload["message"]["text"])

    monkeypatch.setattr("src.api.routes.chat", fake_chat)
    monkeypatch.setattr(messenger, "_send", fake_send)
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger.time, "sleep", lambda *_: None)
    return sent, turns


def test_a_message_is_answered_once(wired):
    sent, turns = wired
    messenger._handle_message("PSID", "I need a trailer", messenger._turn_id_for("m.1"))
    assert turns and sent == ["Sure - what are you hauling?"]


def test_the_psid_goes_in_as_the_session_id_unchanged(wired):
    """chatbot_leads.psid must end up holding the real PSID, so it is not converted here."""
    _, turns = wired
    messenger._handle_message("9876543210987654", "hi", messenger._turn_id_for("m.1"))
    assert turns[0][0] == "9876543210987654"


def test_the_turn_id_reaches_the_chat_request(wired):
    _, turns = wired
    turn_id = messenger._turn_id_for("m.1")
    messenger._handle_message("PSID", "hi", turn_id)
    assert turns[0][1] == turn_id


def test_a_turn_whose_receipt_exists_sends_nothing(wired, monkeypatch):
    """No duplicate reply, and no duplicate alert email, for a delivery already answered."""
    sent, turns = wired
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: True)
    messenger._handle_message("PSID", "I have a complaint", messenger._turn_id_for("m.1"))
    assert turns == [] and sent == []


def test_a_retry_after_a_cold_start_still_sends_nothing(configured, monkeypatch):
    """No in-process cache (fresh container), but chatbot_turns already holds the receipt."""
    sent, turns = [], []
    monkeypatch.setattr("src.api.routes.chat", lambda r: turns.append(r) or _FakeResponse("hi"))
    monkeypatch.setattr(messenger, "_send", lambda p: sent.append(p))
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: True)
    messenger._handle_message("PSID", "I have a complaint", "m.1")
    assert turns == [] and sent == []


def test_two_different_messages_are_both_answered(wired):
    sent, turns = wired
    messenger._handle_message("PSID", "I need a trailer", messenger._turn_id_for("m.1"))
    messenger._handle_message("PSID", "20ft", messenger._turn_id_for("m.2"))
    assert len(turns) == 2 and len(sent) == 2


def test_a_multi_card_reply_is_sent_as_several_bubbles(configured, monkeypatch):
    reply = (
        "Here are two that fit.\n\n"
        "**1. 2024 Big Tex 14GN**\n- Price: $12,000\n\n"
        "**2. 2023 PJ Trailers GB**\n- Price: $14,500\n\n"
        "Want me to check availability?"
    )
    sent = []
    monkeypatch.setattr("src.api.routes.chat", lambda r: _FakeResponse(reply))
    monkeypatch.setattr(messenger, "_send", lambda p: sent.append(p["message"]["text"]) if "message" in p else None)
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger.time, "sleep", lambda *_: None)
    messenger._handle_message("PSID", "show me gooseneck trailers", messenger._turn_id_for("m.1"))
    assert len(sent) > 1
    assert "".join(sent).count("Big Tex") == 1


# ---------------------------------------------------------------------------
# A listing card arrives as three bubbles
# ---------------------------------------------------------------------------

_CARD = (
    "1. [2026 Galyean CATTLE TRAILER 32' - 15079](https://trailerplace.com/inventory/galyean/)\n"
    "   - Category: Livestock\n   - Price: $32,250"
)


def test_a_card_becomes_title_then_url_then_specs():
    assert messenger._bubbles_for_chunk(_CARD) == [
        "1. 2026 Galyean CATTLE TRAILER 32' - 15079",
        "https://trailerplace.com/inventory/galyean/",
        "- Category: Livestock\n- Price: $32,250",
    ]


def test_no_bubble_carries_markdown_link_syntax():
    assert not any("](" in bubble for bubble in messenger._bubbles_for_chunk(_CARD))


def test_a_card_with_no_specs_is_two_bubbles():
    bubbles = messenger._bubbles_for_chunk("1. [2025 Iron Bull DTB](https://x.test/a/)")
    assert bubbles == ["1. 2025 Iron Bull DTB", "https://x.test/a/"]


def test_prose_stays_a_single_bubble():
    assert messenger._bubbles_for_chunk("Thank you for contacting TrailerPlace.") == \
        ["Thank you for contacting TrailerPlace."]


def test_the_reply_is_sent_card_by_card_in_order(configured, monkeypatch):
    reply = (
        "Here are two that fit.\n\n"
        + _CARD
        + "\n\n2. [2025 Iron Bull DTB - 15081](https://trailerplace.com/inventory/dtb/)\n"
        "   - Category: Dump\n   - Price: $9,995\n\n"
        "Do any of these look like a fit?"
    )
    sent = []
    monkeypatch.setattr("src.api.routes.chat", lambda r: _FakeResponse(reply))
    monkeypatch.setattr(messenger, "_send", lambda p: sent.append(p["message"]["text"]) if "message" in p else None)
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger.time, "sleep", lambda *_: None)
    messenger._handle_message("PSID", "show me livestock trailers", messenger._turn_id_for("m.1"))
    assert sent == [
        "Here are two that fit.",
        "1. 2026 Galyean CATTLE TRAILER 32' - 15079",
        "https://trailerplace.com/inventory/galyean/",
        "- Category: Livestock\n- Price: $32,250",
        "2. 2025 Iron Bull DTB - 15081",
        "https://trailerplace.com/inventory/dtb/",
        "- Category: Dump\n- Price: $9,995",
        "Do any of these look like a fit?",
    ]
