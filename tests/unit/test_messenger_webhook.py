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


# These describe the FALLBACK now, not the normal path: with no row to build a card
# from, a listing is still taken apart into text bubbles rather than dropped. An empty
# index is what a lookup miss looks like.


def test_a_card_falls_back_to_title_then_url_then_specs():
    assert messenger._sends_for_chunk(_CARD, {}) == [
        ("text", "1. 2026 Galyean CATTLE TRAILER 32' - 15079"),
        ("text", "https://trailerplace.com/inventory/galyean/"),
        ("text", "- Category: Livestock\n- Price: $32,250"),
    ]


def test_no_bubble_carries_markdown_link_syntax():
    """Messenger renders no markdown, so "[title](url)" would arrive literally."""
    assert not any("](" in payload for _kind, payload in messenger._sends_for_chunk(_CARD, {}))


def test_a_card_with_no_specs_falls_back_to_two_bubbles():
    sends = messenger._sends_for_chunk("1. [2025 Iron Bull DTB](https://x.test/a/)", {})
    assert sends == [("text", "1. 2025 Iron Bull DTB"), ("text", "https://x.test/a/")]


def test_prose_stays_a_single_bubble():
    prose = "Thank you for contacting TrailerPlace."
    assert messenger._sends_for_chunk(prose, {}) == [("text", prose)]


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


# ---------------------------------------------------------------------------
# Listing cards
#
# Messenger renders no link preview for a URL a BOT sends -- that courtesy is
# extended to links a PERSON pastes -- so a trailer's bare URL used to arrive as
# plain text. Each trailer is now a generic template built from the row the
# search returned, never from values parsed back out of the model's prose.
# ---------------------------------------------------------------------------


_LISTING_URL = "https://www.trailerplace.com/inventory/2026-iron-bull-dtb-15081/"
_PHOTO = "https://www.trailerplace.com/wp-content/uploads/2026/04/iron-bull_3072_4.jpg"

_LISTING = {
    "title": "2026 Iron Bull DTB - 15081",
    "url": _LISTING_URL,
    "image_url": _PHOTO,
    "price_display": "$9,995",
    "length": "14 ft 0 in",
    "category": "Utility",
}

_CHUNK = (
    f"1. [2026 Iron Bull DTB - 15081]({_LISTING_URL})\n"
    "   - Category: Utility\n"
    "   - Price: $9,995\n"
    "   - Plenty of deck for a compact tractor without overhanging."
)


def _kinds(sends):
    return [kind for kind, _ in sends]


def test_a_listing_becomes_a_card_and_its_specs(configured):
    sends = messenger._sends_for_chunk(_CHUNK, messenger._listings_by_url([_LISTING]))
    assert _kinds(sends) == ["card", "text"]


def test_the_bare_url_bubble_is_gone(configured):
    """The bubble this whole change exists to remove: Messenger never previewed it."""
    sends = messenger._sends_for_chunk(_CHUNK, messenger._listings_by_url([_LISTING]))
    assert not any(kind == "text" and payload.strip() == _LISTING_URL for kind, payload in sends)


def test_the_card_is_built_from_the_row_not_from_the_prose(configured):
    """The photo appears nowhere in the reply text, so it can only have come from the row."""
    element = messenger._sends_for_chunk(_CHUNK, messenger._listings_by_url([_LISTING]))[0][1]
    assert element["image_url"] == _PHOTO
    assert _PHOTO not in _CHUNK
    assert element["buttons"][0]["url"] == _LISTING_URL
    assert element["title"] == "2026 Iron Bull DTB - 15081"


def test_the_model_written_bullets_still_follow_the_card(configured):
    body = messenger._sends_for_chunk(_CHUNK, messenger._listings_by_url([_LISTING]))[1][1]
    assert "Plenty of deck for a compact tractor" in body


def test_a_listing_with_no_photo_still_makes_a_card(configured):
    """Every row looks like this until the scrape that populates image_url is ingested.
    Meta fetches image_url server-side and rejects the element if it cannot, so the key
    is omitted rather than sent empty."""
    index = messenger._listings_by_url([{**_LISTING, "image_url": None}])
    element = messenger._sends_for_chunk(_CHUNK, index)[0][1]
    assert "image_url" not in element
    assert element["title"] and element["buttons"]


def test_a_url_the_turn_never_presented_falls_back_to_text(configured):
    """A card needs a row to build from. Without one the trailer still reaches the
    customer with a tappable link, which beats dropping it."""
    assert _kinds(messenger._sends_for_chunk(_CHUNK, {})) == ["text", "text", "text"]


def test_cards_can_be_switched_off_without_a_deploy(monkeypatch):
    monkeypatch.setattr(messenger, "settings", Settings(messenger_listing_cards=False))
    index = messenger._listings_by_url([_LISTING])
    assert _kinds(messenger._sends_for_chunk(_CHUNK, index)) == ["text", "text", "text"]


def test_a_trailing_slash_or_capital_does_not_lose_the_match(configured):
    odd = {**_LISTING, "url": "https://WWW.trailerplace.com/inventory/2026-iron-bull-dtb-15081"}
    assert _kinds(messenger._sends_for_chunk(_CHUNK, messenger._listings_by_url([odd]))) == ["card", "text"]


def test_the_intro_and_the_closing_question_are_untouched(configured):
    for prose in ("Thank you for contacting TrailerPlace.", "Do any of these look like a fit?"):
        assert messenger._sends_for_chunk(prose, messenger._listings_by_url([_LISTING])) == [("text", prose)]


def test_a_long_title_is_clipped_to_messengers_limit(configured):
    """Messenger truncates past 80 itself, mid-word and with no ellipsis."""
    element = messenger._card_element({**_LISTING, "title": "Trailer " * 40}, "x", _LISTING_URL)
    assert len(element["title"]) <= 80


def test_the_subtitle_leads_with_the_price(configured):
    assert messenger._card_subtitle(_LISTING).startswith("$9,995")


def test_a_listing_with_no_price_still_has_a_usable_subtitle(configured):
    subtitle = messenger._card_subtitle({**_LISTING, "price_display": None, "price": None})
    assert "14 ft 0 in" in subtitle


def test_the_send_api_receives_a_generic_template(configured, monkeypatch):
    """End to end: the payload that actually reaches Meta."""
    sent = []
    monkeypatch.setattr(messenger, "_send", lambda payload: sent.append(payload))
    monkeypatch.setattr(messenger.conversation_store, "turn_already_handled", lambda *a: False)
    monkeypatch.setattr(messenger.time, "sleep", lambda *_: None)

    class _Response:
        assistant_text = _CHUNK
        listings = [_LISTING]

    monkeypatch.setattr("src.api.routes.chat", lambda request: _Response())
    messenger._handle_message("PSID", "show me a utility trailer", messenger._turn_id_for("m.card"))

    templates = [
        p["message"]["attachment"]["payload"]
        for p in sent
        if "message" in p and "attachment" in p["message"]
    ]
    assert len(templates) == 1
    assert templates[0]["template_type"] == "generic"
    assert templates[0]["elements"][0]["image_url"] == _PHOTO
