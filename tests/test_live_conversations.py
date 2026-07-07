"""Live end-to-end conversation harness.

Drives real conversations through service.handle_chat against the real LLM +
Pinecone stack, asserting on *structural* outcomes (not exact wording, which is
non-deterministic). This is the verification harness for the risky prompt/flow
refactors (E3/E5/E6): run it before and after such a change and compare.

Runs only when RUN_LIVE=1 and API keys are present (the conftest live-call gate
is a no-op under RUN_LIVE). Uses in-memory sessions (persistence forced off) to
avoid the Postgres FK coupling.
"""

from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv

# Load .env so OPENAI_API_KEY/PINECONE_API_KEY are visible when the skip
# condition below is evaluated at collection time.
load_dotenv()
os.environ.setdefault("TRAILERPLACE_PERSIST_CHATS", "0")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE") or not os.getenv("OPENAI_API_KEY"),
    reason="live harness: set RUN_LIVE=1 with real API keys",
)


def _chat(message: str, session_id: str):
    from src.chatbot import service
    from src.models import ChatRequest

    return service.handle_chat(ChatRequest(session_id=session_id, message=message))


def _sid() -> str:
    return str(uuid.uuid4())


def _is_contact_gate(text: str) -> bool:
    t = (text or "").lower()
    return "your name" in t and ("phone" in t or "email" in t)


def _open(opening: str, session_id: str):
    """Send the opening request and clear the turn-1 optional-contact gate."""
    resp = _chat(opening, session_id)
    if _is_contact_gate(resp.assistant_text):
        resp = _chat("no thanks", session_id)  # declining resumes the saved request
    return resp


def _drive_to_cards(opening: str, session_id: str, max_turns: int = 8):
    """Open, then answer 'no preference' until inventory cards appear in the text."""
    resp = _open(opening, session_id)
    for _ in range(max_turns):
        if "trailerplace.com/inventory" in (resp.assistant_text or ""):
            return resp
        resp = _chat("no preference", session_id)
    return resp


def test_faq_contact_reply_contains_phone():
    resp = _open("how do I contact your team?", _sid())
    assert "979-532-1486" in (resp.assistant_text or "")


def test_catalogue_question_lists_real_types():
    text = (_open("what kind of trailers do you carry?", _sid()).assistant_text or "").lower()
    assert sum(t in text for t in ("utility", "dump", "enclosed", "flatbed", "livestock")) >= 2


def test_tilt_use_case_resolves_and_searches():
    text = (_drive_to_cards("I want a tilt trailer to haul a tractor", _sid()).assistant_text or "").lower()
    assert "trailerplace.com/inventory" in text, "expected inventory cards for a tilt request"
    assert "tilt" in text, "expected tilt trailers surfaced"


def _stored_length(resp) -> str:
    """The length the extractor stored, from the last assistant message metadata."""
    msgs = resp.main_prior_messages or []
    meta = msgs[-1] if msgs else {}
    return str(
        (meta.get("metadata_filters_collected") or {}).get("length_ft")
        or (meta.get("slots_collected") or {}).get("trailer_length_ft")
        or ""
    )


def test_range_answer_stores_smallest_value():
    # answer_classification_policy: a numeric range stores only the smallest value.
    length = _stored_length(_open("I need a 15 to 18 ft livestock trailer", _sid()))
    assert length.startswith("15"), f"expected smallest (15), got {length!r}"


def test_number_word_dimension_extracted():
    # answer_classification_policy: unambiguous number words count as numeric.
    length = _stored_length(_open("I want a livestock trailer about twelve feet long", _sid()))
    assert length.startswith("12"), f"expected 12 from 'twelve feet', got {length!r}"


def test_make_search_returns_only_that_make():
    text = (_drive_to_cards("show me Cargo Craft enclosed trailers", _sid()).assistant_text or "").lower()
    assert "cargo craft" in text, "expected Cargo Craft results"
    # no other real brand should appear in the returned cards
    others = ("iron bull", "diamond c", "kaufman", "aluma", "galyean", "stallion")
    assert not [b for b in others if b in text], "expected only Cargo Craft in results"
