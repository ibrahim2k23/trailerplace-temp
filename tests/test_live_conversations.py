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


def _drive_to_results(opening: str, session_id: str, max_turns: int = 8):
    """Send an opening request, then answer 'no preference' until listings appear."""
    resp = _chat(opening, session_id)
    for _ in range(max_turns):
        if resp.listings:
            return resp
        resp = _chat("no preference", session_id)
    return resp


def test_faq_contact_reply_contains_phone():
    resp = _chat("how do I contact your team?", _sid())
    assert "979-532-1486" in (resp.assistant_text or "")


def test_catalogue_question_lists_real_types():
    text = (_chat("what kind of trailers do you carry?", _sid()).assistant_text or "").lower()
    # at least a couple of canonical types should be named
    assert sum(t in text for t in ("utility", "dump", "enclosed", "flatbed", "livestock")) >= 2


def test_tilt_use_case_resolves_and_searches():
    sid = _sid()
    resp = _drive_to_results("I want a tilt trailer to haul a tractor", sid)
    assert resp.listings, "expected listings after qualifying a tilt request"
    cats = {getattr(l, "category_subcategory", "").split(" > ")[0] for l in resp.listings}
    assert cats == {"Tilt"}, f"expected only Tilt listings, got {cats}"


def test_make_search_returns_only_that_make():
    resp = _drive_to_results("show me Cargo Craft enclosed trailers", _sid())
    assert resp.listings, "expected listings for a Cargo Craft search"
    makes = {getattr(l, "make", "") for l in resp.listings}
    assert makes == {"Cargo Craft"}, f"expected only Cargo Craft, got {makes}"
