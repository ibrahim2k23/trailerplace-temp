"""Live single-session harness covering several intents in one conversation.

Unlike test_live_conversations.py (one short conversation per test), this drives
ONE session through the whole funnel in order: onboarding contact gate, category
resolution, qualification, inventory search, listing interest + deferred contact
capture, a general FAQ question, and a category switch back into search.

The point is to catch state bugs that only appear once a session has history --
most notably the contact gate re-asking for details it already holds.

Runs only when RUN_LIVE=1 and API keys are present. Uses in-memory sessions
(persistence forced off). Note: the interest and FAQ steps exercise the real
email path, so whatever EMAIL backend .env points at will receive mail.

Known bug this test steers around (step 8): after a mid-conversation category
switch, answering the new category's required slots with 'no preference' never
resolves them -- the bot acknowledges the non-preference, re-asks, and never
reaches the search node, eventually escalating to sales. A fresh session for the
same category accepts 'no preference' and searches normally. Step 8 therefore
answers concretely; a regression test for the 'no preference' path belongs here
once that is fixed.
"""

from __future__ import annotations

import os
import uuid

import pytest
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TRAILERPLACE_PERSIST_CHATS", "0")

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE") or not os.getenv("OPENAI_API_KEY"),
    reason="live harness: set RUN_LIVE=1 with real API keys",
)

NAME = "Ibrahim"
EMAIL = "ibrahim@esided.ai"


def _chat(message: str, session_id: str):
    from src.chatbot import service
    from src.models import ChatRequest

    return service.handle_chat(ChatRequest(session_id=session_id, message=message))


def _asks_for_name(text: str) -> bool:
    return "your name" in (text or "").lower()


def _asks_for_reachable(text: str) -> bool:
    t = (text or "").lower()
    return "email" in t or "phone" in t


def _has_cards(text: str) -> bool:
    return "trailerplace.com/inventory" in (text or "")


def _listings(resp) -> list[dict]:
    """Main-phase responses leave ChatResponse.listings empty; the structured
    listings ride on the last assistant message instead (service._handle_main)."""
    msgs = resp.main_prior_messages or []
    return (msgs[-1].get("listings") if msgs else None) or []


def _categories(resp) -> list[str]:
    return [str(l.get("category") or "").lower() for l in _listings(resp)]


def _drive_to_cards(session_id: str, answers: list[str], max_turns: int = 6):
    """Answer remaining qualification questions until inventory cards appear.

    ``answers`` is consumed in order, then 'no preference' repeats. Concrete
    answers matter: after a mid-conversation category switch, 'no preference'
    alone never resolves the new category's required slots (see the module
    docstring), so the search node is never reached.
    """
    resp = None
    for turn in range(max_turns):
        reply = answers[turn] if turn < len(answers) else "no preference"
        resp = _chat(reply, session_id)
        if _has_cards(resp.assistant_text):
            return resp
    return resp


def test_multi_intent_single_conversation(capsys):
    session_id = str(uuid.uuid4())
    transcript: list[tuple[str, str]] = []

    def say(message: str):
        resp = _chat(message, session_id)
        transcript.append((message, resp.assistant_text or ""))
        return resp

    # --- 1. Opening: name is supplied inline, so the gate must not re-ask it. ---
    resp = say(f"My name is {NAME} and I am looking for a livestock trailer")
    assert not _asks_for_name(resp.assistant_text), (
        "onboarding gate re-asked for a name that was given in the same message: "
        f"{resp.assistant_text!r}"
    )
    assert _asks_for_reachable(resp.assistant_text), "expected a request for email or phone"
    assert (resp.customer_full_name or "").lower().startswith(NAME.lower())

    # --- 2. Decline contact; the saved trailer request should resume. ---
    resp = say("no")
    assert resp.contact_status == "contact_declined"

    # --- 3. Qualification -> inventory cards. ---
    resp = say("20ft")
    if not _has_cards(resp.assistant_text):
        resp = _drive_to_cards(session_id, [])
        transcript.append(("<no preference x N>", resp.assistant_text or ""))
    assert _has_cards(resp.assistant_text), "expected livestock inventory cards"
    cats = _categories(resp)
    assert cats, "expected structured listings on the last assistant message"
    assert any("livestock" in c for c in cats), f"expected livestock listings, got {cats}"

    # --- 4. Express interest. Name is known, email/phone is not: ask ONLY for the latter. ---
    resp = say("the 2nd one does")
    assert not _asks_for_name(resp.assistant_text), (
        "interest gate re-asked for the already-known name: " f"{resp.assistant_text!r}"
    )
    assert _asks_for_reachable(resp.assistant_text), (
        "interest gate should ask for email or phone: " f"{resp.assistant_text!r}"
    )

    # --- 5. Supply the email; the deferred interest action should fire. ---
    resp = say(f"my email is {EMAIL}")
    assert (resp.customer_email or "").lower() == EMAIL
    assert resp.contact_status == "contact_available"
    assert not _asks_for_reachable(resp.assistant_text) or "logged" in resp.assistant_text.lower()

    # --- 6. General FAQ mid-conversation: contact is on file, so no gate. ---
    resp = say("how do I contact your team?")
    assert "979-532-1486" in (resp.assistant_text or ""), "expected the sales phone number"
    assert not _asks_for_name(resp.assistant_text)

    # --- 7. Non-sales FAQ -> routed to the team, still no contact re-ask. ---
    resp = say("do you offer financing on these trailers?")
    assert not _asks_for_name(resp.assistant_text)
    assert len(resp.assistant_text or "") > 20

    # --- 8. Category switch: new search in the same session. ---
    resp = say("actually, show me dump trailers instead")
    if not _has_cards(resp.assistant_text):
        # Dump requires length, material and haul weight; answer them concretely.
        resp = _drive_to_cards(session_id, ["yes, keep 20 ft", "gravel", "7000 lbs"])
        transcript.append(("<dump qualification>", resp.assistant_text or ""))
    assert _has_cards(resp.assistant_text), "expected dump trailer cards after the category switch"
    cats = _categories(resp)
    assert any("dump" in c for c in cats), f"expected dump listings, got {cats}"

    with capsys.disabled():
        print(f"\n===== live transcript (session {session_id}) =====")
        for user, bot in transcript:
            print(f"\n[user] {user}\n[bot ] {bot[:600]}")
