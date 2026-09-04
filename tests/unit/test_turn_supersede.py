"""Discarding a turn must leave nothing behind - not in memory, not in the database.

When a customer sends another message while we are writing their reply, the turn is
abandoned and rerun with the new message included. That is only safe if abandoning is
complete: a half-applied turn would leave the session holding answers from a reply the
customer never saw, and a queued escalation email would be sent for a turn that never
happened - then sent again by the rerun.
"""
import pytest

from src.api import routes
from src.graph.state import _sessions


@pytest.fixture(autouse=True)
def _clean_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


def test_no_check_means_no_turn_is_ever_discarded():
    """The default for Streamlit and the API: nothing interrupts a turn."""
    assert routes._turn_was_superseded() is False


def test_the_check_decides_whether_the_turn_is_discarded():
    with routes.abandon_turn_if(lambda: True):
        assert routes._turn_was_superseded() is True
    with routes.abandon_turn_if(lambda: False):
        assert routes._turn_was_superseded() is False


def test_passing_none_switches_the_check_off():
    """How the caller stops retrying and lets an answer out after too many interruptions."""
    with routes.abandon_turn_if(None):
        assert routes._turn_was_superseded() is False


def test_the_check_is_cleared_again_afterwards():
    with routes.abandon_turn_if(lambda: True):
        pass
    assert routes._turn_was_superseded() is False


def test_a_failing_check_keeps_the_reply_rather_than_losing_it():
    """Erring towards answering: a broken check must not silence the customer."""
    def broken():
        raise RuntimeError("database down")

    with routes.abandon_turn_if(broken):
        assert routes._turn_was_superseded() is False


def test_the_check_is_restored_to_the_outer_one_on_exit():
    with routes.abandon_turn_if(lambda: True):
        with routes.abandon_turn_if(lambda: False):
            assert routes._turn_was_superseded() is False
        assert routes._turn_was_superseded() is True


# ---------------------------------------------------------------------------
# Rolling the session back
# ---------------------------------------------------------------------------


def test_a_session_the_turn_created_is_removed_again():
    """A first message that gets superseded must not leave a half-built session behind."""
    _sessions["PSID"] = {"session_id": "PSID", "category": "Utility"}
    routes._restore_session("PSID", None)
    assert "PSID" not in _sessions


def test_an_existing_session_is_put_back_as_the_turn_found_it():
    before = {"session_id": "PSID", "category": None, "slots": {}}
    _sessions["PSID"] = {"session_id": "PSID", "category": "Dump", "slots": {"haul_item": "dirt"}}
    routes._restore_session("PSID", before)
    assert _sessions["PSID"] == before


def test_the_restored_copy_is_not_shared_with_the_discarded_turn():
    """The copy is taken with deepcopy because the graph mutates the session in place.

    Restoring a shallow reference would put back a dict the discarded turn had already
    edited, which is the bug this guards.
    """
    import copy

    original = {"session_id": "PSID", "slots": {"haul_item": "ATV"}}
    _sessions["PSID"] = original
    pre_turn = copy.deepcopy(_sessions["PSID"])
    original["slots"]["haul_item"] = "written by the discarded turn"
    routes._restore_session("PSID", pre_turn)
    assert _sessions["PSID"]["slots"] == {"haul_item": "ATV"}


# ---------------------------------------------------------------------------
# The turn itself
# ---------------------------------------------------------------------------


def _request(message="hello"):
    from src.api.schemas import ChatRequest
    return ChatRequest(session_id="PSID", turn_id="0f8fad5b-d9cb-469f-a165-70867728950e", message=message)


def test_an_uninterrupted_turn_returns_its_reply(monkeypatch):
    monkeypatch.setattr(routes.conversation_store, "persistence_enabled", lambda: False)
    monkeypatch.setattr(routes, "_handle_chat_in_memory", lambda r: {"assistant_text": "hi", "listings": []})
    response = routes._run_turn(_request(), "0f8fad5b-d9cb-469f-a165-70867728950e")
    assert response.assistant_text == "hi"


def test_an_interrupted_turn_raises_instead_of_returning_a_reply(monkeypatch):
    """The reply is never returned, so the caller cannot accidentally send it."""
    monkeypatch.setattr(routes.conversation_store, "persistence_enabled", lambda: False)
    monkeypatch.setattr(routes, "_handle_chat_in_memory", lambda r: {"assistant_text": "stale", "listings": []})
    with routes.abandon_turn_if(lambda: True):
        with pytest.raises(routes.TurnSuperseded):
            routes._run_turn(_request(), "0f8fad5b-d9cb-469f-a165-70867728950e")


def test_an_interrupted_turn_leaves_the_session_untouched(monkeypatch):
    """What the customer notices: the rerun must not start from half-applied state."""
    _sessions["PSID"] = {"session_id": "PSID", "category": None, "slots": {}}

    def mutating_turn(request):
        _sessions["PSID"]["category"] = "Utility"
        _sessions["PSID"]["slots"] = {"haul_item": "ATV"}
        return {"assistant_text": "stale", "listings": []}

    monkeypatch.setattr(routes.conversation_store, "persistence_enabled", lambda: False)
    monkeypatch.setattr(routes, "_handle_chat_in_memory", mutating_turn)
    with routes.abandon_turn_if(lambda: True):
        with pytest.raises(routes.TurnSuperseded):
            routes._run_turn(_request(), "0f8fad5b-d9cb-469f-a165-70867728950e")
    assert _sessions["PSID"] == {"session_id": "PSID", "category": None, "slots": {}}


def test_the_emails_of_a_discarded_turn_are_never_sent(monkeypatch):
    """Queued inside the turn's transaction, drained only after it commits.

    An abandoned turn raises out of durable_turn, so the transaction rolls back and the
    outbox rows never exist - which is why a discarded escalation is not emailed, and the
    rerun's escalation is emailed exactly once.
    """
    sends = []
    monkeypatch.setattr(routes.conversation_store, "persistence_enabled", lambda: False)
    monkeypatch.setattr(routes.conversation_store, "deliver_pending_outbox_async",
                        lambda: sends.append("drained"))
    monkeypatch.setattr(routes, "_handle_chat_in_memory", lambda r: {"assistant_text": "x", "listings": []})
    with routes.abandon_turn_if(lambda: True):
        with pytest.raises(routes.TurnSuperseded):
            routes._run_turn(_request(), "0f8fad5b-d9cb-469f-a165-70867728950e")
    assert sends == []
