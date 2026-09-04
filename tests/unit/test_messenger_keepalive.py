"""What the customer sees between sending a message and getting an answer.

A turn takes 25-30 seconds. Left alone that reads as broken: Messenger dismisses a typing
indicator after roughly 20 seconds, so typing visibly stops and then nothing happens.

Two things fill it. The typing indicator is refreshed for as long as the turn runs, and
the search line the graph publishes the moment it starts looking is sent straight away -
the same line the web UI shows, from the same place.

All of it is cosmetic, so the last group of tests is about it never being able to harm
the turn it decorates.
"""
import threading
import time

import pytest

from src import turn_status
from src.api import messenger
from src.config import Settings

PSID = "9876543210987654"
SEARCH_LINE = "Let me see what we have on the lot for you."


@pytest.fixture(autouse=True)
def _clean():
    turn_status.clear(PSID)
    messenger._clear_status_line(PSID)
    yield
    turn_status.clear(PSID)
    messenger._clear_status_line(PSID)


@pytest.fixture
def actions(monkeypatch):
    """Record every sender action and bubble the keep-alive emits."""
    sent = []
    monkeypatch.setattr(messenger, "_send", lambda payload: sent.append(
        payload.get("sender_action") or payload["message"]["text"]))
    return sent


def _fast(**overrides):
    """Real intervals, just short ones - the timing IS the behaviour under test."""
    base = dict(messenger_enabled=True, messenger_app_secret="s",
                messenger_page_access_token="t", messenger_typing_refresh_seconds=0.1)
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Keeping the typing indicator alive
# ---------------------------------------------------------------------------


def test_typing_is_refreshed_while_the_turn_runs(actions, monkeypatch):
    """Messenger drops the indicator after ~20s; a 30s turn must not go quiet."""
    monkeypatch.setattr(messenger, "settings", _fast())
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.65)
    alive.stop()
    assert actions.count("typing_on") >= 2


def test_typing_stops_being_refreshed_once_the_turn_ends(actions, monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast())
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.3)
    alive.stop()
    after_stop = len(actions)
    time.sleep(0.4)
    assert len(actions) == after_stop


def test_a_refresh_interval_of_zero_turns_it_off(actions, monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast(
        messenger_typing_refresh_seconds=0.0, messenger_send_search_status=False))
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.3)
    alive.stop()
    assert actions == []


def test_the_thread_is_joined_so_it_cannot_outlive_the_turn(monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast())
    monkeypatch.setattr(messenger, "_send", lambda payload: None)
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    alive.stop()
    assert alive._thread is not None and not alive._thread.is_alive()


# ---------------------------------------------------------------------------
# The search line
# ---------------------------------------------------------------------------


def test_the_search_line_is_sent_as_soon_as_the_graph_publishes_it(actions, monkeypatch):
    """The point of it: an answer within a second, instead of 30 seconds of silence."""
    monkeypatch.setattr(messenger, "settings", _fast())
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    turn_status.publish(PSID, SEARCH_LINE)   # the search node, mid-turn
    time.sleep(0.9)
    alive.stop()
    assert SEARCH_LINE in actions


def test_the_search_line_is_sent_only_once_per_turn(actions, monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast())
    turn_status.publish(PSID, SEARCH_LINE)
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.9)
    alive.stop()
    assert actions.count(SEARCH_LINE) == 1


def test_a_turn_that_never_searches_sends_no_line(actions, monkeypatch):
    """A qualification turn just asks a question; there is nothing to announce."""
    monkeypatch.setattr(messenger, "settings", _fast())
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.4)
    alive.stop()
    assert all(action in {"typing_on"} for action in actions)


def test_a_discarded_turn_does_not_make_the_customer_read_it_twice(actions, monkeypatch):
    """The interaction with regeneration.

    An attempt that searches, gets interrupted, and is rerun would otherwise publish and
    send the same line again - so the customer sees "Let me see what we have" twice for
    one question.
    """
    monkeypatch.setattr(messenger, "settings", _fast())
    for _ in range(2):   # the discarded attempt, then the rerun
        turn_status.publish(PSID, SEARCH_LINE)
        alive = messenger._TurnKeepAlive(PSID)
        alive.start()
        time.sleep(0.4)
        alive.stop()
    assert actions.count(SEARCH_LINE) == 1


def test_the_next_question_may_show_the_line_again(actions, monkeypatch):
    """Cleared once the answer lands, so the customer's NEXT search announces itself."""
    monkeypatch.setattr(messenger, "settings", _fast())
    for _ in range(2):
        turn_status.publish(PSID, SEARCH_LINE)
        alive = messenger._TurnKeepAlive(PSID)
        alive.start()
        time.sleep(0.4)
        alive.stop()
        messenger._clear_status_line(PSID)   # what _handle_message does after replying
    assert actions.count(SEARCH_LINE) == 2


def test_the_search_line_can_be_switched_off(actions, monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast(messenger_send_search_status=False))
    turn_status.publish(PSID, SEARCH_LINE)
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.4)
    alive.stop()
    assert SEARCH_LINE not in actions


def test_one_customers_line_never_reaches_another(actions, monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast())
    turn_status.publish("someone-else", SEARCH_LINE)
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.4)
    alive.stop()
    assert SEARCH_LINE not in actions


# ---------------------------------------------------------------------------
# It must never harm the turn it decorates
# ---------------------------------------------------------------------------


def test_a_failing_send_does_not_kill_the_keep_alive(monkeypatch):
    """Facebook having a bad moment must not stop the next refresh from trying."""
    calls = []

    def flaky(payload):
        calls.append(payload)
        raise RuntimeError("Graph API said no")

    monkeypatch.setattr(messenger, "settings", _fast())
    monkeypatch.setattr(messenger, "_send", flaky)
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.65)
    alive.stop()
    assert len(calls) >= 2   # it kept going after the first failure


def test_stopping_a_keep_alive_that_never_started_is_safe(monkeypatch):
    monkeypatch.setattr(messenger, "settings", _fast(
        messenger_typing_refresh_seconds=0.0, messenger_send_search_status=False))
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    alive.stop()   # must not raise


def test_it_runs_off_the_turn_s_own_thread(monkeypatch):
    """It must never add latency to the reply, so it cannot run inline."""
    monkeypatch.setattr(messenger, "settings", _fast())
    seen = []
    monkeypatch.setattr(messenger, "_send",
                        lambda payload: seen.append(threading.current_thread().name))
    alive = messenger._TurnKeepAlive(PSID)
    alive.start()
    time.sleep(0.4)
    alive.stop()
    assert seen and all(name != threading.current_thread().name for name in seen)
