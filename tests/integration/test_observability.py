"""Milestone 9 §2-§4 — tracing, per-turn logging, and the cost audit, wired to /chat.

Network-free: the graph is faked, persistence is off, tracing is off (conftest).
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import conversation_log, turn_log
from src.api import routes
from src.api.app import create_app
from src.graph.state import _sessions
from src.llm import usage as llm_usage
from tests.conftest import replace_settings
from tests.unit.llm_helpers import sample_analysis, sample_reply

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cost_report  # noqa: E402


class _RecordingGraph:
    """A graph that records LLM usage the way the real nodes do, on a worker thread."""

    def __init__(self, *, completions=2, embeddings=0, outcome_extra=None, turn=None, reply=None):
        self.completions = completions
        self.embeddings = embeddings
        self.outcome_extra = outcome_extra or {}
        self.turn = turn
        self.reply = reply or sample_reply("ok")

    def invoke(self, state):
        for _ in range(self.completions):
            llm_usage.record_completion("gpt-4o-mini", prompt_tokens=1000, completion_tokens=100)
        for _ in range(self.embeddings):
            llm_usage.record_embedding("text-embedding-3-small", prompt_tokens=20)
        return {
            **state,
            "category": "Dump",
            "turn": self.turn,
            "turn_outcome": {"reply": self.reply, "listings": [], **self.outcome_extra},
        }


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr("src.conversation_store.persistence_enabled", lambda: False)
    _sessions.clear()
    log_file = tmp_path / "turns.jsonl"
    replace_settings(monkeypatch, turn_log_path=str(log_file))
    turn_log.reset_for_tests()

    def _install(graph):
        monkeypatch.setattr(routes, "_get_graph", lambda: graph)
        return TestClient(create_app())

    yield _install, log_file
    turn_log.reset_for_tests()


def _post(client, message="hi"):
    return client.post(
        "/chat",
        json={"session_id": str(uuid.uuid4()), "turn_id": str(uuid.uuid4()), "message": message},
    )


def _lines(log_file):
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_usage_counters_survive_the_graph_executor_thread(offline):
    """Regression: ThreadPoolExecutor does not propagate contextvars.

    Without `contextvars.copy_context().run(...)` in _invoke_graph, the nodes record
    into an empty context, every turn logs chat_completions=0, and cost_report
    silently classifies real turns as receipt replays — auditing nothing.
    """
    install, log_file = offline
    client = install(_RecordingGraph(completions=2))

    assert _post(client).status_code == 200

    record = _lines(log_file)[0]
    assert record["llm_calls"]["chat_completions"] == 2, "usage lost crossing the executor thread"
    assert record["llm_calls"]["total_tokens"] == 2200


def test_turn_log_captures_intent_category_tools_and_latency(offline):
    install, log_file = offline
    graph = _RecordingGraph(completions=2, embeddings=1, outcome_extra={"search_ran": True, "emails_sent": ["Escalation"]})
    client = install(graph)

    assert _post(client).status_code == 200

    record = _lines(log_file)[0]
    assert record["event"] == "chat_turn"
    assert record["category"] == "Dump"
    assert record["tools_fired"] == ["search", "email"]
    assert record["emails_sent"] == ["Escalation"]
    assert record["latency_ms"] > 0
    assert record["llm_calls"]["embeddings"] == 1


def test_each_turn_gets_its_own_usage_scope(offline):
    """Counters must not accumulate across turns."""
    install, log_file = offline
    client = install(_RecordingGraph(completions=2))

    for _ in range(3):
        _post(client)

    records = _lines(log_file)
    assert len(records) == 3
    assert [r["llm_calls"]["chat_completions"] for r in records] == [2, 2, 2]


def test_a_failed_turn_is_logged_with_its_error(offline):
    install, log_file = offline

    class _Boom:
        def invoke(self, state):
            raise RuntimeError("kaboom")

    client = install(_Boom())
    response = _post(client)

    assert response.status_code == 200  # contained into an apology, never a 500
    record = _lines(log_file)[0]
    assert "kaboom" in record["error"]


def test_cost_report_audits_a_real_ten_turn_conversation(offline):
    """M9 acceptance: the cost assertion holds on a 10-turn scripted conversation."""
    install, log_file = offline
    client = install(_RecordingGraph(completions=2))
    for _ in range(8):
        _post(client)

    # Two search turns: each is allowed exactly one embedding.
    routes._GRAPH = None
    searching = _RecordingGraph(completions=2, embeddings=1, outcome_extra={"search_ran": True})
    client2 = install(searching)
    for _ in range(2):
        _post(client2)

    result = cost_report.audit(cost_report.load_turns(log_file))
    assert result["failures"] == []
    assert result["audited"] == 10
    assert result["replays"] == 0
    assert result["total_completions"] == 20  # exactly 2 per turn
    assert result["total_embeddings"] == 2  # only the search turns


def test_cost_report_catches_a_third_llm_call(offline):
    """The audit must actually fail when a turn overspends."""
    install, log_file = offline
    client = install(_RecordingGraph(completions=3))
    _post(client)

    result = cost_report.audit(cost_report.load_turns(log_file))
    assert len(result["failures"]) == 1
    assert "3 chat completions" in result["failures"][0][1][0]


# ---------------------------------------------------------------------------
# Human-readable conversation reasoning + state log, wired to the real route.
# ---------------------------------------------------------------------------

def test_chat_emits_a_full_conversation_reasoning_block(offline, caplog):
    install, _ = offline
    analysis = sample_analysis(intent="qualification_answer", category_mentioned="Dump")
    graph = _RecordingGraph(turn=analysis, reply=sample_reply("Got it, noted."))
    client = install(graph)

    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        resp = _post(client, message="I need a dump trailer for gravel")
    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == conversation_log.CONVERSATION_LOGGER_NAME]
    assert len(records) == 1
    block = records[0].message
    assert "USER: I need a dump trailer for gravel" in block
    assert "ASSISTANT: Got it, noted." in block
    assert "intent: qualification_answer" in block
    assert "category_mentioned: Dump" in block
    assert "STATE AFTER TURN:" in block
    assert "category: Dump" in block  # state was updated by the (fake) graph run


def test_conversation_log_carries_the_real_session_and_turn_id(offline, caplog):
    install, _ = offline
    client = install(_RecordingGraph())
    session_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        resp = client.post("/chat", json={"session_id": session_id, "turn_id": turn_id, "message": "hi"})
    assert resp.status_code == 200

    block = next(r for r in caplog.records if r.name == conversation_log.CONVERSATION_LOGGER_NAME).message
    assert f"session={session_id}" in block
    assert f"turn={turn_id}" in block


def test_create_app_wires_up_logging(monkeypatch):
    """Regression: configure_trailerplace_logging() was defined in src/log_setup.py
    since M0 but never actually called anywhere -- so the root logger had no
    handlers and every logger.info() in the codebase (turn_log, conversation_log,
    every TOOL search/inventory_lookup/email/outbox line) was silently dropped,
    console AND the daily log file, with no error to indicate why. create_app()
    must call it before returning the app."""
    from src.api import app as app_module

    called = []
    monkeypatch.setattr(app_module, "configure_trailerplace_logging", lambda: called.append(True))
    app_module.create_app()
    assert called == [True]


def test_conversation_log_shows_error_on_a_contained_graph_failure(offline, caplog):
    install, _ = offline

    class _Boom:
        def invoke(self, state):
            raise RuntimeError("kaboom")

    client = install(_Boom())
    with caplog.at_level(logging.INFO, logger=conversation_log.CONVERSATION_LOGGER_NAME):
        resp = _post(client)
    assert resp.status_code == 200  # still contained into a 200 apology

    block = next(r for r in caplog.records if r.name == conversation_log.CONVERSATION_LOGGER_NAME).message
    assert "ERROR: kaboom" in block
