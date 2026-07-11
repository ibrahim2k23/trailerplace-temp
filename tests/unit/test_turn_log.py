"""Milestone 9 §3/§4 — per-turn usage counters and the structured turn log."""
from __future__ import annotations

import json
import logging

import pytest

from src import turn_log
from src.llm import usage
from tests.conftest import replace_settings


# ---------------------------------------------------------------------------
# usage counters
# ---------------------------------------------------------------------------

def test_recording_outside_a_scope_is_a_noop():
    # Library code (tests, ingest, smoke scripts) calls the client without a turn.
    usage.record_completion("gpt-4o-mini", 10, 5)
    usage.record_embedding("text-embedding-3-small", 3)
    assert usage.current_usage() is None


def test_usage_scope_counts_a_normal_turn():
    with usage.usage_scope() as turn:
        usage.record_completion("gpt-4o-mini", 1000, 100)  # analyze
        usage.record_completion("gpt-4o-mini", 1500, 200)  # respond
    assert turn.chat_completions == 2
    assert turn.embeddings == 0
    assert turn.prompt_tokens == 2500
    assert turn.completion_tokens == 300
    assert turn.total_tokens == 2800
    assert turn.models == ["gpt-4o-mini"]


def test_usage_scope_counts_a_search_turn():
    with usage.usage_scope() as turn:
        usage.record_completion("gpt-4o-mini", 100, 10)
        usage.record_embedding("text-embedding-3-small", 20)
        usage.record_completion("gpt-4o-mini", 100, 10)
    assert (turn.chat_completions, turn.embeddings) == (2, 1)
    assert turn.total_tokens == 240
    assert turn.models == ["gpt-4o-mini", "text-embedding-3-small"]


def test_scopes_do_not_leak_into_each_other():
    with usage.usage_scope() as first:
        usage.record_completion("m", 1, 1)
        with usage.usage_scope() as second:
            usage.record_completion("m", 5, 5)
        assert second.chat_completions == 1
        usage.record_completion("m", 1, 1)
    assert first.chat_completions == 2
    assert first.prompt_tokens == 2


def test_as_dict_includes_total_tokens():
    with usage.usage_scope() as turn:
        usage.record_completion("m", 7, 3)
    assert turn.as_dict()["total_tokens"] == 10


# ---------------------------------------------------------------------------
# tools_fired
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "outcome,expected",
    [
        ({}, []),
        ({"search_ran": True}, ["search"]),
        ({"inventory_lookup_ran": True}, ["inventory_lookup"]),
        ({"emails_sent": ["Escalation"]}, ["email"]),
        ({"emails_sent": []}, []),
        ({"search_ran": True, "emails_sent": ["Results Shown to User"]}, ["search", "email"]),
    ],
)
def test_tools_fired(outcome, expected):
    assert turn_log.tools_fired(outcome) == expected


# ---------------------------------------------------------------------------
# log_turn
# ---------------------------------------------------------------------------

def test_log_turn_record_shape():
    with usage.usage_scope() as turn_usage:
        usage.record_completion("gpt-4o-mini", 100, 20)
        usage.record_completion("gpt-4o-mini", 100, 20)

    record = turn_log.log_turn(
        session_id="s1",
        turn_id="t1",
        intent="qualification_answer",
        category="Dump",
        latency_ms=1234.567,
        turn_outcome={"search_ran": True, "emails_sent": ["Results Shown to User"]},
        usage=turn_usage,
    )

    assert record["event"] == "chat_turn"
    assert record["session_id"] == "s1"
    assert record["turn_id"] == "t1"
    assert record["intent"] == "qualification_answer"
    assert record["category"] == "Dump"
    assert record["latency_ms"] == 1234.57  # rounded to 2dp
    assert record["tools_fired"] == ["search", "email"]
    assert record["emails_sent"] == ["Results Shown to User"]
    assert record["llm_calls"]["chat_completions"] == 2
    assert record["llm_calls"]["total_tokens"] == 240
    assert "error" not in record


def test_log_turn_without_usage_or_outcome():
    record = turn_log.log_turn(
        session_id="s", turn_id="t", intent=None, category=None, latency_ms=0, turn_outcome={}
    )
    assert record["llm_calls"] is None
    assert record["tools_fired"] == []


def test_log_turn_records_an_error():
    record = turn_log.log_turn(
        session_id="s", turn_id="t", intent=None, category=None, latency_ms=5, turn_outcome={}, error="boom"
    )
    assert record["error"] == "boom"


def test_log_turn_writes_jsonl_when_turn_log_path_set(tmp_path, monkeypatch):
    log_file = tmp_path / "turns.jsonl"
    replace_settings(monkeypatch, turn_log_path=str(log_file))
    turn_log.reset_for_tests()
    try:
        with usage.usage_scope() as turn_usage:
            usage.record_completion("gpt-4o-mini", 10, 2)
        turn_log.log_turn(
            session_id="s1", turn_id="t1", intent="faq", category=None, latency_ms=12,
            turn_outcome={}, usage=turn_usage,
        )
        turn_log.log_turn(
            session_id="s1", turn_id="t2", intent="smalltalk_other", category=None, latency_ms=8,
            turn_outcome={}, usage=turn_usage,
        )
    finally:
        turn_log.reset_for_tests()

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [line["turn_id"] for line in lines] == ["t1", "t2"]
    # Bare JSON per line — cost_report.py json.loads() each one with no prefix to strip.
    assert lines[0]["intent"] == "faq"


def test_log_turn_emits_on_the_turn_logger(caplog):
    with caplog.at_level(logging.INFO, logger=turn_log.TURN_LOGGER_NAME):
        turn_log.log_turn(session_id="s", turn_id="t", intent="faq", category=None, latency_ms=1, turn_outcome={})
    record = next(r for r in caplog.records if r.name == turn_log.TURN_LOGGER_NAME)
    assert record.turn["intent"] == "faq"


def test_json_formatter_inlines_the_turn_payload():
    from src.log_setup import JsonFormatter

    record = logging.LogRecord("trailerplace.turn", logging.INFO, __file__, 1, "chat_turn", None, None)
    record.turn = {"event": "chat_turn", "intent": "faq"}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["intent"] == "faq"
    assert payload["level"] == "INFO"
    assert "ts" in payload
