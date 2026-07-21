"""Milestone 9 §3 — the cost audit. Network-free; parses the turn log JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cost_report  # noqa: E402
from cost_report import audit, load_turns, summarize, violations  # noqa: E402


def _record(
    *, completions=2, feature_reranks=0, embeddings=0, tools=(),
    intent="qualification_answer", tokens=1000
):
    return {
        "event": "chat_turn",
        "session_id": "s1",
        "turn_id": "t1",
        "intent": intent,
        "category": "Dump",
        "latency_ms": 1500.0,
        "tools_fired": list(tools),
        "emails_sent": [],
        "llm_calls": {
            "chat_completions": completions,
            "feature_reranks": feature_reranks,
            "embeddings": embeddings,
            "prompt_tokens": tokens,
            "completion_tokens": 0,
            "total_tokens": tokens,
            "models": ["gpt-4o-mini"],
        },
    }


def _write(tmp_path, records, name="turns.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_load_turns_skips_blanks_and_other_events(tmp_path):
    path = tmp_path / "turns.jsonl"
    path.write_text(
        json.dumps(_record()) + "\n\n" + json.dumps({"event": "startup"}) + "\n" + json.dumps(_record()) + "\n",
        encoding="utf-8",
    )
    assert len(load_turns(path)) == 2


def test_load_turns_rejects_bad_json(tmp_path):
    path = tmp_path / "turns.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_turns(path)


# ---------------------------------------------------------------------------
# the budget itself
# ---------------------------------------------------------------------------

def test_normal_turn_of_two_completions_is_within_budget():
    assert violations(summarize(_record(completions=2))) == []


def test_three_completions_violates_the_two_call_budget():
    problems = violations(summarize(_record(completions=3)))
    assert len(problems) == 1
    assert "3 chat completions" in problems[0]


def test_feature_search_allows_one_tagged_third_completion():
    turn = summarize(
        _record(
            completions=3,
            feature_reranks=1,
            embeddings=1,
            tools=["search", "feature_rerank"],
        )
    )
    assert violations(turn) == []


def test_feature_rerank_on_non_search_turn_is_flagged():
    turn = summarize(_record(completions=3, feature_reranks=1, tools=["feature_rerank"]))
    assert "feature rerank ran on a non-search turn" in violations(turn)


def test_more_than_one_feature_rerank_is_flagged():
    turn = summarize(
        _record(completions=4, feature_reranks=2, embeddings=1, tools=["search", "feature_rerank"])
    )
    assert any("maximum 1" in problem for problem in violations(turn))


def test_search_turn_gets_exactly_one_embedding():
    assert violations(summarize(_record(completions=2, embeddings=1, tools=["search"]))) == []


def test_search_turn_with_no_embedding_is_flagged():
    problems = violations(summarize(_record(completions=2, embeddings=0, tools=["search"])))
    assert "0 embeddings on a search turn" in problems[0]


def test_embedding_on_a_non_search_turn_is_flagged():
    problems = violations(summarize(_record(completions=2, embeddings=1, tools=[])))
    assert "1 embeddings on a non-search turn" in problems[0]


def test_inventory_lookup_turn_adds_zero_calls():
    """The Excel matcher uses no LLM and no embedding (Locked Decision)."""
    turn = summarize(_record(completions=2, embeddings=0, tools=["inventory_lookup"], intent="inventory_lookup"))
    assert violations(turn) == []


def test_inventory_lookup_that_embeds_is_flagged():
    turn = summarize(_record(completions=2, embeddings=1, tools=["inventory_lookup"], intent="inventory_lookup"))
    problems = violations(turn)
    # Both the non-search embedding rule and the lookup-specific rule catch it.
    assert any("must make none" in p for p in problems)


def test_receipt_replay_is_skipped_not_failed():
    """A replayed turn runs no graph, so zero calls is correct, not a violation."""
    turn = summarize(_record(completions=0, embeddings=0))
    assert turn.is_replay
    assert violations(turn) == []


def test_max_completions_is_configurable():
    assert violations(summarize(_record(completions=3)), max_completions=3) == []


# ---------------------------------------------------------------------------
# audit aggregation
# ---------------------------------------------------------------------------

def test_audit_of_a_clean_ten_turn_conversation():
    """The M9 acceptance test: the cost assertion holds on a 10-turn conversation."""
    records = [_record(completions=2, tokens=1000) for _ in range(8)]
    records.append(_record(completions=2, embeddings=1, tools=["search"], tokens=2000))
    records.append(_record(completions=2, embeddings=0, tools=["inventory_lookup"], tokens=1500))

    result = audit(records)
    assert result["failures"] == []
    assert result["audited"] == 10
    assert result["replays"] == 0
    assert result["total_completions"] == 20  # exactly 2 per turn
    assert result["total_embeddings"] == 1  # only the search turn
    assert result["total_tokens"] == 11500
    assert result["avg_tokens_per_turn"] == 1150.0


def test_audit_counts_replays_separately():
    result = audit([_record(), _record(completions=0)])
    assert (result["audited"], result["replays"]) == (1, 1)


def test_audit_collects_every_failing_turn():
    result = audit([_record(), _record(completions=4), _record(embeddings=2)])
    assert len(result["failures"]) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exits_zero_on_a_clean_log(tmp_path, capsys):
    path = _write(tmp_path, [_record(), _record(completions=2, embeddings=1, tools=["search"])])
    assert cost_report.main([str(path)]) == 0
    assert "All turns within budget." in capsys.readouterr().out


def test_cli_exits_nonzero_on_a_violation(tmp_path, capsys):
    path = _write(tmp_path, [_record(completions=5)])
    assert cost_report.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "BUDGET VIOLATIONS" in out
    assert "5 chat completions" in out


def test_cli_missing_file(tmp_path, capsys):
    assert cost_report.main([str(tmp_path / "nope.jsonl")]) == 1
    assert "No such log file" in capsys.readouterr().out


def test_cli_empty_log(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert cost_report.main([str(path)]) == 1
    assert "No chat_turn records" in capsys.readouterr().out


def test_cli_round_trips_a_real_turn_log(tmp_path, monkeypatch):
    """End-to-end: turn_log writes it, cost_report reads it. Guards the format contract."""
    from src import turn_log
    from src.llm import usage
    from tests.conftest import replace_settings

    log_file = tmp_path / "turns.jsonl"
    replace_settings(monkeypatch, turn_log_path=str(log_file))
    turn_log.reset_for_tests()
    try:
        with usage.usage_scope() as turn_usage:
            usage.record_completion("gpt-4o-mini", 900, 100)
            usage.record_embedding("text-embedding-3-small", 12)
            usage.record_completion("gpt-4o-mini", 1200, 150)
        turn_log.log_turn(
            session_id="s1", turn_id="t1", intent="recommendation_request", category="Dump",
            latency_ms=2100, turn_outcome={"search_ran": True}, usage=turn_usage,
        )
    finally:
        turn_log.reset_for_tests()

    result = audit(load_turns(log_file))
    assert result["failures"] == []
    assert result["total_completions"] == 2
    assert result["total_embeddings"] == 1
