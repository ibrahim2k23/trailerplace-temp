from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.llm import usage
from src.search import feature_ranker as ranker
from tests.conftest import replace_settings


def _output(count: int, *, matched_ids: set[str] | None = None) -> ranker.FeatureRerankOutput:
    matched_ids = matched_ids or set()
    ids = [f"C{position:03d}" for position in range(1, count + 1)]
    return ranker.FeatureRerankOutput(
        ranked_candidate_ids=[*matched_ids, *[candidate_id for candidate_id in ids if candidate_id not in matched_ids]],
        assessments=[
            ranker.CandidateFeatureAssessment(
                candidate_id=candidate_id,
                feature_matches=[
                    ranker.FeatureMatch(
                        requested_feature="insulated",
                        matched=candidate_id in matched_ids,
                        evidence="w/insl" if candidate_id in matched_ids else None,
                        reason="The abbreviation states insulation." if candidate_id in matched_ids else "No insulation evidence.",
                    )
                ],
                reasoning_summary="Evidence-grounded feature assessment.",
            )
            for candidate_id in ids
        ],
    )


def test_gpt_ranker_handles_seventy_candidates_with_medium_reasoning_and_logs_everything(
    monkeypatch, caplog
):
    output = _output(70, matched_ids={"C001"})
    calls = []

    class FakeCompletions:
        def parse(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=output, refusal=None))],
                usage=SimpleNamespace(prompt_tokens=5000, completion_tokens=1000),
                _request_id="req-70",
            )

    fake_client = SimpleNamespace(
        beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    )
    monkeypatch.setattr(ranker, "_openai_client", lambda: fake_client)
    replace_settings(
        monkeypatch,
        feature_rerank_model="gpt-5-nano-2025-08-07",
        feature_rerank_reasoning_effort="medium",
        feature_rerank_timeout_seconds=60.0,
    )
    listings = [
        {
            "title": f"Trailer {position}",
            "match_evidence_text": "w/insl" if position == 1 else "standard plywood walls",
        }
        for position in range(1, 71)
    ]
    caplog.set_level(logging.INFO, logger=ranker.__name__)

    with usage.usage_scope() as turn_usage:
        result = ranker.rank_non_metadata_features(listings, ["insulated"])

    assert len(result.assessments_by_id) == 70
    assert result.semantic_rank_by_id["C001"] == 1
    assert calls[0]["model"] == "gpt-5-nano-2025-08-07"
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["response_format"] is ranker.FeatureRerankOutput
    assert calls[0]["timeout"] == 60.0
    assert turn_usage.chat_completions == 1
    assert turn_usage.feature_reranks == 1
    assert turn_usage.total_tokens == 6000
    assert caplog.text.count("feature_rank_input_candidate") == 70
    assert "feature_rank_response" in caplog.text
    assert "feature_rank_validated" in caplog.text


def test_affirmative_match_requires_verbatim_candidate_evidence():
    output = _output(1, matched_ids={"C001"})
    with pytest.raises(ranker.FeatureRerankValidationError, match="non-verbatim evidence"):
        ranker._validate_output(
            output,
            requested_features=["insulated"],
            evidence_by_id={"C001": "standard plywood walls"},
        )


def test_prompt_allows_semantic_equivalence_but_rejects_related_gate_styles():
    prompt = ranker._system_prompt()
    assert "insulated/insulation" in prompt
    assert "w/insl" in prompt
    assert "sliding/swing gates do not prove butterfly gates" in prompt
    assert "uncertain, matched must be false" in prompt
