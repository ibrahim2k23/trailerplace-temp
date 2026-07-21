from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from src.llm import usage
from src.search import feature_ranker as ranker
from tests.conftest import replace_settings


def _output(ids: list[str], *, matched_ids: set[str] | None = None) -> ranker.FeatureRerankOutput:
    matched_ids = matched_ids or set()
    return ranker.FeatureRerankOutput(
        assessments=[
            ranker.CandidateFeatureAssessment(
                candidate_id=candidate_id,
                feature_matches=[
                    ranker.FeatureMatch(
                        requested_feature="insulated",
                        matched=candidate_id in matched_ids,
                        evidence="w/insl" if candidate_id in matched_ids else None,
                        reason_code="abbreviation" if candidate_id in matched_ids else "not_found",
                        reason="The abbreviation states insulation." if candidate_id in matched_ids else None,
                    )
                ],
            )
            for candidate_id in ids
        ],
    )


def test_gpt_ranker_handles_seventy_candidates_with_medium_reasoning_and_logs_everything(
    monkeypatch, caplog
):
    calls = []

    class FakeCompletions:
        def parse(self, **kwargs):
            calls.append(kwargs)
            payload = json.loads(kwargs["messages"][1]["content"])
            ids = [item["candidate_id"] for item in payload["candidates"]]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=_output(ids, matched_ids={"C001"}), refusal=None))],
                usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
                _request_id=f"req-{ids[0]}",
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
        feature_rerank_batch_size=20,
        feature_rerank_max_parallel_batches=4,
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
    assert result.fallback_candidate_ids == frozenset()
    assert result.batch_count == 4
    assert len(calls) == 4
    assert calls[0]["model"] == "gpt-5-nano-2025-08-07"
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["response_format"] is ranker.FeatureRerankOutput
    assert calls[0]["timeout"] == 60.0
    assert turn_usage.chat_completions == 4
    assert turn_usage.feature_reranks == 4
    assert turn_usage.total_tokens == 4800
    assert caplog.text.count("feature_rank_input_candidate") == 70
    assert "feature_rank_response" in caplog.text
    assert "feature_rank_validated" in caplog.text


def test_affirmative_match_requires_verbatim_candidate_evidence():
    output = _output(["C001"], matched_ids={"C001"})
    with pytest.raises(ranker.FeatureRerankValidationError, match="ungrounded evidence"):
        ranker._validate_output(
            output,
            requested_features=["insulated"],
            evidence_by_id={"C001": "standard plywood walls"},
        )


def test_grounding_accepts_html_entities_and_harmless_punctuation_changes():
    output = ranker.FeatureRerankOutput(
        assessments=[ranker.CandidateFeatureAssessment(
            candidate_id="C001",
            feature_matches=[ranker.FeatureMatch(
                requested_feature="insulated",
                matched=True,
                evidence="Enclosed Trailer w/ Insulation Electric",
                reason_code="semantic_equivalent",
                reason="Insulation is explicitly stated.",
            )],
        )]
    )
    validated = ranker._validate_output(
        output,
        requested_features=["insulated"],
        evidence_by_id={"C001": "Enclosed Trailer w/ Insulation &amp; Electric"},
    )
    assert set(validated) == {"C001"}


def test_compact_evidence_removes_metadata_and_keeps_arbitrary_specs():
    evidence = ranker._candidate_evidence({
        "title": "Utility Trailer",
        "model": "Utility Trailer",
        "trim": "Base",
        "features": ["Butterfly Gates"],
        "match_evidence_text": (
            "Utility Trailer | Make: Example | Price: $1,000 | "
            "Rear Door: Butterfly Gates | Url: https://example.test/item"
        ),
    })
    assert "Butterfly Gates" in evidence
    assert "Price:" not in evidence
    assert "Url:" not in evidence
    assert "Make:" not in evidence


def test_one_ungrounded_candidate_falls_back_without_discarding_valid_candidates(monkeypatch):
    output = ranker.FeatureRerankOutput(assessments=[
        ranker.CandidateFeatureAssessment(
            candidate_id="C001",
            feature_matches=[ranker.FeatureMatch(
                requested_feature="insulated", matched=True,
                evidence="w/ insulation", reason_code="explicit",
                reason="Insulation is explicit.",
            )],
        ),
        ranker.CandidateFeatureAssessment(
            candidate_id="C002",
            feature_matches=[ranker.FeatureMatch(
                requested_feature="insulated", matched=True,
                evidence="insulated walls", reason_code="explicit",
                reason="Insulation is explicit.",
            )],
        ),
    ])

    class FakeCompletions:
        def parse(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=output, refusal=None))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                _request_id="req-partial-validation",
            )

    monkeypatch.setattr(
        ranker,
        "_openai_client",
        lambda: SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))),
    )
    replace_settings(
        monkeypatch,
        feature_rerank_batch_size=20,
        feature_rerank_max_parallel_batches=1,
    )
    result = ranker.rank_non_metadata_features(
        [
            {"title": "Enclosed w/ insulation"},
            {"title": "Standard plywood walls"},
        ],
        ["insulated"],
    )
    assert set(result.assessments_by_id) == {"C001"}
    assert result.fallback_candidate_ids == frozenset({"C002"})
    assert "ungrounded evidence" in result.validation_errors_by_id["C002"]


def test_one_failed_parallel_batch_does_not_discard_successful_batches(monkeypatch):
    class FakeCompletions:
        def parse(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            candidate_id = payload["candidates"][0]["candidate_id"]
            if candidate_id == "C002":
                raise TimeoutError("batch timed out")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    parsed=_output([candidate_id]), refusal=None,
                ))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                _request_id="req-successful-batch",
            )

    monkeypatch.setattr(
        ranker,
        "_openai_client",
        lambda: SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))),
    )
    replace_settings(
        monkeypatch,
        feature_rerank_batch_size=1,
        feature_rerank_max_parallel_batches=2,
    )
    result = ranker.rank_non_metadata_features(
        [{"title": "First"}, {"title": "Second"}], ["insulated"]
    )
    assert set(result.assessments_by_id) == {"C001"}
    assert result.fallback_candidate_ids == frozenset({"C002"})
    assert "batch timed out" in result.validation_errors_by_id["C002"]


def test_prompt_allows_semantic_equivalence_but_rejects_related_gate_styles():
    prompt = ranker._system_prompt()
    assert "insulated/insulation" in prompt
    assert "w/insl" in prompt
    assert "sliding/swing gates do not prove butterfly gates" in prompt
    assert "uncertain, matched must be false" in prompt
