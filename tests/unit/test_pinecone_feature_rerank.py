from __future__ import annotations

import logging

import pytest

from src.search import pinecone_search as search
from src.search.feature_ranker import (
    CandidateFeatureAssessment,
    FeatureMatch,
    FeatureRerankOutput,
    ValidatedFeatureRerank,
)


def _listing(
    url: str,
    *,
    features: list[str] | None = None,
    length: str = "20 ft",
    width: str = "8 ft",
    payload: str = "10000 lbs",
    score: float = 0.9,
) -> dict:
    return {
        "title": url,
        "url": url,
        "make": "Diamond C",
        "length": length,
        "width": width,
        "height": None,
        "payload_capacity": payload,
        "gvwr": "14000 lbs",
        "features": features or [],
        "relevance_score": score,
    }


def _combined(listings: list[dict], features: list[str], **requirements):
    defaults = {
        "required_length_ft": 20.0,
        "required_payload_lbs": None,
        "required_width_ft": None,
        "required_height_ft": None,
        "metadata_filter": {"category": {"$eq": "Equipment"}},
        "query_text": "Category: Equipment | Details: ramp gate",
    }
    defaults.update(requirements)
    return search._combined_feature_fit_rerank(
        listings,
        requested_features=features,
        **defaults,
    )


def test_feature_match_dominates_closer_dimension_without_dropping_candidate():
    close_without_feature = _listing("close", length="20 ft", features=[])
    oversized_with_feature = _listing(
        "feature", length="30 ft", features=["Spring-assisted rear ramp gate"]
    )

    ranked, debug = _combined(
        [close_without_feature, oversized_with_feature], ["ramp gate"]
    )

    assert [item["url"] for item in ranked] == ["feature", "close"]
    assert debug["candidate_count"] == 2
    feature_debug = next(x for x in debug["candidates"] if x["url"] == "feature")
    assert feature_debug["feature_coverage"] == 1.0
    assert feature_debug["fit_position"] == 2
    assert feature_debug["final_score"] == pytest.approx(0.85)


def test_dimension_failure_remains_eligible_in_combined_pool():
    fits_without_feature = _listing("fits", width="8 ft")
    narrow_with_feature = _listing(
        "narrow-feature", width="6 ft", features=["electric winch"]
    )

    ranked, debug = _combined(
        [fits_without_feature, narrow_with_feature],
        ["electric winch"],
        required_width_ft=8.0,
    )

    assert len(ranked) == 2
    assert ranked[0]["url"] == "narrow-feature"
    narrow_debug = next(
        x for x in debug["candidates"] if x["url"] == "narrow-feature"
    )
    assert narrow_debug["fit_fail_count"] == 1


def test_multiple_requested_features_use_average_coverage():
    all_features = _listing(
        "all", features=["electric winch", "LED interior lights"]
    )
    one_feature = _listing("one", features=["electric winch"])

    ranked, debug = _combined(
        [one_feature, all_features], ["electric winch", "LED lights"]
    )

    assert ranked[0]["url"] == "all"
    all_debug = next(x for x in debug["candidates"] if x["url"] == "all")
    one_debug = next(x for x in debug["candidates"] if x["url"] == "one")
    assert all_debug["feature_coverage"] > one_debug["feature_coverage"]


def test_below_threshold_scores_zero_and_preserves_fit_order():
    first = _listing("first", features=["spare tire"])
    second = _listing("second", features=[])

    ranked, debug = _combined([first, second], ["insulated interior"])

    assert [item["url"] for item in ranked] == ["first", "second"]
    assert all(item["feature_coverage"] == 0 for item in debug["candidates"])


def test_feature_matching_normalizes_case_and_punctuation():
    coverage, details = search._feature_match_details(
        ["RAMP-GATE"], ["Spring assisted rear ramp gate"]
    )
    assert coverage == 1.0
    assert details[0]["accepted"] is True


def test_feature_matching_handles_inflections_without_a_feature_dictionary():
    coverage, details = search._feature_match_details(
        ["insulated"], ["Interior package: insulation"]
    )

    assert coverage == pytest.approx(0.736842, abs=1e-6)
    assert details[0]["accepted"] is True
    assert details[0]["token_matches"][0]["matched_token"] == "insulation"


def test_every_requested_feature_token_must_match():
    coverage, details = search._feature_match_details(
        ["insulated enclosed"], ["Category: Enclosed"]
    )

    assert coverage == 0.0
    assert details[0]["accepted"] is False


@pytest.mark.parametrize(
    ("listing_overrides", "requested"),
    [
        ({"title": "Insulated Enclosed Trailer"}, "insulated"),
        ({"model": "Blackout Package"}, "blackout"),
        (
            {
                "match_evidence_text": (
                    "Additional specifications: Interior Lining: insulated walls"
                )
            },
            "insulated",
        ),
    ],
)
def test_feature_reranker_matches_title_model_and_embedding_evidence(
    listing_overrides, requested
):
    listing = _listing("candidate", features=[])
    listing.update(listing_overrides)

    ranked, debug = _combined([listing], [requested])

    assert ranked[0]["url"] == "candidate"
    assert debug["candidates"][0]["feature_coverage"] == 1.0
    assert debug["candidates"][0]["match_source_count"] >= 1


def test_legacy_fit_path_still_discards_failing_candidate():
    fits = _listing("fits", width="8 ft")
    fails = _listing("fails", width="6 ft", features=["electric winch"])

    ranked, debug = search._rerank_listings_by_fit(
        [fits, fails],
        required_length_ft=None,
        required_payload_lbs=None,
        required_width_ft=8.0,
        required_height_ft=None,
        warn_ratio=search.RERANK_WARN_RATIO,
        extreme_ratio=search.RERANK_EXTREME_RATIO,
        length_weight=search.RERANK_LENGTH_WEIGHT,
        missing_dim_penalty=search.RERANK_MISSING_DIM_PENALTY,
    )

    assert [item["url"] for item in ranked] == ["fits"]
    assert debug["retained_candidate_count"] == 1


def test_retain_all_keeps_undersize_candidates_and_orders_by_closeness():
    # The category-only relaxed retry (category_only_filters -> retain_all=True). A 50 ft request
    # against a pool where only one row meets 50 ft must NOT collapse to that one row: keep every
    # under-length candidate and rank them closest-first. Reproduces the live "1 of 17" livestock
    # bug where the relaxed pass returned a single trailer.
    meets = _listing("meets", length="50 ft")
    near = _listing("near", length="40 ft")
    far = _listing("far", length="24 ft")

    ranked, debug = search._rerank_listings_by_fit(
        [near, far, meets],
        required_length_ft=50.0,
        required_payload_lbs=None,
        required_width_ft=None,
        required_height_ft=None,
        warn_ratio=search.RERANK_WARN_RATIO,
        extreme_ratio=search.RERANK_EXTREME_RATIO,
        length_weight=search.RERANK_LENGTH_WEIGHT,
        missing_dim_penalty=search.RERANK_MISSING_DIM_PENALTY,
        retain_all=True,
    )

    # Every candidate is retained (not culled to the single 50 ft row) ...
    assert debug["retained_candidate_count"] == 3
    # ... and ordered by fit: the one that meets the length first, then closest under-length.
    assert [item["url"] for item in ranked] == ["meets", "near", "far"]


def test_retain_all_false_still_culls_undersize_candidates():
    # The normal (non-relaxed) path is unchanged: an under-length row is dropped when a fitting
    # row exists.
    meets = _listing("meets", length="50 ft")
    under = _listing("under", length="24 ft")

    ranked, debug = search._rerank_listings_by_fit(
        [under, meets],
        required_length_ft=50.0,
        required_payload_lbs=None,
        required_width_ft=None,
        required_height_ft=None,
        warn_ratio=search.RERANK_WARN_RATIO,
        extreme_ratio=search.RERANK_EXTREME_RATIO,
        length_weight=search.RERANK_LENGTH_WEIGHT,
        missing_dim_penalty=search.RERANK_MISSING_DIM_PENALTY,
    )

    assert [item["url"] for item in ranked] == ["meets"]
    assert debug["retained_candidate_count"] == 1


def test_combined_debug_logs_contain_cosine_features_fit_and_rank(caplog):
    caplog.set_level(logging.INFO, logger=search.__name__)
    _combined(
        [_listing("logged", features=["electric winch"], score=0.87)],
        ["winch"],
    )

    text = caplog.text
    assert "feature_fit_candidate" in text
    assert "pinecone_cosine_similarity" in text
    assert "feature_matches" in text
    assert "fit_order_score" in text
    assert "rank_movement" in text
    assert "feature_fit_summary" in text


def test_public_search_results_strip_internal_features_and_evidence(monkeypatch):
    monkeypatch.setattr(
        search,
        "search_pinecone_listing_result",
        lambda **kwargs: search.PineconeListingSearchResult(
            listings=[
                {
                    "title": "Trailer",
                    "url": "u1",
                    "features": ["electric winch"],
                    "match_evidence_text": "private evidence",
                }
            ],
            query_text="trailer",
            metadata_filter=None,
        ),
    )

    result = search.search_pinecone_listings(category="Utility", slots={})
    assert result == [{"title": "Trailer", "url": "u1"}]


def test_feature_search_reranks_full_pinecone_pool_before_result_limit(monkeypatch):
    class FakeIndex:
        def __init__(self):
            self.query_args = None

        def query(self, **kwargs):
            self.query_args = kwargs
            return {
                "matches": [
                    {
                        "score": 0.99,
                        "metadata": {
                            "title": "closest",
                            "url": "closest",
                            "category": "Equipment",
                            "make": "Diamond C",
                            "length": "20 ft",
                            "features": [],
                        },
                    },
                    {
                        "score": 0.88,
                        "metadata": {
                            "title": "feature",
                            "url": "feature",
                            "category": "Equipment",
                            "make": "P&C",
                            "length": "30 ft",
                            "features": ["spring-assisted rear ramp gate"],
                        },
                    },
                    {
                        "score": 0.77,
                        "metadata": {
                            "title": "third",
                            "url": "third",
                            "category": "Equipment",
                            "make": "Iron Bull Trailers",
                            "length": "22 ft",
                            "features": [],
                        },
                    },
                ]
            }

    fake_index = FakeIndex()
    monkeypatch.setattr(search, "_embed", lambda text: [0.1, 0.2])
    monkeypatch.setattr(search, "_pinecone_index", lambda: fake_index)
    monkeypatch.setattr(
        search,
        "rank_non_metadata_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline fallback")),
    )

    result = search.search_pinecone_listing_result(
        category="Equipment",
        slots={"haul_length_ft": 20.0},
        metadata_filters={"length_ft": 20.0},
        requested_features=["ramp gate"],
        top_k=50,
        max_recommendations=2,
    )

    assert fake_index.query_args["top_k"] == 50
    assert result.match_analysis["candidate_count"] == 3
    assert [item["url"] for item in result.listings] == ["feature", "closest"]
    assert result.make_debug["reason"] == "feature_order_preserved"


def _semantic_result(listings: list[dict], requested: list[str], matched_urls: set[str]):
    assessments = []
    evidence_by_id = {}
    for position, listing in enumerate(listings, 1):
        candidate_id = f"C{position:03d}"
        matched = listing["url"] in matched_urls
        evidence = str(listing.get("match_evidence_text") or listing.get("title") or "")
        assessments.append(
            CandidateFeatureAssessment(
                candidate_id=candidate_id,
                feature_matches=[
                    FeatureMatch(
                        requested_feature=requested[0],
                        matched=matched,
                        evidence=evidence if matched else None,
                        reason_code="semantic_equivalent" if matched else "not_found",
                        reason="semantic equivalent" if matched else None,
                    )
                ],
            )
        )
        evidence_by_id[candidate_id] = evidence
    output = FeatureRerankOutput(assessments=assessments)
    return ValidatedFeatureRerank(
        output=output,
        assessments_by_id={item.candidate_id: item for item in assessments},
        evidence_by_id=evidence_by_id,
        fallback_candidate_ids=frozenset(),
        validation_errors_by_id={},
        model="gpt-5-nano-2025-08-07",
        reasoning_effort="medium",
        latency_ms=12.0,
        request_ids=("req-test",),
        batch_count=1,
    )


def test_semantic_feature_score_is_85_percent_and_unmatched_candidate_gets_zero_feature_component():
    unmatched_best_fit = _listing("best-fit", length="20 ft")
    semantic_match = _listing("abbreviation", length="30 ft")
    semantic_match["match_evidence_text"] = "Title: Enclosed trailer w/insl"
    listings = [unmatched_best_fit, semantic_match]

    ranked, debug = search._combined_feature_fit_rerank(
        listings,
        requested_features=["insulated"],
        required_length_ft=20.0,
        required_payload_lbs=None,
        required_width_ft=None,
        required_height_ft=None,
        metadata_filter={"category": {"$eq": "Enclosed"}},
        query_text="Category: Enclosed | Details: insulated",
        semantic_rerank=_semantic_result(listings, ["insulated"], {"abbreviation"}),
    )

    assert [item["url"] for item in ranked] == ["abbreviation", "best-fit"]
    matched = next(item for item in debug["candidates"] if item["url"] == "abbreviation")
    unmatched = next(item for item in debug["candidates"] if item["url"] == "best-fit")
    assert matched["feature_coverage"] == 1.0
    assert matched["final_score"] >= 0.85
    assert unmatched["feature_coverage"] == 0.0
    assert unmatched["final_score"] == pytest.approx(0.15)
    assert debug["feature_match_mode"] == "gpt_semantic"
    assert debug["semantic_request_id"] == "req-test"


def test_semantic_multiple_features_use_evidence_supported_partial_coverage():
    listing = _listing("partial")
    requested = ["insulated", "air conditioning"]
    assessment = CandidateFeatureAssessment(
        candidate_id="C001",
        feature_matches=[
            FeatureMatch(
                requested_feature="insulated",
                matched=True,
                evidence="w/insl",
                reason_code="abbreviation",
                reason="Catalog abbreviation for insulation.",
            ),
            FeatureMatch(
                requested_feature="air conditioning",
                matched=False,
                evidence=None,
                reason_code="not_found",
                reason=None,
            ),
        ],
    )
    output = FeatureRerankOutput(assessments=[assessment])
    semantic = ValidatedFeatureRerank(
        output=output,
        assessments_by_id={"C001": assessment},
        evidence_by_id={"C001": "w/insl"},
        fallback_candidate_ids=frozenset(),
        validation_errors_by_id={},
        model="gpt-5-nano-2025-08-07",
        reasoning_effort="medium",
        latency_ms=10.0,
        request_ids=("req-partial",),
        batch_count=1,
    )

    _, debug = search._combined_feature_fit_rerank(
        [listing],
        requested_features=requested,
        required_length_ft=None,
        required_payload_lbs=None,
        required_width_ft=None,
        required_height_ft=None,
        metadata_filter=None,
        query_text="Details: insulated; air conditioning",
        semantic_rerank=semantic,
    )

    assert debug["candidates"][0]["feature_coverage"] == 0.5
    assert debug["candidates"][0]["final_score"] == pytest.approx(0.575)


def test_hard_metadata_filters_are_unchanged_for_feature_search_inputs():
    filters = search._metadata_filter(
        "Equipment",
        {"hitch_type": "Gooseneck", "haul_length_ft": 20.0},
        {"make": "Diamond C", "length_ft": 20.0},
    )
    clauses = filters["$and"]
    assert {"category": {"$eq": "Equipment"}} in clauses
    assert any("make" in clause for clause in clauses)
    assert {"hitch_type": {"$eq": "Gooseneck"}} in clauses
    assert {"length_ft_num": {"$gte": 20.0}} in clauses
