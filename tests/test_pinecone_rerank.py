from src.chatbot.tools import pinecone_search as ps


def _mk(title, length, score=0.5, gvwr=None, payload=None, width=None):
    return {
        "title": title,
        "length": length,
        "gvwr": gvwr,
        "payload_capacity": payload,
        "width": width,
        "relevance_score": score,
        "url": f"https://x/{title}",
    }


def test_length_first_rerank_prefers_nearest_overage():
    listings = [
        _mk("L30", "30 ft", 0.9),
        _mk("L16", "16 ft", 0.4),
        _mk("L13", "13 ft", 0.3),
        _mk("L12", "12 ft", 0.2),
        _mk("L25", "25 ft", 0.8),
    ]
    ranked, dbg = ps._rerank_listings_by_fit(
        listings,
        required_length_ft=12.0,
        required_payload_lbs=None,
        required_width_ft=None,
        warn_ratio=1.35,
        extreme_ratio=1.9,
        length_weight=8.0,
        missing_dim_penalty=0.35,
    )
    assert dbg["applied"] is True
    assert [x["title"] for x in ranked[:3]] == ["L12", "L13", "L16"]


def test_under_length_is_not_preferred_over_valid_lengths():
    listings = [
        _mk("L11", "11 ft", 0.95),
        _mk("L12", "12 ft", 0.2),
        _mk("L13", "13 ft", 0.2),
    ]
    ranked, _ = ps._rerank_listings_by_fit(
        listings,
        required_length_ft=12.0,
        required_payload_lbs=None,
        required_width_ft=None,
        warn_ratio=1.35,
        extreme_ratio=1.9,
        length_weight=8.0,
        missing_dim_penalty=0.35,
    )
    assert [x["title"] for x in ranked[:2]] == ["L12", "L13"]


def test_rerank_skips_when_no_requirements():
    listings = [_mk("A", "20 ft", 0.9), _mk("B", "12 ft", 0.1)]
    ranked, dbg = ps._rerank_listings_by_fit(
        listings,
        required_length_ft=None,
        required_payload_lbs=None,
        required_width_ft=None,
        warn_ratio=1.35,
        extreme_ratio=1.9,
        length_weight=8.0,
        missing_dim_penalty=0.35,
    )
    assert dbg["applied"] is False
    assert ranked == listings


def test_payload_capacity_is_used_before_gvwr_for_weight_fit():
    listings = [
        _mk("high-gvwr-low-payload", "12 ft", 0.9, gvwr="14,000 lbs", payload="1,000 lbs"),
        _mk("payload-match", "12 ft", 0.2, gvwr="3,500 lbs", payload="3,000 lbs"),
    ]
    ranked, dbg = ps._rerank_listings_by_fit(
        listings,
        required_length_ft=None,
        required_payload_lbs=3000.0,
        required_width_ft=None,
        warn_ratio=1.35,
        extreme_ratio=1.9,
        length_weight=8.0,
        missing_dim_penalty=0.35,
    )
    assert dbg["required_payload_lbs"] == 3000.0
    assert [x["title"] for x in ranked] == ["payload-match"]
