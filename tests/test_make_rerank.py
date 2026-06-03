from src.chatbot.tools import pinecone_search as ps


def _mk(title, make, length="12 ft", score=0.5):
    return {
        "title": title,
        "make": make,
        "length": length,
        "relevance_score": score,
        "url": f"https://x/{title}",
    }


def test_make_rerank_equipment_preference_order():
    listings = [
        _mk("other", "Some Brand"),
        _mk("ib1", "Iron Bull"),
        _mk("pc1", "p and c"),
        _mk("dc1", "Diamond C Trailers"),
        _mk("ib2", "IRON BULL TRAILERS"),
    ]
    ranked, dbg = ps._apply_category_make_preference(listings, category="Equipment")
    assert dbg["applied"] is True
    assert [x["title"] for x in ranked[:4]] == ["dc1", "ib1", "ib2", "pc1"]
    assert ranked[-1]["title"] == "other"


def test_make_rerank_three_brand_quota_split():
    listings = [
        _mk("dc1", "Diamond C Trailers"),
        _mk("dc2", "Diamond C"),
        _mk("dc3", "diamond c"),
        _mk("dc4", "Diamond C Trailers"),
        _mk("ib1", "Iron Bull"),
        _mk("ib2", "IRON BULL TRAILERS"),
        _mk("ib3", "Iron Bull"),
        _mk("pc1", "p and c"),
        _mk("pc2", "P&C"),
        _mk("other1", "Some Brand"),
        _mk("other2", "Some Brand"),
    ]

    ranked, dbg = ps._apply_category_make_preference(listings, category="Equipment", max_recommendations=6)

    assert dbg["quota_template"] == [3, 2, 1]
    assert [x["title"] for x in ranked] == ["dc1", "dc2", "dc3", "ib1", "ib2", "pc1"]


def test_make_rerank_two_brand_quota_split():
    listings = [
        _mk("dc1", "Diamond C Trailers"),
        _mk("dc2", "Diamond C"),
        _mk("dc3", "diamond c"),
        _mk("dc4", "Diamond C Trailers"),
        _mk("ib1", "Iron Bull"),
        _mk("ib2", "IRON BULL TRAILERS"),
        _mk("ib3", "Iron Bull"),
        _mk("ib4", "IRON BULL TRAILERS"),
    ]

    ranked, dbg = ps._apply_category_make_preference(listings, category="Equipment", max_recommendations=6)

    assert dbg["quota_template"] == [3, 3]
    assert [x["title"] for x in ranked] == ["dc1", "dc2", "dc3", "ib1", "ib2", "ib3"]


def test_make_rerank_backfills_short_brand_counts():
    listings = [
        _mk("dc1", "Diamond C Trailers"),
        _mk("dc2", "Diamond C Trailers"),
        _mk("ib1", "Iron Bull"),
        _mk("ib2", "Iron Bull"),
        _mk("ib3", "Iron Bull"),
        _mk("ib4", "IRON BULL TRAILERS"),
    ]

    ranked, dbg = ps._apply_category_make_preference(listings, category="Equipment", max_recommendations=6)

    assert dbg["quota_template"] == [3, 3]
    assert [x["title"] for x in ranked] == ["dc1", "dc2", "ib1", "ib2", "ib3", "ib4"]


def test_make_rerank_single_brand_returns_all_available_up_to_limit():
    listings = [
        _mk("dc1", "Diamond C Trailers"),
        _mk("dc2", "Diamond C"),
        _mk("dc3", "diamond c"),
        _mk("dc4", "Diamond C Trailers"),
        _mk("dc5", "Diamond C"),
        _mk("dc6", "Diamond C Trailers"),
        _mk("dc7", "Diamond C"),
    ]

    ranked, dbg = ps._apply_category_make_preference(listings, category="Equipment", max_recommendations=6)

    assert dbg["quota_template"] == [6]
    assert len(ranked) == 6
    assert all(ps._canonical_make(x["make"]) == "Diamond C" for x in ranked)


def test_make_alias_normalization_examples():
    assert ps._canonical_make(" Cargo   Craft Trailers ") == "Cargo Craft"
    assert ps._canonical_make("iron bull") == "Iron Bull Trailers"
    assert ps._canonical_make("P and C") == "P&C"


def test_livestock_gooseneck_treated_as_make_value():
    listings = [
        _mk("other", "Random"),
        _mk("g1", "Galyean"),
        _mk("gn1", "Gooseneck"),
        _mk("c1", "Calico"),
    ]
    ranked, _ = ps._apply_category_make_preference(listings, category="Livestock")
    assert [x["title"] for x in ranked[:3]] == ["g1", "gn1", "c1"]


def test_make_rerank_noop_for_unknown_category():
    listings = [_mk("a", "X"), _mk("b", "Y")]
    ranked, dbg = ps._apply_category_make_preference(listings, category="UnknownCat")
    assert dbg["applied"] is False
    assert ranked == listings
