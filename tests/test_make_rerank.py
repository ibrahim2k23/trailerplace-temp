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
