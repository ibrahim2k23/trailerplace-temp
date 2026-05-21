from src.chatbot.tools import pinecone_search as ps


def _filter_parts(filt):
    return filt.get("$and", [filt])


def test_category_only_filter_is_added():
    filt = ps._metadata_filter("Equipment", {}, {})

    assert filt == {"category": {"$eq": "Equipment"}}


def test_length_filter_uses_numeric_metadata_field():
    filt = ps._metadata_filter(
        "Livestock",
        {},
        {"length_ft": "12 feet"},
    )

    assert {"category": {"$eq": "Livestock"}} in filt["$and"]
    assert {"length_ft_num": {"$gte": 12.0}} in filt["$and"]


def test_metadata_length_overrides_slot_length_for_search_constraints():
    filt = ps._metadata_filter(
        "Utility",
        {"trailer_size": "12 feet"},
        {"length_ft": "10 feet"},
    )

    assert {"length_ft_num": {"$gte": 10.0}} in filt["$and"]
    assert {"length_ft_num": {"$gte": 12.0}} not in filt["$and"]


def test_valid_hitch_filter_is_added():
    filt = ps._metadata_filter("Tilt", {}, {"hitch_type": "gooseneck"})

    assert {"category": {"$eq": "Tilt"}} in filt["$and"]
    assert {"hitch_type": {"$eq": "Gooseneck"}} in filt["$and"]


def test_invalid_hitch_filter_is_rejected():
    filt = ps._metadata_filter("Tilt", {}, {"hitch_type": "Tilt"})

    assert filt == {"category": {"$eq": "Tilt"}}


def test_aluminum_subcategory_filter_is_added():
    filt = ps._metadata_filter("Aluminum", {}, {"subcategory": "deckover"})

    assert {"category": {"$eq": "Aluminum"}} in filt["$and"]
    assert {"subcategory": {"$eq": "Deckover"}} in filt["$and"]


def test_non_aluminum_subcategory_filter_is_not_added():
    filt = ps._metadata_filter("Equipment", {}, {"subcategory": "deckover"})

    assert filt == {"category": {"$eq": "Equipment"}}


def test_width_payload_color_and_price_are_not_hard_filters():
    filt = ps._metadata_filter(
        "Equipment",
        {
            "haul_weight_lbs": "3000 lbs",
            "width_ft": "7 feet",
            "color": "black",
            "budget": "12000",
        },
        {
            "width_ft": "6 feet",
            "payload_lbs": "1500 lbs",
            "color": "red",
            "max_price": "10000",
        },
    )
    parts = _filter_parts(filt)

    assert parts == [{"category": {"$eq": "Equipment"}}]
    assert not any("width_ft_num" in part for part in parts)
    assert not any("payload_lbs_num" in part for part in parts)
    assert not any("color" in part for part in parts)
    assert not any("price" in part for part in parts)


def test_full_filter_contract_keeps_only_allowed_hard_filters():
    filt = ps._metadata_filter(
        "Aluminum",
        {
            "haul_weight_lbs": "3000 lbs",
            "width_ft": "7 feet",
            "color": "black",
            "budget": "12000",
            "hitch_type": "bumper pull",
        },
        {
            "length_ft": "14 feet",
            "width_ft": "6 feet",
            "payload_lbs": "1500 lbs",
            "hitch_type": "gooseneck",
            "subcategory": "utility",
            "color": "red",
            "max_price": "10000",
        },
    )

    assert filt == {
        "$and": [
            {"category": {"$eq": "Aluminum"}},
            {"hitch_type": {"$eq": "Gooseneck"}},
            {"subcategory": {"$eq": "Utility"}},
            {"length_ft_num": {"$gte": 14.0}},
        ]
    }
