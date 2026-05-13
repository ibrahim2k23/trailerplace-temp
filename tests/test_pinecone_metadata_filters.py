from src.chatbot.tools import pinecone_search as ps


def test_weight_filter_uses_payload_not_gvwr():
    filt = ps._metadata_filter(
        "Equipment",
        {"haul_weight_lbs": "3000 lbs"},
        {},
    )

    assert {"payload_lbs_num": {"$gte": 3000.0}} in filt["$and"]
    assert {"gvwr_lbs_num": {"$gte": 3000.0}} not in filt["$and"]


def test_width_and_length_filters_use_numeric_metadata_fields():
    filt = ps._metadata_filter(
        "Livestock",
        {},
        {"length_ft": "12 feet", "width_ft": "6 feet"},
    )

    assert {"length_ft_num": {"$gte": 12.0}} in filt["$and"]
    assert {"width_ft_num": {"$gte": 6.0}} in filt["$and"]


def test_metadata_filters_override_slot_filters_for_search_constraints():
    filt = ps._metadata_filter(
        "Utility",
        {"trailer_size": "12 feet", "haul_weight_lbs": "2000 lbs"},
        {"length_ft": "10 feet", "payload_lbs": "1500 lbs"},
    )

    assert {"length_ft_num": {"$gte": 10.0}} in filt["$and"]
    assert {"payload_lbs_num": {"$gte": 1500.0}} in filt["$and"]


def test_valid_hitch_filter_is_added():
    filt = ps._metadata_filter("Tilt", {}, {"hitch_type": "gooseneck"})

    assert {"hitch_type": {"$eq": "Gooseneck"}} in filt["$and"]


def test_invalid_hitch_filter_is_rejected():
    filt = ps._metadata_filter("Tilt", {}, {"hitch_type": "Tilt"})

    assert filt == {"category": {"$eq": "Tilt"}}


def test_subcategory_filter_is_added():
    filt = ps._metadata_filter("Aluminum", {}, {"subcategory": "deckover"})

    assert {"subcategory": {"$eq": "Deckover"}} in filt["$and"]
