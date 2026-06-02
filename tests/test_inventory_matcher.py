from __future__ import annotations

from src.chatbot import inventory_matcher as im


def test_normalize_text_and_model_code_examples():
    assert im.normalize_text("Iron Bull Trailers") == "iron bull trailer"
    assert im.extract_model_code("Dtb 15K 72\"X12'X2'") == "dtb"
    assert im.extract_model_code("Fmax210 W/ Dovetail") == "fmax210"


def test_stock_search_matches_exact_inventory_row():
    extraction = im.TrailerQueryExtraction(
        stock_number="13066",
        user_wants_availability=True,
        search_intent="stock",
    )

    result = im.match_inventory("do you have stock 13066?", extraction)

    assert result["entity_type"] == "STOCK_SEARCH"
    assert result["best_match"]["stock_number"] == "13066"
    assert result["confidence"] == 1.0


def test_year_make_search_returns_inventory_matches():
    extraction = im.TrailerQueryExtraction(
        year=2026,
        possible_make="Diamond C",
        search_intent="year_make",
    )

    result = im.match_inventory("2026 Diamond C trailer", extraction)

    assert result["entity_type"] == "YEAR_MAKE_SEARCH"
    assert result["should_handle_in_chat"] is True
    assert result["top_matches"]
    assert all(str(item["year"]) == "2026" for item in result["top_matches"])
    assert all(item["make"] == "Diamond C" for item in result["top_matches"])


def test_single_year_make_availability_does_not_say_multiple(monkeypatch):
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "0")

    result = im.search_trailers("is the 2014 star trailer available?")

    assert result["entity_type"] == "YEAR_MAKE_SEARCH"
    assert len(result["top_matches"]) == 1
    assert "available in inventory" in result["reply"]
    assert "multiple" not in result["reply"].lower()
    assert "40329" in result["reply"]


def test_model_level_ambiguous_dtb_shows_top_matches_not_single_exact():
    extraction = im.TrailerQueryExtraction(
        year=2026,
        possible_make="Iron Bull Trailers",
        possible_model_code="DTB",
        possible_model_text="DTB",
        search_intent="year_make_model",
    )

    result = im.match_inventory("2026 DTB Iron Bull trailer", extraction)

    assert result["entity_type"] == "YEAR_MAKE_MODEL_SEARCH"
    assert result["best_match"] is None
    assert len(result["top_matches"]) > 1
    assert {item["stock_number"] for item in result["top_matches"]} & {"13066", "12906", "11710", "8242"}


def test_price_response_uses_inventory_price_for_stock_match():
    result = im.search_trailers("how much is stock 13066?")

    assert result["entity_type"] == "STOCK_SEARCH"
    assert "$9,495" in result["reply"]
    assert "13066" in result["reply"]


def test_make_only_search_is_classified_but_not_handled_in_chat():
    extraction = im.TrailerQueryExtraction(
        possible_make="Diamond C",
        search_intent="make",
    )

    result = im.match_inventory("Diamond C trailer", extraction)

    assert result["entity_type"] == "MAKE_SEARCH"
    assert result["should_handle_in_chat"] is False


def test_fuzzy_model_search_for_diamond_c_fmax210():
    extraction = im._fallback_extraction("diamnd c fmax210")

    result = im.match_inventory("diamnd c fmax210", extraction)

    assert result["entity_type"] in {"MODEL_SEARCH", "POSSIBLE_MODEL_SEARCH"}
    assert result["should_handle_in_chat"] is True
    assert any("Fmax210" in item["model"] for item in result["top_matches"])


def test_context_reference_uses_previous_listings():
    last_listings = [
        {"title": "Trailer A", "stock_number": "11111", "price": "$1"},
        {"title": "Trailer B", "stock_number": "22222", "price": "$2"},
    ]

    result = im.search_trailers(
        "what is the price of the second one?",
        last_listings=last_listings,
        for_chat=True,
    )

    assert result["entity_type"] == "STOCK_SEARCH"
    assert "Trailer B" in result["reply"]
    assert "$2" in result["reply"]


def test_ambiguous_context_reference_asks_user_to_choose():
    last_listings = [
        {"title": "Trailer A", "stock_number": "11111", "price": "$1"},
        {"title": "Trailer B", "stock_number": "22222", "price": "$2"},
    ]

    result = im.search_trailers(
        "is that available?",
        last_listings=last_listings,
        for_chat=True,
    )

    assert result["should_handle_in_chat"] is True
    assert "Which trailer do you mean" in result["reply"]
    assert "Trailer A" in result["reply"]
    assert "Trailer B" in result["reply"]
