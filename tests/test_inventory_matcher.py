from __future__ import annotations

import pandas as pd

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


def _feature_df() -> pd.DataFrame:
    return im.prepare_inventory(
        pd.DataFrame(
            [
                {
                    "title": "2024 Aluma Utility Trailer",
                    "url": "https://example.test/aluma-basic",
                    "year": "2024",
                    "make": "Aluma",
                    "model": "8214",
                    "trim": "",
                    "category": "Utility",
                    "subcategory": "",
                    "stock_number": "A100",
                    "price": "$10,995",
                    "condition": "New",
                    "length": "14 ft",
                    "width": "82 in",
                    "gvwr": "2,990 lbs",
                    "payload_capacity": "2,100 lbs",
                    "hitch_type": "Bumper Pull",
                    "dealer_notes": "Lightweight aluminum utility trailer.",
                },
                {
                    "title": "2024 Aluma Utility Trailer With Sliding Gates",
                    "url": "https://example.test/aluma-sliding",
                    "year": "2024",
                    "make": "Aluma",
                    "model": "8214SG",
                    "trim": "",
                    "category": "Utility",
                    "subcategory": "",
                    "stock_number": "A101",
                    "price": "$12,495",
                    "condition": "New",
                    "length": "14 ft",
                    "width": "82 in",
                    "gvwr": "2,990 lbs",
                    "payload_capacity": "2,000 lbs",
                    "hitch_type": "Bumper Pull",
                    "dealer_notes": "Includes sliding gates for flexible loading.",
                },
            ]
        )
    )


def _aluma_alternative_df() -> pd.DataFrame:
    return im.prepare_inventory(
        pd.DataFrame(
            [
                {
                    "title": "2023 Aluma Utility Trailer",
                    "url": "https://example.test/aluma-2023",
                    "year": "2023",
                    "make": "Aluma",
                    "model": "8214",
                    "category": "Utility",
                    "stock_number": "A099",
                    "price": "$9,995",
                    "length": "14 ft",
                },
                {
                    "title": "2024 Alcom Enclosed Trailer",
                    "url": "https://example.test/alcom-2024",
                    "year": "2024",
                    "make": "Alcom",
                    "model": "E716",
                    "category": "Cargo",
                    "stock_number": "C100",
                    "price": "$12,995",
                    "length": "16 ft",
                },
            ]
        )
    )


class _FeatureFramingLLM:
    def __init__(self, decision: im.InventoryFeatureFramingDecision, captured: dict | None = None):
        self.decision = decision
        self.captured = captured

    def invoke(self, messages):
        if self.captured is not None:
            self.captured["system"] = messages[0].content
            self.captured["human"] = messages[1].content
        return self.decision


class _InventoryIntroLLM:
    def __init__(self, text: str, captured: dict | None = None):
        self.text = text
        self.captured = captured

    def invoke(self, messages):
        if self.captured is not None:
            self.captured["system"] = messages[0].content
            self.captured["human"] = messages[1].content
        return type("_Response", (), {"content": self.text})()


def test_direct_year_make_price_exact_match_handles_in_chat(monkeypatch):
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "0")
    extraction = im.TrailerQueryExtraction(
        year=2024,
        possible_make="Aluma",
        user_wants_price=True,
        search_intent="year_make",
    )

    result = im.match_inventory("2024 Aluma price?", extraction, df=_feature_df())
    reply = im.generate_inventory_response("2024 Aluma price?", extraction, result)

    assert result["entity_type"] == "YEAR_MAKE_SEARCH"
    assert result["should_handle_in_chat"] is True
    assert result["exact_match_count"] == 2
    assert result["metadata_filter"] == {"year": "2024"}
    assert result["no_exact_reason"] == ""
    assert "$10,995" in reply


def test_exact_year_make_price_rejects_llm_no_match_intro(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "1")
    monkeypatch.setattr(
        im,
        "_inventory_reply_llm",
        lambda: _InventoryIntroLLM(
            "We do not currently show a matching trailer in the inventory, but we have several 2024 Aluma options available.",
            captured=captured,
        ),
    )
    extraction = im.TrailerQueryExtraction(
        year=2024,
        possible_make="Aluma",
        user_wants_price=True,
        search_intent="year_make",
    )
    result = im.match_inventory("2024 Aluma price?", extraction, df=_feature_df())

    reply = im.generate_inventory_response("2024 Aluma price?", extraction, result)

    assert result["exact_match_count"] == 2
    assert result["no_exact_reason"] == ""
    assert "do not currently show" not in reply
    assert reply.startswith("We have 2 matching trailers in inventory for that year and make.")
    assert "Match status: exact" in captured["human"]
    assert "Exact match count: 2" in captured["human"]
    assert "No exact reason: " in captured["human"]
    assert "If match_status is exact, never use no-match wording" in captured["system"]


def test_direct_year_make_no_exact_still_handles_with_same_make_alternative(monkeypatch):
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "0")
    extraction = im.TrailerQueryExtraction(
        year=2024,
        possible_make="Aluma",
        user_wants_availability=True,
        search_intent="year_make",
    )

    result = im.match_inventory("is 2024 Aluma available?", extraction, df=_aluma_alternative_df())
    reply = im.generate_inventory_response("is 2024 Aluma available?", extraction, result)

    assert result["should_handle_in_chat"] is True
    assert result["exact_match_count"] == 0
    assert result["metadata_filter"] == {"year": "2024"}
    assert result["no_exact_reason"] == "no_exact_year_filtered_match_fallback_discarded_year"
    assert result["top_matches"][0]["make"] == "Aluma"
    assert result["top_matches"][0]["year"] == "2023"
    assert "do not currently show 2024 Aluma" in reply
    assert "Trailer #1: [2023 Aluma Utility Trailer]" in reply


def test_year_is_not_metadata_filter_for_non_price_availability_lookup():
    extraction = im.TrailerQueryExtraction(
        year=2024,
        possible_make="Aluma",
        search_intent="year_make",
    )

    result = im.match_inventory("2024 Aluma trailer", extraction, df=_aluma_alternative_df())

    assert result["metadata_filter"] == {}


def test_stock_id_lookup_direct_handles_short_price_wording(monkeypatch):
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "0")
    stock_df = im.prepare_inventory(
        pd.DataFrame(
            [
                {
                    "title": "2024 Aluma Utility Trailer",
                    "url": "https://example.test/aluma-stock",
                    "year": "2024",
                    "make": "Aluma",
                    "model": "8214",
                    "category": "Utility",
                    "stock_number": "12345",
                    "price": "$10,995",
                }
            ]
        )
    )
    monkeypatch.setattr(im, "prepared_inventory", lambda: stock_df)

    result = im.search_trailers("price for id 12345", for_chat=True)

    assert result["entity_type"] == "STOCK_SEARCH"
    assert result["should_handle_in_chat"] is True
    assert "$10,995" in result["reply"]


def test_make_model_price_lookup_direct_handles(monkeypatch):
    monkeypatch.setenv("INVENTORY_REPLY_LLM_ENABLED", "0")
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            possible_make="Aluma",
            possible_model_code="8214",
            possible_model_text="8214",
            user_wants_price=True,
            search_intent="model",
        ),
    )

    result = im.search_trailers("Aluma 8214 price?", for_chat=True)

    assert result["should_handle_in_chat"] is True
    assert result["top_matches"]
    assert "$10,995" in result["reply"]


def test_make_only_price_lookup_is_not_direct_handled():
    extraction = im._fallback_extraction("Aluma price?", df=_feature_df())

    assert extraction.possible_make == "Aluma"
    assert extraction.user_wants_price is True
    assert im._is_direct_inventory_lookup(extraction) is False
    assert im.should_attempt_chat_lookup("Aluma price?") is False


def test_feature_price_query_frames_unavailable_configuration_with_alternatives(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df().iloc[[0]].copy())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            year=2024,
            possible_make="Aluma",
            user_wants_price=True,
            requested_features=["sliding gates"],
            search_intent="year_make",
        ),
    )
    monkeypatch.setattr(
        im,
        "_inventory_feature_framing_llm",
        lambda: _FeatureFramingLLM(
            im.InventoryFeatureFramingDecision(
                intro_text=(
                    "We do not currently show that exact 2024 Aluma configuration with sliding gates, "
                    "but this is the closest listed Aluma option and its price is shown below."
                ),
                configuration_match_level="no_exact",
                missing_or_unconfirmed_features=["sliding gates"],
                full_match_count=0,
            ),
            captured,
        ),
    )

    result = im.search_trailers("what is the price of a 2024 Aluma with sliding gates?")

    assert result["entity_type"] == "YEAR_MAKE_SEARCH"
    assert result["feature_match_analysis"]["configuration_match_level"] == "no_exact"
    assert result["reply"].startswith("We do not currently show that exact 2024 Aluma configuration")
    assert "Trailer #1: [2024 Aluma Utility Trailer](https://example.test/aluma-basic)" in result["reply"]
    assert "$10,995" in result["reply"]
    assert "match_evidence_text" not in result["top_matches"][0]
    assert "match_evidence_text" not in result["reply"]
    assert "Lightweight aluminum utility trailer" in captured["human"]


def test_feature_price_query_allows_confirmed_feature_match(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df().iloc[[1]].copy())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            year=2024,
            possible_make="Aluma",
            user_wants_price=True,
            requested_features=["sliding gates"],
            search_intent="year_make",
        ),
    )
    monkeypatch.setattr(
        im,
        "_inventory_feature_framing_llm",
        lambda: _FeatureFramingLLM(
            im.InventoryFeatureFramingDecision(
                intro_text="Yes, we currently show a 2024 Aluma option with sliding gates, and its listed price is below.",
                configuration_match_level="full",
                full_match_count=1,
            )
        ),
    )

    result = im.search_trailers("price of 2024 Aluma with sliding gates")

    assert result["feature_match_analysis"]["configuration_match_level"] == "full"
    assert result["reply"].startswith("Yes, we currently show a 2024 Aluma option with sliding gates")
    assert "$12,495" in result["reply"]
    assert "match_evidence_text" not in result["top_matches"][0]


def test_feature_query_multiple_matches_can_frame_full_and_alternatives(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            year=2024,
            possible_make="Aluma",
            user_wants_availability=True,
            requested_features=["sliding gates"],
            search_intent="year_make",
        ),
    )
    monkeypatch.setattr(
        im,
        "_inventory_feature_framing_llm",
        lambda: _FeatureFramingLLM(
            im.InventoryFeatureFramingDecision(
                intro_text="I found one listed 2024 Aluma option that confirms sliding gates, plus another close Aluma alternative to compare.",
                configuration_match_level="partial",
                missing_or_unconfirmed_features=[],
                full_match_count=1,
                listing_match_labels=["alternative", "full"],
            )
        ),
    )

    result = im.search_trailers("is a 2024 Aluma with sliding gates available?")

    assert result["feature_match_analysis"]["full_match_count"] == 1
    assert result["reply"].startswith("I found one listed 2024 Aluma option that confirms sliding gates")
    assert len(result["top_matches"]) == 2
    assert all("match_evidence_text" not in item for item in result["top_matches"])


def test_feature_query_no_full_match_still_shows_closest_inventory(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df().iloc[[0]].copy())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            year=2024,
            possible_make="Aluma",
            user_wants_availability=True,
            requested_features=["sliding gates"],
            search_intent="year_make",
        ),
    )
    monkeypatch.setattr(
        im,
        "_inventory_feature_framing_llm",
        lambda: _FeatureFramingLLM(
            im.InventoryFeatureFramingDecision(
                intro_text="We do not currently show that exact sliding-gate configuration, so I selected the closest listed Aluma option.",
                configuration_match_level="no_exact",
                missing_or_unconfirmed_features=["sliding gates"],
                full_match_count=0,
            )
        ),
    )

    result = im.search_trailers("is a 2024 Aluma with sliding gates available?")

    assert result["feature_match_analysis"]["configuration_match_level"] == "no_exact"
    assert "Trailer #1: [2024 Aluma Utility Trailer](https://example.test/aluma-basic)" in result["reply"]
    assert len(result["top_matches"]) == 1


def test_feature_query_llm_unavailable_uses_neutral_factual_fallback(monkeypatch):
    monkeypatch.setattr(im, "prepared_inventory", lambda: _feature_df().iloc[[0]].copy())
    monkeypatch.setattr(
        im,
        "extract_trailer_query",
        lambda _query: im.TrailerQueryExtraction(
            year=2024,
            possible_make="Aluma",
            user_wants_price=True,
            requested_features=["sliding gates"],
            search_intent="year_make",
        ),
    )
    monkeypatch.setenv("INVENTORY_FEATURE_FRAMING_LLM_ENABLED", "0")

    result = im.search_trailers("price of 2024 Aluma with sliding gates")

    assert result["reply"].startswith("I found the closest inventory match")
    assert "Any listed price below applies to the shown inventory option." in result["reply"]
    assert "sliding gates" not in result["reply"].split("\n\n", 1)[0].lower()
    assert "$10,995" in result["reply"]
    assert result["feature_match_analysis"]["source"] == "neutral_fallback"


def test_feature_evidence_is_not_exposed_for_context_reference():
    last_listings = [
        {
            "title": "Trailer A",
            "stock_number": "11111",
            "price": "$1",
            "match_evidence_text": "Internal notes only",
        }
    ]

    result = im.search_trailers("what is the price of that one?", last_listings=last_listings, for_chat=True)

    assert "match_evidence_text" not in result["reply"]
    assert "Internal notes only" not in result["reply"]
    assert "match_evidence_text" not in result["top_matches"][0]
