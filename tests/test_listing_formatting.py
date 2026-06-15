from src.chatbot import formatting


def test_listing_results_use_plain_visible_numbers_without_fixed_interest_prompt(monkeypatch):
    listings = [
        {
            "title": "Trailer A",
            "url": "https://example.test/a",
            "category": "Utility",
            "length": "12 ft",
            "width": "7 ft",
        },
        {
            "title": "Trailer B",
            "url": "https://example.test/b",
            "category": "Equipment",
            "length": "14 ft",
            "width": "7 ft",
        },
    ]

    text = formatting.format_listing_results(
        listings,
        category="Utility",
        slots={"haul_item": "mower"},
        user_message="show me utility trailers",
    )

    assert "Trailer #1: [Trailer A](https://example.test/a)" in text
    assert "Trailer #2: [Trailer B](https://example.test/b)" in text
    assert "- Category: Utility" in text
    assert "- Category: Equipment" in text
    assert not text.endswith("Are you interested in any of the trailers above?")


def test_why_it_fits_llm_cannot_claim_unconfirmed_requested_feature(monkeypatch):
    listings = [
        {
            "title": "12 Ft Livestock Trailer",
            "url": "https://example.test/alt",
            "category": "Livestock",
            "length": "12 ft",
            "match_validation": {
                "match_level": "alternative",
                "confirmed_requirements": ["12 ft", "livestock"],
                "missing_or_unconfirmed_requirements": ["swing slide gate"],
                "requested_non_metadata_features": ["swing slide gate"],
            },
        }
    ]

    text = formatting.format_listing_results(
        listings,
        category="Livestock",
        slots={},
        user_message="I need a 12 ft livestock trailer with a swing slide gate",
    )

    assert "features a convenient swing slide gate" not in text.lower()
    assert "worth comparing for its confirmed strengths" in text
    assert "partial match" not in text.lower()
    assert "close alternative" not in text.lower()


def test_why_it_fits_prefers_safe_sales_blurb_from_match_validation(monkeypatch):
    listings = [
        {
            "title": "Livestock Trailer",
            "url": "https://example.test/alt",
            "category": "Livestock",
            "length": "16 ft",
            "match_validation": {
                "match_level": "alternative",
                "missing_or_unconfirmed_requirements": ["swing slide gate"],
                "sales_blurb": "This livestock trailer is a strong option to compare, with practical cattle-hauling utility and confirmed specs above.",
            },
        }
    ]

    text = formatting.format_listing_results(
        listings,
        category="Livestock",
        slots={},
        user_message="I need a 12 ft livestock trailer with a swing slide gate",
    )

    assert "strong option to compare" in text
    assert "stronger available options to compare" not in text


def test_why_it_fits_rejects_negative_structured_mismatch_language(monkeypatch):
    listings = [
        {
            "title": "16 Ft Livestock Trailer",
            "url": "https://example.test/alt",
            "category": "Livestock",
            "length": "16 ft",
            "match_validation": {
                "match_level": "alternative",
                "missing_or_unconfirmed_requirements": ["12 ft"],
            },
        }
    ]

    text = formatting.format_listing_results(
        listings,
        category="Livestock",
        slots={},
        user_message="I need a 12 ft livestock trailer",
    )

    assert "exceeds your 12 ft requirement" not in text.lower()
    assert "worth comparing for its confirmed strengths" in text


def test_why_it_fits_rejects_perfect_for_needs_on_non_full_listing(monkeypatch):
    listings = [
        {
            "title": "Livestock Trailer",
            "url": "https://example.test/alt",
            "category": "Livestock",
            "match_validation": {
                "match_level": "alternative",
                "missing_or_unconfirmed_requirements": ["sliding gates"],
            },
        }
    ]

    text = formatting.format_listing_results(
        listings,
        category="Livestock",
        slots={},
        user_message="I need a livestock trailer with sliding gates",
    )

    assert "perfect for your livestock needs" not in text.lower()
    assert "stronger available options to compare" in text
