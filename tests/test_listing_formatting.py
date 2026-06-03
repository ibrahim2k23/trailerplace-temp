from src.chatbot import formatting


def test_listing_results_use_plain_visible_numbers_without_fixed_interest_prompt(monkeypatch):
    monkeypatch.setattr(formatting, "_why_it_fits_llm_enabled", lambda: False)
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
