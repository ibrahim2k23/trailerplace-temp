"""F2 — shared listing-card renderer produces byte-identical output on both paths."""
from src.chatbot import formatting, inventory_matcher

_LISTING = {
    "title": "2024 Big Tex 14GN",
    "url": "https://example.com/14gn",
    "category": "Gooseneck",
    "price_display": "$18,500",
    "length": "24 ft",
    "width": "83 in",
    "gvwr": "14,000 lbs",
    "hitch_type": "Gooseneck",
    "stock_number": "BT-9981",
    "model": "14GN",
}


def test_render_card_no_why_line_matches_old_block_layout():
    # Old inventory_matcher._format_listing_block: line1 + "\n\n" + bullets.
    bullets = inventory_matcher._listing_bullets(_LISTING)
    title = inventory_matcher._item_label(_LISTING)
    url = inventory_matcher._clean_scalar(_LISTING.get("url"))
    expected = "\n\n".join([f"Trailer #1: [{title}]({url})", "\n".join(bullets)])
    assert inventory_matcher._format_listing_block(1, _LISTING) == expected


def test_render_card_with_why_line_matches_old_search_layout():
    # Old format_listing_results: [line1, "", bullets, "", why] joined by "\n".
    card = formatting.render_listing_card(
        2,
        title="2024 Big Tex 14GN",
        url="https://example.com/14gn",
        bullet_lines=["- Length: 24 ft", "- Width: 83 in"],
        why_line="A strong gooseneck match for your needs.",
    )
    expected = "\n".join([
        "Trailer #2: [2024 Big Tex 14GN](https://example.com/14gn)",
        "",
        "- Length: 24 ft\n- Width: 83 in",
        "",
        "A strong gooseneck match for your needs.",
    ])
    assert card == expected


def test_no_url_omits_link_markup():
    card = formatting.render_listing_card(1, title="Plain Trailer", url="", bullet_lines=["- Category: Utility"])
    assert card.splitlines()[0] == "Trailer #1: Plain Trailer"


def test_empty_bullets_fall_back_to_no_specs_line():
    card = formatting.render_listing_card(1, title="Bare", url="", bullet_lines=[])
    assert "*(No spec fields on this listing.)*" in card


def test_format_listing_results_still_renders_cards_and_separator():
    out = formatting.format_listing_results(
        [_LISTING, {**_LISTING, "title": "Second"}],
        category="Gooseneck",
        slots={},
        user_message="need a gooseneck",
    )
    assert out.count("Trailer #") == 2
    assert formatting.LISTING_CARD_SEPARATOR in out
