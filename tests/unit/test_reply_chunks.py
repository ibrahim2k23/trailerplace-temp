"""The reply -> message-bubbles split that the streaming UI sends a turn in."""
from __future__ import annotations

from src.domain.reply_chunks import parse_listing_card, split_reply_into_chunks, urls_in_chunk

LISTING_REPLY = """Here are some options for livestock trailers, including those around 20 ft:

1. [2025 Galyean Cattle Trailer - 15221](https://trailerplace.com/a)
   - Category: Livestock
   - Price: $24,250
   - Ideal for transporting cattle.
2. [2026 Galyean 32' Cattle Trailer - 15079](https://trailerplace.com/b)
   - Category: Livestock
   - Price: $32,250
   - Wider load capacity and butterfly gates.

Do any of these look like a fit, or would you like to see more options?"""


def test_each_trailer_becomes_its_own_message():
    chunks = split_reply_into_chunks(LISTING_REPLY)
    assert len(chunks) == 4
    assert chunks[0].startswith("Here are some options")
    assert chunks[1].startswith("1. [2025 Galyean")
    assert chunks[2].startswith("2. [2026 Galyean")
    assert chunks[3].startswith("Do any of these")
    # Every card keeps its own bullets - a split mid-card would strand them.
    assert "Price: $24,250" in chunks[1] and "Price: $32,250" not in chunks[1]
    assert len([line for line in chunks[1].splitlines() if line.strip().startswith("- ")]) == 3


def test_no_content_is_lost_or_duplicated():
    """Joining the bubbles back up reproduces the reply, so nothing is dropped or repeated."""
    chunks = split_reply_into_chunks(LISTING_REPLY)
    # Blank lines are the seams between bubbles, so only the non-blank lines have to match.
    def lines(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip()]

    assert lines("\n".join(chunks)) == lines(LISTING_REPLY)


def test_opening_line_arrives_as_its_own_message():
    """The mandated first line of the conversation is its own short bubble, then the reply."""
    chunks = split_reply_into_chunks(
        "Thank you for contacting TrailerPlace.\n\nHi Ibrahim, which type are you looking for?"
    )
    assert chunks == [
        "Thank you for contacting TrailerPlace.",
        "Hi Ibrahim, which type are you looking for?",
    ]


def test_a_plain_reply_stays_one_message():
    text = "We carry gooseneck and bumper pull hitches. Which suits your truck?"
    assert split_reply_into_chunks(text) == [text]


def test_a_bulleted_list_is_not_cut_into_bullets():
    text = "We carry these types:\n\n- **Dump** — for dirt and gravel.\n- **Utility** — for general hauling."
    chunks = split_reply_into_chunks(text)
    assert len(chunks) == 2
    assert "Dump" in chunks[1] and "Utility" in chunks[1]


def test_a_bullet_holding_a_link_is_card_body_not_a_new_card():
    text = "1. [A trailer](https://trailerplace.com/a)\n   - See [the listing](https://trailerplace.com/a) for photos."
    assert len(split_reply_into_chunks(text)) == 1


def test_unnumbered_and_bold_card_titles_still_split():
    text = (
        "Options:\n\n"
        "[2026 Iron Bull DTB - 11154](https://trailerplace.com/a)\n"
        "   - Price: $8,995\n"
        "**[2026 Iron Bull DTB - 08242](https://trailerplace.com/b)**\n"
        "   - Price: $9,250\n"
    )
    chunks = split_reply_into_chunks(text)
    assert len(chunks) == 3
    assert chunks[1].startswith("[2026 Iron Bull DTB - 11154]")
    assert chunks[2].startswith("**[2026 Iron Bull DTB - 08242]")


def test_empty_reply_produces_no_messages():
    assert split_reply_into_chunks("") == []
    assert split_reply_into_chunks("   \n\n ") == []


def test_urls_in_chunk_finds_the_listing_link():
    chunks = split_reply_into_chunks(LISTING_REPLY)
    assert urls_in_chunk(chunks[1]) == ["https://trailerplace.com/a"]
    assert urls_in_chunk(chunks[0]) == []


# ---------------------------------------------------------------------------
# Taking a card apart, for channels with no markdown
# ---------------------------------------------------------------------------


def test_a_card_splits_into_marker_title_url_and_body():
    card = (
        "1. [2026 Galyean CATTLE TRAILER 32' - 15079](https://trailerplace.com/inventory/galyean/)\n"
        "   - Category: Livestock\n   - Price: $32,250"
    )
    marker, title, url, body = parse_listing_card(card)
    assert marker == "1."
    assert title == "2026 Galyean CATTLE TRAILER 32' - 15079"
    assert url == "https://trailerplace.com/inventory/galyean/"
    assert body == "- Category: Livestock\n- Price: $32,250"


def test_a_bold_wrapped_title_still_parses():
    marker, title, url, _ = parse_listing_card("**2. [2025 Iron Bull DTB - 15081](https://x.test/a/)**")
    assert (marker, title, url) == ("2.", "2025 Iron Bull DTB - 15081", "https://x.test/a/")


def test_a_card_with_no_number_keeps_an_empty_marker():
    marker, title, _, _ = parse_listing_card("[2025 Iron Bull DTB](https://x.test/a/)")
    assert marker == "" and title == "2025 Iron Bull DTB"


def test_prose_is_not_a_card():
    assert parse_listing_card("Thank you for contacting TrailerPlace.") is None


def test_a_bullet_linking_out_is_not_a_card():
    assert parse_listing_card("- [our website](https://trailerplace.com)") is None


def test_an_empty_chunk_is_not_a_card():
    assert parse_listing_card("   ") is None
