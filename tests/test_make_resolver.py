from src.chatbot.make_inventory import categories_for_make, make_filter_values
from src.chatbot.make_resolver import resolve_make_from_text


def test_resolves_make_typo_without_llm():
    result = resolve_make_from_text("Do you have a Dimond C 6x12?", use_llm_fallback=False)

    assert result.make == "Diamond C"
    assert result.match_type in {"alias", "fuzzy"}


def test_gooseneck_hitch_does_not_resolve_as_make():
    result = resolve_make_from_text("I need a gooseneck flatbed trailer", use_llm_fallback=False)

    assert result.make is None


def test_gooseneck_brand_context_can_resolve_as_make():
    result = resolve_make_from_text("Do you have Gooseneck brand trailers?", use_llm_fallback=False)

    assert result.make == "Gooseneck"


def test_inventory_lookup_normalizes_make_and_filter_variants():
    assert "Flatbed" in categories_for_make("Diamond C")
    assert set(make_filter_values("Diamond C")) >= {"Diamond C", "Diamond C Trailers"}
