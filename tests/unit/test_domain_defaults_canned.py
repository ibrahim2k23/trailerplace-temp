from src.domain import defaults
from src.domain.canned_responses import CANNED_RESPONSES, FAQ_CANNED_RESPONSES, NON_FAQ_CANNED_RESPONSES


def test_defaults_for_missing_and_configured(monkeypatch):
    assert defaults.defaults_for("Nonexistent") == {}
    assert defaults.defaults_for("Flatbed") == {"trailer_width_ft": 8.0}

    monkeypatch.setitem(defaults.CATEGORY_DEFAULTS, "Utility", {"hitch_type": "Bumper Pull"})
    assert defaults.defaults_for("Utility") == {"hitch_type": "Bumper Pull"}


def test_canned_response_keys_and_phone_number():
    assert set(FAQ_CANNED_RESPONSES) == {
        "contact_human",
        "financing",
        "trade_in",
        "service_parts",
        "store_info",
    }
    assert set(NON_FAQ_CANNED_RESPONSES) == {
        "generic_team_request",
        "escalation",
        "listing_interest_selected",
        "listing_interest_unselected",
        "listing_interest_fallback",
    }
    for key in ["contact_human", "financing", "trade_in", "service_parts", "store_info"]:
        assert "979-532-1486" in CANNED_RESPONSES[key]
