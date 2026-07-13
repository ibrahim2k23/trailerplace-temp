from __future__ import annotations

from src.search.ingest import build_flattened_embedding_text


def test_all_flattened_info_and_specification_fields_enter_embedding_text():
    text = build_flattened_embedding_text(
        {
            "make": "Cargo Craft",
            "length": "16 ft",
            "interior_wall_thickness": "1 inch",
            "rear_door_opening": "76 inches",
            "electrical_options": ["30 amp service", "LED outlets"],
        },
        "2026 Cargo Craft Enclosed",
        ["insulated walls"],
    )

    assert "Make: Cargo Craft" in text
    assert "Length: 16 ft" in text
    assert "Interior Wall Thickness: 1 inch" in text
    assert "Rear Door Opening: 76 inches" in text
    assert "Electrical Options: 30 amp service; LED outlets" in text
    assert "Features: insulated walls" in text


def test_msrp_remains_excluded_from_embedding_text():
    text = build_flattened_embedding_text(
        {"price": "$10,000", "msrp": "$15,000"}, "Trailer", []
    )
    assert "Price: $10,000" in text
    assert "15,000" not in text
