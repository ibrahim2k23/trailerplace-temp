import json
import os

import pandas as pd

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")

from src.ingest import build_record, build_vector_id


def test_build_record_adds_numeric_payload_width_and_length_metadata():
    row = pd.Series(
        {
            "category": "Livestock",
            "subcategory": "",
            "make": "Calico",
            "color": "Black",
            "hitch_type": "Gooseneck",
            "condition": "New",
            "title": "Test Livestock Trailer",
            "url": "https://example.test/trailer",
            "price": "$12,000",
            "dealer_notes": "",
            "info_specs_json": json.dumps(
                {
                    "info": {
                        "length": "12 ft",
                        "width": "6 ft",
                        "gvwr": "7,000 lbs",
                        "payload_capacity": "5,200 lbs",
                    }
                }
            ),
        }
    )

    metadata = build_record(row, 1)["metadata"]

    assert metadata["length_ft_num"] == 12.0
    assert metadata["width_ft_num"] == 6.0
    assert metadata["payload_lbs_num"] == 5200.0
    assert metadata["gvwr_lbs_num"] == 7000.0


def test_vector_id_uses_url_hash_to_avoid_stock_number_collisions():
    first = pd.Series(
        {
            "stock_number": "15174",
            "url": "https://www.trailerplace.com/inventory/first/",
        }
    )
    second = pd.Series(
        {
            "stock_number": "15174",
            "url": "https://www.trailerplace.com/inventory/second/",
        }
    )

    assert build_vector_id(first, 1).startswith("stock_15174_")
    assert build_vector_id(second, 2).startswith("stock_15174_")
    assert build_vector_id(first, 1) != build_vector_id(second, 2)
