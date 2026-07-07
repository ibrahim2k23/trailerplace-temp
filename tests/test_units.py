"""Characterization tests for the single-sourced unit parsers.

These lock the numeric behavior of units.py. Because ingest wrote the Pinecone
index using these functions, changing an expected value here means a re-ingest is
required to keep queries consistent with the index.
"""

import math

import pytest

from src.chatbot.units import parse_length_ft, parse_weight_lbs


@pytest.mark.parametrize(
    "value, expected",
    [
        ("7000 lbs", 7000.0),
        ("7000 lb", 7000.0),
        ("7,000 lbs", 7000.0),
        ("14000", 14000.0),
        ("5k", 5000.0),
        ("2.5 ton", 5000.0),
        ("1000 kg", 1000 * 2.2046226218),
        ("3#", 3.0),
        (7000, 7000.0),
        (7000.0, 7000.0),
        # typo tolerance carried from the query-side parser
        ("5000 lbd", 5000.0),
        ("5000 lbss", 5000.0),
        # missing / non-positive / unparseable
        (None, None),
        ("", None),
        ("0", None),
        (float("nan"), None),
        ("no idea", None),
    ],
)
def test_parse_weight_lbs(value, expected):
    got = parse_weight_lbs(value)
    if expected is None:
        assert got is None
    else:
        assert got is not None and math.isclose(got, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("12 ft", 12.0),
        ("12 feet", 12.0),
        ("12'", 12.0),
        ("6 ft 6 in", 6.5),
        ("18 in", 1.5),
        ('18"', 1.5),
        ("5 yd", 15.0),
        ("2 m", 2 * 3.280839895),
        ("150 cm", 150 / 30.48),
        ("500 mm", 500 / 304.8),
        ("14000", 14000.0),
        (12, 12.0),
        # missing / non-positive / unparseable
        (None, None),
        ("", None),
        ("0", None),
        (float("nan"), None),
        ("any length", None),
    ],
)
def test_parse_length_ft(value, expected):
    got = parse_length_ft(value)
    if expected is None:
        assert got is None
    else:
        assert got is not None and math.isclose(got, expected, rel_tol=1e-6)
