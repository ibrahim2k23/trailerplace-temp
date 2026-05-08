"""Category make-priority ordering after fit rerank (``_apply_category_make_priority``)."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from src.agent import _apply_category_make_priority
from src.models import TrailerListing


def _listing(
    listing_id: str,
    make: str,
    *,
    score: float = 0.5,
    category: str = "Utility",
) -> TrailerListing:
    return TrailerListing(
        listing_id=listing_id,
        title=f"Test {listing_id}",
        condition="New",
        price=None,
        payments_from=None,
        category_subcategory=category,
        make=make,
        color="Black",
        hitch_type=None,
        year=None,
        length=None,
        width=None,
        axles=None,
        gvwr=None,
        payload_capacity=None,
        trailer_material=None,
        floor=None,
        url=f"https://example.com/{listing_id}",
        score=score,
    )


class TestApplyCategoryMakePriority(unittest.TestCase):
    def test_orders_preferred_makes_top_to_bottom(self) -> None:
        # Deliberately wrong fit order; make layer should reorder Utility makes.
        rows = [
            _listing("1", "P&C", score=0.99),
            _listing("2", "Other Brand", score=0.98),
            _listing("3", "Diamond C", score=0.5),
            _listing("4", "Iron Bull Trailers", score=0.4),
        ]
        out, dbg = _apply_category_make_priority(rows, "Utility")
        self.assertTrue(dbg.get("applied"))
        self.assertEqual([x.make for x in out[:3]], ["Diamond C", "Iron Bull Trailers", "P&C"])
        self.assertEqual(out[-1].make, "Other Brand")

    def test_stable_within_same_make_bucket(self) -> None:
        rows = [
            _listing("a", "Diamond C", score=0.1),
            _listing("b", "Diamond C", score=0.9),
            _listing("c", "Iron Bull", score=0.5),
        ]
        out, dbg = _apply_category_make_priority(rows, "Utility")
        self.assertTrue(dbg.get("applied"))
        self.assertEqual([x.listing_id for x in out[:2]], ["a", "b"])
        self.assertEqual(out[2].listing_id, "c")

    def test_iron_bull_alias_via_normalize_make(self) -> None:
        rows = [
            _listing("1", "Iron Bull", score=0.99),
            _listing("2", "Diamond C Trailers", score=0.5),
        ]
        out, dbg = _apply_category_make_priority(rows, "Dump")
        self.assertTrue(dbg.get("applied"))
        # Dump order: Iron Bull first, then Diamond C
        self.assertEqual(out[0].make, "Iron Bull")
        self.assertEqual(out[1].make, "Diamond C Trailers")
        self.assertEqual(dbg.get("preferred_count"), 2)

    def test_no_map_category_unchanged(self) -> None:
        rows = [_listing("1", "Any", category="Welding")]
        out, dbg = _apply_category_make_priority(rows, "Welding")
        self.assertFalse(dbg.get("applied"))
        self.assertEqual(out, rows)
        self.assertEqual(dbg.get("reason"), "no_preference_map")

    def test_empty_category_noop(self) -> None:
        rows = [_listing("1", "Diamond C")]
        out, dbg = _apply_category_make_priority(rows, "")
        self.assertFalse(dbg.get("applied"))
        self.assertEqual(out, rows)

    def test_no_preferred_hits_unchanged(self) -> None:
        rows = [
            _listing("1", "Unknown Co", score=0.9),
            _listing("2", "Other", score=0.8),
        ]
        out, dbg = _apply_category_make_priority(rows, "Utility")
        self.assertFalse(dbg.get("applied"))
        self.assertEqual(out, rows)
        self.assertEqual(dbg.get("reason"), "no_preferred_makes_in_results")

    def test_spacing_normalization_on_preferred_list(self) -> None:
        """Preferred keys normalize internal whitespace (see ``_normalize_make_key_for_priority``)."""
        from src.agent import CATEGORY_MAKE_PRIORITY, _make_rank_lookup

        pref = CATEGORY_MAKE_PRIORITY["Car Hauler"]
        rk = _make_rank_lookup(pref)
        self.assertIn("iron bull trailers", rk)


if __name__ == "__main__":
    unittest.main()
