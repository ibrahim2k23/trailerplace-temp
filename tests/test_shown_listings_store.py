"""Tests for URL-first session shown-listings persistence."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from src import shown_listings_store as sls


class TestShownListingsStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        os.environ["TRAILERPLACE_SHOWN_DIR"] = self._tmpdir.name

    def _sid(self) -> str:
        return str(uuid.uuid4())

    def test_add_shown_urls_writes_url_only_json(self) -> None:
        sid = self._sid()
        sls.add_shown_urls(sid, ["https://Example.COM/path/One", "https://example.com/path/two"])
        p = Path(self._tmpdir.name) / f"{sid}.json"
        self.assertTrue(p.is_file())
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertIn("urls", data)
        self.assertNotIn("keys", data)
        self.assertEqual(
            set(data["urls"]),
            {"https://example.com/path/one", "https://example.com/path/two"},
        )
        loaded = sls.load_shown_urls(sid)
        self.assertEqual(
            loaded,
            {"https://example.com/path/one", "https://example.com/path/two"},
        )

    def test_legacy_keys_read_merge_urls_then_url_only_write(self) -> None:
        sid = self._sid()
        p = Path(self._tmpdir.name) / f"{sid}.json"
        p.write_text(
            json.dumps({"keys": ["url:/old"], "urls": ["https://a.com/1"]}),
            encoding="utf-8",
        )
        sls.add_shown_urls(sid, ["https://b.com/2"])
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertNotIn("keys", data)
        self.assertEqual(set(data["urls"]), {"https://a.com/1", "https://b.com/2"})


if __name__ == "__main__":
    unittest.main()
