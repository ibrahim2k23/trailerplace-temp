"""Tests for client-supplied shown URLs (split Streamlit / FastAPI)."""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.shown_listings_store import (
    accumulate_shown_urls_from_chat_messages,
    merge_shown_urls_for_show_more,
    normalize_shown_url,
    sanitize_already_shown_urls,
)


class TestSanitizeAlreadyShown(unittest.TestCase):
    def test_dedupe_and_order(self) -> None:
        out = sanitize_already_shown_urls(
            ["https://A.COM/x", "  https://a.com/x ", "https://b.com/"]
        )
        self.assertEqual(out, ["https://a.com/x", "https://b.com/"])

    def test_keep_last_when_over_cap(self) -> None:
        raw = [f"https://ex.example/{i}" for i in range(450)]
        out = sanitize_already_shown_urls(raw)
        self.assertEqual(len(out), 400)
        self.assertEqual(out[0], "https://ex.example/50")
        self.assertEqual(out[-1], "https://ex.example/449")


class TestNormalizeShownUrl(unittest.TestCase):
    def test_trim_lower(self) -> None:
        self.assertEqual(normalize_shown_url("  HTTPS://X/Y  "), "https://x/y")


class TestAccumulateFromMessages(unittest.TestCase):
    def test_objects_and_dicts(self) -> None:
        from types import SimpleNamespace

        msgs = [
            {"role": "user", "content": "hi", "listings": None},
            {
                "role": "assistant",
                "content": "here",
                "listings": [
                    SimpleNamespace(url="https://A.COM/1"),
                    {"url": "https://b.com/2"},
                ],
            },
        ]
        out = accumulate_shown_urls_from_chat_messages(msgs)
        self.assertEqual(out, ["https://a.com/1", "https://b.com/2"])


class TestMergeShownUrls(unittest.TestCase):
    def test_client_only_without_disk(self) -> None:
        out = merge_shown_urls_for_show_more("", ["https://u.com/a", "https://u.com/b"])
        self.assertEqual(out, {"https://u.com/a", "https://u.com/b"})

    def test_union_with_disk(self) -> None:
        sid = str(uuid.uuid4())
        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"TRAILERPLACE_SHOWN_DIR": tmp}, clear=False):
                from src import shown_listings_store as sls

                sls.add_shown_urls(sid, ["https://disk.only/old"])
                out = merge_shown_urls_for_show_more(
                    sid, ["https://client.only/new", "https://disk.only/old"]
                )
        self.assertEqual(
            out,
            {"https://disk.only/old", "https://client.only/new"},
        )
