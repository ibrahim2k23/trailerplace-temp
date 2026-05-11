"""Tests for FAQ canonical reply matching (shared with hard-lead / interest flows)."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-placeholder")

from src.agent_lg import FAQ_CANONICAL_REPLY_TEXT, _faq_reply_matches_canonical


class TestFaqReplyCanonicalMatch(unittest.TestCase):
    def test_accepts_whitespace_variant(self) -> None:
        canonical = FAQ_CANONICAL_REPLY_TEXT["contact_or_human"]
        spaced = "  " + canonical.replace(" ", "   ") + "\n"
        self.assertTrue(_faq_reply_matches_canonical("contact_or_human", spaced))

    def test_rejects_wrong_script(self) -> None:
        self.assertFalse(
            _faq_reply_matches_canonical("contact_or_human", "wrong answer")
        )


if __name__ == "__main__":
    unittest.main()
