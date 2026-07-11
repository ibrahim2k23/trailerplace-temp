"""Daily-file logging (src/log_setup.py) — console + a same-day log file."""
from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import patch

from src.log_setup import DailyFileHandler, JsonFormatter


def test_writes_to_a_file_named_after_todays_date(tmp_path):
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
        handler.emit(record)
    finally:
        handler.close()

    expected_name = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == [expected_name]
    assert files[0].read_text(encoding="utf-8").strip() == "hello"


def test_creates_the_directory_if_missing(tmp_path):
    target = tmp_path / "nested" / "logs"
    handler = DailyFileHandler(str(target))
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "hi", None, None))
    finally:
        handler.close()
    assert target.is_dir()
    assert len(list(target.iterdir())) == 1


def test_rolls_to_a_new_file_when_the_date_changes(tmp_path):
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        with patch.object(handler, "_today", return_value="2026-01-01"):
            handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "day one", None, None))
        with patch.object(handler, "_today", return_value="2026-01-02"):
            handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "day two", None, None))
    finally:
        handler.close()

    assert (tmp_path / "2026-01-01.log").read_text(encoding="utf-8").strip() == "day one"
    assert (tmp_path / "2026-01-02.log").read_text(encoding="utf-8").strip() == "day two"


def test_setting_formatter_after_first_emit_propagates_to_the_open_file(tmp_path):
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("first: %(message)s"))
    try:
        handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "a", None, None))
        handler.setFormatter(logging.Formatter("second: %(message)s"))
        handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "b", None, None))
    finally:
        handler.close()
    content = list(tmp_path.iterdir())[0].read_text(encoding="utf-8")
    assert "first: a" in content
    assert "second: b" in content


def test_a_broken_emit_does_not_raise(tmp_path):
    """logging.Handler.handleError swallows emit failures by default (no raise)."""
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    bad_record = logging.LogRecord("x", logging.INFO, __file__, 1, "%s", None, None)  # missing arg -> format error
    handler.emit(bad_record)  # must not raise
    handler.close()


def test_multiline_record_preserved_verbatim(tmp_path):
    """Conversation-reasoning blocks are multi-line; the file must keep them intact."""
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(logging.Formatter("%(message)s"))
    block = "line one\nline two\nline three"
    try:
        handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, block, None, None))
    finally:
        handler.close()
    content = list(tmp_path.iterdir())[0].read_text(encoding="utf-8")
    assert "line one\nline two\nline three" in content


def test_json_formatter_still_works_alongside_daily_file_handler(tmp_path):
    handler = DailyFileHandler(str(tmp_path))
    handler.setFormatter(JsonFormatter())
    try:
        record = logging.LogRecord("trailerplace.conversation", logging.INFO, __file__, 1, "hello", None, None)
        handler.emit(record)
    finally:
        handler.close()
    content = list(tmp_path.iterdir())[0].read_text(encoding="utf-8")
    assert '"message": "hello"' in content
