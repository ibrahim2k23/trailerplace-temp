from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path


class DailyDateFileHandler(logging.Handler):
    """Append to ``logs/YYYY-MM-DD.log``; switches file at local midnight."""

    def __init__(self, log_dir: Path | None = None) -> None:
        super().__init__()
        self.terminator = "\n"
        root = Path(__file__).resolve().parent.parent
        self.log_dir = log_dir or (root / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._stream = None
        self._current_date: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = date.today().isoformat()
            if day != self._current_date:
                if self._stream:
                    self._stream.close()
                    self._stream = None
                self._current_date = day
                path = self.log_dir / f"{day}.log"
                self._stream = open(path, "a", encoding="utf-8")
            msg = self.format(record)
            self._stream.write(msg + self.terminator)
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        self._current_date = None
        super().close()


def configure_trailerplace_logging() -> None:
    """Console + daily file under ``logs/YYYY-MM-DD.log`` (project root)."""
    level_name = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    if not any(
        isinstance(h, DailyDateFileHandler) for h in root.handlers
    ):
        file_handler = DailyDateFileHandler()
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, DailyDateFileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(console)
