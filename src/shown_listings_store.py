from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Iterable

_lock = RLock()
_shown_urls: dict[str, set[str]] = defaultdict(set)
_shown_keys: dict[str, set[str]] = defaultdict(set)


def _clean_many(values: Iterable[str]) -> list[str]:
    return [str(v).strip() for v in values if str(v or "").strip()]


def accumulate_shown_urls_from_chat_messages(messages: list[dict]) -> list[str]:
    urls: set[str] = set()
    for msg in messages or []:
        for listing in msg.get("listings") or []:
            url = getattr(listing, "url", None)
            if not url and isinstance(listing, dict):
                url = listing.get("url")
            if url:
                urls.add(str(url))
    return sorted(urls)


def add_shown_urls(session_id: str, urls: Iterable[str]) -> None:
    clean = _clean_many(urls)
    if not session_id or not clean:
        return
    with _lock:
        _shown_urls[session_id].update(clean)


def add_shown_keys_and_urls(
    session_id: str,
    keys: Iterable[str] | None = None,
    urls: Iterable[str] | None = None,
) -> None:
    if not session_id:
        return
    with _lock:
        _shown_keys[session_id].update(_clean_many(keys or []))
        _shown_urls[session_id].update(_clean_many(urls or []))


def get_shown_urls(session_id: str) -> list[str]:
    with _lock:
        return sorted(_shown_urls.get(session_id, set()))


def reset_shown_listings(session_id: str) -> None:
    with _lock:
        _shown_urls.pop(session_id, None)
        _shown_keys.pop(session_id, None)
