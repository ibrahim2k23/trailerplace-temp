"""
Per chat session, persist canonical listing keys and listing URLs already shown in the UI
(for Pinecone exclude-on-more without repeating cards).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Iterable

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.IGNORECASE,
)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _store_dir() -> Path:
    base = (os.getenv("TRAILERPLACE_SHOWN_DIR") or "").strip()
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "shown_listings"


def _path_for_session(session_id: str) -> Path:
    if not _UUID_RE.match((session_id or "").strip()):
        raise ValueError("Invalid session_id for shown_listings store")
    return _store_dir() / f"{session_id.strip()}.json"


def _get_lock(sid: str) -> threading.Lock:
    with _locks_guard:
        if sid not in _locks:
            _locks[sid] = threading.Lock()
        return _locks[sid]


def _normalize_url(url: str) -> str:
    return (url or "").strip().lower()


def _read_store(session_id: str) -> tuple[set[str], set[str]]:
    """Return (keys, urls) from session JSON."""
    if not (session_id or "").strip():
        return set(), set()
    p = _path_for_session(session_id)
    if not p.is_file():
        return set(), set()
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set(), set()
    keys_raw = data.get("keys")
    urls_raw = data.get("urls")
    keys: set[str] = set()
    urls: set[str] = set()
    if isinstance(keys_raw, list):
        keys = {str(k) for k in keys_raw if k}
    if isinstance(urls_raw, list):
        urls = {_normalize_url(str(u)) for u in urls_raw if u}
    return keys, urls


def _write_store(session_id: str, keys: set[str], urls: set[str]) -> None:
    sid = session_id.strip()
    _store_dir().mkdir(parents=True, exist_ok=True)
    p = _path_for_session(sid)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {
        "keys": sorted(keys),
        "urls": sorted(urls),
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=0)
    tmp.replace(p)


def load_shown_keys(session_id: str) -> set[str]:
    keys, _ = _read_store(session_id)
    return keys


def load_shown_urls(session_id: str) -> set[str]:
    _, urls = _read_store(session_id)
    return urls


def add_shown_keys(session_id: str, keys: Iterable[str]) -> None:
    to_add = {k for k in keys if k}
    if not to_add or not (session_id or "").strip():
        return
    sid = session_id.strip()
    with _get_lock(sid):
        current_keys, current_urls = _read_store(sid)
        merged_keys = current_keys | to_add
        if merged_keys == current_keys:
            return
        _write_store(sid, merged_keys, current_urls)


def add_shown_urls(session_id: str, urls: Iterable[str]) -> None:
    to_add = {_normalize_url(u) for u in urls if u}
    if not to_add or not (session_id or "").strip():
        return
    sid = session_id.strip()
    with _get_lock(sid):
        current_keys, current_urls = _read_store(sid)
        merged_urls = current_urls | to_add
        if merged_urls == current_urls:
            return
        _write_store(sid, current_keys, merged_urls)


def add_shown_keys_and_urls(
    session_id: str,
    keys: Iterable[str],
    urls: Iterable[str],
) -> None:
    """Single atomic write for keys and URLs (used from app after each shown turn)."""
    key_set = {k for k in keys if k}
    url_set = {_normalize_url(u) for u in urls if u}
    if (not key_set and not url_set) or not (session_id or "").strip():
        return
    sid = session_id.strip()
    with _get_lock(sid):
        current_keys, current_urls = _read_store(sid)
        merged_keys = current_keys | key_set
        merged_urls = current_urls | url_set
        if merged_keys == current_keys and merged_urls == current_urls:
            return
        _write_store(sid, merged_keys, merged_urls)
