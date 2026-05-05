"""
Per chat session, persist listing URLs already shown in the UI (for Pinecone
exclude-on-more without repeating cards).

JSON shape:
- Preferred: {"urls": ["https://...", ...]}
- Legacy: {"keys": [...], "urls": [...]} — still read for backward compatibility;
  updates via add_shown_urls() rewrite to URL-only.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable

# Caps for HTTP `/chat` payloads and LangGraph state (split UI/API hosts).
SHOWN_URLS_MAX_COUNT = 400
SHOWN_URL_MAX_CHARS = 2048

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


def normalize_shown_url(url: str) -> str:
    """Normalize listing URL for exclude-on-more matching (strip, lower, cap length)."""
    u = (url or "").strip().lower()
    if len(u) > SHOWN_URL_MAX_CHARS:
        return u[:SHOWN_URL_MAX_CHARS]
    return u


def _normalize_url(url: str) -> str:
    return normalize_shown_url(url)


def accumulate_shown_urls_from_chat_messages(messages: list[dict[str, Any]] | None) -> list[str]:
    """
    Collect normalized listing URLs from assistant turns (``listings`` on each message dict).
    Used by Streamlit when calling ``/chat`` so the API can exclude them on show-more.
    """
    raw: list[str] = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for lst in (m.get("listings") or []):
            u = _listing_url_from_message_item(lst)
            if u:
                raw.append(u)
    return sanitize_already_shown_urls(raw)


def _listing_url_from_message_item(item: Any) -> str:
    if item is None:
        return ""
    u = getattr(item, "url", None)
    if u is not None:
        return str(u).strip()
    if isinstance(item, dict):
        return str(item.get("url") or "").strip()
    return ""


def sanitize_already_shown_urls(urls: list[str] | None) -> list[str]:
    """
    Dedupe, normalize, and cap client-supplied URL lists (preserve order; keep last N if over cap).
    """
    if not urls:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        u = normalize_shown_url(str(raw) if raw is not None else "")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
    if len(out) > SHOWN_URLS_MAX_COUNT:
        out = out[-SHOWN_URLS_MAX_COUNT:]
    return out


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
    # URL-only files when no keys (new default). Legacy callers may still write keys.
    if keys:
        payload: dict = {"keys": sorted(keys), "urls": sorted(urls)}
    else:
        payload = {"urls": sorted(urls)}
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=0)
    tmp.replace(p)


def load_shown_keys(session_id: str) -> set[str]:
    keys, _ = _read_store(session_id)
    return keys


def load_shown_urls(session_id: str) -> set[str]:
    _, urls = _read_store(session_id)
    return urls


def merge_shown_urls_for_show_more(session_id: str, client_urls: list[str] | None) -> set[str]:
    """
    Union of disk-backed shown URLs for the session plus client-supplied URLs (split UI/API).
    """
    out: set[str] = set()
    sid = (session_id or "").strip()
    if sid:
        try:
            out |= load_shown_urls(sid)
        except ValueError:
            pass
    for u in sanitize_already_shown_urls(client_urls):
        out.add(u)
    return out


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
        _, current_urls = _read_store(sid)
        merged_urls = current_urls | to_add
        if merged_urls == current_urls:
            return
        # URL-first persistence: drop legacy keys on next write (show-more uses URLs only).
        _write_store(sid, set(), merged_urls)


def add_shown_keys_and_urls(
    session_id: str,
    keys: Iterable[str],
    urls: Iterable[str],
) -> None:
    """Merge keys and URLs in one write. Prefer add_shown_urls() from UI (URL-only file)."""
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
        if key_set:
            _write_store(sid, merged_keys, merged_urls)
        else:
            _write_store(sid, set(), merged_urls)
