"""Tiny in-process TTL cache for read-only, network-backed tool payloads.

Repeated "what's the weather" / "bitcoin price" asks shouldn't re-hit the
network every time. Cached payloads expire quickly; failures are never cached.
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_STORE: dict[str, tuple[float, str]] = {}


def get(key: str) -> str | None:
    with _LOCK:
        hit = _STORE.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        _STORE.pop(key, None)
        return None


def put(key: str, payload: str, ttl_s: float = 300.0) -> None:
    with _LOCK:
        if len(_STORE) > 200:  # bound it
            _STORE.clear()
        _STORE[key] = (time.monotonic() + ttl_s, payload)
