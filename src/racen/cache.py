from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

# Optional Redis import. If missing or connection fails, we fall back to in-memory.
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class CacheBackend:
    """Tiny cache interface for get/set operations using bytes payloads.

    Implementations should be safe for multi-threaded use within a single process.
    """

    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def set(self, key: str, value: bytes, ttl_s: int) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """Process-local in-memory cache with TTL and simple periodic cleanup.

    Not shared across processes; intended as a development fallback.
    """

    def __init__(self) -> None:
        self._store: Dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            exp, val = item
            if exp < now:
                # Expired; delete and miss
                try:
                    del self._store[key]
                except Exception:
                    pass
                return None
            return val

    def set(self, key: str, value: bytes, ttl_s: int) -> None:
        exp = time.time() + max(1, int(ttl_s))
        with self._lock:
            self._store[key] = (exp, value)
            # Opportunistic cleanup of a few expired keys
            if len(self._store) > 2048:
                to_del = [k for k, (e, _) in list(self._store.items())[:512] if e < time.time()]
                for k in to_del:
                    try:
                        del self._store[k]
                    except Exception:
                        pass


class RedisCache(CacheBackend):
    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: bytes, ttl_s: int) -> None:
        try:
            self._client.setex(key, max(1, int(ttl_s)), value)
        except Exception:
            pass


_singleton_lock = threading.Lock()
_singleton_cache: Optional[CacheBackend] = None


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y"}


def _get_prefix() -> str:
    return os.getenv("RACEN_CACHE_PREFIX", "racen:")


def get_cache() -> Optional[CacheBackend]:
    """Return a process-wide cache instance if enabled; otherwise None."""
    global _singleton_cache
    if not _bool_env("RACEN_CACHE_ENABLED", True):
        return None
    with _singleton_lock:
        if _singleton_cache is not None:
            return _singleton_cache
        # Try Redis first
        url = os.getenv("REDIS_URL", "").strip()
        if url and redis is not None:
            try:
                client = redis.from_url(url, decode_responses=False)
                # Small sanity check
                client.ping()
                _singleton_cache = RedisCache(client)
                return _singleton_cache
            except Exception:
                pass
        # Fallback to in-memory
        _singleton_cache = InMemoryCache()
        return _singleton_cache


def _salt() -> str:
    return os.getenv("RACEN_CACHE_SALT", "")


def normalize_text(s: str) -> str:
    t = (s or "").strip().lower()
    # Collapse whitespace
    return " ".join(t.split())


def make_key(parts: Dict[str, Any]) -> str:
    """Build a stable cache key with a configurable prefix and salt.

    Args:
        parts: Dictionary of key parts (will be JSON-encoded deterministically).

    Returns:
        str: Prefixed key string suitable for Redis or in-memory caches.
    """
    # Ensure deterministic ordering
    data = {**parts, "salt": _salt()}
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # Keep key short-ish to be Redis-friendly
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    return f"{_get_prefix()}{digest}"


def get_json(key: str) -> Optional[Any]:
    c = get_cache()
    if c is None:
        return None
    raw = c.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def set_json(key: str, obj: Any, ttl_s: int) -> None:
    c = get_cache()
    if c is None:
        return
    try:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    except Exception:
        return
    c.set(key, raw, ttl_s=max(1, int(ttl_s)))
