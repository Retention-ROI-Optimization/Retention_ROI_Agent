from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, TypeVar

try:
    import redis
except Exception:  # pragma: no cover - redis is optional in local fallback
    redis = None  # type: ignore

T = TypeVar("T")

_DEFAULT_TTL_SECONDS = int(os.getenv("RETENTION_CACHE_TTL_SECONDS", "20"))
_PREFIX = os.getenv("RETENTION_CACHE_PREFIX", "retention-cache")

_CLIENTS: dict[str, Any] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def make_cache_key(*parts: Any) -> str:
    safe = ":".join(str(p).replace(" ", "_").replace("/", "_") for p in parts if p is not None)
    return f"{_PREFIX}:{safe}"


def get_redis_client(redis_url: str | None = None):
    if redis is None:
        return None
    url = redis_url or os.getenv("RETENTION_REDIS_URL")
    if not url:
        return None
    if url not in _CLIENTS:
        try:
            client = redis.from_url(url, decode_responses=True)
            client.ping()
            _CLIENTS[url] = client
        except Exception:
            return None
    return _CLIENTS.get(url)


def cache_get(key: str, redis_url: str | None = None) -> Any | None:
    client = get_redis_client(redis_url)
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl_seconds: int | None = None, redis_url: str | None = None) -> bool:
    client = get_redis_client(redis_url)
    if client is None:
        return False
    try:
        ttl = int(ttl_seconds or _DEFAULT_TTL_SECONDS)
        client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=_json_default))
        return True
    except Exception:
        return False


def cache_delete_pattern(pattern: str, redis_url: str | None = None) -> int:
    client = get_redis_client(redis_url)
    if client is None:
        return 0
    deleted = 0
    try:
        for key in client.scan_iter(pattern):
            deleted += int(client.delete(key) or 0)
    except Exception:
        return deleted
    return deleted


def invalidate_user_live_cache(redis_url: str | None = None) -> int:
    return cache_delete_pattern(make_cache_key("user-live", "*") , redis_url=redis_url)


def cached_json(
    key: str,
    producer: Callable[[], T],
    *,
    ttl_seconds: int | None = None,
    redis_url: str | None = None,
) -> T:
    cached = cache_get(key, redis_url=redis_url)
    if cached is not None:
        return cached  # type: ignore[return-value]
    value = producer()
    cache_set(key, value, ttl_seconds=ttl_seconds, redis_url=redis_url)
    return value
