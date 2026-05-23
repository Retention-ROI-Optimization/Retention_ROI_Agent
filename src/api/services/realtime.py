from __future__ import annotations

from typing import Any, Dict

from src.api.settings import ApiSettings
from src.realtime.scoring import RealtimeStreamConfig, get_current_realtime_scores


def build_realtime_config(settings: ApiSettings) -> RealtimeStreamConfig:
    return RealtimeStreamConfig(
        redis_url=settings.redis_url,
        stream_key=settings.realtime_stream_key,
        consumer_group=settings.realtime_consumer_group,
        consumer_name=settings.realtime_consumer_name,
    )


def load_realtime_payload(settings: ApiSettings, *, top_n: int = 50) -> Dict[str, Any]:
    """Load the latest realtime snapshot.

    User-live mode uses PostgreSQL + Redis cache.  The old Redis Streams replay
    path is intentionally not advanced here; Redis should be cache-aside storage,
    not the source of truth for production-like live customer state.
    """
    config = build_realtime_config(settings)
    return get_current_realtime_scores(settings.resolved_result_dir, config, top_n=top_n)


def advance_realtime_payload(settings: ApiSettings, *, top_n: int = 50, batch_size: int = 250, reset_when_exhausted: bool = True) -> Dict[str, Any]:
    """Compatibility endpoint: refreshes snapshot payload without Redis Streams.

    The dashboard no longer relies on xadd/xread stream replay for user-live demos.
    This function preserves the public endpoint shape for simulator demos while
    preventing Redis from being used as a queue/streaming system in the main app.
    """
    payload = load_realtime_payload(settings, top_n=top_n)
    summary = payload.setdefault("summary", {})
    summary["last_tick_advanced"] = 0
    summary["source"] = summary.get("source", "snapshot")
    summary["stream_disabled"] = True
    summary["message"] = "Redis Streams replay is disabled; Redis is used for cache-aside acceleration."
    return payload
