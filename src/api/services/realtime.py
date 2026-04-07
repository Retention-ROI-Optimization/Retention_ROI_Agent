from __future__ import annotations

from pathlib import Path
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
    config = build_realtime_config(settings)
    return get_current_realtime_scores(settings.resolved_result_dir, config, top_n=top_n)
