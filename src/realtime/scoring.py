from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import redis
except Exception as exc:  # pragma: no cover
    redis = None
    REDIS_IMPORT_ERROR = str(exc)
else:
    REDIS_IMPORT_ERROR = None
def _redis_safe_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, pd.Timestamp):
        ts = _to_utc_timestamp(value)
        return "" if ts is None else ts.isoformat()
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def _redis_safe_mapping(mapping: dict) -> dict:
    return {str(k): _redis_safe_value(v) for k, v in mapping.items()}

EVENT_SIGNAL_MAP: dict[str, str] = {
    'visit': 'visit_signal',
    'browse': 'browse_signal',
    'search': 'search_signal',
    'add_to_cart': 'cart_signal',
    'remove_from_cart': 'cart_remove_signal',
    'purchase': 'purchase_signal',
    'support_contact': 'support_signal',
    'coupon_open': 'coupon_open_signal',
    'coupon_redeem': 'coupon_redeem_signal',
}

TRACKED_SIGNAL_FIELDS = [
    'visit_signal',
    'browse_signal',
    'search_signal',
    'cart_signal',
    'cart_remove_signal',
    'purchase_signal',
    'support_signal',
    'coupon_open_signal',
    'coupon_redeem_signal',
]


@dataclass(frozen=True)
class RealtimeStreamConfig:
    redis_url: str = 'redis://localhost:6379/0'
    stream_key: str = 'retention:events'
    consumer_group: str = 'retention-risk-scorers'
    consumer_name: str = 'retention-risk-worker-1'
    ranking_key: str = 'retention:realtime:risk_ranking'
    summary_key: str = 'retention:realtime:summary'
    state_key_prefix: str = 'retention:realtime:state'
    stream_maxlen: int = 250000
    snapshot_top_n: int = 200
    block_ms: int = 2000
    batch_size: int = 200

    def state_key(self, customer_id: int | str) -> str:
        return f'{self.state_key_prefix}:{int(customer_id)}'


class RealtimeScoringError(RuntimeError):
    pass


def _require_redis() -> None:
    if redis is None:
        message = 'redis Python package가 설치되지 않았습니다. `pip install redis` 후 다시 실행하세요.'
        if REDIS_IMPORT_ERROR:
            message = f'{message} (import error: {REDIS_IMPORT_ERROR})'
        raise RealtimeScoringError(message)


def _redis_client(config: RealtimeStreamConfig):
    _require_redis()
    assert redis is not None
    client = redis.from_url(config.redis_url, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover
        raise RealtimeScoringError(
            f'Redis 연결 실패: {exc}. Redis 서버가 실행 중인지와 REDIS URL({config.redis_url})을 확인하세요.'
        ) from exc
    return client


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _decay(value: float, delta_seconds: float, half_life_hours: float) -> float:
    if value <= 0:
        return 0.0
    if delta_seconds <= 0:
        return float(value)
    half_life_seconds = max(half_life_hours * 3600.0, 1.0)
    return float(value) * (0.5 ** (float(delta_seconds) / half_life_seconds))


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    ts = pd.to_datetime(value, errors='coerce', utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert('UTC') if pd.Timestamp(ts).tzinfo is not None else pd.Timestamp(ts).tz_localize('UTC')


def _parse_timestamp(value: Any) -> pd.Timestamp:
    ts = _to_utc_timestamp(value)
    if ts is None:
        raise RealtimeScoringError(f'유효하지 않은 timestamp입니다: {value}')
    return ts


def _summary_paths(result_dir: Path) -> tuple[Path, Path]:
    return result_dir / 'realtime_scores_snapshot.csv', result_dir / 'realtime_scores_summary.json'


def _seed_state_from_row(row: pd.Series | Dict[str, Any], config: RealtimeStreamConfig) -> Dict[str, Any]:
    customer_id = int(row['customer_id'])
    base_score = min(max(_safe_float(row.get('churn_probability', 0.50), 0.50), 0.001), 0.999)
    snapshot = {
        'customer_id': customer_id,
        'persona': str(row.get('persona', 'unknown')),
        'uplift_segment': str(row.get('uplift_segment', 'unknown')),
        'base_churn_probability': base_score,
        'realtime_churn_score': base_score,
        'score_delta': 0.0,
        'clv': _safe_float(row.get('clv', 0.0), 0.0),
        'expected_roi': _safe_float(row.get('expected_roi', 0.0), 0.0),
        'coupon_cost': _safe_int(row.get('coupon_cost', 0), 0),
        'last_event_type': 'bootstrap',
        'last_event_at': '',
        'total_events_seen': 0,
        'minutes_since_last_event': 0.0,
        'coupon_affinity': _safe_float(row.get('coupon_affinity', 0.0), 0.0),
        'support_contact_propensity': _safe_float(row.get('support_contact_propensity', 0.0), 0.0),
        'updated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
    }
    for field in TRACKED_SIGNAL_FIELDS:
        snapshot[field] = 0.0
    return snapshot


def _event_increment(event_type: str) -> Dict[str, float]:
    increments = {field: 0.0 for field in TRACKED_SIGNAL_FIELDS}
    signal = EVENT_SIGNAL_MAP.get(str(event_type).strip().lower())
    if signal:
        increments[signal] = 1.0
    return increments


def _score_from_state(state: Dict[str, Any], now_ts: pd.Timestamp) -> tuple[float, Dict[str, float]]:
    base = _safe_float(state.get('base_churn_probability', 0.50), 0.50)
    now_ts = _parse_timestamp(now_ts)
    last_event_at = _to_utc_timestamp(state.get('last_event_at'))
    minutes_since_last_event = 0.0
    if last_event_at is not None:
        minutes_since_last_event = max((now_ts - last_event_at).total_seconds() / 60.0, 0.0)

    visit_signal = _safe_float(state.get('visit_signal', 0.0))
    browse_signal = _safe_float(state.get('browse_signal', 0.0))
    search_signal = _safe_float(state.get('search_signal', 0.0))
    cart_signal = _safe_float(state.get('cart_signal', 0.0))
    cart_remove_signal = _safe_float(state.get('cart_remove_signal', 0.0))
    purchase_signal = _safe_float(state.get('purchase_signal', 0.0))
    support_signal = _safe_float(state.get('support_signal', 0.0))
    coupon_open_signal = _safe_float(state.get('coupon_open_signal', 0.0))
    coupon_redeem_signal = _safe_float(state.get('coupon_redeem_signal', 0.0))

    inactivity_signal = _sigmoid((minutes_since_last_event / (60.0 * 24.0) - 7.0) / 3.0)
    behavioral_risk = (
        0.22 * inactivity_signal
        + 0.13 * support_signal
        + 0.06 * cart_remove_signal
        - 0.09 * visit_signal
        - 0.04 * browse_signal
        - 0.05 * search_signal
        - 0.07 * cart_signal
        - 0.16 * purchase_signal
        - 0.05 * coupon_open_signal
        - 0.12 * coupon_redeem_signal
    )
    score = min(max(base + behavioral_risk, 0.001), 0.999)
    diagnostics = {
        'minutes_since_last_event': float(minutes_since_last_event),
        'inactivity_signal': float(inactivity_signal),
        'behavioral_risk': float(behavioral_risk),
    }
    return float(score), diagnostics


def _apply_event_to_state(state: Dict[str, Any], event: Dict[str, Any], event_ts: pd.Timestamp) -> Dict[str, Any]:
    current = {**state}
    event_ts = _parse_timestamp(event_ts)
    previous_ts = (
        _to_utc_timestamp(current.get('last_event_at'))
        or _to_utc_timestamp(current.get('last_event_ts'))
        or _to_utc_timestamp(current.get('updated_at'))
    )
    if previous_ts is None:
        delta_seconds = 0.0
    else:
        delta_seconds = max((event_ts - previous_ts).total_seconds(), 0.0)

    half_life_map = {
        'visit_signal': 18.0,
        'browse_signal': 18.0,
        'search_signal': 18.0,
        'cart_signal': 24.0,
        'cart_remove_signal': 36.0,
        'purchase_signal': 72.0,
        'support_signal': 72.0,
        'coupon_open_signal': 36.0,
        'coupon_redeem_signal': 96.0,
    }
    for field in TRACKED_SIGNAL_FIELDS:
        current[field] = _decay(_safe_float(current.get(field, 0.0)), delta_seconds, half_life_map[field])

    increments = _event_increment(str(event.get('event_type', '')))
    for field, value in increments.items():
        current[field] = _safe_float(current.get(field, 0.0)) + float(value)

    current['last_event_type'] = str(event.get('event_type', 'unknown'))
    current['last_event_at'] = event_ts.floor('s').isoformat()
    current['updated_at'] = pd.Timestamp.now(tz='UTC').floor('s').isoformat()
    current['total_events_seen'] = _safe_int(current.get('total_events_seen', 0)) + 1

    score, diagnostics = _score_from_state(current, event_ts)
    current['realtime_churn_score'] = score
    current['score_delta'] = score - _safe_float(current.get('base_churn_probability', score), score)
    current['minutes_since_last_event'] = diagnostics['minutes_since_last_event']
    current['behavioral_risk'] = diagnostics['behavioral_risk']
    current['inactivity_signal'] = diagnostics['inactivity_signal']
    return current


def _snapshot_from_redis(client, result_dir: Path, config: RealtimeStreamConfig, top_n: int | None = None) -> Dict[str, Any]:
    top_n = int(top_n or config.snapshot_top_n)
    ranking = client.zrevrange(config.ranking_key, 0, max(top_n - 1, 0), withscores=True)
    records: list[Dict[str, Any]] = []
    for customer_id, score in ranking:
        raw_state = client.hgetall(config.state_key(customer_id))
        if not raw_state:
            continue
        raw_state['customer_id'] = _safe_int(raw_state.get('customer_id', customer_id), _safe_int(customer_id))
        raw_state['realtime_churn_score'] = float(score)
        raw_state['base_churn_probability'] = _safe_float(raw_state.get('base_churn_probability', score), score)
        raw_state['score_delta'] = _safe_float(raw_state.get('score_delta', 0.0), 0.0)
        raw_state['total_events_seen'] = _safe_int(raw_state.get('total_events_seen', 0), 0)
        raw_state['minutes_since_last_event'] = _safe_float(raw_state.get('minutes_since_last_event', 0.0), 0.0)
        for field in TRACKED_SIGNAL_FIELDS + ['clv', 'expected_roi', 'coupon_affinity', 'support_contact_propensity', 'behavioral_risk', 'inactivity_signal']:
            raw_state[field] = _safe_float(raw_state.get(field, 0.0), 0.0)
        raw_state['coupon_cost'] = _safe_int(raw_state.get('coupon_cost', 0), 0)
        records.append(raw_state)

    df = pd.DataFrame(records)
    summary = {
        'redis_url': config.redis_url,
        'stream_key': config.stream_key,
        'consumer_group': config.consumer_group,
        'tracked_customers': int(client.zcard(config.ranking_key)),
        'high_risk_customers': int(client.zcount(config.ranking_key, 0.70, '+inf')),
        'critical_risk_customers': int(client.zcount(config.ranking_key, 0.85, '+inf')),
        'snapshot_rows': int(len(df)),
        'generated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
    }
    raw_summary = client.get(config.summary_key)
    if raw_summary:
        try:
            summary.update(json.loads(raw_summary))
        except json.JSONDecodeError:
            pass

    snapshot_csv, snapshot_json = _summary_paths(result_dir)
    if not df.empty:
        df.sort_values(['realtime_churn_score', 'customer_id'], ascending=[False, True]).to_csv(snapshot_csv, index=False)
    else:
        pd.DataFrame(columns=['customer_id', 'realtime_churn_score']).to_csv(snapshot_csv, index=False)
    snapshot_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'summary': summary,
        'records': df.sort_values(['realtime_churn_score', 'customer_id'], ascending=[False, True]).to_dict(orient='records') if not df.empty else [],
    }


def bootstrap_realtime_state(
    data_dir: str | Path,
    result_dir: str | Path,
    config: RealtimeStreamConfig,
    *,
    reset_stream: bool = False,
    batch_size: int = 1000,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    result_dir = _ensure_dir(result_dir)
    summary_path = data_dir / 'customer_summary.csv'
    if not summary_path.exists():
        raise RealtimeScoringError(f'필수 파일이 없습니다: {summary_path}')

    client = _redis_client(config)
    if reset_stream:
        for key in [config.ranking_key, config.summary_key, config.stream_key]:
            client.delete(key)
        for match in client.scan_iter(match=f'{config.state_key_prefix}:*'):
            client.delete(match)

    df = pd.read_csv(summary_path)
    pipe = client.pipeline(transaction=False)
    buffered = 0
    for _, row in df.iterrows():
        state = _seed_state_from_row(row, config)
        pipe.hset(config.state_key(state['customer_id']), mapping=_redis_safe_mapping(state))       
        pipe.zadd(config.ranking_key, {str(state['customer_id']): float(state['realtime_churn_score'])})
        buffered += 1
        if buffered >= batch_size:
            pipe.execute()
            buffered = 0
    if buffered:
        pipe.execute()

    summary = {
        'bootstrapped_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
        'bootstrapped_customers': int(len(df)),
        'processed_events': 0,
        'last_produced_event_at': None,
        'last_consumed_event_at': None,
    }
    client.set(config.summary_key, json.dumps(summary, ensure_ascii=False))
    payload = _snapshot_from_redis(client, result_dir, config)
    payload['summary'].update(summary)
    return payload


def produce_events_to_stream(
    data_dir: str | Path,
    result_dir: str | Path,
    config: RealtimeStreamConfig,
    *,
    limit: int | None = None,
    sleep_ms: int = 0,
    reset_stream: bool = False,
    event_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    result_dir = _ensure_dir(result_dir)
    events_path = data_dir / 'events.csv'
    if not events_path.exists():
        raise RealtimeScoringError(f'필수 파일이 없습니다: {events_path}')

    client = _redis_client(config)
    if reset_stream:
        client.delete(config.stream_key)

    usecols = ['customer_id', 'timestamp', 'event_type', 'session_id', 'item_category', 'quantity']
    events = pd.read_csv(events_path, usecols=usecols, parse_dates=['timestamp']).sort_values('timestamp')
    if event_types:
        normalized = {str(item).strip().lower() for item in event_types}
        events = events[events['event_type'].astype(str).str.lower().isin(normalized)]
    if limit is not None and int(limit) > 0:
        events = events.head(int(limit))

    produced = 0
    last_ts: Optional[str] = None
    for _, row in events.iterrows():
        payload = {
            'customer_id': str(_safe_int(row['customer_id'], 0)),
            'timestamp': _parse_timestamp(row['timestamp']).floor('s').isoformat(),
            'event_type': str(row.get('event_type', 'unknown')),
            'session_id': '' if pd.isna(row.get('session_id')) else str(row.get('session_id')),
            'item_category': '' if pd.isna(row.get('item_category')) else str(row.get('item_category')),
            'quantity': str(_safe_int(row.get('quantity', 0), 0)),
        }
        client.xadd(config.stream_key, payload, maxlen=config.stream_maxlen, approximate=True)
        produced += 1
        last_ts = payload['timestamp']
        if sleep_ms > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    raw_summary = client.get(config.summary_key)
    summary = json.loads(raw_summary) if raw_summary else {}
    summary.update(
        {
            'last_produced_event_at': last_ts,
            'produced_events_total': _safe_int(summary.get('produced_events_total', 0), 0) + produced,
            'producer_updated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
        }
    )
    client.set(config.summary_key, json.dumps(summary, ensure_ascii=False))
    _snapshot_from_redis(client, result_dir, config)
    return {'produced_events': produced, 'last_event_at': last_ts, 'summary': summary}


def _ensure_consumer_group(client, config: RealtimeStreamConfig) -> None:
    try:
        client.xgroup_create(config.stream_key, config.consumer_group, id='0', mkstream=True)
    except Exception as exc:
        text = str(exc)
        if 'BUSYGROUP' not in text:
            raise RealtimeScoringError(f'Redis consumer group 생성 실패: {exc}') from exc


def consume_stream_events(
    data_dir: str | Path,
    result_dir: str | Path,
    config: RealtimeStreamConfig,
    *,
    max_events: int | None = None,
    idle_cycles: int = 2,
    snapshot_every: int = 250,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    result_dir = _ensure_dir(result_dir)
    client = _redis_client(config)

    if not client.exists(config.ranking_key):
        bootstrap_realtime_state(data_dir, result_dir, config)

    _ensure_consumer_group(client, config)

    processed = 0
    last_consumed_event_at: Optional[str] = None
    idle_seen = 0
    while True:
        if max_events is not None and processed >= int(max_events):
            break

        count = min(config.batch_size, int(max_events) - processed) if max_events is not None else config.batch_size
        response = client.xreadgroup(
            groupname=config.consumer_group,
            consumername=config.consumer_name,
            streams={config.stream_key: '>'},
            count=max(count, 1),
            block=config.block_ms,
        )
        if not response:
            idle_seen += 1
            if idle_seen >= max(idle_cycles, 1):
                break
            continue
        idle_seen = 0

        pipe = client.pipeline(transaction=False)
        for _, messages in response:
            for message_id, payload in messages:
                customer_id = _safe_int(payload.get('customer_id', 0), 0)
                event_ts = _parse_timestamp(payload.get('timestamp'))
                raw_state = client.hgetall(config.state_key(customer_id))
                if not raw_state:
                    baseline_path = data_dir / 'customer_summary.csv'
                    if not baseline_path.exists():
                        raise RealtimeScoringError(f'필수 파일이 없습니다: {baseline_path}')
                    baseline_df = pd.read_csv(baseline_path)
                    row = baseline_df.loc[baseline_df['customer_id'] == customer_id]
                    if row.empty:
                        raw_state = _seed_state_from_row({'customer_id': customer_id, 'churn_probability': 0.50}, config)
                    else:
                        raw_state = _seed_state_from_row(row.iloc[0], config)
                updated = _apply_event_to_state(raw_state, payload, event_ts)
                pipe.hset(config.state_key(customer_id), mapping=_redis_safe_mapping(updated))                
                pipe.zadd(config.ranking_key, {str(customer_id): float(updated['realtime_churn_score'])})
                pipe.xack(config.stream_key, config.consumer_group, message_id)
                processed += 1
                last_consumed_event_at = event_ts.floor('s').isoformat()
                if snapshot_every > 0 and processed % int(snapshot_every) == 0:
                    pipe.execute()
                    pipe = client.pipeline(transaction=False)
        pipe.execute()

    raw_summary = client.get(config.summary_key)
    summary = json.loads(raw_summary) if raw_summary else {}
    summary.update(
        {
            'processed_events': _safe_int(summary.get('processed_events', 0), 0) + processed,
            'last_consumed_event_at': last_consumed_event_at,
            'consumer_name': config.consumer_name,
            'consumer_updated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
        }
    )
    client.set(config.summary_key, json.dumps(summary, ensure_ascii=False))
    payload = _snapshot_from_redis(client, result_dir, config)
    payload['summary'].update(summary)
    return payload


def get_current_realtime_scores(
    result_dir: str | Path,
    config: RealtimeStreamConfig,
    *,
    top_n: int = 50,
) -> Dict[str, Any]:
    result_dir = _ensure_dir(result_dir)
    try:
        client = _redis_client(config)
        if client.exists(config.ranking_key):
            payload = _snapshot_from_redis(client, result_dir, config, top_n=top_n)
            payload['summary']['source'] = 'redis'
            return payload
    except Exception:
        pass

    snapshot_csv, snapshot_json = _summary_paths(result_dir)
    summary: Dict[str, Any] = {}
    if snapshot_json.exists():
        try:
            summary = json.loads(snapshot_json.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            summary = {}
    if snapshot_csv.exists():
        df = pd.read_csv(snapshot_csv).head(int(top_n))
    else:
        df = pd.DataFrame()
    summary['source'] = 'snapshot'
    return {'summary': summary, 'records': df.to_dict(orient='records') if not df.empty else []}
