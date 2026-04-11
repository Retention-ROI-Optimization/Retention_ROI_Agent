from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.optimization.policy import build_intensity_action_candidates
from src.optimization.timing import load_survival_predictions

try:  # pragma: no cover - optional dependency
    import redis
except Exception as exc:  # pragma: no cover
    redis = None
    REDIS_IMPORT_ERROR = str(exc)
else:
    REDIS_IMPORT_ERROR = None


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

QUEUE_STATE_FIELDS = [
    'action_queue_status',
    'queued_recommended_action',
    'queued_intervention_intensity',
    'queued_coupon_cost',
    'queued_expected_profit',
    'queued_expected_roi',
    'action_queue_priority',
    'latest_trigger_reason',
    'reoptimization_count',
    'last_reoptimized_at',
    'dispatched_intensity_history',
]


@dataclass(frozen=True)
class RealtimeStreamConfig:
    redis_url: str = 'redis://localhost:6379/0'
    stream_key: str = 'retention:events'
    consumer_group: str = 'retention-risk-scorers'
    consumer_name: str = 'retention-risk-worker-1'
    ranking_key: str = 'retention:realtime:risk_ranking'
    action_queue_key: str = 'retention:realtime:action_queue'
    trigger_log_key: str = 'retention:realtime:trigger_log'
    summary_key: str = 'retention:realtime:summary'
    state_key_prefix: str = 'retention:realtime:state'
    stream_maxlen: int = 250000
    snapshot_top_n: int = 200
    block_ms: int = 2000
    batch_size: int = 200
    default_budget_limit: int = 5_000_000
    daily_channel_capacity: int = 500
    reoptimize_customer_threshold: float = 0.65
    reoptimize_high_risk_threshold: float = 0.80
    reoptimize_score_delta_threshold: float = 0.12

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


def _redis_safe_value(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, pd.Timestamp):
        ts = _to_utc_timestamp(value)
        return '' if ts is None else ts.isoformat()
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    return value


def _redis_safe_mapping(mapping: dict) -> dict:
    return {str(k): _redis_safe_value(v) for k, v in mapping.items()}


def _summary_paths(result_dir: Path) -> tuple[Path, Path]:
    return result_dir / 'realtime_scores_snapshot.csv', result_dir / 'realtime_scores_summary.json'


def _queue_paths(result_dir: Path) -> tuple[Path, Path]:
    return result_dir / 'realtime_action_queue_snapshot.csv', result_dir / 'realtime_action_queue_summary.json'


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(column, default), errors='coerce').fillna(float(default))


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce').fillna(0.0)
    if numeric.empty:
        return numeric.astype(float)
    low = float(numeric.min())
    high = float(numeric.max())
    if abs(high - low) < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - low) / (high - low)


def _load_baseline_customer_summary(data_dir: Path) -> pd.DataFrame:
    summary_path = data_dir / 'customer_summary.csv'
    if not summary_path.exists():
        raise RealtimeScoringError(f'필수 파일이 없습니다: {summary_path}')
    df = pd.read_csv(summary_path, low_memory=False)
    if 'customer_id' not in df.columns:
        raise RealtimeScoringError(f'customer_id 컬럼이 없습니다: {summary_path}')
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    df = df.dropna(subset=['customer_id']).copy()
    df['customer_id'] = df['customer_id'].astype(int)
    for column in ['churn_probability', 'clv', 'expected_roi', 'coupon_cost', 'coupon_affinity', 'support_contact_propensity']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0)
    return df


def _initial_summary(config: RealtimeStreamConfig) -> Dict[str, Any]:
    return {
        'bootstrapped_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
        'bootstrapped_customers': 0,
        'processed_events': 0,
        'last_produced_event_at': None,
        'last_consumed_event_at': None,
        'closed_loop_budget_limit': int(config.default_budget_limit),
        'closed_loop_budget_spent': 0,
        'daily_channel_capacity': int(config.daily_channel_capacity),
        'daily_channel_allocated': 0,
        'triggered_reoptimizations': 0,
        'queued_actions_total': 0,
        'deferred_actions_total': 0,
        'action_queue_size': 0,
    }


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
        'action_queue_status': 'idle',
        'queued_recommended_action': '',
        'queued_intervention_intensity': '',
        'queued_coupon_cost': 0,
        'queued_expected_profit': 0.0,
        'queued_expected_roi': 0.0,
        'action_queue_priority': 0.0,
        'latest_trigger_reason': '',
        'reoptimization_count': 0,
        'last_reoptimized_at': '',
        'dispatched_intensity_history': '',
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


def _parse_history_set(state: Dict[str, Any]) -> set[str]:
    raw = str(state.get('dispatched_intensity_history', '') or '').strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(',') if item.strip()}


def _format_history(history: set[str]) -> str:
    return ','.join(sorted(history))


def _should_trigger_reoptimization(
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
    event: Dict[str, Any],
    config: RealtimeStreamConfig,
) -> tuple[bool, str]:
    prev_score = _safe_float(previous_state.get('realtime_churn_score', previous_state.get('base_churn_probability', 0.5)), 0.5)
    current_score = _safe_float(current_state.get('realtime_churn_score', current_state.get('base_churn_probability', 0.5)), 0.5)
    score_jump = current_score - prev_score
    event_type = str(event.get('event_type', '')).strip().lower()

    reasons: list[str] = []
    if current_score >= config.reoptimize_high_risk_threshold and prev_score < config.reoptimize_high_risk_threshold:
        reasons.append('critical_risk_cross')
    if current_score >= config.reoptimize_customer_threshold and prev_score < config.reoptimize_customer_threshold:
        reasons.append('risk_threshold_cross')
    if score_jump >= config.reoptimize_score_delta_threshold:
        reasons.append('score_spike')
    if event_type in {'remove_from_cart', 'support_contact'} and current_score >= config.reoptimize_customer_threshold:
        reasons.append(f'{event_type}_event')

    purchase_signal = _safe_float(current_state.get('purchase_signal', 0.0))
    support_signal = _safe_float(current_state.get('support_signal', 0.0))
    cart_remove_signal = _safe_float(current_state.get('cart_remove_signal', 0.0))
    inactivity_signal = _safe_float(current_state.get('inactivity_signal', 0.0))
    if support_signal >= 0.85 and cart_remove_signal >= 0.85 and purchase_signal <= 0.15:
        reasons.append('support_and_cart_remove_without_purchase')
    if inactivity_signal >= 0.75 and purchase_signal <= 0.10 and current_score >= config.reoptimize_customer_threshold:
        reasons.append('extended_inactivity')

    if not reasons:
        return False, ''
    return True, ', '.join(dict.fromkeys(reasons))


def _reoptimize_customer_action(
    *,
    current_state: Dict[str, Any],
    baseline_row: pd.Series | None,
    survival_predictions: pd.DataFrame,
    remaining_budget: int,
    remaining_capacity: int,
) -> Dict[str, Any]:
    if baseline_row is None:
        return {
            'action_queue_status': 'hold',
            'latest_trigger_reason': 'baseline_missing',
            'action_queue_priority': 0.0,
        }

    base = pd.DataFrame([baseline_row.to_dict()])
    base['customer_id'] = pd.to_numeric(base['customer_id'], errors='coerce').fillna(_safe_int(current_state.get('customer_id', 0), 0)).astype(int)
    base['churn_probability'] = float(_safe_float(current_state.get('realtime_churn_score', current_state.get('base_churn_probability', 0.50)), 0.50))
    if 'persona' not in base.columns:
        base['persona'] = str(current_state.get('persona', 'unknown'))
    if 'uplift_segment' not in base.columns:
        base['uplift_segment'] = str(current_state.get('uplift_segment', 'unknown'))

    candidates = build_intensity_action_candidates(base, survival_predictions=survival_predictions)
    if candidates.empty:
        return {
            'action_queue_status': 'hold',
            'latest_trigger_reason': 'candidate_actions_missing',
            'action_queue_priority': 0.0,
        }

    blocked = _parse_history_set(current_state)
    if current_state.get('queued_intervention_intensity'):
        blocked.add(str(current_state.get('queued_intervention_intensity')))

    candidates = candidates.copy()
    candidates['coupon_cost'] = pd.to_numeric(candidates.get('coupon_cost', 0.0), errors='coerce').fillna(0.0)
    candidates['expected_incremental_profit'] = pd.to_numeric(candidates.get('expected_incremental_profit', 0.0), errors='coerce').fillna(0.0)
    candidates['expected_roi'] = pd.to_numeric(candidates.get('expected_roi', 0.0), errors='coerce').fillna(0.0)
    candidates['timing_urgency_score'] = pd.to_numeric(candidates.get('timing_urgency_score', 0.0), errors='coerce').fillna(0.0)
    candidates = candidates[
        (candidates['coupon_cost'] > 0.0)
        & (candidates['expected_incremental_profit'] > 0.0)
        & ~candidates.get('intervention_intensity', pd.Series(index=candidates.index, dtype=object)).astype(str).isin(blocked)
    ].copy()

    if candidates.empty:
        return {
            'action_queue_status': 'hold',
            'latest_trigger_reason': 'all_intensities_already_used',
            'action_queue_priority': 0.0,
        }

    profit_rank = _normalize(candidates['expected_incremental_profit'])
    roi_rank = _normalize(candidates['expected_roi'].clip(lower=0.0))
    candidates['action_priority'] = (
        0.34 * profit_rank
        + 0.28 * roi_rank
        + 0.23 * candidates['timing_urgency_score'].clip(lower=0.0, upper=1.0)
        + 0.15 * float(_safe_float(current_state.get('realtime_churn_score', 0.0), 0.0))
    )
    candidates = candidates.sort_values(
        ['action_priority', 'expected_incremental_profit', 'expected_roi', 'coupon_cost'],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    if remaining_budget <= 0:
        best = candidates.iloc[0]
        return {
            'action_queue_status': 'deferred_budget_guardrail',
            'queued_recommended_action': str(best.get('recommended_action', '')),
            'queued_intervention_intensity': str(best.get('intervention_intensity', '')),
            'queued_coupon_cost': int(round(float(best.get('coupon_cost', 0.0)))),
            'queued_expected_profit': round(float(best.get('expected_incremental_profit', 0.0)), 2),
            'queued_expected_roi': round(float(best.get('expected_roi', 0.0)), 6),
            'action_queue_priority': round(float(best.get('action_priority', 0.0)), 6),
        }
    if remaining_capacity <= 0:
        best = candidates.iloc[0]
        return {
            'action_queue_status': 'deferred_channel_guardrail',
            'queued_recommended_action': str(best.get('recommended_action', '')),
            'queued_intervention_intensity': str(best.get('intervention_intensity', '')),
            'queued_coupon_cost': int(round(float(best.get('coupon_cost', 0.0)))),
            'queued_expected_profit': round(float(best.get('expected_incremental_profit', 0.0)), 2),
            'queued_expected_roi': round(float(best.get('expected_roi', 0.0)), 6),
            'action_queue_priority': round(float(best.get('action_priority', 0.0)), 6),
        }

    affordable = candidates[candidates['coupon_cost'] <= float(remaining_budget)].copy()
    if affordable.empty:
        best = candidates.iloc[0]
        return {
            'action_queue_status': 'deferred_budget_guardrail',
            'queued_recommended_action': str(best.get('recommended_action', '')),
            'queued_intervention_intensity': str(best.get('intervention_intensity', '')),
            'queued_coupon_cost': int(round(float(best.get('coupon_cost', 0.0)))),
            'queued_expected_profit': round(float(best.get('expected_incremental_profit', 0.0)), 2),
            'queued_expected_roi': round(float(best.get('expected_roi', 0.0)), 6),
            'action_queue_priority': round(float(best.get('action_priority', 0.0)), 6),
        }

    best = affordable.iloc[0]
    history = _parse_history_set(current_state)
    chosen_intensity = str(best.get('intervention_intensity', '')).strip()
    if chosen_intensity:
        history.add(chosen_intensity)
    return {
        'action_queue_status': 'queued',
        'queued_recommended_action': str(best.get('recommended_action', '')),
        'queued_intervention_intensity': chosen_intensity,
        'queued_coupon_cost': int(round(float(best.get('coupon_cost', 0.0)))),
        'queued_expected_profit': round(float(best.get('expected_incremental_profit', 0.0)), 2),
        'queued_expected_roi': round(float(best.get('expected_roi', 0.0)), 6),
        'action_queue_priority': round(float(best.get('action_priority', 0.0)), 6),
        'dispatched_intensity_history': _format_history(history),
    }


def _read_summary(client, config: RealtimeStreamConfig) -> Dict[str, Any]:
    raw_summary = client.get(config.summary_key)
    if raw_summary:
        try:
            return json.loads(raw_summary)
        except json.JSONDecodeError:
            pass
    return _initial_summary(config)


def _update_running_summary_for_action(
    summary: Dict[str, Any],
    previous_state: Dict[str, Any],
    updated_state: Dict[str, Any],
    *,
    triggered: bool,
) -> None:
    if triggered:
        summary['triggered_reoptimizations'] = _safe_int(summary.get('triggered_reoptimizations', 0), 0) + 1

    previous_status = str(previous_state.get('action_queue_status', '') or '')
    new_status = str(updated_state.get('action_queue_status', '') or '')
    prev_cost = _safe_int(previous_state.get('queued_coupon_cost', 0), 0)
    new_cost = _safe_int(updated_state.get('queued_coupon_cost', 0), 0)

    budget_spent = _safe_int(summary.get('closed_loop_budget_spent', 0), 0)
    allocated = _safe_int(summary.get('daily_channel_allocated', 0), 0)

    if previous_status == 'queued' and new_status != 'queued':
        budget_spent = max(0, budget_spent - prev_cost)
        allocated = max(0, allocated - 1)
    elif previous_status != 'queued' and new_status == 'queued':
        budget_spent += new_cost
        allocated += 1
        summary['queued_actions_total'] = _safe_int(summary.get('queued_actions_total', 0), 0) + 1
    elif previous_status == 'queued' and new_status == 'queued':
        budget_spent = max(0, budget_spent - prev_cost + new_cost)

    if triggered and new_status.startswith('deferred'):
        summary['deferred_actions_total'] = _safe_int(summary.get('deferred_actions_total', 0), 0) + 1

    summary['closed_loop_budget_spent'] = int(max(0, budget_spent))
    summary['daily_channel_allocated'] = int(max(0, allocated))


def _queue_snapshot_from_redis(client, result_dir: Path, config: RealtimeStreamConfig, top_n: int | None = None) -> Dict[str, Any]:
    top_n = int(top_n or config.snapshot_top_n)
    members = client.zrevrange(config.action_queue_key, 0, max(top_n - 1, 0), withscores=True)
    records: list[Dict[str, Any]] = []
    for customer_id, priority in members:
        raw_state = client.hgetall(config.state_key(customer_id))
        if not raw_state:
            continue
        record = {
            'customer_id': _safe_int(raw_state.get('customer_id', customer_id), _safe_int(customer_id)),
            'persona': str(raw_state.get('persona', 'unknown')),
            'uplift_segment': str(raw_state.get('uplift_segment', 'unknown')),
            'realtime_churn_score': _safe_float(raw_state.get('realtime_churn_score', 0.0), 0.0),
            'action_queue_status': str(raw_state.get('action_queue_status', 'idle')),
            'queued_recommended_action': str(raw_state.get('queued_recommended_action', '')),
            'queued_intervention_intensity': str(raw_state.get('queued_intervention_intensity', '')),
            'queued_coupon_cost': _safe_int(raw_state.get('queued_coupon_cost', 0), 0),
            'queued_expected_profit': _safe_float(raw_state.get('queued_expected_profit', 0.0), 0.0),
            'queued_expected_roi': _safe_float(raw_state.get('queued_expected_roi', 0.0), 0.0),
            'action_queue_priority': float(priority),
            'latest_trigger_reason': str(raw_state.get('latest_trigger_reason', '')),
            'last_reoptimized_at': str(raw_state.get('last_reoptimized_at', '')),
            'reoptimization_count': _safe_int(raw_state.get('reoptimization_count', 0), 0),
        }
        records.append(record)

    queue_df = pd.DataFrame(records)
    if not queue_df.empty:
        queue_df = queue_df.sort_values(['action_queue_priority', 'queued_expected_profit', 'customer_id'], ascending=[False, False, True])
    queue_summary = {
        'queue_size': int(client.zcard(config.action_queue_key)),
        'high_priority_queue_size': int(client.zcount(config.action_queue_key, 0.70, '+inf')),
        'generated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
    }
    queue_csv, queue_json = _queue_paths(result_dir)
    if not queue_df.empty:
        queue_df.to_csv(queue_csv, index=False)
    else:
        pd.DataFrame(columns=['customer_id', 'queued_recommended_action']).to_csv(queue_csv, index=False)
    queue_json.write_text(json.dumps(queue_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'summary': queue_summary, 'records': queue_df.to_dict(orient='records') if not queue_df.empty else []}


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
        for field in TRACKED_SIGNAL_FIELDS + ['clv', 'expected_roi', 'coupon_affinity', 'support_contact_propensity', 'behavioral_risk', 'inactivity_signal', 'queued_expected_profit', 'queued_expected_roi', 'action_queue_priority']:
            raw_state[field] = _safe_float(raw_state.get(field, 0.0), 0.0)
        for field in ['coupon_cost', 'queued_coupon_cost', 'reoptimization_count']:
            raw_state[field] = _safe_int(raw_state.get(field, 0), 0)
        for field in ['action_queue_status', 'queued_recommended_action', 'queued_intervention_intensity', 'latest_trigger_reason', 'last_reoptimized_at', 'dispatched_intensity_history']:
            raw_state[field] = str(raw_state.get(field, ''))
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

    queue_payload = _queue_snapshot_from_redis(client, result_dir, config, top_n=top_n)
    summary.update({
        'action_queue_size': int(queue_payload['summary'].get('queue_size', 0)),
        'high_priority_queue_size': int(queue_payload['summary'].get('high_priority_queue_size', 0)),
    })

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
    df = _load_baseline_customer_summary(data_dir)

    client = _redis_client(config)
    if reset_stream:
        for key in [config.ranking_key, config.action_queue_key, config.summary_key, config.stream_key, config.trigger_log_key]:
            client.delete(key)
        for match in client.scan_iter(match=f'{config.state_key_prefix}:*'):
            client.delete(match)

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

    summary = _initial_summary(config)
    summary['bootstrapped_customers'] = int(len(df))
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

    summary = _read_summary(client, config)
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
    baseline_df = _load_baseline_customer_summary(data_dir)
    baseline_indexed = baseline_df.set_index('customer_id', drop=False)
    survival_predictions = load_survival_predictions(result_dir)
    summary = _read_summary(client, config)

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
                    row = baseline_indexed.loc[customer_id] if customer_id in baseline_indexed.index else None
                    if row is None:
                        raw_state = _seed_state_from_row({'customer_id': customer_id, 'churn_probability': 0.50}, config)
                    else:
                        raw_state = _seed_state_from_row(row, config)

                previous_state = dict(raw_state)
                updated = _apply_event_to_state(raw_state, payload, event_ts)
                triggered, reason = _should_trigger_reoptimization(previous_state, updated, payload, config)
                if triggered:
                    remaining_budget = _safe_int(summary.get('closed_loop_budget_limit', config.default_budget_limit), config.default_budget_limit) - _safe_int(summary.get('closed_loop_budget_spent', 0), 0)
                    remaining_capacity = _safe_int(summary.get('daily_channel_capacity', config.daily_channel_capacity), config.daily_channel_capacity) - _safe_int(summary.get('daily_channel_allocated', 0), 0)
                    baseline_row = baseline_indexed.loc[customer_id] if customer_id in baseline_indexed.index else None
                    action_update = _reoptimize_customer_action(
                        current_state=updated,
                        baseline_row=baseline_row,
                        survival_predictions=survival_predictions,
                        remaining_budget=max(remaining_budget, 0),
                        remaining_capacity=max(remaining_capacity, 0),
                    )
                    updated.update(action_update)
                    updated['latest_trigger_reason'] = reason if reason else str(updated.get('latest_trigger_reason', ''))
                    updated['reoptimization_count'] = _safe_int(previous_state.get('reoptimization_count', 0), 0) + 1
                    updated['last_reoptimized_at'] = event_ts.floor('s').isoformat()
                else:
                    updated['latest_trigger_reason'] = str(previous_state.get('latest_trigger_reason', ''))
                    updated['reoptimization_count'] = _safe_int(previous_state.get('reoptimization_count', 0), 0)
                    updated['last_reoptimized_at'] = str(previous_state.get('last_reoptimized_at', ''))

                _update_running_summary_for_action(summary, previous_state, updated, triggered=triggered)
                pipe.hset(config.state_key(customer_id), mapping=_redis_safe_mapping(updated))
                pipe.zadd(config.ranking_key, {str(customer_id): float(updated['realtime_churn_score'])})
                if str(updated.get('action_queue_status', '')) == 'queued':
                    pipe.zadd(config.action_queue_key, {str(customer_id): float(_safe_float(updated.get('action_queue_priority', 0.0), 0.0))})
                else:
                    pipe.zrem(config.action_queue_key, str(customer_id))
                if triggered:
                    pipe.xadd(
                        config.trigger_log_key,
                        _redis_safe_mapping(
                            {
                                'customer_id': customer_id,
                                'event_type': str(payload.get('event_type', 'unknown')),
                                'timestamp': event_ts.floor('s').isoformat(),
                                'realtime_churn_score': updated.get('realtime_churn_score', 0.0),
                                'trigger_reason': updated.get('latest_trigger_reason', ''),
                                'action_queue_status': updated.get('action_queue_status', 'idle'),
                                'queued_intervention_intensity': updated.get('queued_intervention_intensity', ''),
                                'queued_recommended_action': updated.get('queued_recommended_action', ''),
                            }
                        ),
                        maxlen=50000,
                        approximate=True,
                    )
                pipe.xack(config.stream_key, config.consumer_group, message_id)
                processed += 1
                last_consumed_event_at = event_ts.floor('s').isoformat()
                if snapshot_every > 0 and processed % int(snapshot_every) == 0:
                    summary['action_queue_size'] = int(client.zcard(config.action_queue_key))
                    pipe.set(config.summary_key, json.dumps(summary, ensure_ascii=False))
                    pipe.execute()
                    pipe = client.pipeline(transaction=False)
        pipe.execute()

    summary.update(
        {
            'processed_events': _safe_int(summary.get('processed_events', 0), 0) + processed,
            'last_consumed_event_at': last_consumed_event_at,
            'consumer_name': config.consumer_name,
            'consumer_updated_at': pd.Timestamp.now(tz='UTC').floor('s').isoformat(),
            'action_queue_size': int(client.zcard(config.action_queue_key)),
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
    queue_csv, queue_json = _queue_paths(result_dir)
    summary: Dict[str, Any] = {}
    if snapshot_json.exists():
        try:
            summary = json.loads(snapshot_json.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            summary = {}
    if queue_json.exists():
        try:
            queue_summary = json.loads(queue_json.read_text(encoding='utf-8'))
            summary.update({
                'action_queue_size': int(queue_summary.get('queue_size', summary.get('action_queue_size', 0))),
                'high_priority_queue_size': int(queue_summary.get('high_priority_queue_size', summary.get('high_priority_queue_size', 0))),
            })
        except json.JSONDecodeError:
            pass
    if snapshot_csv.exists():
        df = pd.read_csv(snapshot_csv).head(int(top_n))
    else:
        df = pd.DataFrame()
    if queue_csv.exists() and 'action_queue_size' not in summary:
        try:
            queue_df = pd.read_csv(queue_csv)
            summary['action_queue_size'] = int(len(queue_df))
        except Exception:
            pass
    summary['source'] = 'snapshot'
    return {'summary': summary, 'records': df.to_dict(orient='records') if not df.empty else []}
