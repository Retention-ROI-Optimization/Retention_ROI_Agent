from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.engineering import build_feature_dataset, _compute_data_span_days, auto_adjust_horizon_days


if TYPE_CHECKING:  # pragma: no cover - typing only
    from lifelines import CoxPHFitter as CoxPHFitterType
    from lifelines import KaplanMeierFitter as KaplanMeierFitterType
else:  # pragma: no cover - runtime fallback for optional dependency
    CoxPHFitterType = Any
    KaplanMeierFitterType = Any

try:  # pragma: no cover - optional dependency
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.exceptions import ConvergenceError
    from lifelines.utils import concordance_index
except Exception as exc:  # pragma: no cover
    CoxPHFitter = None
    KaplanMeierFitter = None
    ConvergenceError = RuntimeError
    concordance_index = None
    LIFELINES_IMPORT_ERROR = str(exc)
else:
    LIFELINES_IMPORT_ERROR = None


@dataclass
class SurvivalArtifacts:
    model_path: str
    metrics_path: str
    predictions_path: str
    coefficients_path: str
    risk_plot_path: str
    metrics: Dict[str, Any]


class SurvivalModelError(RuntimeError):
    pass


def _require_lifelines() -> None:
    if CoxPHFitter is None or KaplanMeierFitter is None or concordance_index is None:
        message = 'lifelines 패키지가 설치되지 않았습니다. `pip install lifelines` 후 다시 실행하세요.'
        if LIFELINES_IMPORT_ERROR:
            message = f'{message} (import error: {LIFELINES_IMPORT_ERROR})'
        raise SurvivalModelError(message)


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_preprocessing_metadata(data_dir: Path) -> Dict[str, Any]:
    metadata_path = data_dir / 'preprocessing_metadata.json'
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _load_activity_times(data_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    events_path = data_dir / 'events.csv'
    if events_path.exists():
        events = pd.read_csv(events_path, low_memory=False)
        if {'customer_id', 'timestamp'}.issubset(events.columns):
            cols = ['customer_id', 'timestamp'] + (['event_type'] if 'event_type' in events.columns else [])
            events = events[cols].copy()
            events['activity_time'] = pd.to_datetime(events['timestamp'], errors='coerce')
            if 'event_type' in events.columns:
                events['activity_kind'] = events['event_type'].astype(str).str.lower()
                if events['activity_kind'].isin(['visit', 'purchase']).any():
                    events = events[events['activity_kind'].isin(['visit', 'purchase'])]
            pieces.append(events[['customer_id', 'activity_time']])
    orders_path = data_dir / 'orders.csv'
    if orders_path.exists():
        orders = pd.read_csv(orders_path, low_memory=False)
        if {'customer_id', 'order_time'}.issubset(orders.columns):
            orders = orders[['customer_id', 'order_time']].copy()
            orders['activity_time'] = pd.to_datetime(orders['order_time'], errors='coerce')
            pieces.append(orders[['customer_id', 'activity_time']])
    if not pieces:
        return pd.DataFrame(columns=['customer_id', 'activity_time'])
    activity = pd.concat(pieces, ignore_index=True).dropna(subset=['customer_id', 'activity_time'])
    if activity.empty:
        return pd.DataFrame(columns=['customer_id', 'activity_time'])
    activity['customer_id'] = pd.to_numeric(activity['customer_id'], errors='coerce')
    activity = activity.dropna(subset=['customer_id'])
    activity['customer_id'] = activity['customer_id'].astype(int)
    return activity.drop_duplicates().sort_values(['customer_id', 'activity_time'])


def _load_customer_signup_dates(data_dir: Path) -> pd.Series:
    customers_path = data_dir / 'customers.csv'
    if not customers_path.exists():
        return pd.Series(dtype='datetime64[ns]')
    customers = pd.read_csv(customers_path, low_memory=False)
    if not {'customer_id', 'signup_date'}.issubset(customers.columns):
        return pd.Series(dtype='datetime64[ns]')
    customers['customer_id'] = pd.to_numeric(customers['customer_id'], errors='coerce')
    customers = customers.dropna(subset=['customer_id'])
    customers['customer_id'] = customers['customer_id'].astype(int)
    customers['signup_date'] = pd.to_datetime(customers['signup_date'], errors='coerce')
    return customers.drop_duplicates('customer_id', keep='last').set_index('customer_id')['signup_date']


def _activity_gap_events(
    data_dir: Path,
    customer_ids: pd.Series,
    landmark_date: pd.Timestamp,
    horizon_days: int,
) -> tuple[pd.DataFrame | None, pd.Index | None, Dict[str, Any]]:
    preprocessing_metadata = _load_preprocessing_metadata(data_dir)
    if preprocessing_metadata.get('source') != 'user_upload':
        return None, None, {'survival_event_source': 'state_snapshots_status'}

    activity = _load_activity_times(data_dir)
    if activity.empty:
        return None, None, {'survival_event_source': 'state_snapshots_status', 'activity_rows': 0}

    threshold_days = int(preprocessing_metadata.get('churn_inactivity_threshold_days') or 30)
    threshold = pd.Timedelta(days=max(threshold_days, 1))
    horizon_end = landmark_date + pd.Timedelta(days=int(horizon_days))
    min_event_date = landmark_date + pd.Timedelta(days=1)
    signup_dates = _load_customer_signup_dates(data_dir)
    activity_by_customer = {
        int(customer_id): group['activity_time'].sort_values().tolist()
        for customer_id, group in activity.groupby('customer_id', sort=False)
    }

    rows: list[Dict[str, Any]] = []
    eligible_customer_ids: list[int] = []
    prevalent_event_count = 0
    for raw_customer_id in customer_ids:
        customer_id = int(raw_customer_id)
        times = activity_by_customer.get(customer_id, [])
        signup_date = pd.Timestamp(signup_dates.get(customer_id, landmark_date))
        if pd.isna(signup_date):
            signup_date = landmark_date

        historical = [ts for ts in times if ts <= landmark_date]
        if historical:
            last_activity = pd.Timestamp(historical[-1])
        elif signup_date <= landmark_date:
            last_activity = signup_date
        else:
            last_activity = landmark_date

        event_date: pd.Timestamp | None = None
        if last_activity + threshold <= landmark_date:
            prevalent_event_count += 1
            continue
        else:
            eligible_customer_ids.append(customer_id)
            future_times = [pd.Timestamp(ts) for ts in times if landmark_date < ts <= horizon_end]
            for next_activity in future_times:
                gap_event_date = last_activity + threshold
                if gap_event_date < next_activity and gap_event_date <= horizon_end:
                    event_date = max(gap_event_date, min_event_date)
                    break
                last_activity = next_activity
            if event_date is None:
                gap_event_date = last_activity + threshold
                if gap_event_date <= horizon_end:
                    event_date = max(gap_event_date, min_event_date)

        if event_date is not None:
            rows.append({'customer_id': customer_id, 'event_date': event_date})

    return pd.DataFrame(rows, columns=['customer_id', 'event_date']), pd.Index(eligible_customer_ids, dtype='int64'), {
        'survival_event_source': 'activity_gap_rule',
        'activity_rows': int(len(activity)),
        'churn_inactivity_threshold_days': int(threshold_days),
        'excluded_prevalent_churn_rows': int(prevalent_event_count),
        'landmark_at_risk_rows': int(len(eligible_customer_ids)),
    }


def _build_landmark_dataset(
    data_dir: Path,
    feature_store_dir: Path,
    *,
    as_of_date: str | pd.Timestamp | None,
    horizon_days: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    built = build_feature_dataset(
        data_dir=data_dir,
        feature_store_dir=feature_store_dir,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
    )
    features = built.features.copy()
    metadata = dict(built.metadata)
    landmark_date = pd.Timestamp(metadata['as_of_date'])

    first_event, eligible_customer_ids, event_metadata = _activity_gap_events(
        data_dir,
        features['customer_id'],
        landmark_date,
        int(horizon_days),
    )
    if eligible_customer_ids is not None:
        features = features[features['customer_id'].astype(int).isin(eligible_customer_ids)].copy()
        if features.empty:
            raise SurvivalModelError(
                '기준일 시점에 이미 이탈 조건을 충족한 고객을 제외한 뒤 생존분석 대상 고객이 없습니다. '
                'as_of_date를 더 이른 날짜로 지정하거나 이탈 기준일 수를 늘려주세요.'
            )
    if first_event is None:
        snapshots = pd.read_csv(
            data_dir / 'state_snapshots.csv',
            parse_dates=['snapshot_date', 'last_visit_date', 'last_purchase_date'],
        )
        future = snapshots.loc[
            (snapshots['snapshot_date'] > landmark_date)
            & (snapshots['snapshot_date'] <= landmark_date + pd.Timedelta(days=horizon_days))
            & (snapshots['current_status'].astype(str).isin(['churn_risk', 'churned'])),
            ['customer_id', 'snapshot_date'],
        ].copy()
        first_event = future.groupby('customer_id', as_index=False)['snapshot_date'].min().rename(
            columns={'snapshot_date': 'event_date'}
        )
        event_metadata = {'survival_event_source': 'state_snapshots_status'}
    metadata.update(event_metadata)
    features = features.merge(first_event, on='customer_id', how='left')
    duration = np.where(
        features['event_date'].notna(),
        (pd.to_datetime(features['event_date']) - landmark_date).dt.days.clip(lower=1),
        int(horizon_days),
    )
    features['duration_days'] = pd.Series(duration, index=features.index).astype(int)
    features['event_observed'] = features['event_date'].notna().astype(int)
    features.drop(columns=['event_date'], inplace=True)
    return features, metadata


def _collapse_rare_categories(
    series: pd.Series,
    *,
    max_levels: int,
    min_frequency: float,
) -> pd.Series:
    normalized = series.astype('object').where(series.notna(), 'unknown').astype(str)
    value_share = normalized.value_counts(normalize=True, dropna=False)
    if value_share.empty:
        return normalized
    keep = set(value_share.head(max_levels).index.tolist())
    keep |= set(value_share[value_share >= float(min_frequency)].index.tolist())
    return normalized.where(normalized.isin(keep), '__rare__')


def _filter_problematic_encoded_columns(
    encoded: pd.DataFrame,
    event_observed: pd.Series,
    *,
    min_variance: float = 1e-4,
    min_prevalence: float = 0.005,
    max_features: int = 140,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        'dropped_zero_or_low_variance': [],
        'dropped_extreme_prevalence': [],
        'dropped_complete_separation': [],
        'trimmed_by_variance_rank': [],
    }
    working = encoded.copy()

    variance = working.var(axis=0)
    keep = variance[variance > float(min_variance)].index.tolist()
    metadata['dropped_zero_or_low_variance'] = [col for col in working.columns if col not in keep]
    working = working.loc[:, keep]
    if working.empty:
        return working, metadata

    prevalence = working.mean(axis=0)
    keep = prevalence[prevalence.between(float(min_prevalence), 1.0 - float(min_prevalence))].index.tolist()
    metadata['dropped_extreme_prevalence'] = [col for col in working.columns if col not in keep]
    working = working.loc[:, keep]
    if working.empty:
        return working, metadata

    event_mask = event_observed.astype(int).eq(1)
    censored_mask = ~event_mask
    separation_drop: list[str] = []
    if event_mask.any() and censored_mask.any():
        for col in working.columns:
            series = working[col]
            if series.nunique(dropna=False) > 2:
                continue
            event_values = series.loc[event_mask]
            censored_values = series.loc[censored_mask]
            if event_values.empty or censored_values.empty:
                continue
            event_mean = float(event_values.mean())
            censored_mean = float(censored_values.mean())
            event_var = float(event_values.var(ddof=0))
            censored_var = float(censored_values.var(ddof=0))
            if min(event_var, censored_var) < 1e-8 and abs(event_mean - censored_mean) >= 0.20:
                separation_drop.append(col)
    if separation_drop:
        metadata['dropped_complete_separation'] = separation_drop
        working = working.drop(columns=separation_drop, errors='ignore')
    if working.empty:
        return working, metadata

    if working.shape[1] > int(max_features):
        ranked = working.var(axis=0).sort_values(ascending=False)
        keep = ranked.head(int(max_features)).index.tolist()
        metadata['trimmed_by_variance_rank'] = [col for col in working.columns if col not in keep]
        working = working.loc[:, keep]

    return working, metadata


def _prepare_survival_frame(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base = feature_df.copy()
    base = base.drop(columns=['label'], errors='ignore')
    base = base.replace([np.inf, -np.inf], np.nan)

    id_cols = ['customer_id']
    target_cols = ['duration_days', 'event_observed']
    feature_cols = [c for c in base.columns if c not in id_cols + target_cols]

    prepared = base[['customer_id', 'duration_days', 'event_observed']].copy()
    raw_features = base[id_cols + feature_cols].copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(base[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_frame = pd.DataFrame(index=base.index)
    if numeric_cols:
        numeric_frame = base[numeric_cols].apply(pd.to_numeric, errors='coerce')
        fill_values = {
            col: float(numeric_frame[col].median()) if numeric_frame[col].notna().any() else 0.0
            for col in numeric_frame.columns
        }
        numeric_frame = numeric_frame.fillna(fill_values).astype(float)

    categorical_map: dict[str, pd.Series] = {}
    for col in categorical_cols:
        if col.startswith('recent_event_sequence_'):
            categorical_map[col] = _collapse_rare_categories(base[col], max_levels=6, min_frequency=0.02)
        elif base[col].nunique(dropna=True) > 20:
            categorical_map[col] = _collapse_rare_categories(base[col], max_levels=10, min_frequency=0.01)
        else:
            categorical_map[col] = base[col].astype('object').where(base[col].notna(), 'unknown').astype(str)
    categorical_frame = pd.DataFrame(categorical_map, index=base.index) if categorical_map else pd.DataFrame(index=base.index)

    encoded_source = pd.concat([numeric_frame, categorical_frame], axis=1)
    encoded = pd.get_dummies(encoded_source, drop_first=True, dtype=float)
    if encoded.empty:
        raise SurvivalModelError('생존분석용 feature가 비어 있습니다.')

    encoded, filtering_meta = _filter_problematic_encoded_columns(encoded, prepared['event_observed'])
    if encoded.empty:
        raise SurvivalModelError('생존분석용 feature가 모두 제거되었습니다. 입력 feature 구성을 다시 확인하세요.')

    prepared = pd.concat([prepared, encoded], axis=1)
    prepared['duration_days'] = prepared['duration_days'].astype(float).clip(lower=1.0)
    prepared['event_observed'] = prepared['event_observed'].astype(int)

    prep_meta = {
        'numeric_feature_count': int(len(numeric_cols)),
        'categorical_feature_count': int(len(categorical_cols)),
        'encoded_feature_count': int(encoded.shape[1]),
        **filtering_meta,
    }
    return prepared, raw_features, prep_meta


def _median_survival_time(curve: pd.Series, fallback: int) -> float:
    below = curve[curve <= 0.5]
    if below.empty:
        return float(fallback)
    return float(below.index[0])


def _plot_risk_groups(df: pd.DataFrame, output_path: Path, horizon_days: int) -> None:
    _require_lifelines()
    assert KaplanMeierFitter is not None

    plt.figure(figsize=(8, 5))
    kmf_cls = cast(type[KaplanMeierFitterType], KaplanMeierFitter)
    kmf = kmf_cls()
    for group_name in ['Low risk', 'Mid risk', 'High risk']:
        subset = df[df['risk_group'] == group_name]
        if subset.empty:
            continue
        kmf.fit(
            durations=subset['duration_days'],
            event_observed=subset['event_observed'],
            label=group_name,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.xlim(0, horizon_days)
    plt.ylim(0, 1.0)
    plt.xlabel('Days from landmark date')
    plt.ylabel('Survival probability')
    plt.title('Survival curve by predicted risk group')
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _fit_cox_with_retry(
    train_df: pd.DataFrame,
    *,
    base_penalizer: float,
) -> tuple[CoxPHFitterType, float, list[Dict[str, Any]]]:
    _require_lifelines()
    assert CoxPHFitter is not None

    cox_cls = cast(type[CoxPHFitterType], CoxPHFitter)

    attempts: list[Dict[str, Any]] = []
    penalties = []
    for value in [base_penalizer, max(base_penalizer, 0.25), max(base_penalizer, 0.5), max(base_penalizer, 1.0)]:
        if value not in penalties:
            penalties.append(float(value))

    last_error: Exception | None = None
    for penalty in penalties:
        model = cox_cls(penalizer=float(penalty))
        try:
            model.fit(
                train_df,
                duration_col='duration_days',
                event_col='event_observed',
                robust=True,
                show_progress=False,
            )
            if not np.isfinite(model.params_.values).all():
                raise SurvivalModelError('추정된 Cox 계수에 NaN/inf가 포함되었습니다.')
            attempts.append({'penalizer': penalty, 'status': 'success'})
            return model, float(penalty), attempts
        except (ConvergenceError, ValueError, np.linalg.LinAlgError, SurvivalModelError) as exc:
            attempts.append({'penalizer': penalty, 'status': 'failed', 'error': str(exc)})
            last_error = exc
    raise SurvivalModelError(
        'CoxPH 수렴에 실패했습니다. 사용 feature를 더 줄이거나 penalizer를 높여야 합니다. '
        f'시도 내역: {attempts}'
    ) from last_error


def _prepare_current_features_for_prediction(
    current_features: pd.DataFrame,
    training_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = current_features.copy()
    base = base.drop(columns=['label', 'duration_days', 'event_observed'], errors='ignore')
    base = base.replace([np.inf, -np.inf], np.nan)

    id_cols = ['customer_id']
    feature_cols = [c for c in base.columns if c not in id_cols]
    raw_features = base[id_cols + feature_cols].copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(base[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_frame = pd.DataFrame(index=base.index)
    if numeric_cols:
        numeric_frame = base[numeric_cols].apply(pd.to_numeric, errors='coerce')
        fill_values = {
            col: float(numeric_frame[col].median()) if numeric_frame[col].notna().any() else 0.0
            for col in numeric_frame.columns
        }
        numeric_frame = numeric_frame.fillna(fill_values).astype(float)

    categorical_map: dict[str, pd.Series] = {}
    for col in categorical_cols:
        if col.startswith('recent_event_sequence_'):
            categorical_map[col] = _collapse_rare_categories(base[col], max_levels=6, min_frequency=0.02)
        elif base[col].nunique(dropna=True) > 20:
            categorical_map[col] = _collapse_rare_categories(base[col], max_levels=10, min_frequency=0.01)
        else:
            categorical_map[col] = base[col].astype('object').where(base[col].notna(), 'unknown').astype(str)
    categorical_frame = pd.DataFrame(categorical_map, index=base.index) if categorical_map else pd.DataFrame(index=base.index)

    encoded_source = pd.concat([numeric_frame, categorical_frame], axis=1)
    encoded = pd.get_dummies(encoded_source, drop_first=True, dtype=float)

    aligned = encoded.reindex(columns=training_columns, fill_value=0.0)

    return aligned, raw_features


def run_survival_pipeline(
    data_dir: str | Path,
    model_dir: str | Path,
    result_dir: str | Path,
    *,
    feature_store_dir: str | Path | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    horizon_days: int = 90,
    test_size: float = 0.20,
    random_state: int = 42,
    penalizer: float = 0.25,
) -> SurvivalArtifacts:
    _require_lifelines()
    assert concordance_index is not None

    data_dir = Path(data_dir)
    model_dir = _ensure_dir(model_dir)
    result_dir = _ensure_dir(result_dir)
    survival_feature_dir = _ensure_dir(Path(feature_store_dir or Path('data/feature_store')) / 'survival')

    import logging
    _surv_logger = logging.getLogger(__name__)

    data_span = _compute_data_span_days(data_dir)
    original_horizon = int(horizon_days)
    if data_span is not None:
        adjusted, warning = auto_adjust_horizon_days(data_span, original_horizon)
        if warning:
            _surv_logger.warning("[생존분석 horizon 조정] %s", warning)
        if adjusted == 0:
            raise SurvivalModelError(
                f"데이터 기간이 {data_span}일로 너무 짧아 생존분석을 수행할 수 없습니다. "
                f"최소 60일 이상의 데이터가 필요합니다. "
                f"이탈 확률(churn probability)은 정상 제공되며, 이탈 시점 예측만 비활성화됩니다."
            )
        horizon_days = adjusted
        if adjusted != original_horizon:
            _surv_logger.info(
                "[생존분석] horizon_days 자동 조정: %d일 → %d일 (데이터 기간: %d일)",
                original_horizon, adjusted, data_span,
            )
    else:
        _surv_logger.info("[생존분석] 데이터 기간을 감지할 수 없어 기본 horizon(%d일)을 사용합니다.", original_horizon)

    feature_df, feature_metadata = _build_landmark_dataset(
        data_dir=data_dir,
        feature_store_dir=survival_feature_dir,
        as_of_date=as_of_date,
        horizon_days=int(horizon_days),
    )
    training_frame, raw_features, prep_meta = _prepare_survival_frame(feature_df)

    train_idx, test_idx = train_test_split(
        training_frame.index,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=training_frame['event_observed'],
    )
    train_df = training_frame.loc[train_idx].copy()
    test_df = training_frame.loc[test_idx].copy()

    model, fitted_penalizer, fit_attempts = _fit_cox_with_retry(train_df, base_penalizer=float(penalizer))

    test_features = test_df.drop(columns=['duration_days', 'event_observed'])
    test_partial_hazard = model.predict_partial_hazard(test_features).values.ravel()
    c_index = float(
        concordance_index(
            test_df['duration_days'].values,
            -test_partial_hazard,
            test_df['event_observed'].values,
        )
    )

    survival_times = sorted({1, min(30, horizon_days), min(60, horizon_days), min(90, horizon_days), int(horizon_days)})

    training_pred_frame = training_frame.drop(columns=['duration_days', 'event_observed'])
    training_pred_columns = training_pred_frame.columns.tolist()

    _landmark_hazard = model.predict_partial_hazard(training_pred_frame).values.ravel()
    _plot_df = pd.DataFrame({
        'duration_days': feature_df['duration_days'].astype(int),
        'event_observed': feature_df['event_observed'].astype(int),
        'predicted_hazard_ratio': _landmark_hazard.astype(float),
    })
    _plot_df['risk_group'] = pd.qcut(
        _plot_df['predicted_hazard_ratio'].rank(method='first'),
        q=3, labels=['Low risk', 'Mid risk', 'High risk'],
    )

    today = pd.Timestamp.today().floor('D')
    prediction_base_date = feature_metadata.get('as_of_date')  # fallback: landmark

    try:
        current_built = build_feature_dataset(
            data_dir=data_dir,
            feature_store_dir=survival_feature_dir,
            as_of_date=today,
            horizon_days=int(horizon_days),
        )
        current_aligned, current_raw = _prepare_current_features_for_prediction(
            current_built.features, training_pred_columns,
        )

        _pred_hazard = model.predict_partial_hazard(current_aligned).values.ravel()
        _pred_full = model.predict_survival_function(
            current_aligned, times=list(range(1, int(horizon_days) + 1))
        ).T
        _pred_medians = _pred_full.apply(
            lambda row: _median_survival_time(row, int(horizon_days)), axis=1
        )
        _pred_surv = model.predict_survival_function(current_aligned, times=survival_times).T
        _pred_surv.columns = [f'survival_prob_{int(col)}d' for col in _pred_surv.columns]

        _pred_ids = current_built.features['customer_id'].astype(int)
        _pred_raw = current_raw
        prediction_base_date = str(today.date())
        _surv_logger.info(
            "[생존분석] 오늘(%s) 기준 재예측 완료: %d명", today.date(), len(current_aligned),
        )
    except Exception as exc:
        _surv_logger.warning("[생존분석] 재예측 실패, landmark 예측 유지: %s", exc)
        _pred_hazard = _landmark_hazard
        _pred_full = model.predict_survival_function(
            training_pred_frame, times=list(range(1, int(horizon_days) + 1))
        ).T
        _pred_medians = _pred_full.apply(
            lambda row: _median_survival_time(row, int(horizon_days)), axis=1
        )
        _pred_surv = model.predict_survival_function(training_pred_frame, times=survival_times).T
        _pred_surv.columns = [f'survival_prob_{int(col)}d' for col in _pred_surv.columns]

        _pred_ids = feature_df['customer_id'].astype(int)
        _pred_raw = raw_features

    predictions = pd.DataFrame({
        'customer_id': _pred_ids,
        'predicted_hazard_ratio': _pred_hazard.astype(float),
        'predicted_median_time_to_churn_days': _pred_medians.astype(float),
    })
    predictions = pd.concat([predictions, _pred_surv.reset_index(drop=True)], axis=1)
    if 'survival_prob_30d' not in predictions.columns:
        predictions['survival_prob_30d'] = predictions.filter(like='survival_prob_').iloc[:, 0]
    predictions['risk_percentile'] = predictions['predicted_hazard_ratio'].rank(pct=True, method='average')
    predictions['risk_group'] = pd.qcut(
        predictions['predicted_hazard_ratio'].rank(method='first'),
        q=3,
        labels=['Low risk', 'Mid risk', 'High risk'],
    )
    predictions = predictions.merge(
        _pred_raw[['customer_id'] + [c for c in ['persona', 'region', 'device_type', 'acquisition_channel'] if c in _pred_raw.columns]],
        on='customer_id',
        how='left',
    )
    predictions.sort_values(['predicted_hazard_ratio', 'customer_id'], ascending=[False, True], inplace=True)

    top_coefficients = (
        model.summary.reset_index()
        .rename(columns={'covariate': 'feature'})
        .assign(abs_coef=lambda df: df['coef'].abs())
        .sort_values('abs_coef', ascending=False)
        .loc[:, ['feature', 'coef', 'exp(coef)', 'p', 'abs_coef']]
    )

    model_path = model_dir / 'survival_cox_model.joblib'
    metrics_path = result_dir / 'survival_metrics.json'
    predictions_path = result_dir / 'survival_predictions.csv'
    coefficients_path = result_dir / 'survival_top_coefficients.csv'
    risk_plot_path = result_dir / 'survival_risk_stratification.png'

    joblib.dump(model, model_path)
    predictions.to_csv(predictions_path, index=False)
    top_coefficients.head(30).to_csv(coefficients_path, index=False)
    _plot_risk_groups(_plot_df, risk_plot_path, int(horizon_days))

    metrics = {
        'model_name': 'CoxPHFitter',
        'model_path': str(model_path),
        'predictions_path': str(predictions_path),
        'coefficients_path': str(coefficients_path),
        'risk_plot_path': str(risk_plot_path),
        'landmark_as_of_date': feature_metadata.get('as_of_date'),
        'prediction_as_of_date': prediction_base_date,
        'horizon_days': int(horizon_days),
        'horizon_days_requested': original_horizon,
        'horizon_auto_adjusted': (int(horizon_days) != original_horizon),
        'data_span_days': data_span,
        'row_count': int(len(training_frame)),
        'event_rate': float(training_frame['event_observed'].mean()),
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'test_concordance_index': c_index,
        'test_event_rate': float(test_df['event_observed'].mean()),
        'feature_count_before_encoding': int(len(raw_features.columns) - 1),
        'feature_count_after_encoding': int(training_frame.shape[1] - 3),
        'requested_penalizer': float(penalizer),
        'fitted_penalizer': float(fitted_penalizer),
        'fit_attempts': fit_attempts,
        'preprocessing': prep_meta,
        'test_size': float(test_size),
        'random_state': int(random_state),
        'top_coefficients': top_coefficients.head(10).to_dict(orient='records'),
        'highest_risk_customers': predictions.head(20).to_dict(orient='records'),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    return SurvivalArtifacts(
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        predictions_path=str(predictions_path),
        coefficients_path=str(coefficients_path),
        risk_plot_path=str(risk_plot_path),
        metrics=metrics,
    )
