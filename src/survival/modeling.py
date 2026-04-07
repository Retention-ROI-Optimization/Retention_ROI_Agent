from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.engineering import build_feature_dataset

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
    kmf = KaplanMeierFitter()
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
) -> tuple[CoxPHFitter, float, list[Dict[str, Any]]]:
    _require_lifelines()
    assert CoxPHFitter is not None

    attempts: list[Dict[str, Any]] = []
    penalties = []
    for value in [base_penalizer, max(base_penalizer, 0.25), max(base_penalizer, 0.5), max(base_penalizer, 1.0)]:
        if value not in penalties:
            penalties.append(float(value))

    last_error: Exception | None = None
    for penalty in penalties:
        model = CoxPHFitter(penalizer=float(penalty))
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

    prediction_frame = training_frame.drop(columns=['duration_days', 'event_observed'])
    partial_hazard_all = model.predict_partial_hazard(prediction_frame).values.ravel()
    survival_times = sorted({1, min(30, horizon_days), min(60, horizon_days), min(90, horizon_days), int(horizon_days)})
    survival_fn = model.predict_survival_function(prediction_frame, times=survival_times).T
    survival_fn.columns = [f'survival_prob_{int(col)}d' for col in survival_fn.columns]
    full_curve = model.predict_survival_function(prediction_frame, times=list(range(1, int(horizon_days) + 1))).T
    median_times = full_curve.apply(lambda row: _median_survival_time(row, int(horizon_days)), axis=1)

    predictions = pd.DataFrame(
        {
            'customer_id': feature_df['customer_id'].astype(int),
            'duration_days': feature_df['duration_days'].astype(int),
            'event_observed': feature_df['event_observed'].astype(int),
            'predicted_hazard_ratio': partial_hazard_all.astype(float),
            'predicted_median_time_to_churn_days': median_times.astype(float),
        }
    )
    predictions = pd.concat([predictions, survival_fn.reset_index(drop=True)], axis=1)
    if 'survival_prob_30d' not in predictions.columns:
        predictions['survival_prob_30d'] = predictions.filter(like='survival_prob_').iloc[:, 0]
    predictions['risk_percentile'] = predictions['predicted_hazard_ratio'].rank(pct=True, method='average')
    predictions['risk_group'] = pd.qcut(
        predictions['predicted_hazard_ratio'].rank(method='first'),
        q=3,
        labels=['Low risk', 'Mid risk', 'High risk'],
    )
    predictions = predictions.merge(
        raw_features[['customer_id'] + [c for c in ['persona', 'region', 'device_type', 'acquisition_channel'] if c in raw_features.columns]],
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
    _plot_risk_groups(predictions, risk_plot_path, int(horizon_days))

    metrics = {
        'model_name': 'CoxPHFitter',
        'model_path': str(model_path),
        'predictions_path': str(predictions_path),
        'coefficients_path': str(coefficients_path),
        'risk_plot_path': str(risk_plot_path),
        'landmark_as_of_date': feature_metadata.get('as_of_date'),
        'horizon_days': int(horizon_days),
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
