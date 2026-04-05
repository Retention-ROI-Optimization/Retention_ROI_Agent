from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

DEFAULT_CUSTOMER_COLUMNS = [
    'customer_id', 'persona', 'uplift_segment_true', 'acquisition_month', 'recency_days', 'frequency', 'monetary',
    'churn_probability', 'uplift_score', 'clv', 'coupon_cost', 'expected_incremental_profit',
    'expected_roi', 'uplift_segment', 'treatment_group', 'inactivity_days',
]

DEFAULT_SEGMENT_ORDER = [
    'Persuadables',
    'Sure Things',
    'Sleeping Dogs',
    'Lost Causes',
]


def get_churn_status(customers: pd.DataFrame, threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    df = customers.copy()
    if df.empty:
        summary = {
            'total_customers': 0,
            'at_risk_customers': 0,
            'risk_rate': 0.0,
            'avg_churn_prob': 0.0,
        }
        return summary, df

    df['churn_probability'] = pd.to_numeric(df['churn_probability'], errors='coerce').fillna(0.0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)
    df['is_churn_risk'] = df['churn_probability'] >= float(threshold)

    summary = {
        'total_customers': int(len(df)),
        'at_risk_customers': int(df['is_churn_risk'].sum()),
        'risk_rate': float(df['is_churn_risk'].mean()) if len(df) else 0.0,
        'avg_churn_prob': float(df['churn_probability'].mean()) if len(df) else 0.0,
    }
    risk_customers = df[df['is_churn_risk']].sort_values(['churn_probability', 'clv'], ascending=[False, False])
    return summary, risk_customers


def get_top_high_value_customers(customers: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    df = customers.copy()
    if df.empty:
        return df.head(0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)
    df['uplift_score'] = pd.to_numeric(df.get('uplift_score', 0.0), errors='coerce').fillna(0.0)
    df['value_score'] = df['clv'] * df['uplift_score']
    return df.sort_values(['value_score', 'clv', 'customer_id'], ascending=[False, False, True]).head(top_n)


def get_retention_targets(customers: pd.DataFrame, threshold: float, top_n: int = 30) -> pd.DataFrame:
    df = customers.copy()
    if df.empty:
        return df.head(0)

    df['churn_probability'] = pd.to_numeric(df.get('churn_probability', 0.0), errors='coerce').fillna(0.0)
    df['uplift_score'] = pd.to_numeric(df.get('uplift_score', 0.0), errors='coerce').fillna(0.0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)

    condition = (
        (df['churn_probability'] >= float(threshold))
        & (df['uplift_score'] > 0.08)
        & (df['clv'] > df['clv'].median())
        & (df['uplift_segment'] != 'Sleeping Dogs')
    )
    target = df[condition].copy()
    if target.empty:
        return target

    max_clv = max(float(target['clv'].max()), 1.0)
    target['priority_score'] = (
        0.45 * target['churn_probability']
        + 0.25 * target['uplift_score']
        + 0.30 * (target['clv'] / max_clv)
    )
    return target.sort_values(['priority_score', 'expected_roi', 'customer_id'], ascending=[False, False, True]).head(top_n)


def _segment_order(customers: pd.DataFrame) -> List[str]:
    present = [str(x) for x in customers.get('uplift_segment', pd.Series(dtype=object)).dropna().unique()]
    ordered = [seg for seg in DEFAULT_SEGMENT_ORDER if seg in present]
    remaining = sorted(seg for seg in present if seg not in ordered)
    if not ordered and not remaining:
        return DEFAULT_SEGMENT_ORDER.copy()
    return ordered + remaining


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce').fillna(default)


def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    series = pd.to_numeric(series, errors='coerce').fillna(0.0)
    low = float(series.min())
    high = float(series.max())
    if high - low < 1e-12:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    return (series - low) / (high - low)


def _build_candidate_pool(customers: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if customers.empty:
        return customers.head(0).copy()

    df = customers.copy()
    df['churn_probability'] = _safe_series(df, 'churn_probability')
    df['uplift_score'] = _safe_series(df, 'uplift_score')
    df['clv'] = _safe_series(df, 'clv')
    df['coupon_cost'] = _safe_series(df, 'coupon_cost')
    df['expected_incremental_profit'] = _safe_series(df, 'expected_incremental_profit')
    df['expected_roi'] = _safe_series(df, 'expected_roi')

    candidate = df[
        (df['churn_probability'] >= float(threshold))
        & (df['uplift_score'] > 0.0)
        & (df['expected_incremental_profit'] > 0.0)
        & (df['coupon_cost'] > 0.0)
    ].copy()

    if candidate.empty:
        return candidate

    candidate['roi_rank_score'] = _normalize(candidate['expected_roi'])
    candidate['profit_rank_score'] = _normalize(candidate['expected_incremental_profit'])
    candidate['clv_rank_score'] = _normalize(candidate['clv'])

    candidate['priority_score'] = (
        0.35 * candidate['roi_rank_score']
        + 0.25 * candidate['profit_rank_score']
        + 0.20 * candidate['churn_probability']
        + 0.10 * candidate['uplift_score']
        + 0.10 * candidate['clv_rank_score']
    )

    candidate = candidate.sort_values(
        ['priority_score', 'expected_roi', 'expected_incremental_profit', 'clv', 'customer_id'],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return candidate


def budget_allocation_by_segment(
    selected_customers: pd.DataFrame,
    all_segments: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    all_segments = list(all_segments or DEFAULT_SEGMENT_ORDER)

    if selected_customers.empty:
        return pd.DataFrame(
            {
                'uplift_segment': all_segments,
                'customer_count': [0] * len(all_segments),
                'allocated_budget': [0.0] * len(all_segments),
                'expected_profit': [0.0] * len(all_segments),
            }
        )

    grouped = (
        selected_customers.groupby('uplift_segment', as_index=False)
        .agg(
            customer_count=('customer_id', 'count'),
            allocated_budget=('coupon_cost', 'sum'),
            expected_profit=('expected_incremental_profit', 'sum'),
        )
        .set_index('uplift_segment')
    )

    grouped = grouped.reindex(all_segments, fill_value=0).reset_index()
    return grouped


def allocate_budget(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    selected, summary, _ = get_budget_result(
        customers=customers,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    )
    return selected, summary


def get_budget_result(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    if customers.empty or budget <= 0:
        empty = customers.head(0).copy()
        summary = {
            'budget': int(budget),
            'spent': 0,
            'remaining': int(max(budget, 0)),
            'num_targeted': 0,
            'candidate_customers': 0,
            'expected_incremental_profit': 0.0,
            'overall_roi': 0.0,
            'threshold': float(threshold),
            'max_customers_cap': int(max_customers or 0),
            'candidate_segment_counts': {seg: 0 for seg in _segment_order(customers)},
        }
        return empty, summary, budget_allocation_by_segment(empty, _segment_order(customers))

    all_segments = _segment_order(customers)
    candidate = _build_candidate_pool(customers, threshold=threshold)

    if max_customers is not None and max_customers > 0:
        candidate = candidate.head(int(max_customers)).copy()

    if candidate.empty:
        summary = {
            'budget': int(budget),
            'spent': 0,
            'remaining': int(budget),
            'num_targeted': 0,
            'candidate_customers': 0,
            'expected_incremental_profit': 0.0,
            'overall_roi': 0.0,
            'threshold': float(threshold),
            'max_customers_cap': int(max_customers or 0),
            'candidate_segment_counts': {seg: 0 for seg in all_segments},
        }
        return candidate, summary, budget_allocation_by_segment(candidate, all_segments)

    candidate_segment_counts = (
        candidate['uplift_segment'].value_counts().reindex(all_segments, fill_value=0).astype(int).to_dict()
    )

    cumulative_cost = candidate['coupon_cost'].cumsum()
    selected = candidate[cumulative_cost <= budget].copy()

    spent = float(selected['coupon_cost'].sum()) if not selected.empty else 0.0
    expected_profit = float(selected['expected_incremental_profit'].sum()) if not selected.empty else 0.0
    overall_roi = float(expected_profit / spent) if spent > 0 else 0.0

    summary = {
        'budget': int(budget),
        'spent': int(round(spent)),
        'remaining': int(round(budget - spent)),
        'num_targeted': int(len(selected)),
        'candidate_customers': int(len(candidate)),
        'expected_incremental_profit': round(expected_profit, 2),
        'overall_roi': round(overall_roi, 6),
        'threshold': float(threshold),
        'max_customers_cap': int(max_customers or len(candidate)),
        'candidate_segment_counts': candidate_segment_counts,
    }
    segment_allocation = budget_allocation_by_segment(selected, all_segments=all_segments)
    return selected, summary, segment_allocation


def distribution_table(df: pd.DataFrame, column: str, limit: int | None = None) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=['name', 'count', 'share'])
    counts = df[column].fillna('Unknown').astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    total = max(int(len(df)), 1)
    return pd.DataFrame(
        {
            'name': counts.index.tolist(),
            'count': counts.astype(int).tolist(),
            'share': (counts / total).astype(float).tolist(),
        }
    )
