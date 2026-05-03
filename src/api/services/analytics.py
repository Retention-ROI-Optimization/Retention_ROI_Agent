from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

import pandas as pd

from src.optimization.policy import build_intensity_action_candidates, normalize, safe_numeric
from src.optimization.timing import load_survival_predictions

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

INTENSITY_ORDER = ['low', 'mid', 'high']


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


def _ensure_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(float(default))
    return pd.Series(float(default), index=df.index, dtype='float64')


def get_top_high_value_customers(customers: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    df = customers.copy()
    if df.empty:
        return df.head(0)

    if 'customer_id' not in df.columns:
        df['customer_id'] = df.index.astype(str)

    df['clv'] = _ensure_numeric_column(df, 'clv', 0.0)
    df['uplift_score'] = _ensure_numeric_column(df, 'uplift_score', 0.0)
    df['churn_probability'] = _ensure_numeric_column(df, 'churn_probability', 0.0)
    df['coupon_cost'] = _ensure_numeric_column(df, 'coupon_cost', 0.0)

    # A few uploaded datasets produce only clv/uplift columns. The dashboard still
    # expects expected_incremental_profit and expected_roi for hover text, tables,
    # and summaries, so derive safe proxy values when they are absent.
    if 'expected_incremental_profit' not in df.columns:
        df['expected_incremental_profit'] = (df['clv'] * df['uplift_score'].clip(lower=0.0) - df['coupon_cost']).fillna(0.0)
    else:
        df['expected_incremental_profit'] = _ensure_numeric_column(df, 'expected_incremental_profit', 0.0)

    if 'expected_roi' not in df.columns:
        denom = df['coupon_cost'].where(df['coupon_cost'] > 0, 1.0)
        df['expected_roi'] = (df['expected_incremental_profit'] / denom).replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
    else:
        df['expected_roi'] = _ensure_numeric_column(df, 'expected_roi', 0.0)

    if 'uplift_segment' not in df.columns:
        df['uplift_segment'] = 'Unknown'
    if 'persona' not in df.columns:
        df['persona'] = 'unknown'

    df['value_score'] = df['clv'] * df['uplift_score']
    ranked = df.sort_values(['value_score', 'clv', 'customer_id'], ascending=[False, False, True])
    if top_n is None or int(top_n) <= 0:
        return ranked
    return ranked.head(int(top_n))


def get_retention_targets(customers: pd.DataFrame, threshold: float, top_n: int | None = None) -> pd.DataFrame:
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
    if 'expected_roi' not in target.columns:
        target['expected_roi'] = 0.0
    ranked = target.sort_values(['priority_score', 'expected_roi', 'customer_id'], ascending=[False, False, True])
    if top_n is None or int(top_n) <= 0:
        return ranked
    return ranked.head(int(top_n))


def _segment_order(customers: pd.DataFrame) -> List[str]:
    present = [str(x) for x in customers.get('uplift_segment', pd.Series(dtype=object)).dropna().unique()]
    ordered = [seg for seg in DEFAULT_SEGMENT_ORDER if seg in present]
    remaining = sorted(seg for seg in present if seg not in ordered)
    if not ordered and not remaining:
        return DEFAULT_SEGMENT_ORDER.copy()
    return ordered + remaining




def _build_candidate_pool(
    customers: pd.DataFrame,
    threshold: float,
    survival_predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if customers.empty:
        return customers.head(0).copy()

    df = customers.copy()
    df["churn_probability"] = safe_numeric(df.get("churn_probability"), default=0.0)
    df["uplift_score"] = safe_numeric(df.get("uplift_score"), default=0.0)
    df["clv"] = safe_numeric(df.get("clv"), default=0.0)
    df["coupon_cost"] = safe_numeric(df.get("coupon_cost"), default=0.0)
    df["expected_incremental_profit"] = safe_numeric(df.get("expected_incremental_profit"), default=0.0)
    df["expected_roi"] = safe_numeric(df.get("expected_roi"), default=0.0)

    candidate = build_intensity_action_candidates(df, survival_predictions=survival_predictions)
    candidate["optimization_score"] = candidate["expected_incremental_profit"] / candidate["coupon_cost"].where(
        candidate["coupon_cost"] > 0,
        1.0,
    )
    candidate = candidate[
        (candidate["churn_probability"] >= float(threshold))
        & (candidate["uplift_score"] > 0.0)
        & (candidate["expected_incremental_profit"] > 0.0)
        & (candidate["coupon_cost"] > 0.0)
    ].copy()

    if candidate.empty:
        return candidate

    candidate["roi_rank_score"] = normalize(candidate["expected_roi"])
    candidate["profit_rank_score"] = normalize(candidate["expected_incremental_profit"])
    candidate["clv_rank_score"] = normalize(safe_numeric(candidate.get("clv"), default=0.0))
    candidate["timing_rank_score"] = normalize(candidate["timing_urgency_score"])
    candidate["window_rank_score"] = 1.0 - normalize(candidate["intervention_window_days"])
    candidate["intensity_fit_rank_score"] = normalize(candidate["intensity_effect_multiplier"])
    candidate["optimization_rank_score"] = normalize(candidate["optimization_score"])

    candidate["priority_score"] = (
        0.18 * candidate["roi_rank_score"]
        + 0.18 * candidate["profit_rank_score"]
        + 0.14 * candidate["churn_probability"]
        + 0.10 * candidate["uplift_score"]
        + 0.10 * candidate["clv_rank_score"]
        + 0.12 * candidate["timing_rank_score"]
        + 0.08 * candidate["window_rank_score"]
        + 0.10 * candidate["intensity_fit_rank_score"]
    )

    candidate["selection_score"] = 0.55 * candidate["priority_score"] + 0.45 * candidate["optimization_rank_score"]

    candidate = candidate.sort_values(
        [
            "selection_score",
            "priority_score",
            "optimization_score",
            "timing_urgency_score",
            "expected_roi",
            "expected_incremental_profit",
            "intervention_window_days",
            "coupon_cost",
            "customer_id",
        ],
        ascending=[False, False, False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    return candidate


def budget_allocation_by_segment(
    selected_customers: pd.DataFrame,
    all_segments: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    all_segments = list(all_segments or DEFAULT_SEGMENT_ORDER)

    if selected_customers.empty:
        return pd.DataFrame(
            [
                {
                    "uplift_segment": segment,
                    "intervention_intensity": intensity,
                    "customer_count": 0,
                    "allocated_budget": 0.0,
                    "expected_profit": 0.0,
                }
                for segment in all_segments
                for intensity in INTENSITY_ORDER
            ]
        )

    grouped = (
        selected_customers.groupby(["uplift_segment", "intervention_intensity"], as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            allocated_budget=("coupon_cost", "sum"),
            expected_profit=("expected_incremental_profit", "sum"),
        )
        .reset_index(drop=True)
    )
    full_index = pd.MultiIndex.from_product([all_segments, INTENSITY_ORDER], names=["uplift_segment", "intervention_intensity"])
    grouped = grouped.set_index(["uplift_segment", "intervention_intensity"]).reindex(full_index, fill_value=0).reset_index()
    grouped["customer_count"] = grouped["customer_count"].astype(int)
    grouped["allocated_budget"] = pd.to_numeric(grouped["allocated_budget"], errors="coerce").fillna(0.0)
    grouped["expected_profit"] = pd.to_numeric(grouped["expected_profit"], errors="coerce").fillna(0.0)
    grouped["intensity_order"] = grouped["intervention_intensity"].map({key: idx for idx, key in enumerate(INTENSITY_ORDER)})
    grouped = grouped.sort_values(["uplift_segment", "intensity_order"]).drop(columns=["intensity_order"]).reset_index(drop=True)
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
    survival_predictions: Optional[pd.DataFrame] = None,
    result_dir: Optional[str | Path] = None,
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
            'survival_enriched': False,
            'dose_response_enriched': False,
            'dose_response_model_version': None,
        }
        return empty, summary, budget_allocation_by_segment(empty, _segment_order(customers))

    all_segments = _segment_order(customers)
    resolved_survival = survival_predictions
    if resolved_survival is None and result_dir is not None:
        resolved_survival = load_survival_predictions(result_dir)
    candidate = _build_candidate_pool(customers, threshold=threshold, survival_predictions=resolved_survival)

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
            'survival_enriched': False,
            'dose_response_enriched': False,
            'dose_response_model_version': None,
        }
        return candidate, summary, budget_allocation_by_segment(candidate, all_segments)

    candidate_segment_counts = (
        candidate['uplift_segment'].value_counts().reindex(all_segments, fill_value=0).astype(int).to_dict()
    )

    selected_rows: list[dict] = []
    used_customers: set[str] = set()
    spent = 0.0
    selection_cap = int(max_customers) if max_customers is not None and int(max_customers) > 0 else None

    intensity_seed_order = ["high", "mid", "low"]
    for intensity in intensity_seed_order:
        if selection_cap is not None and len(selected_rows) >= selection_cap:
            break
        seed_pool = candidate[candidate["intervention_intensity"].astype(str).str.lower() == intensity]
        if seed_pool.empty:
            continue
        for row in seed_pool.itertuples(index=False):
            customer_id = str(getattr(row, "customer_id"))
            cost = float(getattr(row, "coupon_cost", 0.0))
            if customer_id in used_customers or cost <= 0 or spent + cost > float(budget):
                continue
            selected_rows.append(row._asdict())
            used_customers.add(customer_id)
            spent += cost
            break

    for row in candidate.itertuples(index=False):
        if selection_cap is not None and len(selected_rows) >= selection_cap:
            break
        customer_id = str(getattr(row, "customer_id"))
        cost = float(getattr(row, "coupon_cost", 0.0))
        if customer_id in used_customers:
            continue
        if cost <= 0:
            continue
        if spent + cost > float(budget):
            continue
        selected_rows.append(row._asdict())
        used_customers.add(customer_id)
        spent += cost

    if selected_rows:
        selected = pd.DataFrame(selected_rows)
    else:
        selected = candidate.head(0).copy()

    spent = float(selected['coupon_cost'].sum()) if not selected.empty else 0.0
    expected_profit = float(selected['expected_incremental_profit'].sum()) if not selected.empty else 0.0
    overall_roi = float(expected_profit / spent) if spent > 0 else 0.0

    summary = {
        'budget': int(budget),
        'spent': int(round(spent)),
        'remaining': int(round(budget - spent)),
        'num_targeted': int(len(selected)),
        'candidate_customers': int(candidate['customer_id'].nunique()),
        'candidate_actions': int(len(candidate)),
        'expected_incremental_profit': round(expected_profit, 2),
        'overall_roi': round(overall_roi, 6),
        'threshold': float(threshold),
        'max_customers_cap': int(max_customers or len(candidate)),
        'candidate_segment_counts': candidate_segment_counts,
        'survival_enriched': bool(resolved_survival is not None and not resolved_survival.empty),
        'dose_response_enriched': bool(candidate.get('dose_response_enabled', pd.Series(dtype=bool)).fillna(False).any()) if not candidate.empty else False,
        'dose_response_model_version': str(candidate['dose_response_model_version'].iloc[0]) if not candidate.empty and 'dose_response_model_version' in candidate.columns else None,
        'selected_intensity_counts': selected['intervention_intensity'].value_counts().to_dict() if not selected.empty else {},
        'avg_timing_urgency_score': round(float(candidate['timing_urgency_score'].mean()), 6),
        'avg_intervention_window_days': round(float(candidate['intervention_window_days'].mean()), 2),
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
