from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RecommendationArtifacts:
    recommendations_path: str
    summary_path: str


TARGET_META_COLUMNS = [
    'customer_id',
    'priority_score',
    'expected_incremental_profit',
    'expected_roi',
    'coupon_cost',
    'uplift_segment',
    'persona',
    'predicted_median_time_to_churn_days',
    'timing_urgency_score',
    'intervention_window_days',
    'recommended_intervention_window',
    'timing_priority_bucket',
    'short_term_churn_probability',
    'intervention_intensity',
    'intervention_intensity_label',
    'recommended_action',
    'selection_score',
]



def _load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer_summary = pd.read_csv(data_dir / 'customer_summary.csv')
    orders = pd.read_csv(data_dir / 'orders.csv', parse_dates=['order_time'])
    events = pd.read_csv(data_dir / 'events.csv', parse_dates=['timestamp'])
    return customer_summary, orders, events


def _weighted_category_preferences(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=['customer_id', 'item_category', 'customer_pref_score'])
    max_time = orders['order_time'].max()
    tmp = orders.copy()
    tmp['days_ago'] = (max_time - tmp['order_time']).dt.days.clip(lower=0)
    tmp['recency_weight'] = np.exp(-tmp['days_ago'] / 90.0)
    tmp['customer_pref_score'] = (
        tmp['net_amount'].fillna(0.0) * tmp['recency_weight']
        + tmp['quantity'].fillna(0.0) * 5000.0
    )
    return tmp.groupby(['customer_id', 'item_category'], as_index=False)['customer_pref_score'].sum()


def _segment_popularity(customer_summary: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    merged = orders.merge(
        customer_summary[['customer_id', 'persona', 'uplift_segment']],
        on='customer_id',
        how='left',
    )
    if merged.empty:
        return pd.DataFrame(columns=['persona', 'uplift_segment', 'item_category', 'segment_popularity'])
    seg = merged.groupby(['persona', 'uplift_segment', 'item_category'], as_index=False).agg(
        segment_popularity=('net_amount', 'sum'),
        segment_orders=('order_id', 'count'),
    )
    seg['segment_popularity'] = seg['segment_popularity'] + seg['segment_orders'] * 3000.0
    return seg[['persona', 'uplift_segment', 'item_category', 'segment_popularity']]


def _global_popularity(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=['item_category', 'global_popularity'])
    out = orders.groupby('item_category', as_index=False).agg(
        global_popularity=('net_amount', 'sum'),
        order_count=('order_id', 'count'),
    )
    out['global_popularity'] = out['global_popularity'] + out['order_count'] * 2000.0
    return out[['item_category', 'global_popularity']]


def _build_candidate_customers(customer_summary: pd.DataFrame) -> pd.DataFrame:
    df = customer_summary.copy()
    df['churn_probability'] = pd.to_numeric(df.get('churn_probability', 0.0), errors='coerce').fillna(0.0)
    df['uplift_score'] = pd.to_numeric(df.get('uplift_score', 0.0), errors='coerce').fillna(0.0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)
    df['coupon_affinity'] = pd.to_numeric(df.get('coupon_affinity', 0.0), errors='coerce').fillna(0.0)
    df['recommendation_priority'] = (
        0.45 * df['churn_probability']
        + 0.25 * df['uplift_score']
        + 0.30 * (df['clv'] / max(float(df['clv'].max()), 1.0))
    )
    df['target_priority_score'] = df['recommendation_priority']
    return df[df['churn_probability'] >= 0.45].sort_values(
        ['target_priority_score', 'clv'],
        ascending=[False, False],
    )


def _prepare_target_customers(
    customer_summary: pd.DataFrame,
    target_customers: Optional[pd.DataFrame],
    candidate_limit: int,
) -> tuple[pd.DataFrame, str]:
    if target_customers is None or target_customers.empty:
        return _build_candidate_customers(customer_summary).head(candidate_limit).copy(), 'risk_candidates'

    meta_cols = [col for col in TARGET_META_COLUMNS if col in target_customers.columns]
    target_meta = target_customers[meta_cols].copy()
    target_meta['customer_id'] = pd.to_numeric(target_meta['customer_id'], errors='coerce')
    target_meta = target_meta.dropna(subset=['customer_id']).copy()
    target_meta['customer_id'] = target_meta['customer_id'].astype(int)

    base = customer_summary.copy()
    base['customer_id'] = pd.to_numeric(base['customer_id'], errors='coerce')
    base = base.dropna(subset=['customer_id']).copy()
    base['customer_id'] = base['customer_id'].astype(int)

    candidates = base.merge(target_meta, on='customer_id', how='inner', suffixes=('', '_target'))
    if candidates.empty:
        return candidates, 'optimized_targets'

    for col in ['churn_probability', 'uplift_score', 'clv', 'coupon_affinity']:
        candidates[col] = pd.to_numeric(candidates.get(col, 0.0), errors='coerce').fillna(0.0)

    if 'persona_target' in candidates.columns:
        candidates['persona'] = candidates['persona_target'].fillna(candidates.get('persona'))
    if 'uplift_segment_target' in candidates.columns:
        candidates['uplift_segment'] = candidates['uplift_segment_target'].fillna(candidates.get('uplift_segment'))

    candidates['priority_score'] = pd.to_numeric(candidates.get('priority_score', 0.0), errors='coerce').fillna(0.0)
    candidates['expected_incremental_profit'] = pd.to_numeric(
        candidates.get('expected_incremental_profit', 0.0),
        errors='coerce',
    ).fillna(0.0)
    candidates['expected_roi'] = pd.to_numeric(candidates.get('expected_roi', 0.0), errors='coerce').fillna(0.0)
    candidates['coupon_cost'] = pd.to_numeric(candidates.get('coupon_cost', 0.0), errors='coerce').fillna(0.0)

    candidates['predicted_median_time_to_churn_days'] = pd.to_numeric(candidates.get('predicted_median_time_to_churn_days', 90.0), errors='coerce').fillna(90.0)
    candidates['timing_urgency_score'] = pd.to_numeric(candidates.get('timing_urgency_score', 0.0), errors='coerce').fillna(0.0)
    candidates['intervention_window_days'] = pd.to_numeric(candidates.get('intervention_window_days', candidates['predicted_median_time_to_churn_days']), errors='coerce').fillna(candidates['predicted_median_time_to_churn_days'])
    candidates['short_term_churn_probability'] = pd.to_numeric(candidates.get('short_term_churn_probability', 0.0), errors='coerce').fillna(0.0)
    max_clv = max(float(candidates['clv'].max()), 1.0)
    candidates['recommendation_priority'] = (
        0.32 * candidates['priority_score']
        + 0.18 * candidates['churn_probability']
        + 0.12 * candidates['uplift_score']
        + 0.14 * (candidates['clv'] / max_clv)
        + 0.10 * candidates['expected_roi'].clip(lower=0.0)
        + 0.14 * candidates['timing_urgency_score']
    )
    candidates['target_priority_score'] = candidates['priority_score']

    candidates = candidates.sort_values(
        [
            'target_priority_score',
            'expected_incremental_profit',
            'expected_roi',
            'recommendation_priority',
            'intervention_window_days',
            'clv',
            'customer_id',
        ],
        ascending=[False, False, False, False, True, False, True],
    ).head(candidate_limit)
    return candidates, 'optimized_targets'


def _recent_interest_scores(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['customer_id', 'item_category', 'view_score'])

    recent_view = events.dropna(subset=['item_category']).copy()
    if 'event_type' in recent_view.columns:
        recent_view['event_type'] = recent_view['event_type'].astype(str).str.lower()
        recent_view = recent_view[
            recent_view['event_type'].isin({'view', 'browse', 'page_view', 'search', 'product_view', 'add_to_cart'})
        ].copy()

    if recent_view.empty:
        return pd.DataFrame(columns=['customer_id', 'item_category', 'view_score'])

    max_ts = recent_view['timestamp'].max()
    recent_view['days_ago'] = (max_ts - recent_view['timestamp']).dt.days.clip(lower=0)
    recent_view['view_score'] = np.exp(-recent_view['days_ago'] / 60.0)
    return recent_view.groupby(['customer_id', 'item_category'], as_index=False)['view_score'].sum()


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() - series.min() < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def run_personalized_recommendation_pipeline(
    data_dir: Path,
    result_dir: Path,
    per_customer: int = 3,
    candidate_limit: int = 100,
    target_customers: Optional[pd.DataFrame] = None,
    target_source: Optional[str] = None,
) -> RecommendationArtifacts:
    result_dir.mkdir(parents=True, exist_ok=True)
    customer_summary, orders, events = _load_inputs(data_dir)
    candidates, resolved_target_source = _prepare_target_customers(
        customer_summary=customer_summary,
        target_customers=target_customers,
        candidate_limit=candidate_limit,
    )
    if target_source:
        resolved_target_source = target_source

    pref = _weighted_category_preferences(orders)
    seg = _segment_popularity(customer_summary, orders)
    glob = _global_popularity(orders)
    view_pref = _recent_interest_scores(events)

    all_categories = sorted(
        set(glob['item_category'].dropna().astype(str).tolist())
        | set(orders['item_category'].dropna().astype(str).tolist())
        | set(view_pref['item_category'].dropna().astype(str).tolist())
    )
    rows: List[Dict] = []

    if not candidates.empty and all_categories:
        for _, customer in candidates.iterrows():
            base = pd.DataFrame({'item_category': all_categories})
            customer_id = int(customer['customer_id'])
            merged = base.merge(
                pref[pref['customer_id'] == customer_id],
                on='item_category',
                how='left',
            )
            merged = merged.merge(
                view_pref[view_pref['customer_id'] == customer_id],
                on='item_category',
                how='left',
            )
            merged = merged.merge(
                seg[(seg['persona'] == customer['persona']) & (seg['uplift_segment'] == customer['uplift_segment'])],
                on='item_category',
                how='left',
            )
            merged = merged.merge(glob, on='item_category', how='left')
            score_cols = ['customer_pref_score', 'view_score', 'segment_popularity', 'global_popularity']
            merged[score_cols] = merged[score_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

            merged['score'] = (
                0.50 * _normalize(merged['customer_pref_score'])
                + 0.15 * _normalize(merged['view_score'])
                + 0.20 * _normalize(merged['segment_popularity'])
                + 0.15 * _normalize(merged['global_popularity'])
                + 0.05 * float(customer.get('coupon_affinity', 0.0))
            )
            merged = merged.sort_values(['score', 'item_category'], ascending=[False, True]).head(per_customer)
            for rank, (_, rec) in enumerate(merged.iterrows(), start=1):
                reason_bits = []
                if rec['customer_pref_score'] > 0:
                    reason_bits.append('own_purchase_history')
                if rec['view_score'] > 0:
                    reason_bits.append('recent_browse_signal')
                if rec['segment_popularity'] > 0:
                    reason_bits.append('segment_popularity')
                if not reason_bits:
                    reason_bits.append('global_popularity')
                rows.append(
                    {
                        'customer_id': customer_id,
                        'persona': customer.get('persona'),
                        'uplift_segment': customer.get('uplift_segment'),
                        'churn_probability': float(customer.get('churn_probability', 0.0)),
                        'uplift_score': float(customer.get('uplift_score', 0.0)),
                        'clv': float(customer.get('clv', 0.0)),
                        'recommendation_priority': float(customer.get('recommendation_priority', 0.0)),
                        'target_priority_score': float(customer.get('target_priority_score', 0.0)),
                        'expected_incremental_profit': float(customer.get('expected_incremental_profit', 0.0)),
                        'expected_roi': float(customer.get('expected_roi', 0.0)),
                        'coupon_cost': float(customer.get('coupon_cost', 0.0)),
                        'predicted_median_time_to_churn_days': float(customer.get('predicted_median_time_to_churn_days', 0.0)),
                        'timing_urgency_score': float(customer.get('timing_urgency_score', 0.0)),
                        'intervention_window_days': float(customer.get('intervention_window_days', 0.0)),
                        'recommended_intervention_window': customer.get('recommended_intervention_window'),
                        'timing_priority_bucket': customer.get('timing_priority_bucket'),
                        'short_term_churn_probability': float(customer.get('short_term_churn_probability', 0.0)),
                        'recommendation_rank': rank,
                        'recommended_category': rec['item_category'],
                        'recommendation_score': round(float(rec['score']), 6),
                        'reason_tags': ', '.join(reason_bits),
                    }
                )

    rec_df = pd.DataFrame(rows)
    summary = {
        'rows': int(len(rec_df)),
        'customers_covered': int(rec_df['customer_id'].nunique()) if not rec_df.empty else 0,
        'per_customer': int(per_customer),
        'candidate_limit': int(candidate_limit),
        'target_source': resolved_target_source,
        'top_categories': rec_df['recommended_category'].value_counts().head(10).to_dict() if not rec_df.empty else {},
    }
    recommendations_path = result_dir / 'personalized_recommendations.csv'
    summary_path = result_dir / 'personalized_recommendation_summary.json'
    rec_df.to_csv(recommendations_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return RecommendationArtifacts(str(recommendations_path), str(summary_path))
