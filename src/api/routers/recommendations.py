from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_settings
from src.api.schemas import RecommendationResponse
from src.api.settings import ApiSettings
from src.workflows.pipeline_runner import run_recommendation_pipeline

router = APIRouter(prefix='/recommendations', tags=['recommendations'])


MAX_RECOMMENDATION_CUSTOMERS = 100000


def _should_rebuild(
    summary: dict,
    *,
    limit: int,
    per_customer: int,
    budget: int,
    threshold: float,
    max_customers: int,
) -> bool:
    if not summary:
        return True

    budget_context = summary.get('budget_context', {}) or {}

    return any([
        int(summary.get('per_customer', 0)) != int(per_customer),
        int(summary.get('candidate_limit', 0)) != int(limit),
        str(summary.get('target_source', '')) not in {'optimized_targets', 'current_budget_threshold_targets'},
        int(budget_context.get('budget', -1)) != int(budget),
        float(budget_context.get('max_customers_cap', -1)) != float(max_customers),
        abs(float(budget_context.get('threshold', threshold)) - float(threshold)) > 1e-12,
    ])


@router.get('/personalized', response_model=RecommendationResponse)
def personalized_recommendations(
    limit: int = Query(default=20, ge=1, le=MAX_RECOMMENDATION_CUSTOMERS),
    per_customer: int = Query(default=3, ge=1, le=5),
    budget: int = Query(default=5000000, ge=1),
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    max_customers: int = Query(default=1000, ge=1, le=MAX_RECOMMENDATION_CUSTOMERS),
    rebuild: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
) -> RecommendationResponse:
    requested_limit = min(int(limit), int(max_customers), MAX_RECOMMENDATION_CUSTOMERS)

    result_path = settings.resolved_result_dir / 'personalized_recommendations.csv'
    summary_path = settings.resolved_result_dir / 'personalized_recommendation_summary.json'

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            summary = {}

    if rebuild or (not result_path.exists()) or _should_rebuild(
        summary,
        limit=requested_limit,
        per_customer=per_customer,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    ):
        pipeline_result = run_recommendation_pipeline(
            data_dir=settings.resolved_data_dir,
            result_dir=settings.resolved_result_dir,
            budget=budget,
            threshold=threshold,
            max_customers=max_customers,
            per_customer=per_customer,
            candidate_limit=requested_limit,
        )
        summary = dict(pipeline_result.get('metadata', {}) or {})
        budget_context = dict(summary.get('budget_context', {}) or {})
        budget_context['budget'] = int(budget)
        budget_context['threshold'] = float(threshold)
        budget_context['max_customers_cap'] = int(max_customers)
        summary['budget_context'] = budget_context
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    if not result_path.exists():
        raise HTTPException(status_code=404, detail='personalized_recommendations.csv not found')

    df = pd.read_csv(result_path)
    if df.empty:
        return RecommendationResponse(rows=0, summary=summary, records=[])

    order_column = 'target_priority_score' if 'target_priority_score' in df.columns else 'recommendation_priority'
    if order_column not in df.columns:
        order_column = 'recommendation_score' if 'recommendation_score' in df.columns else None

    if order_column and 'customer_id' in df.columns:
        customer_order = (
            df.groupby('customer_id')[order_column]
            .max()
            .sort_values(ascending=False)
            .head(requested_limit)
            .index.tolist()
        )
        df = df[df['customer_id'].isin(customer_order)].copy()
        df['customer_sort'] = df['customer_id'].map({cid: idx for idx, cid in enumerate(customer_order)})
        sort_cols = ['customer_sort']
        if 'recommendation_rank' in df.columns:
            sort_cols.append('recommendation_rank')
        df = df.sort_values(sort_cols).drop(columns=['customer_sort'])
    else:
        df = df.head(requested_limit * int(per_customer)).copy()

    return RecommendationResponse(rows=int(len(df)), summary=summary, records=df.to_dict(orient='records'))
