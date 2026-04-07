from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd

from src.api.dependencies import get_repository
from src.api.schemas import (
    BudgetResponse,
    ChurnResponse,
    CohortRetentionResponse,
    CustomerListResponse,
    DashboardSummaryResponse,
    PaginationMeta,
)
from src.api.services.analytics import (
    budget_allocation_by_segment,
    distribution_table,
    get_budget_result,
    get_churn_status,
    get_retention_targets,
    get_top_high_value_customers,
)
from src.api.services.repository import DataRepository
from src.api.services.serialization import dataframe_to_records

router = APIRouter(prefix='/analytics', tags=['analytics'])

ALLOWED_SORT_COLUMNS = {
    'customer_id', 'churn_probability', 'uplift_score', 'clv', 'expected_roi',
    'expected_incremental_profit', 'recency_days', 'frequency', 'monetary',
}


_TRUE_VALUES = {'true', '1', 'yes', 'y', 't'}
_FALSE_VALUES = {'false', '0', 'no', 'n', 'f'}


def _coerce_bool_series(series: pd.Series, default: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)

    def _parse(value):
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        text = str(value).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        return default

    return series.map(_parse).astype(bool)


def _filter_cohort_retention(
    cohort: pd.DataFrame,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> pd.DataFrame:
    df = cohort.copy()
    if df.empty:
        return df

    if 'activity_definition' in df.columns and activity_definition:
        candidate = df[df['activity_definition'].astype(str) == activity_definition].copy()
        if not candidate.empty:
            df = candidate
    elif 'activity_definition' in df.columns:
        for preferred in ['core_engagement', 'all_activity']:
            candidate = df[df['activity_definition'].astype(str) == preferred].copy()
            if not candidate.empty:
                df = candidate
                break

    if 'retention_mode' in df.columns and retention_mode:
        candidate = df[df['retention_mode'].astype(str) == retention_mode].copy()
        if not candidate.empty:
            df = candidate
    elif 'retention_mode' in df.columns:
        for preferred in ['rolling', 'point']:
            candidate = df[df['retention_mode'].astype(str) == preferred].copy()
            if not candidate.empty:
                df = candidate
                break

    if 'observed' in df.columns:
        df['observed'] = _coerce_bool_series(df['observed'], default=True)

    return df.reset_index(drop=True)


def _load_customer_summary(repository: DataRepository) -> pd.DataFrame:
    try:
        return repository.require_customer_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _load_cohort_retention(repository: DataRepository) -> pd.DataFrame:
    try:
        return repository.require_cohort_retention()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/summary', response_model=DashboardSummaryResponse)
def dashboard_summary(
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    budget: int = Query(default=5000000, ge=1),
    max_customers: Optional[int] = Query(default=None, ge=1, le=5000),
    repository: DataRepository = Depends(get_repository),
) -> DashboardSummaryResponse:
    customers = _load_customer_summary(repository)
    cohort = _load_cohort_retention(repository)
    churn_summary, _ = get_churn_status(customers, threshold=threshold)
    _, budget_summary, _ = get_budget_result(customers, budget=budget, threshold=threshold, max_customers=max_customers)
    persona_distribution = dataframe_to_records(distribution_table(customers, 'persona'))
    uplift_segment_distribution = dataframe_to_records(distribution_table(customers, 'uplift_segment'))
    return DashboardSummaryResponse(
        customer_count=int(len(customers)),
        cohort_row_count=int(len(cohort)),
        threshold=float(threshold),
        budget=int(budget),
        churn_summary=churn_summary,
        budget_summary=budget_summary,
        persona_distribution=persona_distribution,
        uplift_segment_distribution=uplift_segment_distribution,
    )


@router.get('/customers', response_model=CustomerListResponse)
def list_customers(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    persona: Optional[str] = Query(default=None),
    uplift_segment: Optional[str] = Query(default=None),
    treatment_group: Optional[str] = Query(default=None),
    min_churn_probability: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    sort_by: str = Query(default='customer_id'),
    sort_order: str = Query(default='asc', pattern='^(asc|desc)$'),
    repository: DataRepository = Depends(get_repository),
) -> CustomerListResponse:
    customers = _load_customer_summary(repository)
    if persona:
        customers = customers[customers['persona'] == persona]
    if uplift_segment:
        customers = customers[customers['uplift_segment'] == uplift_segment]
    if treatment_group:
        customers = customers[customers['treatment_group'] == treatment_group]
    if min_churn_probability is not None:
        customers = customers[customers['churn_probability'] >= min_churn_probability]
    sort_by = sort_by if sort_by in customers.columns and sort_by in ALLOWED_SORT_COLUMNS else 'customer_id'
    ascending = sort_order == 'asc'
    customers = customers.sort_values(sort_by, ascending=ascending)
    paged = customers.iloc[offset: offset + limit]
    return CustomerListResponse(
        meta=PaginationMeta(total=int(len(customers)), limit=limit, offset=offset, returned=int(len(paged))),
        records=dataframe_to_records(paged),
    )


@router.get('/churn', response_model=ChurnResponse)
def churn_view(
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=500),
    repository: DataRepository = Depends(get_repository),
) -> ChurnResponse:
    customers = _load_customer_summary(repository)
    summary, top_at_risk = get_churn_status(customers, threshold=threshold)
    return ChurnResponse(threshold=float(threshold), summary=summary, top_at_risk=dataframe_to_records(top_at_risk.head(limit)))


@router.get('/cohort-retention', response_model=CohortRetentionResponse)
def cohort_retention(
    activity_definition: Optional[str] = Query(default=None),
    retention_mode: Optional[str] = Query(default=None),
    repository: DataRepository = Depends(get_repository),
) -> CohortRetentionResponse:
    cohort = _filter_cohort_retention(
        _load_cohort_retention(repository),
        activity_definition=activity_definition,
        retention_mode=retention_mode,
    )
    keep_cols = [
        'cohort_month',
        'period',
        'cohort_size',
        'retained_customers',
        'retention_rate',
        'observed',
    ]
    records = dataframe_to_records(cohort, columns=keep_cols)
    observed = cohort[cohort['observed']] if not cohort.empty and 'observed' in cohort.columns else cohort
    periods = int(observed['period'].max()) + 1 if not observed.empty else 0
    return CohortRetentionResponse(periods=periods, records=records)


@router.get('/uplift/top', response_model=CustomerListResponse)
def uplift_top_customers(
    limit: int = Query(default=20, ge=1, le=200),
    repository: DataRepository = Depends(get_repository),
) -> CustomerListResponse:
    customers = _load_customer_summary(repository)
    ranked = get_top_high_value_customers(customers, top_n=limit)
    return CustomerListResponse(
        meta=PaginationMeta(total=int(len(ranked)), limit=limit, offset=0, returned=int(len(ranked))),
        records=dataframe_to_records(ranked),
    )


@router.get('/retention-targets', response_model=CustomerListResponse)
def retention_targets(
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    limit: int = Query(default=30, ge=1, le=300),
    repository: DataRepository = Depends(get_repository),
) -> CustomerListResponse:
    customers = _load_customer_summary(repository)
    target = get_retention_targets(customers, threshold=threshold, top_n=limit)
    return CustomerListResponse(
        meta=PaginationMeta(total=int(len(target)), limit=limit, offset=0, returned=int(len(target))),
        records=dataframe_to_records(target),
    )


@router.get('/optimization/budget', response_model=BudgetResponse)
def budget_optimization(
    budget: int = Query(default=5000000, ge=1),
    threshold: float = Query(default=0.50, ge=0.0, le=1.0),
    max_customers: Optional[int] = Query(default=None, ge=1, le=5000),
    repository: DataRepository = Depends(get_repository),
) -> BudgetResponse:
    customers = _load_customer_summary(repository)
    selected, summary, segment_allocation = get_budget_result(
        customers,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    )
    return BudgetResponse(
        budget=int(budget),
        summary=summary,
        selected_customers=dataframe_to_records(selected),
        segment_allocation=dataframe_to_records(segment_allocation),
    )


@router.get('/segments/persona', response_model=CustomerListResponse)
def persona_segments(repository: DataRepository = Depends(get_repository)) -> CustomerListResponse:
    customers = _load_customer_summary(repository)
    dist = distribution_table(customers, 'persona')
    return CustomerListResponse(
        meta=PaginationMeta(total=int(len(dist)), limit=int(len(dist)), offset=0, returned=int(len(dist))),
        records=dataframe_to_records(dist),
    )


@router.get('/segments/uplift', response_model=CustomerListResponse)
def uplift_segments(repository: DataRepository = Depends(get_repository)) -> CustomerListResponse:
    customers = _load_customer_summary(repository)
    dist = distribution_table(customers, 'uplift_segment')
    return CustomerListResponse(
        meta=PaginationMeta(total=int(len(dist)), limit=int(len(dist)), offset=0, returned=int(len(dist))),
        records=dataframe_to_records(dist),
    )
