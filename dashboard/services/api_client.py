from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import requests

DEFAULT_TIMEOUT = 120
DEFAULT_API_BASE_URL = os.getenv('RETENTION_API_BASE_URL', 'http://localhost:8000').rstrip('/')


class DashboardApiError(RuntimeError):
    pass


def get_api_base_url() -> str:
    return os.getenv('RETENTION_API_BASE_URL', DEFAULT_API_BASE_URL).rstrip('/')


def _request_json(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{get_api_base_url()}{path}"
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DashboardApiError(f'API 요청 실패: {exc}') from exc
    return response.json()


def fetch_health() -> Dict[str, Any]:
    return _request_json('/health')


def fetch_dashboard_summary(threshold: float, budget: int) -> Dict[str, Any]:
    return _request_json('/api/v1/analytics/summary', {'threshold': threshold, 'budget': budget})


def fetch_churn_view(threshold: float, limit: int) -> tuple[Dict[str, Any], pd.DataFrame]:
    data = _request_json('/api/v1/analytics/churn', {'threshold': threshold, 'limit': limit})
    return data['summary'], pd.DataFrame(data['top_at_risk'])


def fetch_cohort_retention(
    activity_definition: str | None = None,
    retention_mode: str | None = None,
) -> pd.DataFrame:
    params: Dict[str, Any] = {}
    if activity_definition:
        params['activity_definition'] = activity_definition
    if retention_mode:
        params['retention_mode'] = retention_mode
    data = _request_json('/api/v1/analytics/cohort-retention', params or None)
    return pd.DataFrame(data['records'])


def fetch_uplift_top(limit: int) -> pd.DataFrame:
    data = _request_json('/api/v1/analytics/uplift/top', {'limit': limit})
    return pd.DataFrame(data['records'])


def fetch_budget_optimization(
    budget: int,
    threshold: float = 0.50,
    max_customers: int | None = None,
) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    params: Dict[str, Any] = {'budget': budget, 'threshold': threshold}
    if max_customers is not None:
        params['max_customers'] = max_customers
    data = _request_json('/api/v1/analytics/optimization/budget', params)
    return data['summary'], pd.DataFrame(data['selected_customers']), pd.DataFrame(data['segment_allocation'])


def fetch_retention_targets(threshold: float, limit: int) -> pd.DataFrame:
    data = _request_json('/api/v1/analytics/retention-targets', {'threshold': threshold, 'limit': limit})
    return pd.DataFrame(data['records'])


def fetch_personalized_recommendations(
    limit: int,
    per_customer: int,
    budget: int,
    threshold: float,
    max_customers: int,
    rebuild: bool = True,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    data = _request_json(
        '/api/v1/recommendations/personalized',
        {
            'limit': limit,
            'per_customer': per_customer,
            'budget': budget,
            'threshold': threshold,
            'max_customers': max_customers,
            'rebuild': str(bool(rebuild)).lower(),
        },
    )
    return data['summary'], pd.DataFrame(data['records'])


def fetch_training_artifacts(rebuild: bool = False) -> Dict[str, Any]:
    return _request_json('/api/v1/artifacts/training', {'rebuild': str(bool(rebuild)).lower()})


def fetch_saved_results_artifacts(budget: int, rebuild: bool = False) -> Dict[str, Any]:
    return _request_json(
        '/api/v1/artifacts/saved-results',
        {'budget': budget, 'rebuild': str(bool(rebuild)).lower()},
    )
