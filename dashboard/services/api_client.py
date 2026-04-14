from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import requests
from requests.utils import urlparse

DEFAULT_TIMEOUT = 120
DEFAULT_API_BASE_URL = os.getenv('RETENTION_API_BASE_URL', 'http://localhost:8000').rstrip('/')


class DashboardApiError(RuntimeError):
    pass


def get_api_base_url() -> str:
    return os.getenv('RETENTION_API_BASE_URL', DEFAULT_API_BASE_URL).rstrip('/')


def _candidate_api_base_urls() -> list[str]:
    configured = get_api_base_url()
    candidates: list[str] = []

    def _append(url: str | None) -> None:
        if not url:
            return
        normalized = str(url).rstrip('/')
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _append(configured)
    parsed = urlparse(configured)
    host = (parsed.hostname or '').strip().lower()
    scheme = parsed.scheme or 'http'
    port = parsed.port or 8000

    if host == 'api':
        _append(f'{scheme}://localhost:{port}')
        _append(f'{scheme}://127.0.0.1:{port}')
        _append(f'{scheme}://host.docker.internal:{port}')
    elif host in {'localhost', '127.0.0.1'}:
        _append(f'{scheme}://api:{port}')

    return candidates


def _request_json(path: str, params: Dict[str, Any] | None = None, *, method: str = 'GET') -> Dict[str, Any]:
    last_exc: Exception | None = None
    attempted: list[str] = []
    for base_url in _candidate_api_base_urls():
        url = f"{base_url}{path}"
        attempted.append(url)
        try:
            response = requests.request(method.upper(), url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            continue
    attempted_text = ', '.join(attempted)
    raise DashboardApiError(f'API 요청 실패: {last_exc}. attempted={attempted_text}') from last_exc


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
    safe_limit = max(int(limit), 1)
    safe_per_customer = min(max(int(per_customer), 1), 5)
    safe_budget = max(int(budget), 1)
    safe_threshold = min(max(float(threshold), 0.0), 1.0)
    safe_max_customers = max(int(max_customers), 1)
    data = _request_json(
        '/api/v1/recommendations/personalized',
        {
            'limit': safe_limit,
            'per_customer': safe_per_customer,
            'budget': safe_budget,
            'threshold': safe_threshold,
            'max_customers': safe_max_customers,
            'rebuild': str(bool(rebuild)).lower(),
        },
    )
    return data['summary'], pd.DataFrame(data['records'])


def fetch_training_artifacts() -> Dict[str, Any]:
    return _request_json('/api/v1/artifacts/training')


def fetch_saved_results_artifacts(
    budget: int,
    threshold: float = 0.50,
    max_customers: int | None = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        'budget': budget,
        'threshold': threshold,
        'rebuild': str(bool(rebuild)).lower(),
    }
    if max_customers is not None:
        params['max_customers'] = max_customers
    return _request_json('/api/v1/artifacts/saved-results', params)


def fetch_realtime_scores(limit: int = 50) -> tuple[Dict[str, Any], pd.DataFrame]:
    data = _request_json('/api/v1/realtime/scores', {'top_n': limit})
    return data.get('summary', {}), pd.DataFrame(data.get('records', []))


def fetch_survival_summary(limit: int = 50) -> tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    data = _request_json('/api/v1/survival/summary', {'top_n': limit})
    return (
        data.get('metrics', {}),
        pd.DataFrame(data.get('predictions', [])),
        pd.DataFrame(data.get('coefficients', [])),
        data.get('image_paths', {}),
    )


def advance_realtime_stream(batch_size: int = 250, top_n: int = 50, reset_when_exhausted: bool = True) -> Dict[str, Any]:
    return _request_json(
        '/api/v1/realtime/tick',
        {
            'batch_size': int(batch_size),
            'top_n': int(top_n),
            'reset_when_exhausted': str(bool(reset_when_exhausted)).lower(),
        },
        method='POST',
    )
