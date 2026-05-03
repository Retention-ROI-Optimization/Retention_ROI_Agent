from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.autoops_universal.dashboard_contract import ensure_dashboard_contract
from dashboard.data.mock_data import generate_mock_cohort_retention, generate_mock_customers

RAW_CUSTOMER_SUMMARY = "customer_summary.csv"
RAW_COHORT_RETENTION = "cohort_retention.csv"
OPTIONAL_RAW_FILES = {
    "customers": {"filename": "customers.csv", "usecols": ["customer_id", "persona", "region", "device_type", "acquisition_channel", "price_sensitivity", "coupon_affinity", "support_contact_propensity"]},
    "events": {"filename": "events.csv", "usecols": ["customer_id", "timestamp", "event_type", "item_category", "quantity"]},
    "orders": {"filename": "orders.csv", "usecols": ["customer_id", "order_time", "item_category", "net_amount", "coupon_used"]},
    "campaign_exposures": {"filename": "campaign_exposures.csv", "usecols": ["customer_id", "exposure_time", "campaign_type", "coupon_cost"]},
    "state_snapshots": {"filename": "state_snapshots.csv", "usecols": ["customer_id", "snapshot_date", "current_status", "inactivity_days", "coupon_fatigue_score", "discount_dependency_score"]},
    "treatment_assignments": {"filename": "treatment_assignments.csv", "usecols": ["customer_id", "treatment_group", "campaign_type", "coupon_cost", "assigned_at"]},
}


@dataclass(frozen=True)
class DashboardDataBundle:
    customer_summary: pd.DataFrame
    cohort_retention: pd.DataFrame
    customers: pd.DataFrame
    events: pd.DataFrame
    orders: pd.DataFrame
    campaign_exposures: pd.DataFrame
    state_snapshots: pd.DataFrame
    treatment_assignments: pd.DataFrame
    source_dir: Optional[str]
    used_mock: bool


def _parse_date_columns(path: Path) -> List[str]:
    mapping = {
        RAW_CUSTOMER_SUMMARY: ["signup_date", "assigned_at"],
        "customers.csv": ["signup_date"],
        "events.csv": ["timestamp"],
        "orders.csv": ["order_time"],
        "campaign_exposures.csv": ["exposure_time"],
        "state_snapshots.csv": ["snapshot_date", "last_visit_date", "last_purchase_date"],
        "treatment_assignments.csv": ["assigned_at"],
    }
    return mapping.get(path.name, [])


def _read_csv(path: Path, usecols: List[str] | None = None) -> pd.DataFrame:
    parse_dates = _parse_date_columns(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    existing_usecols = [col for col in (usecols or header) if col in header]
    existing_parse_dates = [col for col in parse_dates if col in header and col in existing_usecols]
    return pd.read_csv(path, parse_dates=existing_parse_dates or None, usecols=existing_usecols or None, low_memory=False)


def _candidate_data_dirs(data_dir: str) -> List[Path]:
    raw = Path(data_dir)
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        raw,
        Path.cwd() / data_dir,
        project_root / data_dir,
        project_root / "data" / "raw",
    ]

    seen = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def load_dashboard_bundle(
    data_dir: str = "data/raw",
    fallback_to_mock: bool = True,
    mock_n_customers: int = 500,
    seed: int = 42,
    include_optional: bool = False,
) -> DashboardDataBundle:
    """
    Dashboard의 현재 6개 화면에 꼭 필요한 파일은 customer_summary.csv와 cohort_retention.csv다.
    나머지 raw 파일은 향후 drill-down 확장을 위해 있으면 함께 로드하지만,
    현재 화면 렌더링에는 필수는 아니다.
    """
    for base in _candidate_data_dirs(data_dir):
        customer_path = base / RAW_CUSTOMER_SUMMARY
        cohort_path = base / RAW_COHORT_RETENTION
        if customer_path.exists() and cohort_path.exists():
            optionals: Dict[str, pd.DataFrame] = {}
            for key, config in OPTIONAL_RAW_FILES.items():
                if not include_optional:
                    optionals[key] = _empty_df()
                    continue
                path = base / str(config["filename"])
                optionals[key] = _read_csv(path, usecols=list(config.get("usecols", []))) if path.exists() else _empty_df()

            return DashboardDataBundle(
                customer_summary=ensure_dashboard_contract(_read_csv(customer_path)),
                cohort_retention=_read_csv(cohort_path),
                customers=optionals["customers"],
                events=optionals["events"],
                orders=optionals["orders"],
                campaign_exposures=optionals["campaign_exposures"],
                state_snapshots=optionals["state_snapshots"],
                treatment_assignments=optionals["treatment_assignments"],
                source_dir=base.as_posix(),
                used_mock=False,
            )

    if not fallback_to_mock:
        searched = ", ".join(path.as_posix() for path in _candidate_data_dirs(data_dir))
        raise FileNotFoundError(
            "Required dashboard data files are missing. "
            f"Searched directories: {searched}"
        )

    return DashboardDataBundle(
        customer_summary=ensure_dashboard_contract(generate_mock_customers(n_customers=mock_n_customers, seed=seed)),
        cohort_retention=generate_mock_cohort_retention(seed=seed),
        customers=_empty_df(),
        events=_empty_df(),
        orders=_empty_df(),
        campaign_exposures=_empty_df(),
        state_snapshots=_empty_df(),
        treatment_assignments=_empty_df(),
        source_dir=None,
        used_mock=True,
    )


def load_dashboard_data(
    data_dir: str = "data/raw",
    fallback_to_mock: bool = True,
    mock_n_customers: int = 500,
    seed: int = 42,
    include_optional: bool = False,
):
    bundle = load_dashboard_bundle(
        data_dir=data_dir,
        fallback_to_mock=fallback_to_mock,
        mock_n_customers=mock_n_customers,
        seed=seed,
        include_optional=include_optional,
    )
    label = "mock data 사용 중" if bundle.used_mock else "실제 시뮬레이터 산출물 사용 중"
    return bundle.customer_summary, bundle.cohort_retention, label
