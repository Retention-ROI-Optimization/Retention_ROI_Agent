from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, SimulationConfig
from .customer_generator import generate_customers
from .event_engine import simulate_events
from .exporter import export_tables
from .personas import DEFAULT_PERSONAS
from .treatment import assign_treatment


def _safe_div(numer, denom):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    return numer / np.maximum(denom, 1.0)


def _build_customer_summary(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
    exposures: pd.DataFrame,
    final_state: pd.DataFrame,
    config: SimulationConfig,
) -> pd.DataFrame:
    end_ts = pd.Timestamp(config.end_date)

    visit_events = events[events["event_type"] == "visit"].copy() if not events.empty else pd.DataFrame(columns=["customer_id", "timestamp"])
    if not visit_events.empty:
        visit_events["date"] = pd.to_datetime(visit_events["timestamp"]).dt.normalize()

    purchase_events = orders.copy()
    if not purchase_events.empty:
        purchase_events["date"] = pd.to_datetime(purchase_events["order_time"]).dt.normalize()

    def _count_in_window(df: pd.DataFrame, date_col: str, start_days_ago: int, end_days_ago: int) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        start = end_ts - pd.Timedelta(days=start_days_ago)
        end = end_ts - pd.Timedelta(days=end_days_ago)
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        return df.loc[mask].groupby("customer_id").size()

    visits_last_7 = _count_in_window(visit_events, "date", 6, 0).rename("visits_last_7")
    visits_prev_7 = _count_in_window(visit_events, "date", 13, 7).rename("visits_prev_7")
    purchase_last_30 = _count_in_window(purchase_events, "date", 29, 0).rename("purchase_last_30")
    purchase_prev_30 = _count_in_window(purchase_events, "date", 59, 30).rename("purchase_prev_30")

    coupon_redeem_count = (
        events.loc[events["event_type"] == "coupon_redeem"].groupby("customer_id").size().rename("coupon_redeem_count")
        if not events.empty else pd.Series(dtype=float)
    )
    exposure_count = (
        exposures.groupby("customer_id").size().rename("coupon_exposure_count")
        if not exposures.empty else pd.Series(dtype=float)
    )

    summary = customers.merge(assignments, on="customer_id", how="left").merge(final_state, on="customer_id", how="left")

    for series in [visits_last_7, visits_prev_7, purchase_last_30, purchase_prev_30, coupon_redeem_count, exposure_count]:
        summary = summary.merge(series, on="customer_id", how="left")

    fill_zero_cols = [
        "frequency",
        "monetary",
        "coupon_exposures",
        "coupon_opens",
        "coupon_redeemed",
        "visits_last_7",
        "visits_prev_7",
        "purchase_last_30",
        "purchase_prev_30",
        "coupon_redeem_count",
        "coupon_exposure_count",
    ]
    for col in fill_zero_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    summary["visit_change_rate"] = _safe_div(summary["visits_last_7"] - summary["visits_prev_7"], summary["visits_prev_7"])
    summary["purchase_change_rate"] = _safe_div(summary["purchase_last_30"] - summary["purchase_prev_30"], summary["purchase_prev_30"])

    observed_coupon_response = _safe_div(summary["coupon_redeem_count"], summary["coupon_exposure_count"])
    latent_uplift = (
        summary["treatment_lift_base"]
        + 0.06 * summary["coupon_affinity"]
        - 0.04 * summary["price_sensitivity"]
        - 0.03 * (summary["persona"] == "sure_thing").astype(float)
        - 0.08 * (summary["persona"] == "lost_cause").astype(float)
    )
    summary["uplift_score"] = np.clip(
        0.45 * latent_uplift + 0.25 * observed_coupon_response + 0.10 * (summary["coupon_exposure_count"] > 0).astype(float),
        -0.12,
        0.38,
    )

    monetary_scaled = np.clip(summary["monetary"] / np.maximum(summary["monetary"].quantile(0.95), 1), 0, 1)
    frequency_scaled = np.clip(summary["frequency"] / np.maximum(summary["frequency"].quantile(0.95), 1), 0, 1)
    recency_scaled = np.clip(summary["recency_days"] / max(config.churn_inactivity_days * 1.5, 1), 0, 1)

    persona_boost = (
        np.where(summary["persona"] == "vip", -0.12, 0.0)
        + np.where(summary["persona"] == "coupon_sensitive", 0.04, 0.0)
        + np.where(summary["persona"] == "churn_risk", 0.16, 0.0)
        + np.where(summary["persona"] == "sure_thing", -0.08, 0.0)
        + np.where(summary["persona"] == "lost_cause", 0.18, 0.0)
    )

    base_churn = (
        0.34 * recency_scaled
        + 0.20 * (1 - frequency_scaled)
        + 0.16 * (1 - monetary_scaled)
        + 0.13 * (summary["visit_change_rate"] < 0).astype(float)
        + 0.13 * (summary["purchase_change_rate"] < 0).astype(float)
        + 0.04 * np.clip(summary["inactivity_days"] / max(config.churn_inactivity_days, 1), 0, 1)
    )
    summary["churn_probability"] = np.clip(base_churn + persona_boost, 0.01, 0.99)

    avg_order_value = _safe_div(summary["monetary"], summary["frequency"])
    retention_factor = np.clip(1.15 - summary["churn_probability"], 0.20, 1.15)
    summary["clv"] = (
        summary["monetary"] * (1.30 + 1.25 * retention_factor)
        + summary["frequency"] * np.maximum(avg_order_value, 20000) * 0.55
    ).clip(lower=15000)

    summary["expected_incremental_profit"] = np.maximum(summary["clv"] * summary["uplift_score"], -50000)
    summary["expected_roi"] = _safe_div(summary["expected_incremental_profit"] - summary["coupon_cost"], summary["coupon_cost"])

    summary["uplift_segment"] = np.select(
        [
            summary["uplift_score"] >= 0.15,
            (summary["uplift_score"] >= 0.05) & (summary["uplift_score"] < 0.15),
            (summary["uplift_score"] >= -0.02) & (summary["uplift_score"] < 0.05),
        ],
        ["Persuadables", "Sure Things", "Lost Causes"],
        default="Sleeping Dogs",
    )

    columns = [
        "customer_id",
        "persona",
        "acquisition_month",
        "recency_days",
        "frequency",
        "monetary",
        "visits_last_7",
        "visits_prev_7",
        "visit_change_rate",
        "purchase_last_30",
        "purchase_prev_30",
        "purchase_change_rate",
        "churn_probability",
        "uplift_score",
        "clv",
        "coupon_cost",
        "expected_incremental_profit",
        "expected_roi",
        "uplift_segment",
        "signup_date",
        "region",
        "device_type",
        "acquisition_channel",
        "treatment_group",
        "treatment_flag",
        "coupon_exposure_count",
        "coupon_redeem_count",
        "inactivity_days",
    ]

    ordered = [c for c in columns if c in summary.columns]
    others = [c for c in summary.columns if c not in ordered]
    summary = summary[ordered + others].sort_values("customer_id").reset_index(drop=True)
    return summary


def _build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = 7,
) -> pd.DataFrame:
    if events.empty:
        cohort_months = sorted(customers["acquisition_month"].unique())
        return pd.DataFrame(
            [
                {"cohort_month": month, "period": p, "retention_rate": (1.0 if p == 0 else 0.0)}
                for month in cohort_months
                for p in range(periods)
            ]
        )

    visit_events = events[events["event_type"] == "visit"][["customer_id", "timestamp"]].copy()
    visit_events["event_month"] = pd.to_datetime(visit_events["timestamp"]).dt.to_period("M")

    base = customers[["customer_id", "acquisition_month", "signup_date"]].copy()
    base["cohort_month"] = pd.PeriodIndex(base["acquisition_month"], freq="M")

    merged = visit_events.merge(base[["customer_id", "cohort_month"]], on="customer_id", how="left")
    merged["period"] = (merged["event_month"] - merged["cohort_month"]).apply(lambda x: x.n if pd.notna(x) else None)
    merged = merged[(merged["period"] >= 0) & (merged["period"] < periods)]

    cohort_sizes = base.groupby("cohort_month")["customer_id"].nunique()
    active_counts = merged.groupby(["cohort_month", "period"])["customer_id"].nunique()

    rows = []
    for cohort_month, size in cohort_sizes.items():
        for p in range(periods):
            count = int(active_counts.get((cohort_month, p), 0))
            rate = count / max(int(size), 1)
            rows.append(
                {
                    "cohort_month": str(cohort_month),
                    "period": int(p),
                    "retention_rate": float(rate),
                }
            )

    return pd.DataFrame(rows).sort_values(["cohort_month", "period"]).reset_index(drop=True)


def run_simulation(
    config: Optional[SimulationConfig] = None,
    export: bool = False,
    output_dir: Optional[str] = None,
    file_format: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full simulator pipeline.

    Returned tables:
    - customers
    - treatment_assignments
    - campaign_exposures
    - events
    - orders
    - state_snapshots
    - customer_summary
    - cohort_retention
    """
    config = config or DEFAULT_CONFIG
    rng = np.random.default_rng(config.random_seed)

    customers = generate_customers(config=config, personas=DEFAULT_PERSONAS, rng=rng)
    assignments = assign_treatment(customers=customers, config=config, rng=rng)
    events, orders, exposures, state_snapshots, final_state = simulate_events(
        customers=customers,
        assignments=assignments,
        config=config,
        rng=rng,
    )

    customer_summary = _build_customer_summary(
        customers=customers,
        assignments=assignments,
        events=events,
        orders=orders,
        exposures=exposures,
        final_state=final_state,
        config=config,
    )
    cohort_retention = _build_cohort_retention(customers=customers, events=events, periods=7)

    tables: Dict[str, pd.DataFrame] = {
        "customers": customers,
        "treatment_assignments": assignments,
        "campaign_exposures": exposures,
        "events": events,
        "orders": orders,
        "state_snapshots": state_snapshots,
        "customer_summary": customer_summary,
        "cohort_retention": cohort_retention,
    }

    if export:
        export_tables(
            tables=tables,
            output_dir=output_dir or config.default_export_dir,
            file_format=file_format or config.default_file_format,
        )

    return tables


def run_simulation_for_dashboard(
    config: Optional[SimulationConfig] = None,
    export: bool = False,
    output_dir: Optional[str] = None,
    file_format: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper for the current Streamlit UI.

    Returns:
    - customer_summary
    - cohort_retention
    """
    tables = run_simulation(
        config=config,
        export=export,
        output_dir=output_dir,
        file_format=file_format,
    )
    return tables["customer_summary"], tables["cohort_retention"]
