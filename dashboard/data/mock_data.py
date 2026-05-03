from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PERSONAS: List[str] = [
    "vip",
    "loyal",
    "coupon_sensitive",
    "new_customer",
    "at_risk",
]

UPLIFT_SEGMENTS: List[str] = [
    "Persuadables",
    "Sure Things",
    "Lost Causes",
    "Sleeping Dogs",
]

CITIES: List[str] = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon"]
DEVICES: List[str] = ["mobile", "desktop", "tablet"]
ACQUISITION_CHANNELS: List[str] = ["organic", "paid_ads", "email", "referral", "social"]


@dataclass(frozen=True)
class PersonaProfile:
    clv_mean: float
    clv_std: float
    churn_low: float
    churn_high: float
    uplift_mean: float
    uplift_std: float
    coupon_low: int
    coupon_high: int


PERSONA_CONFIG: Dict[str, PersonaProfile] = {
    "vip": PersonaProfile(130_000, 22_000, 0.18, 0.45, 0.11, 0.04, 14_000, 26_000),
    "loyal": PersonaProfile(95_000, 18_000, 0.10, 0.32, 0.08, 0.03, 11_000, 20_000),
    "coupon_sensitive": PersonaProfile(72_000, 15_000, 0.18, 0.52, 0.20, 0.05, 13_000, 24_000),
    "new_customer": PersonaProfile(58_000, 12_000, 0.22, 0.48, 0.10, 0.05, 9_000, 18_000),
    "at_risk": PersonaProfile(64_000, 14_000, 0.45, 0.82, 0.16, 0.06, 10_000, 20_000),
}


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _assign_uplift_segment(churn_probability: float, uplift_score: float) -> str:
    if uplift_score >= 0.12 and churn_probability >= 0.45:
        return "Persuadables"
    if uplift_score < 0.12 and churn_probability < 0.45:
        return "Sure Things"
    if uplift_score < 0.08 and churn_probability >= 0.45:
        return "Lost Causes"
    return "Sleeping Dogs"


def generate_mock_customers(n_customers: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    persona_probs = np.array([0.15, 0.22, 0.28, 0.15, 0.20])

    for idx in range(n_customers):
        persona = str(rng.choice(PERSONAS, p=persona_probs))
        profile = PERSONA_CONFIG[persona]

        clv = _clip(rng.normal(profile.clv_mean, profile.clv_std), 25_000, 200_000)
        churn_probability = _clip(
            rng.uniform(profile.churn_low, profile.churn_high) + rng.normal(0, 0.03),
            0.01,
            0.95,
        )
        uplift_score = _clip(
            rng.normal(profile.uplift_mean, profile.uplift_std),
            -0.08,
            0.45,
        )

        uplift_segment = _assign_uplift_segment(churn_probability, uplift_score)
        coupon_cost = int(rng.integers(profile.coupon_low, profile.coupon_high + 1))

        incremental_margin = max(uplift_score, 0.0) * clv * rng.uniform(0.9, 1.6)
        expected_incremental_profit = float(incremental_margin - coupon_cost)
        expected_roi = float(expected_incremental_profit / coupon_cost)

        rows.append(
            {
                "customer_id": 1000 + idx,
                "persona": persona,
                "signup_month": f"2025-{int(rng.integers(1, 7)):02d}",
                "city": str(rng.choice(CITIES)),
                "device": str(rng.choice(DEVICES, p=[0.58, 0.30, 0.12])),
                "acquisition_channel": str(rng.choice(ACQUISITION_CHANNELS)),
                "churn_probability": round(churn_probability, 6),
                "retention_probability": round(1.0 - churn_probability, 6),
                "uplift_score": round(uplift_score, 6),
                "uplift_segment": uplift_segment,
                "clv": round(clv, 2),
                "coupon_cost": coupon_cost,
                "expected_incremental_profit": round(expected_incremental_profit, 2),
                "expected_roi": round(expected_roi, 6),
            }
        )

    return pd.DataFrame(rows)


def generate_mock_cohort_retention(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cohorts = pd.period_range("2025-01", periods=6, freq="M").astype(str)
    periods = list(range(6))
    rows = []

    for cohort_idx, cohort_month in enumerate(cohorts):
        base = 0.74 - cohort_idx * 0.025 + rng.normal(0, 0.01)
        for period in periods:
            decay = 0.11 * period + 0.018 * max(period - 1, 0)
            retention_rate = _clip(base - decay + rng.normal(0, 0.015), 0.12, 0.92)
            rows.append(
                {
                    "cohort_month": cohort_month,
                    "period": period,
                    "retention_rate": round(retention_rate, 4),
                }
            )

    return pd.DataFrame(rows)


def allocate_budget(customers: pd.DataFrame, budget: int) -> Tuple[pd.DataFrame, dict]:
    if customers.empty or budget <= 0:
        empty = customers.head(0).copy()
        return empty, {
            "budget": int(budget),
            "spent": 0,
            "remaining": int(budget),
            "num_targeted": 0,
            "expected_incremental_profit": 0.0,
            "overall_roi": 0.0,
        }

    eligible = customers.copy()
    eligible = eligible[
        (eligible["uplift_score"] > 0.05)
        & (eligible["expected_incremental_profit"] > 0)
        & (eligible["uplift_segment"] != "Lost Causes")
    ].copy()

    if eligible.empty:
        return eligible, {
            "budget": int(budget),
            "spent": 0,
            "remaining": int(budget),
            "num_targeted": 0,
            "expected_incremental_profit": 0.0,
            "overall_roi": 0.0,
        }

    eligible = eligible.sort_values(
        ["expected_roi", "expected_incremental_profit", "clv"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    cumulative_cost = eligible["coupon_cost"].cumsum()
    selected = eligible[cumulative_cost <= budget].copy()

    spent = float(selected["coupon_cost"].sum()) if not selected.empty else 0.0
    expected_profit = float(selected["expected_incremental_profit"].sum()) if not selected.empty else 0.0
    overall_roi = float(expected_profit / spent) if spent > 0 else 0.0

    summary = {
        "budget": int(budget),
        "spent": int(round(spent)),
        "remaining": int(round(budget - spent)),
        "num_targeted": int(len(selected)),
        "expected_incremental_profit": round(expected_profit, 2),
        "overall_roi": round(overall_roi, 6),
    }
    return selected, summary


def budget_allocation_by_segment(selected_customers: pd.DataFrame) -> pd.DataFrame:
    if selected_customers.empty:
        return pd.DataFrame(
            columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"]
        )

    grouped = (
        selected_customers.groupby("uplift_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            allocated_budget=("coupon_cost", "sum"),
            expected_profit=("expected_incremental_profit", "sum"),
        )
        .sort_values("allocated_budget", ascending=False)
        .reset_index(drop=True)
    )
    return grouped