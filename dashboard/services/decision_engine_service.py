from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_BASELINE_COLUMNS = [
    "customer_id",
    "persona",
    "uplift_segment",
    "churn_probability",
    "uplift_score",
    "clv",
    "coupon_cost",
    "expected_incremental_profit",
    "expected_roi",
]


DECISION_ENGINE_FACTORS = [
    {
        "factor": "이탈 확률",
        "legacy_engine": "반영",
        "enhanced_engine": "반영",
        "user_meaning": "누가 지금 위험 고객인지 먼저 거릅니다.",
    },
    {
        "factor": "업리프트 점수",
        "legacy_engine": "반영",
        "enhanced_engine": "반영",
        "user_meaning": "개입했을 때 실제로 반응할 가능성이 높은 고객을 우선합니다.",
    },
    {
        "factor": "CLV / 기대 이익",
        "legacy_engine": "반영",
        "enhanced_engine": "반영",
        "user_meaning": "같은 반응 가능성이라면 더 큰 가치의 고객에게 예산을 우선 씁니다.",
    },
    {
        "factor": "예상 ROI",
        "legacy_engine": "반영",
        "enhanced_engine": "반영",
        "user_meaning": "예산 1원당 기대되는 수익이 큰 고객을 선호합니다.",
    },
    {
        "factor": "이탈 시점 위험도(Survival)",
        "legacy_engine": "미반영",
        "enhanced_engine": "반영",
        "user_meaning": "누가 더 빨리 떠날지를 고려해 늦기 전에 개입합니다.",
    },
    {
        "factor": "개입 가능 시점 / intervention window",
        "legacy_engine": "미반영",
        "enhanced_engine": "반영",
        "user_meaning": "지금 당장 대응해야 하는 고객과 추후 관리할 고객을 구분합니다.",
    },
    {
        "factor": "개입 강도(low/mid/high)",
        "legacy_engine": "미반영",
        "enhanced_engine": "반영",
        "user_meaning": "고객마다 필요한 쿠폰/혜택 강도를 달리해 과투자와 과소투자를 줄입니다.",
    },
]



def _safe_numeric(series: pd.Series | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(float(default))



def _normalize(series: pd.Series) -> pd.Series:
    numeric = _safe_numeric(series, default=0.0)
    if numeric.empty:
        return numeric.astype(float)
    low = float(numeric.min())
    high = float(numeric.max())
    if abs(high - low) < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - low) / (high - low)



def get_baseline_budget_result(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    if customers.empty or budget <= 0:
        empty = customers.head(0).copy()
        return (
            empty,
            {
                "budget": int(max(budget, 0)),
                "spent": 0,
                "remaining": int(max(budget, 0)),
                "num_targeted": 0,
                "candidate_customers": 0,
                "expected_incremental_profit": 0.0,
                "overall_roi": 0.0,
                "engine_name": "baseline_profit_uplift_only",
            },
            pd.DataFrame(columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"]),
        )

    df = customers.copy()
    for column in REQUIRED_BASELINE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["churn_probability"] = _safe_numeric(df.get("churn_probability"), default=0.0).clip(0.0, 1.0)
    df["uplift_score"] = _safe_numeric(df.get("uplift_score"), default=0.0)
    df["clv"] = _safe_numeric(df.get("clv"), default=0.0)
    df["coupon_cost"] = _safe_numeric(df.get("coupon_cost"), default=0.0)
    df["expected_incremental_profit"] = _safe_numeric(df.get("expected_incremental_profit"), default=0.0)
    df["expected_roi"] = _safe_numeric(df.get("expected_roi"), default=0.0)
    df["uplift_segment"] = df.get("uplift_segment", pd.Series(index=df.index, dtype=object)).fillna("Unknown").astype(str)

    candidate = df[
        (df["churn_probability"] >= float(threshold))
        & (df["uplift_score"] > 0.0)
        & (df["coupon_cost"] > 0.0)
        & (df["expected_incremental_profit"] > 0.0)
    ].copy()

    if candidate.empty:
        summary = {
            "budget": int(budget),
            "spent": 0,
            "remaining": int(budget),
            "num_targeted": 0,
            "candidate_customers": 0,
            "expected_incremental_profit": 0.0,
            "overall_roi": 0.0,
            "engine_name": "baseline_profit_uplift_only",
        }
        allocation = pd.DataFrame(columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"])
        return candidate, summary, allocation

    candidate["baseline_priority_score"] = (
        0.40 * _normalize(candidate["expected_roi"])
        + 0.25 * _normalize(candidate["expected_incremental_profit"])
        + 0.15 * candidate["churn_probability"]
        + 0.10 * _normalize(candidate["uplift_score"])
        + 0.10 * _normalize(candidate["clv"])
    )
    candidate = candidate.sort_values(
        [
            "baseline_priority_score",
            "expected_roi",
            "expected_incremental_profit",
            "churn_probability",
            "clv",
            "customer_id",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    if max_customers is not None and max_customers > 0:
        candidate = candidate.head(int(max_customers)).copy()

    selected_rows = []
    spent = 0.0
    for row in candidate.itertuples(index=False):
        cost = float(getattr(row, "coupon_cost", 0.0))
        if cost <= 0:
            continue
        if spent + cost > float(budget):
            continue
        selected_rows.append(row._asdict())
        spent += cost

    selected = pd.DataFrame(selected_rows) if selected_rows else candidate.head(0).copy()
    spent = float(selected["coupon_cost"].sum()) if not selected.empty else 0.0
    expected_profit = float(selected["expected_incremental_profit"].sum()) if not selected.empty else 0.0
    overall_roi = (expected_profit / spent) if spent > 0 else 0.0

    summary = {
        "budget": int(budget),
        "spent": int(round(spent)),
        "remaining": int(round(float(budget) - spent)),
        "num_targeted": int(len(selected)),
        "candidate_customers": int(candidate["customer_id"].nunique()),
        "expected_incremental_profit": round(expected_profit, 2),
        "overall_roi": round(float(overall_roi), 6),
        "engine_name": "baseline_profit_uplift_only",
    }

    allocation = (
        selected.groupby("uplift_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            allocated_budget=("coupon_cost", "sum"),
            expected_profit=("expected_incremental_profit", "sum"),
        )
        .sort_values(["allocated_budget", "expected_profit"], ascending=[False, False])
        .reset_index(drop=True)
        if not selected.empty
        else pd.DataFrame(columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"])
    )
    return selected, summary, allocation



def aggregate_enhanced_segment_allocation(segment_allocation: pd.DataFrame) -> pd.DataFrame:
    if segment_allocation.empty:
        return pd.DataFrame(columns=["uplift_segment", "customer_count", "allocated_budget", "expected_profit"])
    frame = segment_allocation.copy()
    aggregations = {"customer_count": "sum", "allocated_budget": "sum"}
    if "expected_profit" in frame.columns:
        aggregations["expected_profit"] = "sum"
    return (
        frame.groupby("uplift_segment", as_index=False)
        .agg(aggregations)
        .sort_values(["allocated_budget", "expected_profit"], ascending=[False, False])
        .reset_index(drop=True)
    )



def get_decision_engine_factor_table() -> pd.DataFrame:
    return pd.DataFrame(DECISION_ENGINE_FACTORS)
