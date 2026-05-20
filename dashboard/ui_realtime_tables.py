"""Real-time table preparation helpers.

These helpers prepare display-only DataFrames for the operations monitor. They
do not mutate source data or change backend scoring/optimization logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.ui_labels import drop_duplicate_metric_columns


def _num_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        # duplicate columns can return DataFrame; keep the first visible one
        value = df[column]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _first_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    for col in columns:
        if col in df.columns:
            value = df[col]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return value
    return None


def _format_money(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if np.isnan(x) or np.isinf(x):
        return ""
    return f"₩{x:,.0f}"


def _format_roi(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if np.isnan(x) or np.isinf(x):
        return ""
    # Keep the same ratio scale the platform already uses, but make it compact.
    return f"{x:.3f}"


def _derive_investment_amount(df: pd.DataFrame) -> pd.Series:
    """Derive per-customer investment from cost fields or profit/ROI.

    The live action queue sometimes contains expected_profit and expected_roi but
    omits coupon_cost. Since expected_roi = expected_profit / cost, the displayed
    investment can be reconstructed as expected_profit / expected_roi.
    """
    cost_candidates = ["recommended_investment_amount", "coupon_cost", "queued_coupon_cost", "intervention_cost"]
    for col in cost_candidates:
        if col in df.columns:
            cost = _num_series(df, col)
            if (cost > 0).any():
                return cost

    profit = _num_series(df, "expected_incremental_profit")
    if not (profit > 0).any():
        profit = _num_series(df, "expected_profit")
    if not (profit > 0).any():
        profit = _num_series(df, "queued_expected_profit")

    roi = _num_series(df, "expected_roi")
    if not (roi > 0).any():
        roi = _num_series(df, "queued_expected_roi")

    derived = (profit / roi.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return derived


def _ensure_common_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    fixed = drop_duplicate_metric_columns(df.copy()).reset_index(drop=True)

    if "recommended_action" not in fixed.columns and "queued_recommended_action" in fixed.columns:
        fixed["recommended_action"] = fixed["queued_recommended_action"]
    if "intervention_intensity" not in fixed.columns and "queued_intervention_intensity" in fixed.columns:
        fixed["intervention_intensity"] = fixed["queued_intervention_intensity"]
    if "expected_profit" not in fixed.columns and "queued_expected_profit" in fixed.columns:
        fixed["expected_profit"] = fixed["queued_expected_profit"]
    if "expected_profit" not in fixed.columns and "expected_incremental_profit" in fixed.columns:
        fixed["expected_profit"] = fixed["expected_incremental_profit"]
    if "action_status" not in fixed.columns and "action_queue_status" in fixed.columns:
        fixed["action_status"] = fixed["action_queue_status"]
    if "trigger_reason" not in fixed.columns and "latest_trigger_reason" in fixed.columns:
        fixed["trigger_reason"] = fixed["latest_trigger_reason"]

    fixed["recommended_investment_amount"] = _derive_investment_amount(fixed)
    return drop_duplicate_metric_columns(fixed)


def prepare_live_action_queue_table(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact live action-queue table shown in user-live mode."""
    if not isinstance(actions_df, pd.DataFrame) or actions_df.empty:
        return pd.DataFrame()

    fixed = _ensure_common_action_columns(actions_df)

    for col in ["recommended_investment_amount", "expected_profit", "expected_incremental_profit"]:
        if col in fixed.columns:
            fixed[col] = _num_series(fixed, col).map(_format_money)
    if "expected_roi" in fixed.columns:
        fixed["expected_roi"] = _num_series(fixed, "expected_roi").map(_format_roi)

    wanted = [
        "customer_id",
        "persona",
        "recommended_action",
        "intervention_intensity",
        "recommended_investment_amount",
        "expected_profit",
        "expected_incremental_profit",
        "expected_roi",
        "action_status",
        "trigger_reason",
    ]
    cols = [col for col in wanted if col in fixed.columns]
    return fixed[cols].copy() if cols else fixed.copy()


def prepare_realtime_queue_table(queue_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact re-optimized queue table shown in the operations monitor."""
    if not isinstance(queue_df, pd.DataFrame) or queue_df.empty:
        return pd.DataFrame()

    fixed = _ensure_common_action_columns(queue_df)

    if "realtime_churn_score" in fixed.columns:
        fixed["realtime_churn_score"] = _num_series(fixed, "realtime_churn_score").map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
    for col in ["recommended_investment_amount", "queued_coupon_cost", "queued_expected_profit", "expected_profit", "expected_incremental_profit"]:
        if col in fixed.columns:
            fixed[col] = _num_series(fixed, col).map(_format_money)
    for col in ["queued_expected_roi", "expected_roi"]:
        if col in fixed.columns:
            fixed[col] = _num_series(fixed, col).map(_format_roi)

    wanted = [
        "customer_id",
        "persona",
        "uplift_segment",
        "realtime_churn_score",
        "recommended_action",
        "intervention_intensity",
        "recommended_investment_amount",
        "expected_profit",
        "queued_expected_profit",
        "expected_roi",
        "queued_expected_roi",
        "action_status",
        "latest_trigger_reason",
        "trigger_reason",
        "reoptimization_count",
    ]
    cols = [col for col in wanted if col in fixed.columns]
    return drop_duplicate_metric_columns(fixed[cols].copy() if cols else fixed.copy())
