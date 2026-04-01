from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ["cohort_month", "period", "retention_rate"]
_OPTIONAL_COLUMNS = ["cohort_size", "retained_customers", "observed"]


def _ensure_schema(cohort_df: pd.DataFrame) -> pd.DataFrame:
    if cohort_df.empty:
        columns = _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS
        return pd.DataFrame(columns=columns)

    df = cohort_df.copy()
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"cohort dataframe is missing required columns: {missing}")

    if "cohort_size" not in df.columns:
        df["cohort_size"] = np.nan
    if "retained_customers" not in df.columns:
        df["retained_customers"] = np.nan
    if "observed" not in df.columns:
        df["observed"] = True

    df["cohort_month"] = df["cohort_month"].astype(str)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["retention_rate"] = pd.to_numeric(df["retention_rate"], errors="coerce")
    df["cohort_size"] = pd.to_numeric(df["cohort_size"], errors="coerce")
    df["retained_customers"] = pd.to_numeric(df["retained_customers"], errors="coerce")
    df["observed"] = df["observed"].fillna(True).astype(bool)
    return df.sort_values(["cohort_month", "period"]).reset_index(drop=True)


def get_cohort_curve(cohort_df: pd.DataFrame, min_cohort_size: int = 0) -> pd.DataFrame:
    df = _ensure_schema(cohort_df)
    if min_cohort_size > 0 and "cohort_size" in df.columns:
        df = df[df["cohort_size"].fillna(0) >= min_cohort_size].copy()
    return df[df["observed"]].reset_index(drop=True)


def get_cohort_pivot(cohort_df: pd.DataFrame, min_cohort_size: int = 0) -> pd.DataFrame:
    df = get_cohort_curve(cohort_df, min_cohort_size=min_cohort_size)
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="cohort_month", columns="period", values="retention_rate")
    return pivot.sort_index().sort_index(axis=1)


def get_cohort_display_table(cohort_df: pd.DataFrame, min_cohort_size: int = 0) -> pd.DataFrame:
    pivot = get_cohort_pivot(cohort_df, min_cohort_size=min_cohort_size)
    if pivot.empty:
        return pd.DataFrame()
    display = pivot.reset_index()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return display


def get_cohort_summary(cohort_df: pd.DataFrame, min_cohort_size: int = 0) -> Dict:
    df = get_cohort_curve(cohort_df, min_cohort_size=min_cohort_size)
    if df.empty:
        return {
            "cohort_count": 0,
            "avg_cohort_size": 0,
            "observed_periods": 0,
            "month1_avg_retention": np.nan,
            "last_observed_avg_retention": np.nan,
            "best_last_cohort": None,
            "worst_last_cohort": None,
        }

    cohort_sizes = df.groupby("cohort_month")["cohort_size"].max(min_count=1)
    last_observed = (
        df.sort_values(["cohort_month", "period"])
        .groupby("cohort_month", as_index=False)
        .tail(1)
        .sort_values("retention_rate", ascending=False)
        .reset_index(drop=True)
    )
    comparable_last_observed = last_observed[last_observed["period"] >= 1].reset_index(drop=True)
    if comparable_last_observed.empty:
        comparable_last_observed = last_observed.copy()

    month1 = df[df["period"] == 1]["retention_rate"]

    def _row_to_dict(frame: pd.DataFrame):
        if frame.empty:
            return None
        row = frame.iloc[0]
        return {
            "cohort_month": str(row["cohort_month"]),
            "period": int(row["period"]),
            "retention_rate": float(row["retention_rate"]),
            "cohort_size": None if pd.isna(row["cohort_size"]) else int(row["cohort_size"]),
        }

    return {
        "cohort_count": int(df["cohort_month"].nunique()),
        "avg_cohort_size": float(cohort_sizes.mean()) if not cohort_sizes.dropna().empty else np.nan,
        "observed_periods": int(df["period"].nunique()),
        "month1_avg_retention": float(month1.mean()) if not month1.empty else np.nan,
        "last_observed_avg_retention": float(comparable_last_observed["retention_rate"].mean()) if not comparable_last_observed.empty else np.nan,
        "best_last_cohort": _row_to_dict(comparable_last_observed.head(1)),
        "worst_last_cohort": _row_to_dict(comparable_last_observed.tail(1)),
    }
