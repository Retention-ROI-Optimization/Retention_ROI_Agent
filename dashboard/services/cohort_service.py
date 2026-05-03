from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ["cohort_month", "period", "retention_rate"]
_OPTIONAL_COLUMNS = [
    "cohort_size",
    "retained_customers",
    "observed",
    "activity_definition",
    "retention_mode",
    "min_events_per_period",
]

_ACTIVITY_LABELS = {
    "all_activity": "전체 활동(쿠폰/지원 포함)",
    "core_engagement": "핵심 참여(방문·검색·장바구니·구매)",
    "purchase_only": "구매만",
    "purchase_or_redeem": "구매 또는 쿠폰 사용",
}

_RETENTION_MODE_LABELS = {
    "point": "해당 월 재방문율",
    "rolling": "이후 재활성 포함 롤링 리텐션",
}

_TRUE_VALUES = {"true", "1", "yes", "y", "t"}
_FALSE_VALUES = {"false", "0", "no", "n", "f"}


def _coerce_bool(series: pd.Series, default: bool = True) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)

    def _parse(value):
        if pd.isna(value):
            return default
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer, float, np.floating)):
            return bool(int(value))
        text = str(value).strip().lower()
        if text in _TRUE_VALUES:
            return True
        if text in _FALSE_VALUES:
            return False
        return default

    return series.map(_parse).astype(bool)


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
    if "activity_definition" not in df.columns:
        df["activity_definition"] = "all_activity"
    if "retention_mode" not in df.columns:
        df["retention_mode"] = "point"
    if "min_events_per_period" not in df.columns:
        df["min_events_per_period"] = 1

    df["cohort_month"] = df["cohort_month"].astype(str)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["retention_rate"] = pd.to_numeric(df["retention_rate"], errors="coerce")
    df["cohort_size"] = pd.to_numeric(df["cohort_size"], errors="coerce")
    df["retained_customers"] = pd.to_numeric(df["retained_customers"], errors="coerce")
    df["min_events_per_period"] = pd.to_numeric(df["min_events_per_period"], errors="coerce").fillna(1).astype(int)
    df["activity_definition"] = df["activity_definition"].fillna("all_activity").astype(str)
    df["retention_mode"] = df["retention_mode"].fillna("point").astype(str)
    df["observed"] = _coerce_bool(df["observed"], default=True)
    return df.sort_values(["activity_definition", "retention_mode", "cohort_month", "period"]).reset_index(drop=True)


def get_available_activity_definitions(cohort_df: pd.DataFrame) -> list[str]:
    df = _ensure_schema(cohort_df)
    values = [str(x) for x in df["activity_definition"].dropna().unique()]
    preferred = [key for key in ["core_engagement", "purchase_only", "purchase_or_redeem", "all_activity"] if key in values]
    remaining = sorted(key for key in values if key not in preferred)
    return preferred + remaining


def get_available_retention_modes(cohort_df: pd.DataFrame) -> list[str]:
    df = _ensure_schema(cohort_df)
    values = [str(x) for x in df["retention_mode"].dropna().unique()]
    preferred = [key for key in ["rolling", "point"] if key in values]
    remaining = sorted(key for key in values if key not in preferred)
    return preferred + remaining


def get_activity_definition_label(value: str) -> str:
    return _ACTIVITY_LABELS.get(value, value.replace("_", " ").title())


def get_retention_mode_label(value: str) -> str:
    return _RETENTION_MODE_LABELS.get(value, value.replace("_", " ").title())


def _resolve_default_selection(
    cohort_df: pd.DataFrame,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> tuple[str, str]:
    activity_options = get_available_activity_definitions(cohort_df)
    mode_options = get_available_retention_modes(cohort_df)

    selected_activity = activity_definition if activity_definition in activity_options else (activity_options[0] if activity_options else "all_activity")
    selected_mode = retention_mode if retention_mode in mode_options else (mode_options[0] if mode_options else "point")
    return selected_activity, selected_mode


def _filtered_cohort_df(
    cohort_df: pd.DataFrame,
    min_cohort_size: int = 0,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
    observed_only: bool = True,
) -> pd.DataFrame:
    df = _ensure_schema(cohort_df)
    selected_activity, selected_mode = _resolve_default_selection(
        df,
        activity_definition=activity_definition,
        retention_mode=retention_mode,
    )
    df = df[
        (df["activity_definition"] == selected_activity)
        & (df["retention_mode"] == selected_mode)
    ].copy()
    if min_cohort_size > 0 and "cohort_size" in df.columns:
        df = df[df["cohort_size"].fillna(0) >= min_cohort_size].copy()
    if observed_only:
        df = df[df["observed"]].copy()
    return df.reset_index(drop=True)


def _comparable_period(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    max_observed = df.groupby("cohort_month")["period"].max()
    mature = max_observed[max_observed >= 1]
    if mature.empty:
        return 0 if not max_observed.empty else None
    return int(mature.min())


def _comparable_slice(df: pd.DataFrame) -> pd.DataFrame:
    period = _comparable_period(df)
    if period is None:
        return df.head(0).copy()
    max_observed = df.groupby("cohort_month")["period"].max()
    eligible_cohorts = max_observed[max_observed >= period].index
    return df[(df["cohort_month"].isin(eligible_cohorts)) & (df["period"] == period)].copy()


def _count_non_monotonic_cohorts(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    count = 0
    for _, group in df.sort_values(["cohort_month", "period"]).groupby("cohort_month"):
        series = pd.to_numeric(group["retention_rate"], errors="coerce").dropna()
        if len(series) >= 3 and (series.diff().fillna(0) > 1e-12).any():
            count += 1
    return count


def get_cohort_curve(
    cohort_df: pd.DataFrame,
    min_cohort_size: int = 0,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> pd.DataFrame:
    return _filtered_cohort_df(
        cohort_df,
        min_cohort_size=min_cohort_size,
        activity_definition=activity_definition,
        retention_mode=retention_mode,
        observed_only=True,
    )


def get_cohort_pivot(
    cohort_df: pd.DataFrame,
    min_cohort_size: int = 0,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> pd.DataFrame:
    df = get_cohort_curve(
        cohort_df,
        min_cohort_size=min_cohort_size,
        activity_definition=activity_definition,
        retention_mode=retention_mode,
    )
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="cohort_month", columns="period", values="retention_rate")
    return pivot.sort_index().sort_index(axis=1)


def get_cohort_display_table(
    cohort_df: pd.DataFrame,
    min_cohort_size: int = 0,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> pd.DataFrame:
    pivot = get_cohort_pivot(
        cohort_df,
        min_cohort_size=min_cohort_size,
        activity_definition=activity_definition,
        retention_mode=retention_mode,
    )
    if pivot.empty:
        return pd.DataFrame()
    display = pivot.reset_index()
    for col in display.columns[1:]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return display


def get_cohort_summary(
    cohort_df: pd.DataFrame,
    min_cohort_size: int = 0,
    activity_definition: Optional[str] = None,
    retention_mode: Optional[str] = None,
) -> Dict:
    selected_activity, selected_mode = _resolve_default_selection(
        cohort_df,
        activity_definition=activity_definition,
        retention_mode=retention_mode,
    )
    df = get_cohort_curve(
        cohort_df,
        min_cohort_size=min_cohort_size,
        activity_definition=selected_activity,
        retention_mode=selected_mode,
    )
    if df.empty:
        return {
            "cohort_count": 0,
            "avg_cohort_size": 0,
            "observed_periods": 0,
            "month1_avg_retention": np.nan,
            "last_observed_avg_retention": np.nan,
            "comparable_period": None,
            "comparable_avg_retention": np.nan,
            "best_last_cohort": None,
            "worst_last_cohort": None,
            "best_comparable_cohort": None,
            "worst_comparable_cohort": None,
            "selected_activity_definition": selected_activity,
            "selected_activity_label": get_activity_definition_label(selected_activity),
            "selected_retention_mode": selected_mode,
            "selected_retention_mode_label": get_retention_mode_label(selected_mode),
            "non_monotonic_cohort_count": 0,
        }

    cohort_sizes = df.groupby("cohort_month")["cohort_size"].max(min_count=1)
    last_observed = (
        df.sort_values(["cohort_month", "period"])
        .groupby("cohort_month", as_index=False)
        .tail(1)
        .sort_values("retention_rate", ascending=False)
        .reset_index(drop=True)
    )
    comparable_slice = (
        _comparable_slice(df)
        .sort_values("retention_rate", ascending=False)
        .reset_index(drop=True)
    )
    month1 = df[df["period"] == 1]["retention_rate"]
    comparable_period = _comparable_period(df)

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
        "last_observed_avg_retention": float(last_observed["retention_rate"].mean()) if not last_observed.empty else np.nan,
        "comparable_period": comparable_period,
        "comparable_avg_retention": float(comparable_slice["retention_rate"].mean()) if not comparable_slice.empty else np.nan,
        "best_last_cohort": _row_to_dict(last_observed.head(1)),
        "worst_last_cohort": _row_to_dict(last_observed.tail(1)),
        "best_comparable_cohort": _row_to_dict(comparable_slice.head(1)),
        "worst_comparable_cohort": _row_to_dict(comparable_slice.tail(1)),
        "selected_activity_definition": selected_activity,
        "selected_activity_label": get_activity_definition_label(selected_activity),
        "selected_retention_mode": selected_mode,
        "selected_retention_mode_label": get_retention_mode_label(selected_mode),
        "non_monotonic_cohort_count": _count_non_monotonic_cohorts(df) if selected_mode == "point" else 0,
    }


__all__ = [
    "get_activity_definition_label",
    "get_available_activity_definitions",
    "get_available_retention_modes",
    "get_cohort_curve",
    "get_cohort_display_table",
    "get_cohort_pivot",
    "get_cohort_summary",
    "get_retention_mode_label",
]
