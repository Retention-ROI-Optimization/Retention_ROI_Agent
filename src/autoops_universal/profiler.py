from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mapper import normalize_column_name


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(df))
    columns = int(len(df.columns))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    datetime_like: list[str] = []
    date_name_tokens = {"date", "datetime", "timestamp", "day", "week", "일자", "일"}
    for col in df.columns:
        name = normalize_column_name(col)
        sample = df[col].dropna().head(200)
        name_suggests_date = (
            any(token in name.split("_") for token in date_name_tokens)
            or "date" in name
            or "timestamp" in name
            or "일자" in name
        )
        if name_suggests_date:
            datetime_like.append(str(col))
        elif len(sample) > 0 and (pd.api.types.is_object_dtype(sample) or pd.api.types.is_string_dtype(sample)):
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.75:
                datetime_like.append(str(col))

    duplicate_rows = int(df.duplicated().sum()) if rows else 0
    missing_ratio = df.isna().mean().sort_values(ascending=False).head(30).to_dict()
    nunique = df.nunique(dropna=True).sort_values(ascending=False).head(30).to_dict()
    return {
        "rows": rows,
        "columns": columns,
        "numeric_columns": numeric_cols,
        "categorical_columns": object_cols,
        "datetime_like_columns": datetime_like,
        "duplicate_rows": duplicate_rows,
        "missing_ratio_top30": {str(k): float(v) for k, v in missing_ratio.items()},
        "nunique_top30": {str(k): int(v) for k, v in nunique.items()},
    }


def infer_grain(df: pd.DataFrame, *, customer_col: str | None, transaction_col: str | None, transaction_date_col: str | None, amount_col: str | None) -> dict[str, Any]:
    rows = int(len(df))
    if not customer_col or customer_col not in df.columns or rows == 0:
        return {"grain": "row_level", "reason": "customer_id not mapped", "duplicate_customer_ratio": 0.0, "unique_customers": rows}
    unique_customers = int(df[customer_col].nunique(dropna=True))
    duplicate_ratio = 1.0 - unique_customers / rows if rows else 0.0
    has_transaction_date = transaction_date_col is not None
    has_amount = amount_col is not None
    has_transaction_id = transaction_col is not None
    transaction_signals = [has_transaction_id, has_transaction_date, has_amount]
    if duplicate_ratio >= 0.10:
        grain = "transaction"
        reason = "repeated customer_id rows detected"
    elif has_transaction_id and (has_transaction_date or has_amount):
        grain = "transaction"
        reason = "transaction identifier with date/amount columns detected"
    else:
        grain = "customer"
        reason = "mostly one row per customer or insufficient transaction evidence"
    return {
        "grain": grain,
        "reason": reason,
        "input_rows": rows,
        "unique_customers": unique_customers,
        "duplicate_customer_ratio": float(duplicate_ratio),
        "transaction_signals": {
            "transaction_id": transaction_col,
            "transaction_date": transaction_date_col,
            "amount": amount_col,
        },
    }
