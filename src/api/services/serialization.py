from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def to_builtin(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if hasattr(value, 'item'):
        try:
            item = value.item()
            return to_builtin(item)
        except Exception:
            pass
    return str(value)


def dataframe_to_records(df: pd.DataFrame, columns: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    target = df.copy()
    if columns is not None:
        existing = [column for column in columns if column in target.columns]
        target = target[existing]
    return [to_builtin(row) for row in target.to_dict(orient='records')]
