from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str, bytes)) else False:
        return None
    return value


def read_csv_safely(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read CSV with several common encodings without mutating user data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    errors: list[str] = []
    for kwargs in [
        {"encoding": "utf-8-sig", "low_memory": False},
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "cp949", "low_memory": False},
        {"encoding": "euc-kr", "low_memory": False},
    ]:
        try:
            return pd.read_csv(path, nrows=nrows, **kwargs)
        except Exception as exc:  # noqa: BLE001 - keep trying common encodings
            errors.append(f"{kwargs.get('encoding')}: {exc}")

    # Last resort: delimiter sniffing. It is slower, so only do it after simple
    # comma-separated attempts fail.
    try:
        return pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"sniffed delimiter: {exc}")
        raise ValueError("CSV parsing failed. Tried utf-8-sig, utf-8, cp949, euc-kr, and delimiter sniffing. " + " | ".join(errors)) from exc


def safe_numeric(series: pd.Series | Any, *, default: float = 0.0, nonnegative: bool = False) -> pd.Series:
    """Convert to numeric, remove infinities, winsorize, and clip extreme values.

    This is the main guard against `overflow encountered in matmul` from user
    datasets with huge IDs, timestamps, monetary values, or division artifacts.
    """
    if not isinstance(series, pd.Series):
        return pd.Series(default)
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    if len(valid) >= 20:
        lo = valid.quantile(0.001)
        hi = valid.quantile(0.999)
        if pd.notna(lo) and pd.notna(hi) and float(lo) < float(hi):
            numeric = numeric.clip(lower=float(lo), upper=float(hi))
    if nonnegative:
        numeric = numeric.clip(lower=0)
    # Hard clip avoids overflow in downstream matrix multiplications.
    numeric = numeric.clip(lower=0 if nonnegative else -1e9, upper=1e9)
    return numeric.fillna(default)


def minmax(series: pd.Series, *, default: float = 0.0) -> pd.Series:
    numeric = safe_numeric(series, default=default)
    if numeric.empty:
        return numeric
    lo = float(numeric.min())
    hi = float(numeric.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=numeric.index)
    return ((numeric - lo) / (hi - lo)).clip(0, 1)


def safe_text(series: pd.Series | Any, *, default: str = "unknown", index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        out = series.astype(str).replace({"nan": default, "NaN": default, "None": default, "": default}).fillna(default)
        return out
    if index is None:
        return pd.Series([default])
    return pd.Series(default, index=index)
