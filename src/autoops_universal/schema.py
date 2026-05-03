from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ID_SYNONYMS = [
    "customer_id", "customerid", "cust_id", "user_id", "userid", "member_id", "memberid", "client_id", "account_id", "id", "user", "고객id", "고객_id", "회원id"
]
LABEL_SYNONYMS = [
    "label", "target", "churn", "is_churn", "churned", "is_churned", "churn_label", "attrition", "left", "cancelled", "이탈", "이탈여부"
]
RECENCY_SYNONYMS = [
    "recency_days", "days_since_last_purchase", "days_since_last_order", "days_since_last_event", "inactivity_days", "inactive_days", "last_purchase_days", "미구매일수", "비활성일수"
]
FREQUENCY_SYNONYMS = [
    "frequency", "frequency_90d", "purchase_count", "order_count", "orders", "transactions", "total_orders", "visits", "visit_count", "구매횟수", "주문횟수"
]
MONETARY_SYNONYMS = [
    "monetary", "monetary_90d", "revenue", "sales", "total_spend", "total_amount", "amount", "net_amount", "clv", "ltv", "gmv", "총구매액", "매출"
]
DATE_SYNONYMS = [
    "signup_date", "join_date", "created_at", "registration_date", "first_seen_at", "가입일"
]
LAST_DATE_SYNONYMS = [
    "last_purchase_date", "last_order_date", "last_event_date", "last_active_date", "last_seen_at", "transaction_date", "order_date", "event_date", "purchase_date", "timestamp", "created_time", "최근구매일", "최근활동일"
]
SEGMENT_SYNONYMS = ["segment", "persona", "customer_segment", "cluster", "grade", "등급", "세그먼트"]
REGION_SYNONYMS = ["region", "city", "location", "area", "country", "지역"]
CHANNEL_SYNONYMS = ["acquisition_channel", "channel", "source", "유입채널"]
DEVICE_SYNONYMS = ["device", "device_type", "platform", "os"]
CATEGORY_SYNONYMS = ["category", "item_category", "favorite_category", "preferred_category", "카테고리"]


@dataclass(frozen=True)
class ColumnMapping:
    customer_id: str | None
    label: str | None
    recency_days: str | None
    frequency: str | None
    monetary: str | None
    signup_date: str | None
    last_activity_date: str | None
    segment: str | None
    region: str | None
    channel: str | None
    device: str | None
    category: str | None

    def as_dict(self) -> dict[str, str | None]:
        return self.__dict__.copy()


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    original = list(columns)
    lookup = {_normalize_name(col): col for col in original}
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    for col in original:
        norm = _normalize_name(col)
        if any(_normalize_name(candidate) in norm for candidate in candidates):
            return col
    return None


def infer_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    columns = list(df.columns)
    return ColumnMapping(
        customer_id=_first_existing(columns, ID_SYNONYMS),
        label=_first_existing(columns, LABEL_SYNONYMS),
        recency_days=_first_existing(columns, RECENCY_SYNONYMS),
        frequency=_first_existing(columns, FREQUENCY_SYNONYMS),
        monetary=_first_existing(columns, MONETARY_SYNONYMS),
        signup_date=_first_existing(columns, DATE_SYNONYMS),
        last_activity_date=_first_existing(columns, LAST_DATE_SYNONYMS),
        segment=_first_existing(columns, SEGMENT_SYNONYMS),
        region=_first_existing(columns, REGION_SYNONYMS),
        channel=_first_existing(columns, CHANNEL_SYNONYMS),
        device=_first_existing(columns, DEVICE_SYNONYMS),
        category=_first_existing(columns, CATEGORY_SYNONYMS),
    )


def read_customer_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", low_memory=False)


def _safe_numeric(series: Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = numeric.dropna()
        if len(valid) >= 20:
            lo = valid.quantile(0.001)
            hi = valid.quantile(0.999)
            if pd.notna(lo) and pd.notna(hi) and float(lo) < float(hi):
                numeric = numeric.clip(lower=float(lo), upper=float(hi))
        numeric = numeric.clip(lower=-1e12, upper=1e12)
        return numeric.fillna(default)
    return pd.Series(default)


def _minmax(series: pd.Series, default: float = 0.0) -> pd.Series:
    numeric = _safe_numeric(series, default=default)
    if numeric.empty:
        return numeric
    lo = float(numeric.min())
    hi = float(numeric.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=numeric.index)
    return (numeric - lo) / (hi - lo)


def _parse_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        unique = sorted(set(numeric.dropna().astype(float).unique().tolist()))
        if set(unique).issubset({0.0, 1.0}):
            return numeric.astype(int).clip(0, 1)
        threshold = float(numeric.median())
        return (numeric > threshold).astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "true", "yes", "y", "churn", "churned", "left", "cancel", "cancelled", "탈퇴", "이탈", "해지"}
    negative_values = {"0", "false", "no", "n", "active", "retained", "stay", "유지", "정상"}
    out = lowered.map(lambda x: 1 if x in positive_values else (0 if x in negative_values else np.nan))
    if out.notna().sum() == 0:
        return pd.Series(0, index=series.index, dtype=int)
    return out.fillna(0).astype(int)


def _safe_text(df: pd.DataFrame, col: str | None, default: str) -> pd.Series:
    if col and col in df.columns:
        return df[col].astype(str).replace({"nan": default, "None": default, "": default}).fillna(default)
    return pd.Series(default, index=df.index)



def _find_date_column(df: pd.DataFrame, mapping: ColumnMapping) -> str | None:
    if mapping.last_activity_date and mapping.last_activity_date in df.columns:
        return mapping.last_activity_date
    return _first_existing(df.columns, LAST_DATE_SYNONYMS)


def _aggregate_transaction_rows(df: pd.DataFrame, mapping: ColumnMapping) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate transaction-level rows to one row per customer when needed.

    Many users upload transaction logs instead of customer-level feature tables.
    Modeling directly on one row per transaction can be slow, create duplicated
    customers, and produce unstable numeric scales. This guard converts repeated
    customer rows into recency/frequency/monetary style customer features.
    """
    if not mapping.customer_id or mapping.customer_id not in df.columns:
        return df, {"transaction_aggregation_applied": False, "reason": "no_customer_id"}

    customer_col = mapping.customer_id
    unique_customers = int(df[customer_col].nunique(dropna=True))
    rows = int(len(df))
    duplicate_ratio = 1.0 - (unique_customers / rows) if rows else 0.0
    should_aggregate = rows >= 1000 and unique_customers > 0 and duplicate_ratio >= 0.10
    if not should_aggregate:
        return df, {
            "transaction_aggregation_applied": False,
            "input_rows": rows,
            "unique_customers": unique_customers,
            "duplicate_ratio": duplicate_ratio,
        }

    date_col = _find_date_column(df, mapping)
    amount_col = mapping.monetary if mapping.monetary and mapping.monetary in df.columns else _first_existing(df.columns, MONETARY_SYNONYMS)
    label_col = mapping.label if mapping.label and mapping.label in df.columns else None

    work = df.copy()
    work[customer_col] = work[customer_col].astype(str)
    now = pd.Timestamp.now(tz=None).floor("D")
    if date_col and date_col in work.columns:
        work["__event_ts"] = pd.to_datetime(work[date_col], errors="coerce")
    else:
        work["__event_ts"] = pd.NaT
    if amount_col and amount_col in work.columns:
        work["__amount"] = _safe_numeric(work[amount_col], default=0.0).clip(lower=0)
    else:
        work["__amount"] = 0.0

    grouped = work.groupby(customer_col, dropna=False)
    agg = grouped.agg(
        frequency_90d=(customer_col, "size"),
        monetary_90d=("__amount", "sum"),
        avg_order_amount=("__amount", "mean"),
        last_activity_date=("__event_ts", "max"),
        first_activity_date=("__event_ts", "min"),
    ).reset_index().rename(columns={customer_col: "customer_id"})

    agg["last_activity_date"] = pd.to_datetime(agg["last_activity_date"], errors="coerce").fillna(now)
    agg["signup_date"] = pd.to_datetime(agg["first_activity_date"], errors="coerce").fillna(now - pd.Timedelta(days=365))
    agg["recency_days"] = (now - agg["last_activity_date"]).dt.days.clip(lower=0)

    if label_col:
        labels = grouped[label_col].agg(lambda x: _parse_binary_label(x).max()).reset_index().rename(columns={customer_col: "customer_id", label_col: "label"})
        agg = agg.merge(labels, on="customer_id", how="left")

    for source_col, target_col, default in [
        (mapping.segment, "segment", "general"),
        (mapping.region, "region", "unknown"),
        (mapping.channel, "channel", "unknown"),
        (mapping.device, "device", "unknown"),
        (mapping.category, "category", "general"),
    ]:
        if source_col and source_col in work.columns:
            first_values = grouped[source_col].first().reset_index().rename(columns={customer_col: "customer_id", source_col: target_col})
            agg = agg.merge(first_values, on="customer_id", how="left")
            agg[target_col] = agg[target_col].fillna(default)
        else:
            agg[target_col] = default

    diagnostics = {
        "transaction_aggregation_applied": True,
        "input_rows": rows,
        "output_customer_rows": int(len(agg)),
        "unique_customers": unique_customers,
        "duplicate_ratio": duplicate_ratio,
        "date_column": date_col,
        "amount_column": amount_col,
    }
    return agg, diagnostics


def normalize_customer_dataset(df: pd.DataFrame, mapping: ColumnMapping | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize an arbitrary customer CSV into a model-ready customer-level table.

    The function accepts either already aggregated customer features or rough CRM-style columns.
    Missing operational fields are synthesized conservatively so the existing dashboard can refresh.
    """
    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    mapping = mapping or infer_column_mapping(df)
    aggregation_diagnostics: dict[str, Any] = {"transaction_aggregation_applied": False}
    df, aggregation_diagnostics = _aggregate_transaction_rows(df, mapping)
    mapping = infer_column_mapping(df)

    out = pd.DataFrame(index=df.index)
    if mapping.customer_id and mapping.customer_id in df.columns:
        out["customer_id"] = df[mapping.customer_id]
    else:
        out["customer_id"] = np.arange(1, len(df) + 1)
    out["customer_id"] = out["customer_id"].astype(str).str.replace(r"[^0-9A-Za-z_-]", "", regex=True)
    out.loc[out["customer_id"].eq(""), "customer_id"] = [f"C{i+1}" for i in range(out["customer_id"].eq("").sum())]

    out["recency_days"] = _safe_numeric(df[mapping.recency_days], default=np.nan) if mapping.recency_days else pd.Series(np.nan, index=df.index)
    out["frequency_90d"] = _safe_numeric(df[mapping.frequency], default=np.nan) if mapping.frequency else pd.Series(np.nan, index=df.index)
    out["monetary_90d"] = _safe_numeric(df[mapping.monetary], default=np.nan) if mapping.monetary else pd.Series(np.nan, index=df.index)

    now = pd.Timestamp.now(tz=None).floor("D")
    if mapping.last_activity_date and mapping.last_activity_date in df.columns:
        last_ts = pd.to_datetime(df[mapping.last_activity_date], errors="coerce")
        inferred_recency = (now - last_ts).dt.days.clip(lower=0)
        out["recency_days"] = out["recency_days"].fillna(inferred_recency)
    if mapping.signup_date and mapping.signup_date in df.columns:
        signup_ts = pd.to_datetime(df[mapping.signup_date], errors="coerce")
    else:
        signup_ts = now - pd.to_timedelta(np.maximum(out["recency_days"].fillna(60).astype(float) + 180, 30), unit="D")

    out["recency_days"] = out["recency_days"].fillna(float(out["recency_days"].median()) if out["recency_days"].notna().any() else 60.0).clip(lower=0)
    out["frequency_90d"] = out["frequency_90d"].fillna(float(out["frequency_90d"].median()) if out["frequency_90d"].notna().any() else 3.0).clip(lower=0)
    out["monetary_90d"] = out["monetary_90d"].fillna(float(out["monetary_90d"].median()) if out["monetary_90d"].notna().any() else 100000.0).clip(lower=0)

    out["avg_order_value_90d"] = out["monetary_90d"] / out["frequency_90d"].replace(0, np.nan)
    out["avg_order_value_90d"] = out["avg_order_value_90d"].replace([np.inf, -np.inf], np.nan).fillna(out["monetary_90d"])
    out["clv"] = np.maximum(out["monetary_90d"] * 1.8, out["avg_order_value_90d"] * 2.0).clip(lower=1000)

    out["persona"] = _safe_text(df, mapping.segment, "general")
    out["city"] = _safe_text(df, mapping.region, "unknown")
    out["region"] = out["city"]
    out["device"] = _safe_text(df, mapping.device, "unknown")
    out["device_type"] = out["device"]
    out["acquisition_channel"] = _safe_text(df, mapping.channel, "unknown")
    out["preferred_category"] = _safe_text(df, mapping.category, "general")
    out["signup_date"] = pd.to_datetime(signup_ts, errors="coerce").fillna(now - pd.Timedelta(days=365)).dt.floor("D")
    out["last_activity_date"] = now - pd.to_timedelta(out["recency_days"].round().astype(int), unit="D")

    recency_risk = _minmax(out["recency_days"])
    frequency_safety = _minmax(out["frequency_90d"])
    monetary_safety = _minmax(out["monetary_90d"])
    proxy_risk = (0.62 * recency_risk + 0.25 * (1.0 - frequency_safety) + 0.13 * (1.0 - monetary_safety)).clip(0, 1)
    out["proxy_churn_risk"] = proxy_risk

    if mapping.label and mapping.label in df.columns:
        out["label"] = _parse_binary_label(df[mapping.label])
    else:
        cutoff = float(proxy_risk.quantile(0.78)) if len(proxy_risk) > 5 else 0.55
        out["label"] = (proxy_risk >= cutoff).astype(int)

    if out["label"].nunique(dropna=True) < 2 and len(out) >= 5:
        cutoff = float(proxy_risk.quantile(0.78))
        out["label"] = (proxy_risk >= cutoff).astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:80]:
        safe_col = f"src_{str(col).strip().replace(' ', '_').replace('-', '_')}"
        if safe_col not in out.columns and col not in {mapping.customer_id, mapping.label}:
            out[safe_col] = _safe_numeric(df[col], default=0.0)

    diagnostics = {
        "input_rows": int(len(df)),
        "input_columns": list(map(str, df.columns)),
        "mapping": mapping.as_dict(),
        "label_source": "provided" if mapping.label else "proxy_generated",
        "positive_rate": float(out["label"].mean()) if len(out) else 0.0,
        "unique_customers": int(out["customer_id"].nunique()),
        "missing_values_total": int(df.isna().sum().sum()),
        "aggregation": aggregation_diagnostics,
    }
    return out.reset_index(drop=True), diagnostics
