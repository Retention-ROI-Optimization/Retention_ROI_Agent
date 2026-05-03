from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import minmax, safe_numeric, safe_text
from .mapper import infer_schema_mapping, mapping_table, normalize_column_name
from .profiler import infer_grain, profile_dataframe
from .schema_registry import LEGACY_COMPATIBILITY_FIELDS, SchemaMapping


@dataclass
class CanonicalizationResult:
    customer_table: pd.DataFrame
    feature_table: pd.DataFrame
    mapping: SchemaMapping
    diagnostics: dict[str, Any]
    mapping_table: pd.DataFrame


def _parse_date(series: pd.Series | Any, *, default: pd.Timestamp) -> pd.Series:
    if isinstance(series, pd.Series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.fillna(default)
    return pd.Series(default)


def _parse_event_datetime(series: pd.Series | Any, *, column_name: str | None, default: pd.Timestamp) -> pd.Series:
    """Parse uploaded event dates, including relative retail DAY/WEEK_NO columns."""
    if not isinstance(series, pd.Series):
        return pd.Series(default)
    name = normalize_column_name(column_name or "")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.90:
        values = numeric.fillna(numeric.median() if numeric.notna().any() else 0.0)
        if name in {"day", "trans_day", "transaction_day", "purchase_day", "order_day"} or name.endswith("_day"):
            max_day = float(values.max()) if len(values) else 0.0
            offsets = (max_day - values).clip(lower=0, upper=3650)
            return default - pd.to_timedelta(offsets, unit="D")
        if name in {"week", "week_no", "week_number", "purchase_week", "order_week"}:
            max_week = float(values.max()) if len(values) else 0.0
            offsets = ((max_week - values) * 7).clip(lower=0, upper=3650)
            return default - pd.to_timedelta(offsets, unit="D")
        if "time" in name and "date" not in name and "day" not in name:
            return pd.Series(default, index=series.index)
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.fillna(default)


def _parse_binary(series: pd.Series | Any, *, index: pd.Index, default: int = 0) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(default, index=index, dtype=int)
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        unique = set(num.dropna().astype(float).unique().tolist())
        if unique and unique.issubset({0.0, 1.0}):
            return num.fillna(default).astype(int).clip(0, 1)
        if num.notna().sum() > 0:
            return (num.fillna(num.median()) > num.median()).astype(int)
        return pd.Series(default, index=index, dtype=int)
    lowered = series.astype(str).str.strip().str.lower()
    positives = {"1", "true", "yes", "y", "churn", "churned", "left", "cancel", "cancelled", "canceled", "탈퇴", "이탈", "해지", "전환", "구매", "성공"}
    negatives = {"0", "false", "no", "n", "active", "retained", "stay", "normal", "유지", "정상", "미전환", "실패"}
    out = lowered.map(lambda x: 1 if x in positives else (0 if x in negatives else np.nan))
    return out.fillna(default).astype(int).clip(0, 1)


def _get(df: pd.DataFrame, mapping: SchemaMapping, field: str) -> pd.Series | None:
    col = mapping.get(field)
    if col and col in df.columns:
        return df[col]
    return None


def _derive_proxy_label(customer: pd.DataFrame) -> tuple[pd.Series, str]:
    """Create a clearly marked churn proxy when no actual label exists."""
    recency_score = minmax(customer["recency_days"], default=0.0)
    low_frequency = 1.0 - minmax(customer["frequency_90d"], default=1.0)
    low_monetary = 1.0 - minmax(customer["monetary_90d"], default=1.0)
    risk = (0.55 * recency_score + 0.25 * low_frequency + 0.20 * low_monetary).clip(0, 1)
    threshold = float(risk.quantile(0.80)) if len(risk) >= 10 else 0.55
    return (risk >= threshold).astype(int), "proxy_from_recency_frequency_monetary_top20pct"


def _first_non_null(grouped: pd.core.groupby.generic.DataFrameGroupBy, col: str | None, target: str, default: Any) -> pd.DataFrame:
    key = grouped.grouper.names[0]
    if col:
        tmp = grouped[col].first().reset_index().rename(columns={key: "customer_id", col: target})
        tmp[target] = tmp[target].fillna(default)
        return tmp
    return pd.DataFrame({"customer_id": grouped.size().index.astype(str), target: default})


def _canonicalize_transaction_table(df: pd.DataFrame, mapping: SchemaMapping, diagnostics: dict[str, Any]) -> pd.DataFrame:
    now = pd.Timestamp.now(tz=None).floor("D")
    cid_col = mapping.get("customer_id")
    if cid_col is None or cid_col not in df.columns:
        work = df.copy()
        work["__customer_id"] = [f"row_{i:08d}" for i in range(len(work))]
        cid_col = "__customer_id"
        diagnostics["customer_id_generated"] = True
    else:
        work = df.copy()
        diagnostics["customer_id_generated"] = False
    work[cid_col] = work[cid_col].astype(str).fillna("unknown")

    date_col = mapping.get("transaction_date") or mapping.get("last_activity_date")
    if date_col and date_col in work.columns:
        work["__event_ts"] = _parse_event_datetime(work[date_col], column_name=date_col, default=now)
    else:
        # Stable fallback: spread rows over the previous 90 days.
        work["__event_ts"] = now - pd.to_timedelta(np.arange(len(work)) % 90, unit="D")

    amount_col = mapping.get("amount") or mapping.get("monetary_90d")
    if amount_col and amount_col in work.columns:
        work["__amount"] = safe_numeric(work[amount_col], default=0.0, nonnegative=True)
    else:
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        fallback = numeric_cols[0] if numeric_cols else None
        work["__amount"] = safe_numeric(work[fallback], default=0.0, nonnegative=True) if fallback else 0.0
        diagnostics["amount_fallback_column"] = fallback

    qty_col = mapping.get("quantity")
    work["__quantity"] = safe_numeric(work[qty_col], default=1.0, nonnegative=True) if qty_col and qty_col in work.columns else 1.0

    grouped = work.groupby(cid_col, dropna=False)
    customer = grouped.agg(
        frequency_90d=(cid_col, "size"),
        monetary_90d=("__amount", "sum"),
        avg_order_amount=("__amount", "mean"),
        total_quantity=("__quantity", "sum"),
        last_activity_date=("__event_ts", "max"),
        signup_date=("__event_ts", "min"),
    ).reset_index().rename(columns={cid_col: "customer_id"})

    customer["last_activity_date"] = pd.to_datetime(customer["last_activity_date"], errors="coerce").fillna(now)
    signup_src = mapping.get("signup_date")
    if signup_src and signup_src in work.columns:
        signup = grouped[signup_src].first().reset_index().rename(columns={cid_col: "customer_id", signup_src: "__signup"})
        customer = customer.merge(signup, on="customer_id", how="left")
        customer["signup_date"] = pd.to_datetime(customer["__signup"], errors="coerce").fillna(customer["signup_date"]).fillna(now - pd.Timedelta(days=365))
        customer = customer.drop(columns=["__signup"])
    customer["signup_date"] = pd.to_datetime(customer["signup_date"], errors="coerce").fillna(now - pd.Timedelta(days=365))
    customer["recency_days"] = (now - customer["last_activity_date"]).dt.days.clip(lower=0)
    customer["tenure_days"] = (now - customer["signup_date"]).dt.days.clip(lower=1)

    for field, target, default in [
        ("segment", "segment", "general"),
        ("region", "region", "unknown"),
        ("channel", "channel", "unknown"),
        ("device", "device", "unknown"),
        ("category", "category", "general"),
        ("treatment_group", "treatment_group", "control"),
    ]:
        src = mapping.get(field)
        if src and src in work.columns:
            vals = grouped[src].first().reset_index().rename(columns={cid_col: "customer_id", src: target})
            customer = customer.merge(vals, on="customer_id", how="left")
            customer[target] = customer[target].fillna(default)
        else:
            customer[target] = default

    label_src = mapping.get("label")
    if label_src and label_src in work.columns:
        labels = grouped[label_src].agg(lambda s: int(_parse_binary(s, index=s.index).max())).reset_index().rename(columns={cid_col: "customer_id", label_src: "label"})
        customer = customer.merge(labels, on="customer_id", how="left")
        customer["label"] = customer["label"].fillna(0).astype(int)
        diagnostics["label_source"] = f"mapped:{label_src}"
    else:
        label, source = _derive_proxy_label(customer)
        customer["label"] = label
        diagnostics["label_source"] = source

    for field, target in [("converted", "converted"), ("campaign_exposed", "campaign_exposed")]:
        src = mapping.get(field)
        if src and src in work.columns:
            vals = grouped[src].agg(lambda s: float(_parse_binary(s, index=s.index).mean())).reset_index().rename(columns={cid_col: "customer_id", src: target})
            customer = customer.merge(vals, on="customer_id", how="left")
            customer[target] = customer[target].fillna(0.0)
        else:
            customer[target] = 0.0

    return customer


def _canonicalize_customer_table(df: pd.DataFrame, mapping: SchemaMapping, diagnostics: dict[str, Any]) -> pd.DataFrame:
    now = pd.Timestamp.now(tz=None).floor("D")
    index = df.index
    customer = pd.DataFrame(index=index)
    cid = _get(df, mapping, "customer_id")
    if cid is None:
        customer["customer_id"] = [f"row_{i:08d}" for i in range(len(df))]
        diagnostics["customer_id_generated"] = True
    else:
        customer["customer_id"] = cid.astype(str).fillna("unknown")
        diagnostics["customer_id_generated"] = False

    signup = _get(df, mapping, "signup_date")
    last = _get(df, mapping, "last_activity_date")
    if last is None:
        last = _get(df, mapping, "transaction_date")
    customer["signup_date"] = _parse_date(signup, default=now - pd.Timedelta(days=365)) if signup is not None else now - pd.Timedelta(days=365)
    if last is not None:
        customer["last_activity_date"] = _parse_date(last, default=now)
    else:
        recency_src = _get(df, mapping, "recency_days")
        recency = safe_numeric(recency_src, default=30.0, nonnegative=True) if recency_src is not None else pd.Series(30.0, index=index)
        customer["last_activity_date"] = now - pd.to_timedelta(recency.clip(0, 3650), unit="D")

    customer["recency_days"] = safe_numeric(_get(df, mapping, "recency_days"), default=np.nan, nonnegative=True) if _get(df, mapping, "recency_days") is not None else (now - pd.to_datetime(customer["last_activity_date"], errors="coerce")).dt.days
    customer["recency_days"] = safe_numeric(customer["recency_days"], default=30.0, nonnegative=True)
    customer["tenure_days"] = (now - pd.to_datetime(customer["signup_date"], errors="coerce")).dt.days.clip(lower=1).fillna(365)
    customer["frequency_90d"] = safe_numeric(_get(df, mapping, "frequency_90d"), default=1.0, nonnegative=True) if _get(df, mapping, "frequency_90d") is not None else 1.0
    amount_src = _get(df, mapping, "monetary_90d")
    if amount_src is None:
        amount_src = _get(df, mapping, "amount")
    customer["monetary_90d"] = safe_numeric(amount_src, default=0.0, nonnegative=True) if amount_src is not None else 0.0
    customer["avg_order_amount"] = (customer["monetary_90d"] / customer["frequency_90d"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(customer["monetary_90d"])
    customer["total_quantity"] = safe_numeric(_get(df, mapping, "quantity"), default=customer["frequency_90d"].mean() if len(customer) else 1.0, nonnegative=True) if _get(df, mapping, "quantity") is not None else customer["frequency_90d"]

    for field, target, default in [
        ("segment", "segment", "general"),
        ("region", "region", "unknown"),
        ("channel", "channel", "unknown"),
        ("device", "device", "unknown"),
        ("category", "category", "general"),
        ("treatment_group", "treatment_group", "control"),
    ]:
        src = _get(df, mapping, field)
        customer[target] = safe_text(src, default=default, index=index) if src is not None else default

    label_src = _get(df, mapping, "label")
    if label_src is not None:
        customer["label"] = _parse_binary(label_src, index=index)
        diagnostics["label_source"] = f"mapped:{mapping.get('label')}"
    else:
        label, source = _derive_proxy_label(customer)
        customer["label"] = label
        diagnostics["label_source"] = source

    for field in ["converted", "campaign_exposed"]:
        src = _get(df, mapping, field)
        customer[field] = _parse_binary(src, index=index) if src is not None else 0

    return customer.reset_index(drop=True)


def add_compatibility_columns(customer: pd.DataFrame) -> pd.DataFrame:
    out = customer.copy()
    now = pd.Timestamp.now(tz=None).floor("D")

    # Standard names used by both AutoOps and legacy services.
    out["churn"] = out["label"].astype(int)
    out["is_churned"] = out["label"].astype(int)
    out["last_purchase_date"] = pd.to_datetime(out["last_activity_date"], errors="coerce").fillna(now)
    out["assigned_at"] = now - pd.Timedelta(days=7)

    out["persona"] = safe_text(out.get("segment"), default="general", index=out.index)
    out["customer_segment"] = out["persona"]
    out["device_type"] = safe_text(out.get("device"), default="unknown", index=out.index)
    out["acquisition_channel"] = safe_text(out.get("channel"), default="unknown", index=out.index)
    out["preferred_category"] = safe_text(out.get("category"), default="general", index=out.index)

    recency_norm = minmax(out["recency_days"], default=30.0)
    freq_norm = minmax(out["frequency_90d"], default=1.0)
    money_norm = minmax(out["monetary_90d"], default=0.0)
    out["price_sensitivity"] = (0.55 * recency_norm + 0.45 * (1 - money_norm)).clip(0, 1)
    out["coupon_affinity"] = (0.35 + 0.45 * out["price_sensitivity"] + 0.20 * (1 - freq_norm)).clip(0, 1)
    out["discount_pressure_score"] = (0.55 * out["coupon_affinity"] + 0.45 * recency_norm).clip(0, 1)
    out["brand_sensitivity"] = (1 - out["price_sensitivity"]).clip(0, 1)
    out["support_contact_propensity"] = (0.05 + 0.25 * recency_norm).clip(0, 1)

    out["treatment_group"] = safe_text(out.get("treatment_group"), default="control", index=out.index)
    out["treatment_flag"] = out["treatment_group"].astype(str).str.lower().isin(["treatment", "treated", "b", "variant", "test", "1", "실험군"]).astype(int)
    out.loc[out["campaign_exposed"].astype(float) > 0, "treatment_flag"] = 1
    out["campaign_type"] = "universal_autoops"
    out["coupon_cost"] = (3000 + (out["price_sensitivity"] * 7000)).round(0).astype(int)

    # Risk/probability proxies are overwritten by the trainer when available.
    out["proxy_churn_risk"] = (0.55 * recency_norm + 0.25 * (1 - freq_norm) + 0.20 * (1 - money_norm)).clip(0, 1)
    out["churn_probability"] = out["proxy_churn_risk"]
    out["retention_probability"] = (1 - out["churn_probability"]).clip(0, 1)
    out["predicted_uplift"] = (0.20 * out["coupon_affinity"] + 0.15 * out["churn_probability"] - 0.05 * money_norm).clip(-0.1, 0.35)
    out["uplift_score"] = out["predicted_uplift"]
    out["clv"] = (out["monetary_90d"] * (1.0 + out["retention_probability"]) + out["avg_order_amount"] * 2.0).clip(0, 1e9)

    for col in LEGACY_COMPATIBILITY_FIELDS:
        if col not in out.columns:
            out[col] = "unknown"
    return out


def canonicalize_dataframe(df: pd.DataFrame, *, manual_mapping: dict[str, str] | None = None) -> CanonicalizationResult:
    profile = profile_dataframe(df)
    mapping = infer_schema_mapping(df, manual_mapping=manual_mapping)
    grain = infer_grain(
        df,
        customer_col=mapping.get("customer_id"),
        transaction_col=mapping.get("transaction_id"),
        transaction_date_col=mapping.get("transaction_date"),
        amount_col=mapping.get("amount") or mapping.get("monetary_90d"),
    )
    diagnostics: dict[str, Any] = {
        "profile": profile,
        "grain_detection": grain,
        "mapping": mapping.as_dict(),
    }

    if grain["grain"] == "transaction":
        customer = _canonicalize_transaction_table(df, mapping, diagnostics)
    else:
        customer = _canonicalize_customer_table(df, mapping, diagnostics)

    customer = add_compatibility_columns(customer)
    # Keep customer_id unique after canonicalization.
    customer = customer.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)

    feature = customer.copy()
    diagnostics.update({
        "canonical_rows": int(len(customer)),
        "canonical_columns": int(len(customer.columns)),
        "label_rate": float(customer["label"].mean()) if len(customer) else 0.0,
        "churn_proxy_used": str(diagnostics.get("label_source", "")).startswith("proxy"),
    })
    return CanonicalizationResult(
        customer_table=customer,
        feature_table=feature,
        mapping=mapping,
        diagnostics=diagnostics,
        mapping_table=mapping_table(mapping),
    )
