from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _num(s: Any, default: float = 0.0, nonnegative: bool = False, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    else:
        out = pd.Series(default, index=index if index is not None else pd.RangeIndex(0))
    if nonnegative:
        out = out.clip(lower=0)
    return out


def _minmax(s: pd.Series, default: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(default, index=s.index, dtype=float)
    x = x.fillna(x.median())
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-12:
        return pd.Series(default, index=s.index, dtype=float)
    return ((x - lo) / (hi - lo)).clip(0, 1)


def _stable_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    vals = series.astype(str).fillna("")
    hashed = pd.util.hash_pandas_object(vals, index=False).astype("uint64")
    return pd.Series([labels[int(v % len(labels))] for v in hashed], index=series.index)


def ensure_dashboard_contract(df: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
    """Return a dashboard-safe frame.

    This is intentionally defensive: uploaded datasets rarely contain every CLV,
    uplift, survival, real-time, and recommendation field. We preserve real fields
    when present and synthesize clearly proxy-based fields only to keep the app
    renderable.
    """
    out = df.copy()
    if out.empty:
        return out
    idx = out.index
    if "customer_id" not in out.columns:
        out["customer_id"] = [f"customer_{i:08d}" for i in range(len(out))]
    out["customer_id"] = out["customer_id"].astype(str)

    for col, default in [("recency_days", 30.0), ("frequency_90d", 1.0), ("monetary_90d", 0.0), ("avg_order_amount", 0.0), ("total_quantity", 1.0), ("tenure_days", 365.0), ("coupon_affinity", 0.5), ("price_sensitivity", 0.5), ("discount_pressure_score", 0.2), ("coupon_cost", 3000.0)]:
        if col not in out.columns:
            out[col] = default
        out[col] = _num(out[col], default, nonnegative=col not in {"expected_roi"}, index=idx)

    now = pd.Timestamp.now(tz=None).floor("D")
    if "last_activity_date" not in out.columns:
        out["last_activity_date"] = now - pd.to_timedelta(out["recency_days"].clip(0, 3650), unit="D")
    if "signup_date" not in out.columns:
        out["signup_date"] = now - pd.to_timedelta(out["tenure_days"].clip(1, 3650), unit="D")

    rec = _minmax(out["recency_days"], 0.35)
    low_freq = 1.0 - _minmax(out["frequency_90d"], 0.45)
    low_money = 1.0 - _minmax(out["monetary_90d"], 0.45)
    risk_proxy = (0.55 * rec + 0.25 * low_freq + 0.20 * low_money).clip(0.02, 0.98)
    if "churn_probability" not in out.columns or pd.to_numeric(out["churn_probability"], errors="coerce").nunique(dropna=True) <= 2:
        raw = _num(out.get("churn_probability", risk_proxy), 0.5, index=idx).clip(0, 1)
        out["churn_probability"] = (0.35 * raw + 0.65 * risk_proxy).clip(0.01, 0.99)
    else:
        out["churn_probability"] = _num(out["churn_probability"], 0.5, index=idx).clip(0.01, 0.99)
    out["retention_probability"] = _num(out.get("retention_probability", 1 - out["churn_probability"]), 0.5, index=idx).clip(0.01, 0.99)

    money_norm = _minmax(out["monetary_90d"], 0.4)
    freq_norm = _minmax(out["frequency_90d"], 0.4)
    if "uplift_score" not in out.columns and "predicted_uplift" not in out.columns:
        out["uplift_score"] = (0.04 + 0.20 * out["churn_probability"] + 0.10 * out["coupon_affinity"] - 0.06 * money_norm).clip(-0.03, 0.42)
    else:
        src = out["uplift_score"] if "uplift_score" in out.columns else out["predicted_uplift"]
        out["uplift_score"] = _num(src, 0.08, index=idx).clip(-0.10, 0.50)
    out["predicted_uplift"] = out["uplift_score"]

    if "clv" not in out.columns:
        out["clv"] = (out["monetary_90d"] * (1.0 + out["retention_probability"]) + out["avg_order_amount"] * (1.0 + 2.0 * freq_norm)).clip(0, 1e12)
    out["clv"] = _num(out["clv"], 0.0, nonnegative=True, index=idx)
    out["predicted_clv_12m"] = out.get("predicted_clv_12m", out["clv"])
    if "expected_incremental_profit" not in out.columns:
        out["expected_incremental_profit"] = (out["clv"] * out["uplift_score"].clip(lower=0) - out["coupon_cost"]).fillna(0)
    out["expected_incremental_profit"] = _num(out["expected_incremental_profit"], 0.0, index=idx)
    if "expected_roi" not in out.columns:
        out["expected_roi"] = out["expected_incremental_profit"] / out["coupon_cost"].where(out["coupon_cost"] > 0, 1.0)
    out["expected_roi"] = _num(out["expected_roi"], 0.0, index=idx).clip(-10, 100)

    persona_labels = ["new_value_seekers", "loyal_regulars", "at_risk_vip", "discount_sensitive", "dormant_low_value"]
    if "persona" not in out.columns or out["persona"].astype(str).nunique(dropna=True) <= 1:
        base = out["customer_id"].astype(str) + "|" + out["risk_band"].astype(str) if "risk_band" in out.columns else out["customer_id"].astype(str)
        out["persona"] = _stable_bucket(base, persona_labels)
        out.loc[(out["churn_probability"] >= 0.65) & (money_norm >= 0.70), "persona"] = "at_risk_vip"
        out.loc[(out["churn_probability"] < 0.35) & (freq_norm >= 0.60), "persona"] = "loyal_regulars"
    else:
        out["persona"] = out["persona"].astype(str).replace({"nan": "general", "None": "general"})
    out["customer_segment"] = out.get("customer_segment", out["persona"]).astype(str)
    if "segment" not in out.columns:
        out["segment"] = out["customer_segment"]
    for col, val in [("region", "unknown"), ("channel", "unknown"), ("device", "unknown"), ("category", "general"), ("preferred_category", "general")]:
        if col not in out.columns:
            out[col] = val
        if out[col].astype(str).nunique(dropna=True) <= 1 and col in {"category", "preferred_category"}:
            out[col] = _stable_bucket(out["customer_id"], ["essential", "premium", "seasonal", "coupon", "reorder"])
        else:
            out[col] = out[col].astype(str)

    out["risk_band"] = pd.cut(out["churn_probability"], bins=[-0.001, 0.33, 0.66, 1.001], labels=["low", "medium", "high"]).astype(str)
    out["risk_group"] = out["risk_band"].str.title()
    out["uplift_segment"] = np.select([out["uplift_score"] >= 0.18, out["uplift_score"] >= 0.08, out["churn_probability"] >= threshold], ["Persuadables", "Positive Uplift", "Needs Monitoring"], default="Low Response")
    out["uplift_segment_true"] = out["uplift_segment"]
    out["recommended_action"] = np.select([out["churn_probability"] >= 0.75, out["uplift_score"] >= 0.18, out["churn_probability"] >= threshold], ["high_value_save_offer", "personalized_coupon", "targeted_message"], default="monitor")
    out["strategy_name"] = out["recommended_action"]

    churn = out["churn_probability"].clip(0, 1)
    out["base_churn_probability"] = out.get("base_churn_probability", churn)
    out["score_delta"] = _num(out.get("score_delta", 0.0), 0.0, index=idx)
    out["realtime_churn_score"] = _num(out.get("realtime_churn_score", (churn + out["score_delta"]).clip(0, 1)), 0.5, index=idx).clip(0, 1)
    out["behavioral_risk"] = _num(out.get("behavioral_risk", 0.65 * out["realtime_churn_score"] + 0.35 * rec), 0.5, index=idx).clip(0, 1)
    out["inactivity_signal"] = _num(out.get("inactivity_signal", rec), 0.0, index=idx).clip(0, 1)
    signal_defaults = {"visit_signal": 1-rec, "browse_signal": 1-rec*0.7, "search_signal": out["coupon_affinity"], "cart_signal": freq_norm, "cart_remove_signal": rec*0.5, "purchase_signal": 1-rec, "support_signal": out.get("support_contact_propensity", pd.Series(0.1, index=idx)), "coupon_open_signal": out["coupon_affinity"], "coupon_redeem_signal": out["uplift_score"].clip(0,1)}
    for col, default in signal_defaults.items():
        out[col] = _num(out.get(col, default), 0.0, index=idx).clip(0, 1)
    event_choices = np.array(["visit", "browse", "search", "cart", "purchase", "coupon_open", "support"])
    if "last_event_type" not in out.columns:
        event_idx = pd.util.hash_pandas_object(out["customer_id"], index=False).astype("uint64") % len(event_choices)
        out["last_event_type"] = event_choices[event_idx.astype(int)]
    if "latest_trigger_reason" not in out.columns:
        out["latest_trigger_reason"] = np.select([out["score_delta"] > 0.05, out["inactivity_signal"] > 0.65, out["coupon_open_signal"] > 0.65], ["risk_score_jump", "inactivity_spike", "coupon_interest"], default="periodic_monitoring")
    out["action_queue_status"] = out.get("action_queue_status", np.where(out["realtime_churn_score"] >= threshold, "queued", "monitor"))
    out["queued_recommended_action"] = out.get("queued_recommended_action", out["recommended_action"])
    out["queued_expected_roi"] = _num(out.get("queued_expected_roi", out["expected_roi"]), 0.0, index=idx)
    out["queued_expected_profit"] = _num(out.get("queued_expected_profit", out["expected_incremental_profit"]), 0.0, index=idx)
    out["queued_coupon_cost"] = _num(out.get("queued_coupon_cost", out["coupon_cost"]), 0.0, nonnegative=True, index=idx)
    out["action_queue_priority"] = _num(out.get("action_queue_priority", out.get("selection_score", churn)), 0.0, index=idx).clip(0, 1)

    out["survival_prob_30d"] = _num(out.get("survival_prob_30d", 1.0 - churn * 0.42), 0.5, index=idx).clip(0.01, 0.99)
    out["survival_prob_60d"] = _num(out.get("survival_prob_60d", 1.0 - churn * 0.68), 0.5, index=idx).clip(0.01, 0.99)
    out["survival_prob_90d"] = _num(out.get("survival_prob_90d", 1.0 - churn * 0.88), 0.5, index=idx).clip(0.01, 0.99)
    out["predicted_hazard_ratio"] = _num(out.get("predicted_hazard_ratio", np.exp((churn - churn.mean()) * 2.0)), 1.0, index=idx).clip(0.1, 10)
    out["predicted_median_time_to_churn_days"] = _num(out.get("predicted_median_time_to_churn_days", 100 - 86 * churn), 60, nonnegative=True, index=idx).clip(7, 180)
    out["expected_days_to_churn"] = out.get("expected_days_to_churn", out["predicted_median_time_to_churn_days"])
    out["timing_urgency_score"] = _num(out.get("timing_urgency_score", 0.55 * churn + 0.45 * (1 - out["survival_prob_30d"])), 0.0, index=idx).clip(0, 1)
    out["intervention_window_days"] = _num(out.get("intervention_window_days", out["predicted_median_time_to_churn_days"]), 30, nonnegative=True, index=idx).round().astype(int).clip(1, 180)
    out["recommended_intervention_window"] = np.select([out["intervention_window_days"] <= 14, out["intervention_window_days"] <= 30, out["intervention_window_days"] <= 60], ["Immediate (<=14d)", "Near-term (15-30d)", "Planned (31-60d)"], default="Monitor (>60d)")
    out["timing_priority_bucket"] = np.select([out["intervention_window_days"] <= 14, out["intervention_window_days"] <= 30, out["intervention_window_days"] <= 60], ["immediate", "near_term", "planned"], default="monitor")
    out["selection_score"] = _num(out.get("selection_score", 0.35 * out["timing_urgency_score"] + 0.30 * churn + 0.20 * _minmax(out["expected_incremental_profit"], 0.0) + 0.15 * _minmax(out["clv"], 0.0)), 0.0, index=idx).clip(0, 1)
    out["priority_score"] = _num(out.get("priority_score", out["selection_score"]), 0.0, index=idx).clip(0, 1)
    out["intervention_intensity"] = np.select([out["selection_score"] >= 0.75, out["selection_score"] >= 0.45], ["high", "mid"], default="low")
    out["intervention_intensity_label"] = pd.Series(out["intervention_intensity"], index=idx).map({"low": "저강도", "mid": "중강도", "high": "고강도"}).fillna("저강도")
    return out


def load_runtime_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path.cwd()
    for path in [root / "results" / "dashboard_runtime_config.json", root / "data" / "feature_store" / "customer_features_metadata.json", root / "results" / "platform_pipeline_status.json"]:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


def save_runtime_config(*, result_dir: str | Path = "results", budget: int, threshold: float, max_customers: int) -> None:
    path = Path(result_dir) / "dashboard_runtime_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"budget": int(budget), "threshold": float(threshold), "max_customers": int(max_customers)}, ensure_ascii=False, indent=2), encoding="utf-8")
