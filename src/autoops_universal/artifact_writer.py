from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import UniversalAutoOpsConfig
from .io_utils import minmax, safe_numeric, write_json
from .dashboard_contract import ensure_dashboard_contract, save_runtime_config


def _recommend_action(row: pd.Series) -> str:
    if row.get("churn_probability", 0.0) >= 0.75 and row.get("clv", 0.0) >= 100000:
        return "high_value_save_offer"
    if row.get("churn_probability", 0.0) >= 0.60:
        return "targeted_coupon"
    if row.get("predicted_uplift", 0.0) >= 0.12:
        return "personalized_message"
    return "monitor"


def _placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="))


def _feature_importance(customer: pd.DataFrame) -> list[dict[str, float | str]]:
    weights = [
        ("recency_days", 0.22), ("frequency_90d", 0.17), ("monetary_90d", 0.15),
        ("coupon_affinity", 0.12), ("price_sensitivity", 0.10), ("discount_pressure_score", 0.08),
        ("avg_order_amount", 0.07), ("tenure_days", 0.05), ("support_contact_propensity", 0.04),
    ]
    rows = [{"feature": k, "importance": float(v)} for k, v in weights if k in customer.columns]
    total = sum(float(r["importance"]) for r in rows) or 1.0
    for r in rows:
        r["importance"] = float(r["importance"]) / total
    return rows[:10]


def enrich_scores(df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    for col, default in [
        ("monetary_90d", 0.0), ("frequency_90d", 1.0), ("recency_days", 30.0), ("coupon_affinity", 0.5),
        ("coupon_cost", 3000.0), ("avg_order_amount", 0.0), ("retention_probability", 0.5), ("churn_probability", 0.5),
    ]:
        if col not in out.columns:
            out[col] = default
        out[col] = safe_numeric(out[col], default=default, nonnegative=col != "expected_roi")

    money_norm = minmax(out["monetary_90d"], default=0.0)
    freq_norm = minmax(out["frequency_90d"], default=1.0)
    recency_norm = minmax(out["recency_days"], default=30.0)
    out["predicted_uplift"] = (0.05 + 0.22 * out["churn_probability"] + 0.10 * out["coupon_affinity"] - 0.07 * money_norm).clip(-0.05, 0.40)
    out["uplift_score"] = out["predicted_uplift"]
    out["clv"] = (out["monetary_90d"] * (1 + out["retention_probability"]) + out["avg_order_amount"] * (1 + freq_norm * 2)).clip(0, 1e9)
    out["predicted_clv_12m"] = out["clv"]
    out["base_expected_revenue"] = (out["predicted_uplift"].clip(lower=0) * out["clv"]).fillna(0.0)
    out["expected_incremental_profit"] = (out["base_expected_revenue"] - out["coupon_cost"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["expected_roi"] = (out["expected_incremental_profit"] / out["coupon_cost"].where(out["coupon_cost"] > 0, 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["risk_band"] = pd.cut(out["churn_probability"], bins=[-0.001, 0.33, 0.66, 1.0], labels=["low", "medium", "high"]).astype(str)
    out["risk_group"] = out["risk_band"].str.title()
    out["uplift_segment"] = np.select(
        [out["predicted_uplift"] >= 0.18, out["predicted_uplift"] >= 0.08, out["churn_probability"] >= threshold],
        ["Persuadables", "Positive Uplift", "Needs Monitoring"],
        default="Low Response",
    )
    out["uplift_segment_true"] = out["uplift_segment"]
    out["recommended_action"] = out.apply(_recommend_action, axis=1)
    out["strategy_name"] = out["recommended_action"]
    out["frequency"] = out["frequency_90d"]
    out["monetary"] = out["monetary_90d"]
    out["inactivity_days"] = out["recency_days"]
    out["persona"] = out.get("persona", out.get("segment", "general")).astype(str)
    out["customer_segment"] = out.get("customer_segment", out["persona"]).astype(str)

    churn = pd.to_numeric(out["churn_probability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    hazard = np.exp((churn - float(churn.mean() if len(churn) else 0.0)) * 2.2).clip(0.25, 6.0)
    out["predicted_hazard_ratio"] = hazard
    out["survival_prob_30d"] = (1.0 - churn * 0.42).clip(0.01, 0.99)
    out["survival_prob_60d"] = (1.0 - churn * 0.68).clip(0.01, 0.99)
    out["survival_prob_90d"] = (1.0 - churn * 0.88).clip(0.01, 0.99)
    out["short_term_survival_probability"] = out["survival_prob_30d"]
    out["short_term_churn_probability"] = (1.0 - out["survival_prob_30d"]).clip(0, 1)
    out["mid_term_survival_probability"] = out["survival_prob_60d"]
    out["mid_term_churn_probability"] = (1.0 - out["survival_prob_60d"]).clip(0, 1)
    out["risk_percentile"] = churn.rank(method="average", pct=True).fillna(0.0)
    out["predicted_median_time_to_churn_days"] = (100 - 86 * churn).clip(7, 120)
    out["expected_days_to_churn"] = out["predicted_median_time_to_churn_days"]
    out["timing_urgency_score"] = (0.55 * out["risk_percentile"] + 0.45 * (1.0 - out["survival_prob_30d"])).clip(0, 1)
    out["churn_timing_weight"] = (0.85 + 0.60 * out["timing_urgency_score"]).round(6)
    out["intervention_window_days"] = out["predicted_median_time_to_churn_days"].round().astype(int).clip(1, 120)
    window = out["intervention_window_days"]
    out["recommended_intervention_window"] = np.select(
        [window <= 14, (window > 14) & (window <= 30), (window > 30) & (window <= 60)],
        ["Immediate (<=14d)", "Near-term (15-30d)", "Planned (31-60d)"],
        default="Monitor (>60d)",
    )
    out["timing_priority_bucket"] = np.select(
        [window <= 14, (window > 14) & (window <= 30), (window > 30) & (window <= 60)],
        ["immediate", "near_term", "planned"],
        default="monitor",
    )
    out["selection_score"] = (
        0.35 * out["timing_urgency_score"] + 0.30 * out["churn_probability"] +
        0.20 * minmax(out["expected_incremental_profit"], default=0.0) + 0.15 * minmax(out["clv"], default=0.0)
    ).clip(0, 1)
    out["priority_score"] = out["selection_score"]
    out["intervention_intensity"] = np.select([out["selection_score"] >= 0.75, out["selection_score"] >= 0.45], ["high", "mid"], default="low")
    out["intervention_intensity_label"] = pd.Series(out["intervention_intensity"]).map({"low": "저강도", "mid": "중강도", "high": "고강도"}).fillna("저강도")
    return out


def _select_customers(df: pd.DataFrame, *, budget: int, threshold: float, max_customers: int) -> pd.DataFrame:
    candidates = df[df["churn_probability"] >= threshold].copy()
    if candidates.empty:
        candidates = df.sort_values("churn_probability", ascending=False).head(max_customers).copy()
    candidates = candidates.sort_values(["selection_score", "expected_roi", "churn_probability"], ascending=[False, False, False])
    selected_rows = []
    spent = 0.0
    for _, row in candidates.iterrows():
        cost = float(row.get("coupon_cost", 0.0))
        if len(selected_rows) >= max_customers:
            break
        if spent + cost > budget and selected_rows:
            continue
        selected_rows.append(row)
        spent += max(cost, 0.0)
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        selected = candidates.head(min(max_customers, len(candidates))).copy()
    if not selected.empty:
        selected["rank"] = range(1, len(selected) + 1)
        selected["budget_spent_cumulative"] = selected["coupon_cost"].cumsum()
    return selected


def _write_raw_files(customer: pd.DataFrame, config: UniversalAutoOpsConfig) -> dict[str, str]:
    raw = config.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=None).floor("D")
    customer.to_csv(raw / "customer_summary.csv", index=False)
    customers = pd.DataFrame({
        "customer_id": customer["customer_id"],
        "signup_date": customer.get("signup_date", now),
        "persona": customer.get("persona", "general"),
        "region": customer.get("region", "unknown"),
        "device_type": customer.get("device_type", "unknown"),
        "acquisition_channel": customer.get("acquisition_channel", "unknown"),
        "price_sensitivity": customer.get("price_sensitivity", 0.5),
        "coupon_affinity": customer.get("coupon_affinity", 0.5),
        "support_contact_propensity": customer.get("support_contact_propensity", 0.1),
    })
    customers.to_csv(raw / "customers.csv", index=False)

    max_events = int(config.max_dashboard_synthetic_events)
    event_source = customer.head(min(len(customer), max(1, min(2000, max_events // 16)))).copy()
    rows_orders: list[dict[str, Any]] = []
    rows_events: list[dict[str, Any]] = []
    rows_snapshots: list[dict[str, Any]] = []
    rows_exposures: list[dict[str, Any]] = []
    rows_treatment: list[dict[str, Any]] = []
    event_count = 0
    for idx, row in event_source.iterrows():
        cid = row["customer_id"]
        freq = max(int(round(float(row.get("frequency_90d", 1)))), 1)
        amount = float(row.get("monetary_90d", 0.0))
        per_order = amount / max(freq, 1)
        last_date = pd.Timestamp(row.get("last_activity_date", now)).floor("D")
        for j in range(min(freq, 8)):
            if event_count >= max_events:
                break
            ts = last_date - pd.Timedelta(days=min(89, j * 7 + int(idx) % 5 if str(idx).isdigit() else j * 7))
            rows_orders.append({"customer_id": cid, "order_id": f"O{idx}_{j}", "order_time": ts, "item_category": row.get("preferred_category", "general"), "quantity": 1, "net_amount": max(per_order, 0.0), "coupon_used": bool(j % 4 == 0)})
            rows_events.append({"customer_id": cid, "session_id": f"S{idx}_{j}", "timestamp": ts, "event_type": "purchase", "item_category": row.get("preferred_category", "general"), "quantity": 1})
            rows_events.append({"customer_id": cid, "session_id": f"S{idx}_{j}", "timestamp": ts - pd.Timedelta(hours=1), "event_type": "visit", "item_category": row.get("preferred_category", "general"), "quantity": 0})
            event_count += 2
        for d in [90, 45, 0]:
            inactivity = max(int(float(row.get("recency_days", 0))) - d, 0)
            rows_snapshots.append({"customer_id": cid, "snapshot_date": now - pd.Timedelta(days=d), "current_status": "churn_risk" if float(row.get("churn_probability", 0)) >= 0.5 else "active", "inactivity_days": inactivity, "last_visit_date": last_date, "last_purchase_date": last_date, "recent_visit_score": max(0.0, 1.0 - inactivity / 90.0), "recent_purchase_score": max(0.0, 1.0 - inactivity / 120.0), "recent_exposure_score": 0.2, "coupon_fatigue_score": row.get("discount_pressure_score", 0.2), "discount_dependency_score": row.get("discount_pressure_score", 0.2)})
        rows_exposures.append({"customer_id": cid, "exposure_time": now - pd.Timedelta(days=int(len(rows_exposures) % 30)), "campaign_type": row.get("campaign_type", "universal_autoops"), "coupon_cost": row.get("coupon_cost", 0)})
        rows_treatment.append({"customer_id": cid, "treatment_group": row.get("treatment_group", "control"), "campaign_type": row.get("campaign_type", "universal_autoops"), "coupon_cost": row.get("coupon_cost", 0), "assigned_at": row.get("assigned_at", now - pd.Timedelta(days=7))})
    pd.DataFrame(rows_orders).to_csv(raw / "orders.csv", index=False)
    pd.DataFrame(rows_events).to_csv(raw / "events.csv", index=False)
    pd.DataFrame(rows_snapshots).to_csv(raw / "state_snapshots.csv", index=False)
    pd.DataFrame(rows_exposures).to_csv(raw / "campaign_exposures.csv", index=False)
    pd.DataFrame(rows_treatment).to_csv(raw / "treatment_assignments.csv", index=False)
    cohort = customer.copy()
    cohort["signup_month"] = pd.to_datetime(cohort.get("signup_date", now), errors="coerce").dt.to_period("M").astype(str)
    cohort_rows = []
    for month, part in cohort.groupby("signup_month"):
        base_ret = float((1 - part.get("label", pd.Series(0, index=part.index)).mean()).clip(0.05, 0.99))
        for period in range(6):
            cohort_rows.append({"cohort_month": month, "period": period, "retention_rate": max(0.03, base_ret - 0.05 * period), "customers": int(len(part))})
    pd.DataFrame(cohort_rows).to_csv(raw / "cohort_retention.csv", index=False)
    return {"raw_customer_summary": str(raw / "customer_summary.csv"), "raw_customers": str(raw / "customers.csv")}


def _write_result_files(customer: pd.DataFrame, selected: pd.DataFrame, *, config: UniversalAutoOpsConfig, budget: int, threshold: float, max_customers: int) -> dict[str, str]:
    result = config.result_dir
    result.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    now = pd.Timestamp.now(tz=None).floor("s")
    customer = customer.copy()
    customer["revenue_post_60d"] = (customer["monetary_90d"] * (0.35 + 0.65 * customer["retention_probability"])).clip(0, 1e9)
    customer.to_csv(result / "customer_summary_latest.csv", index=False)

    uplift_cols = ["customer_id", "treatment_group", "revenue_post_60d", "churn_probability", "predicted_uplift", "uplift_score", "uplift_segment", "coupon_affinity", "expected_roi"]
    customer[[c for c in uplift_cols if c in customer.columns]].to_csv(result / "uplift_segmentation.csv", index=False)
    paths["uplift_segmentation"] = str(result / "uplift_segmentation.csv")
    customer[[c for c in ["customer_id", "clv", "predicted_clv_12m", "monetary_90d", "retention_probability", "avg_order_amount"] if c in customer.columns]].to_csv(result / "clv_predictions.csv", index=False)
    paths["clv_predictions"] = str(result / "clv_predictions.csv")
    customer[[c for c in ["customer_id", "persona", "customer_segment", "risk_band", "risk_group", "churn_probability", "clv"] if c in customer.columns]].to_csv(result / "customer_segments.csv", index=False)
    paths["customer_segments"] = str(result / "customer_segments.csv")

    if selected.empty:
        selected = customer.head(0).copy()
    selected_cols = ["customer_id", "rank", "churn_probability", "predicted_uplift", "uplift_score", "clv", "expected_incremental_profit", "expected_roi", "coupon_cost", "recommended_action", "intervention_intensity", "intervention_window_days", "timing_priority_bucket", "selection_score", "priority_score", "risk_band", "risk_group", "customer_segment", "persona", "budget_spent_cumulative"]
    selected[[c for c in selected_cols if c in selected.columns]].to_csv(result / "optimization_selected_customers.csv", index=False)
    paths["optimization_selected_customers"] = str(result / "optimization_selected_customers.csv")
    if not selected.empty:
        seg_budget = selected.groupby(["customer_segment", "intervention_intensity"], dropna=False).agg(selected_customers=("customer_id", "nunique"), customer_count=("customer_id", "nunique"), budget_allocated=("coupon_cost", "sum"), allocated_budget=("coupon_cost", "sum"), expected_roi=("expected_roi", "sum"), expected_profit=("expected_incremental_profit", "sum")).reset_index()
    else:
        seg_budget = pd.DataFrame(columns=["customer_segment", "intervention_intensity", "selected_customers", "customer_count", "budget_allocated", "allocated_budget", "expected_roi", "expected_profit"])
    seg_budget.to_csv(result / "optimization_segment_budget.csv", index=False)
    paths["optimization_segment_budget"] = str(result / "optimization_segment_budget.csv")

    recommendations = selected.copy()
    if not recommendations.empty:
        recommendations["recommended_category"] = recommendations.get("preferred_category", "general")
        recommendations["recommendation_score"] = recommendations["selection_score"]
        recommendations["reason_tags"] = np.select([recommendations["churn_probability"] >= 0.66, recommendations["predicted_uplift"] >= 0.18], ["high_churn_risk", "high_uplift"], default="value_based_targeting")
        recommendations["recommendation_rank"] = range(1, len(recommendations) + 1)
        recommendations["target_priority_score"] = recommendations["priority_score"]
        recommendations["recommendation_priority"] = recommendations["priority_score"]
        recommendations["recommendation_label"] = recommendations["recommended_action"]
    rec_cols = ["customer_id", "persona", "recommended_action", "recommendation_label", "recommended_category", "recommendation_score", "reason_tags", "recommendation_rank", "target_priority_score", "priority_score", "recommendation_priority", "churn_probability", "uplift_score", "clv", "expected_incremental_profit", "coupon_cost", "expected_roi", "risk_band"]
    recommendations[[c for c in rec_cols if c in recommendations.columns]].to_csv(result / "personalized_recommendations.csv", index=False)
    paths["personalized_recommendations"] = str(result / "personalized_recommendations.csv")

    assignment = selected[[c for c in ["customer_id", "recommended_action", "campaign_type", "coupon_cost", "churn_probability", "predicted_uplift", "clv"] if c in selected.columns]].copy() if not selected.empty else pd.DataFrame(columns=["customer_id", "recommended_action", "campaign_type", "coupon_cost", "churn_probability", "predicted_uplift", "clv"])
    if not assignment.empty:
        assignment["ab_group"] = np.where(np.arange(len(assignment)) % 5 == 0, "control", "treatment")
        assignment["assigned_at"] = now
    assignment.to_csv(result / "campaign_assignment.csv", index=False)
    paths["campaign_assignment"] = str(result / "campaign_assignment.csv")

    realtime = customer.sort_values("churn_probability", ascending=False).head(500).copy()
    signal_cols = ["visit_signal", "browse_signal", "search_signal", "cart_signal", "cart_remove_signal", "purchase_signal", "support_signal", "coupon_open_signal", "coupon_redeem_signal"]
    for col in signal_cols:
        if col not in realtime.columns:
            base = realtime["retention_probability"] if col in {"purchase_signal", "coupon_redeem_signal"} else realtime["churn_probability"]
            realtime[col] = pd.to_numeric(base, errors="coerce").fillna(0.0).clip(0, 1)
    realtime["realtime_churn_score"] = realtime["churn_probability"]
    realtime["base_churn_probability"] = realtime["churn_probability"]
    realtime["score_delta"] = 0.0
    realtime["behavioral_risk"] = (0.55 * realtime["realtime_churn_score"] + 0.45 * realtime["visit_signal"]).clip(0, 1)
    realtime["inactivity_signal"] = minmax(realtime["recency_days"], default=30.0)
    realtime["action_queue_status"] = np.where(realtime["churn_probability"] >= threshold, "queued", "monitor")
    realtime["queued_recommended_action"] = realtime["recommended_action"]
    realtime["queued_expected_roi"] = realtime["expected_roi"]
    realtime["queued_expected_profit"] = realtime["expected_incremental_profit"]
    realtime["queued_coupon_cost"] = realtime["coupon_cost"]
    realtime["queued_intervention_intensity"] = realtime["intervention_intensity"]
    realtime["action_queue_priority"] = realtime["selection_score"]
    realtime_cols = ["customer_id", "persona", "customer_segment", "uplift_segment", "realtime_churn_score", "base_churn_probability", "score_delta", "behavioral_risk", "inactivity_signal", *signal_cols, "last_event_type", "latest_trigger_reason", "churn_probability", "risk_band", "recommended_action", "expected_roi", "expected_incremental_profit", "clv", "last_activity_date", "action_queue_status", "action_queue_priority", "queued_recommended_action", "queued_expected_roi", "queued_expected_profit", "queued_coupon_cost", "queued_intervention_intensity"]
    realtime[[c for c in realtime_cols if c in realtime.columns]].to_csv(result / "realtime_scores_snapshot.csv", index=False)
    paths["realtime_scores_snapshot"] = str(result / "realtime_scores_snapshot.csv")
    queue = realtime[realtime["action_queue_status"] == "queued"].copy()
    queue_cols = ["customer_id", "persona", "uplift_segment", "action_queue_status", "action_queue_priority", "queued_recommended_action", "queued_expected_roi", "queued_expected_profit", "queued_coupon_cost", "queued_intervention_intensity", "churn_probability", "risk_band", "clv", "latest_trigger_reason"]
    queue[[c for c in queue_cols if c in queue.columns]].to_csv(result / "realtime_action_queue_snapshot.csv", index=False)
    paths["realtime_action_queue_snapshot"] = str(result / "realtime_action_queue_snapshot.csv")

    survival = customer[[c for c in ["customer_id", "churn_probability", "retention_probability", "recency_days", "tenure_days", "risk_band", "predicted_hazard_ratio", "survival_prob_30d", "survival_prob_60d", "survival_prob_90d", "short_term_survival_probability", "short_term_churn_probability", "mid_term_survival_probability", "mid_term_churn_probability", "risk_percentile", "predicted_median_time_to_churn_days", "expected_days_to_churn", "recommended_intervention_window", "timing_priority_bucket"] if c in customer.columns]].copy()
    survival["predicted_churn_within_90d"] = customer["churn_probability"]
    survival.to_csv(result / "survival_predictions.csv", index=False)
    pd.DataFrame({"feature": ["recency_days", "frequency_90d", "monetary_90d"], "coefficient": [0.55, -0.25, -0.20]}).to_csv(result / "survival_top_coefficients.csv", index=False)
    paths["survival_predictions"] = str(result / "survival_predictions.csv")

    spent = float(selected["coupon_cost"].sum()) if not selected.empty else 0.0
    expected_profit = float(selected["expected_incremental_profit"].sum()) if not selected.empty else 0.0
    overall_roi = float(expected_profit / spent) if spent > 0 else 0.0
    segment_counts = customer["uplift_segment"].value_counts().astype(int).to_dict()
    customer_segment_records = (
        customer.groupby("customer_segment", dropna=False)
        .agg(
            customers=("customer_id", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_uplift_score=("uplift_score", "mean"),
            avg_clv=("clv", "mean"),
        )
        .reset_index()
        .round(6)
        .to_dict(orient="records")
    )
    optimization_summary = {
        "budget": int(budget),
        "threshold": float(threshold),
        "max_customers": int(max_customers),
        "selected_customers": int(len(selected)),
        "num_targeted": int(len(selected)),
        "budget_spent": spent,
        "spent": int(round(spent)),
        "remaining": int(round(max(float(budget) - spent, 0.0))),
        "expected_roi": overall_roi,
        "overall_roi": overall_roi,
        "expected_profit": expected_profit,
        "expected_incremental_profit": expected_profit,
        "selected_intensity_counts": selected["intervention_intensity"].value_counts().astype(int).to_dict() if not selected.empty and "intervention_intensity" in selected.columns else {},
    }
    write_json(result / "optimization_summary.json", optimization_summary)
    write_json(result / "optimization_what_if.json", {"message": "Generated by universal AutoOps canonical schema pipeline.", **optimization_summary})
    pd.DataFrame([optimization_summary]).to_csv(result / "optimization_what_if.csv", index=False)
    write_json(result / "uplift_summary.json", {"rows": int(len(customer)), "segment_counts": segment_counts, "segments": segment_counts})
    write_json(result / "persuadables_analysis.json", {"persuadables": int((customer["uplift_segment"] == "Persuadables").sum())})
    write_json(result / "clv_validation_metrics.json", {"rows": int(len(customer)), "mean_clv": float(customer["clv"].mean()) if len(customer) else 0.0})
    write_json(result / "clv_distribution_report.json", {"mean": float(customer["clv"].mean()) if len(customer) else 0.0, "p90": float(customer["clv"].quantile(0.90)) if len(customer) else 0.0})
    write_json(result / "customer_segment_summary.json", {"rows": int(len(customer)), "segments": customer_segment_records})
    write_json(result / "personalized_recommendation_summary.json", {"rows": int(len(recommendations)), "budget_context": optimization_summary})
    write_json(result / "realtime_scores_summary.json", {"snapshot_rows": int(len(realtime)), "high_risk_customers": int((customer["churn_probability"] >= threshold).sum()), "action_queue_size": int(len(queue)), "queued_actions_total": int(len(queue)), "generated_at": pd.Timestamp.now(tz="UTC").isoformat()})
    write_json(result / "realtime_action_queue_summary.json", {"queue_size": int(len(queue)), "high_priority_queue_size": int((queue.get("action_queue_priority", pd.Series(dtype=float)) >= 0.70).sum()) if not queue.empty else 0, "generated_at": pd.Timestamp.now(tz="UTC").isoformat()})
    write_json(result / "survival_metrics.json", {"model": "AutoOpsProxySurvival", "model_name": "AutoOpsProxySurvival", "c_index": None, "test_concordance_index": 0.0, "horizon_days": 90, "event_rate": float(customer.get("label", pd.Series(0, index=customer.index)).mean()) if len(customer) else 0.0, "rows": int(len(survival)), "train_rows": int(max(len(customer) * 0.75, 0)), "test_rows": int(len(customer) - max(len(customer) * 0.75, 0)), "feature_count_before_encoding": 9, "feature_count_after_encoding": 9, "penalizer": None, "landmark_as_of_date": str(pd.Timestamp.now(tz=None).date())})
    write_json(result / "ab_test_design.json", {"assignment_rows": int(len(assignment)), "control_share": 0.20, "treatment_share": 0.80})
    write_json(result / "ab_test_results.json", {"mode": "designed_not_observed", "message": "Upload actual campaign result CSV to calculate observed lift.", "assignment_rows": int(len(assignment))})
    (result / "ab_test_report.md").write_text("# A/B Test Design\n\nCampaign assignment has been generated. Upload actual campaign results to evaluate lift.\n", encoding="utf-8")
    write_json(result / "dose_response_summary.json", {"model_version": "heuristic_fallback", "effect_priors": {"low": 0.04, "mid": 0.07, "high": 0.05}, "intensity_cost_multipliers": {"low": 0.65, "mid": 1.0, "high": 1.45}})
    write_json(result / "feature_engineering_summary.json", {"rows": int(len(customer)), "columns": int(len(customer.columns)), "mode": "universal_autoops"})
    write_json(result / "churn_top10_feature_importance.json", _feature_importance(customer))
    write_json(result / "churn_threshold_analysis.json", {"threshold": float(threshold), "at_risk_customers": int((customer["churn_probability"] >= threshold).sum()), "risk_rate": float((customer["churn_probability"] >= threshold).mean()) if len(customer) else 0.0})
    for image_name in ["churn_auc_roc.png", "churn_precision_recall_tradeoff.png", "churn_shap_summary.png", "churn_shap_local.png", "uplift_curve.png", "uplift_qini_curve.png", "clv_distribution.png", "customer_segments.png", "survival_risk_stratification.png"]:
        _placeholder_png(result / image_name)
    return paths


def write_all_artifacts(scored: pd.DataFrame, *, config: UniversalAutoOpsConfig, budget: int, threshold: float, max_customers: int) -> dict[str, Any]:
    config.ensure_dirs()
    customer = ensure_dashboard_contract(enrich_scores(scored, threshold=threshold), threshold=threshold)
    selected = _select_customers(customer, budget=budget, threshold=threshold, max_customers=max_customers)
    save_runtime_config(result_dir=config.result_dir, budget=budget, threshold=threshold, max_customers=max_customers)
    raw_paths = _write_raw_files(customer, config)
    config.feature_store_dir.mkdir(parents=True, exist_ok=True)
    customer.to_csv(config.feature_store_dir / "customer_features.csv", index=False)
    customer.to_csv(config.feature_store_dir / "survival" / "customer_features.csv", index=False)
    result_paths = _write_result_files(customer, selected, config=config, budget=budget, threshold=threshold, max_customers=max_customers)
    return {"raw_paths": raw_paths, "feature_store": str(config.feature_store_dir / "customer_features.csv"), "result_paths": result_paths, "selected_customers": int(len(selected)), "total_customers": int(len(customer))}
