from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.autoops_universal.dashboard_contract import ensure_dashboard_contract


INTENSITY_ORDER = ['low', 'mid', 'high']


@dataclass(frozen=True)
class DashboardInsightBundle:
    result_dir: str
    data_dir: str
    customer_summary: pd.DataFrame
    customers: pd.DataFrame
    events: pd.DataFrame
    orders: pd.DataFrame
    campaign_exposures: pd.DataFrame
    state_snapshots: pd.DataFrame
    treatment_assignments: pd.DataFrame
    optimization_selected_customers: pd.DataFrame
    personalized_recommendations: pd.DataFrame
    realtime_scores: pd.DataFrame
    realtime_action_queue: pd.DataFrame
    survival_predictions: pd.DataFrame
    survival_coefficients: pd.DataFrame
    survival_metrics: Dict[str, Any]
    uplift_segmentation: pd.DataFrame
    top_feature_importance: pd.DataFrame
    ab_test_results: Dict[str, Any]
    dose_response_summary: Dict[str, Any]
    customer_segment_summary: Dict[str, Any]
    persuadables_analysis: Dict[str, Any]
    optimization_summary: Dict[str, Any]
    realtime_scores_summary: Dict[str, Any]
    realtime_action_queue_summary: Dict[str, Any]
    personalized_recommendation_summary: Dict[str, Any]
    clv_validation_metrics: Dict[str, Any]
    feature_engineering_summary: Dict[str, Any]
    churn_metrics: Dict[str, Any]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _complete_intensity_mix(selected: pd.DataFrame) -> pd.DataFrame:
    counts = {intensity: 0 for intensity in INTENSITY_ORDER}
    if not selected.empty and 'intervention_intensity' in selected.columns:
        observed = selected['intervention_intensity'].fillna('unknown').astype(str).str.lower().value_counts()
        for intensity in INTENSITY_ORDER:
            counts[intensity] = int(observed.get(intensity, 0))
    return pd.DataFrame({
        'intervention_intensity': INTENSITY_ORDER,
        'count': [counts[intensity] for intensity in INTENSITY_ORDER],
    })


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


CSV_USECOLS: dict[str, list[str]] = {
    "customer_summary.csv": [
        "customer_id", "persona", "region", "device_type", "acquisition_channel",
        "churn_probability", "uplift_score", "clv", "coupon_cost", "expected_incremental_profit",
        "expected_roi", "uplift_segment", "treatment_group", "coupon_exposure_count",
        "coupon_redeem_count", "inactivity_days", "coupon_fatigue_score", "discount_dependency_score",
        "discount_pressure_score", "discount_effect_penalty", "price_sensitivity", "coupon_affinity",
        "support_contact_propensity", "coupon_exposures", "coupon_opens", "coupon_redeemed",
    ],
    "customers.csv": ["customer_id", "persona", "region", "device_type", "acquisition_channel"],
    "events.csv": ["customer_id", "event_type"],
    "orders.csv": ["customer_id"],
    "campaign_exposures.csv": ["customer_id", "coupon_cost"],
    "state_snapshots.csv": ["customer_id"],
    "treatment_assignments.csv": ["customer_id"],
    "optimization_selected_customers.csv": [
        "customer_id", "recommended_action", "intervention_intensity", "intervention_window_days",
        "expected_incremental_profit", "selection_score", "timing_priority_bucket", "expected_roi", "priority_score"
    ],
    "personalized_recommendations.csv": [
        "customer_id", "persona", "recommended_category", "recommendation_score", "reason_tags",
        "recommendation_rank", "target_priority_score", "recommendation_priority", "churn_probability",
        "uplift_score", "clv", "expected_incremental_profit", "coupon_cost", "expected_roi"
    ],
    "realtime_scores_snapshot.csv": [
        "customer_id", "persona", "uplift_segment", "realtime_churn_score", "action_queue_status",
        "latest_trigger_reason", "visit_signal", "browse_signal", "search_signal", "cart_signal",
        "cart_remove_signal", "purchase_signal", "support_signal", "coupon_open_signal",
        "coupon_redeem_signal", "behavioral_risk", "inactivity_signal", "queued_recommended_action",
        "queued_expected_roi", "queued_expected_profit", "queued_coupon_cost", "queued_intervention_intensity",
        "base_churn_probability", "score_delta", "last_event_type", "clv", "coupon_cost", "expected_roi",
        "reoptimization_count"
    ],
    "realtime_action_queue_snapshot.csv": [
        "customer_id", "persona", "uplift_segment", "realtime_churn_score", "action_queue_status",
        "queued_recommended_action", "queued_intervention_intensity", "queued_coupon_cost",
        "queued_expected_profit", "queued_expected_roi", "action_queue_priority",
        "latest_trigger_reason", "last_reoptimized_at", "reoptimization_count",
    ],
    "survival_predictions.csv": [
        "customer_id", "predicted_median_time_to_churn_days", "risk_group", "survival_prob_30d", "predicted_hazard_ratio", "persona", "risk_percentile"
    ],
    "uplift_segmentation.csv": ["customer_id", "treatment_group", "revenue_post_60d", "uplift_segment"],
}


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in CSV_USECOLS.get(path.name, header) if col in header]
    return pd.read_csv(path, usecols=usecols or None, low_memory=False)


def _safe_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    return pd.DataFrame()


def load_dashboard_insight_bundle(
    result_dir: str = "results",
    data_dir: str = "data/raw",
) -> DashboardInsightBundle:
    result_base = _resolve_dir(result_dir)
    data_base = _resolve_dir(data_dir)

    return DashboardInsightBundle(
        result_dir=str(result_base),
        data_dir=str(data_base),
        customer_summary=ensure_dashboard_contract(_safe_csv(data_base / "customer_summary.csv"), frame_name="insight_customer_summary"),
        customers=_safe_csv(data_base / "customers.csv"),
        events=_safe_csv(data_base / "events.csv"),
        orders=_safe_csv(data_base / "orders.csv"),
        campaign_exposures=_safe_csv(data_base / "campaign_exposures.csv"),
        state_snapshots=_safe_csv(data_base / "state_snapshots.csv"),
        treatment_assignments=_safe_csv(data_base / "treatment_assignments.csv"),
        optimization_selected_customers=ensure_dashboard_contract(_safe_csv(result_base / "optimization_selected_customers.csv"), frame_name="optimization_selected_customers"),
        personalized_recommendations=ensure_dashboard_contract(_safe_csv(result_base / "personalized_recommendations.csv"), frame_name="personalized_recommendations"),
        realtime_scores=ensure_dashboard_contract(_safe_csv(result_base / "realtime_scores_snapshot.csv"), frame_name="realtime_scores"),
        realtime_action_queue=ensure_dashboard_contract(_safe_csv(result_base / "realtime_action_queue_snapshot.csv"), frame_name="realtime_action_queue"),
        survival_predictions=ensure_dashboard_contract(_safe_csv(result_base / "survival_predictions.csv"), frame_name="survival_predictions"),
        survival_coefficients=_safe_csv(result_base / "survival_top_coefficients.csv"),
        survival_metrics=_safe_json(result_base / "survival_metrics.json"),
        uplift_segmentation=ensure_dashboard_contract(_safe_csv(result_base / "uplift_segmentation.csv"), frame_name="uplift_segmentation"),
        top_feature_importance=_safe_json_df(result_base / "churn_top10_feature_importance.json"),
        ab_test_results=_safe_json(result_base / "ab_test_results.json"),
        dose_response_summary=_safe_json(result_base / "dose_response_summary.json"),
        customer_segment_summary=_safe_json(result_base / "customer_segment_summary.json"),
        persuadables_analysis=_safe_json(result_base / "persuadables_analysis.json"),
        optimization_summary=_safe_json(result_base / "optimization_summary.json"),
        realtime_scores_summary=_safe_json(result_base / "realtime_scores_summary.json"),
        realtime_action_queue_summary=_safe_json(result_base / "realtime_action_queue_summary.json"),
        personalized_recommendation_summary=_safe_json(result_base / "personalized_recommendation_summary.json"),
        clv_validation_metrics=_safe_json(result_base / "clv_validation_metrics.json"),
        feature_engineering_summary=_safe_json(result_base / "feature_engineering_summary.json"),
        churn_metrics=_safe_json(result_base / "churn_metrics.json"),
    )


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _summary_records(value: Any, *, name_column: str, count_column: str = "customers") -> list[dict[str, Any]]:
    """Normalize summary JSON values that may be saved as records or as a mapping.

    Older pipeline outputs stored e.g. {"segments": {"A": 10, "B": 20}},
    while some dashboard views expect [{"segment": "A", "customers": 10}, ...].
    This helper accepts both formats so Streamlit does not crash while rendering.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for key, raw in value.items():
            if isinstance(raw, dict):
                row = {name_column: key}
                row.update(raw)
            else:
                row = {name_column: key, count_column: raw}
            rows.append(row)
        return rows
    return []


def _id_values(df: pd.DataFrame, id_col: str = "customer_id") -> set[str]:
    if df.empty or id_col not in df.columns:
        return set()
    return set(df[id_col].dropna().astype(str).tolist())


def _safe_nunique(df: pd.DataFrame, column: str = "customer_id") -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].nunique())


def _normalize_feature_name(raw_name: str) -> str:
    name = str(raw_name)
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", " ")


def build_global_feature_table(bundle: DashboardInsightBundle) -> pd.DataFrame:
    df = bundle.top_feature_importance.copy()
    if df.empty:
        return pd.DataFrame()
    if "feature" not in df.columns or "importance" not in df.columns:
        return pd.DataFrame()
    df = df[["feature", "importance"]].copy()
    df["feature_display"] = df["feature"].map(_normalize_feature_name)
    total_importance = float(df["importance"].sum()) or 1.0
    df["importance_share"] = df["importance"] / total_importance
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def build_operational_overview(
    customers: pd.DataFrame,
    selected_customers: pd.DataFrame,
    optimize_summary: Dict[str, Any],
    recommendation_summary: Dict[str, Any],
    realtime_summary: Dict[str, Any],
    survival_metrics: Dict[str, Any],
    insight_bundle: DashboardInsightBundle,
) -> Dict[str, Any]:
    customers = ensure_dashboard_contract(customers, frame_name="operational_customers")
    selected_customers = ensure_dashboard_contract(selected_customers, frame_name="operational_selected")
    total_customers = int(len(customers))
    risk_count = int((pd.to_numeric(customers.get("churn_probability"), errors="coerce") >= 0.5).sum()) if not customers.empty and "churn_probability" in customers.columns else 0
    selected_count = int(len(selected_customers))
    recommended_rows = int(recommendation_summary.get("rows", 0) or 0)
    queued_actions = int(realtime_summary.get("queued_actions_total", 0) or 0)
    avg_window = _safe_num(optimize_summary.get("avg_intervention_window_days", 0.0))
    avg_hazard = _safe_num(survival_metrics.get("test_concordance_index", 0.0))
    event_count = int(len(insight_bundle.events))
    order_count = int(len(insight_bundle.orders))

    funnel_rows: List[Dict[str, Any]] = []
    if not insight_bundle.events.empty and "event_type" in insight_bundle.events.columns:
        counts = insight_bundle.events["event_type"].value_counts()
        for event_type, count in counts.items():
            funnel_rows.append(
                {
                    "stage": str(event_type),
                    "events": int(count),
                    "share": int(count) / max(len(insight_bundle.events), 1),
                }
            )
    funnel_df = pd.DataFrame(funnel_rows)

    persona_df = pd.DataFrame()
    if not customers.empty and "persona" in customers.columns:
        persona_df = (
            customers.groupby("persona", dropna=False)
            .agg(
                customers=("customer_id", "count"),
                avg_churn_probability=("churn_probability", "mean"),
                avg_uplift_score=("uplift_score", "mean"),
                avg_clv=("clv", "mean"),
            )
            .reset_index()
            .sort_values(["avg_churn_probability", "avg_uplift_score"], ascending=False)
        )

    segment_rows = _summary_records(
        insight_bundle.customer_segment_summary.get("segments", []),
        name_column="customer_segment",
        count_column="customers",
    )
    segment_df = pd.DataFrame(segment_rows)

    summary_cards = {
        "total_customers": total_customers,
        "risk_count": risk_count,
        "selected_count": selected_count,
        "expected_profit": _safe_num(optimize_summary.get("expected_incremental_profit", 0.0)),
        "overall_roi": _safe_num(optimize_summary.get("overall_roi", 0.0)),
        "recommended_rows": recommended_rows,
        "queued_actions": queued_actions,
        "avg_window": avg_window,
        "survival_c_index": avg_hazard,
        "event_count": event_count,
        "order_count": order_count,
    }

    return {
        "summary_cards": summary_cards,
        "funnel_df": funnel_df,
        "persona_df": persona_df,
        "segment_df": segment_df,
    }


def build_experiment_overview(bundle: DashboardInsightBundle) -> Dict[str, Any]:
    ab = bundle.ab_test_results or {}
    sample_sizes = ab.get("sample_sizes", {}) if isinstance(ab, dict) else {}
    rates = ab.get("rates", {}) if isinstance(ab, dict) else {}
    power = ab.get("power_analysis", {}) if isinstance(ab, dict) else {}
    hypothesis = ab.get("hypothesis_test", {}) if isinstance(ab, dict) else {}
    z_test = hypothesis.get("z_test", {}) if isinstance(hypothesis, dict) else {}

    treatment_n = _safe_int(sample_sizes.get("treatment", 0))
    control_n = _safe_int(sample_sizes.get("control", 0))
    treatment_retention = _safe_num(rates.get("treatment_retention_rate", 0.0))
    control_retention = _safe_num(rates.get("control_retention_rate", 0.0))
    incremental_retention = treatment_retention - control_retention
    incremental_retained_customers = incremental_retention * treatment_n

    coupon_spend_total = _safe_num(bundle.campaign_exposures.get("coupon_cost", pd.Series(dtype=float)).sum())
    incremental_cpic = coupon_spend_total / incremental_retained_customers if incremental_retained_customers > 0 else np.nan

    revenue_increment = 0.0
    if not bundle.uplift_segmentation.empty and {"treatment_group", "revenue_post_60d"}.issubset(bundle.uplift_segmentation.columns):
        revenue_by_group = bundle.uplift_segmentation.groupby("treatment_group")["revenue_post_60d"].mean()
        revenue_increment = _safe_num(revenue_by_group.get("treatment", 0.0)) - _safe_num(revenue_by_group.get("control", 0.0))
        revenue_increment *= max(treatment_n, 1)

    dose_rows: List[Dict[str, Any]] = []
    arm_summary = bundle.dose_response_summary.get("arm_summary", {}) if isinstance(bundle.dose_response_summary, dict) else {}
    priors = bundle.dose_response_summary.get("effect_priors", {}) if isinstance(bundle.dose_response_summary, dict) else {}
    multipliers = bundle.dose_response_summary.get("intensity_cost_multipliers", {}) if isinstance(bundle.dose_response_summary, dict) else {}
    for arm, stats in arm_summary.items():
        dose_rows.append(
            {
                "arm": arm,
                "samples": _safe_int(stats.get("samples", 0)),
                "retention_rate": _safe_num(stats.get("retention_rate", 0.0)),
                "avg_coupon_cost": _safe_num(stats.get("avg_coupon_cost", 0.0)),
                "avg_revenue_post_horizon": _safe_num(stats.get("avg_revenue_post_horizon", 0.0)),
                "effect_prior": _safe_num(priors.get(arm, 0.0)),
                "cost_multiplier": _safe_num(multipliers.get(arm, 1.0)),
            }
        )
    dose_df = pd.DataFrame(dose_rows)

    persuadables = bundle.persuadables_analysis or {}
    numeric_deltas = pd.DataFrame(persuadables.get("top_numeric_deltas", []))

    metrics = {
        "treatment_n": treatment_n,
        "control_n": control_n,
        "incremental_retention": incremental_retention,
        "incremental_retained_customers": incremental_retained_customers,
        "coupon_spend_total": coupon_spend_total,
        "incremental_cpic": incremental_cpic,
        "incremental_revenue": revenue_increment,
        "p_value": _safe_num(z_test.get("p_value", np.nan), default=np.nan),
        "achieved_power": _safe_num(power.get("achieved_power_with_current_sample", 0.0)),
    }

    return {
        "metrics": metrics,
        "dose_df": dose_df,
        "numeric_deltas": numeric_deltas,
        "persuadables": persuadables,
        "ab_test": ab,
    }


def build_realtime_monitor_overview(bundle: DashboardInsightBundle, fallback_scores: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    scores = fallback_scores.copy() if isinstance(fallback_scores, pd.DataFrame) and not fallback_scores.empty else bundle.realtime_scores.copy()
    queue = bundle.realtime_action_queue.copy()
    summary = bundle.realtime_scores_summary or {}

    status_df = pd.DataFrame()
    trigger_df = pd.DataFrame()
    signal_df = pd.DataFrame()

    if not scores.empty:
        if "action_queue_status" in scores.columns:
            status_df = scores["action_queue_status"].fillna("unknown").value_counts().rename_axis("status").reset_index(name="count")
        if "latest_trigger_reason" in scores.columns:
            trigger_df = scores["latest_trigger_reason"].fillna("unknown").value_counts().rename_axis("trigger_reason").reset_index(name="count")
        signal_columns = [
            col for col in [
                "visit_signal",
                "browse_signal",
                "search_signal",
                "cart_signal",
                "cart_remove_signal",
                "purchase_signal",
                "support_signal",
                "coupon_open_signal",
                "coupon_redeem_signal",
                "behavioral_risk",
                "inactivity_signal",
            ] if col in scores.columns
        ]
        if signal_columns:
            signal_df = pd.DataFrame(
                {
                    "signal": signal_columns,
                    "mean_value": [float(pd.to_numeric(scores[col], errors="coerce").fillna(0).mean()) for col in signal_columns],
                }
            ).sort_values("mean_value", ascending=False)

    if queue.empty and not scores.empty and "action_queue_status" in scores.columns:
        queue = scores[scores["action_queue_status"].astype(str).eq("queued")].copy()

    return {
        "summary": summary,
        "status_df": status_df,
        "trigger_df": trigger_df,
        "signal_df": signal_df,
        "queue_df": queue,
    }


def build_coupon_risk_overview(bundle: DashboardInsightBundle) -> Dict[str, Any]:
    customers = bundle.customer_summary.copy()
    exposures = bundle.campaign_exposures.copy()
    recommendations = bundle.personalized_recommendations.copy()
    selected = bundle.optimization_selected_customers.copy()
    dose_summary = bundle.dose_response_summary or {}

    exposure_col = "coupon_exposure_count" if "coupon_exposure_count" in customers.columns else "coupon_exposures"
    open_col = "coupon_opens" if "coupon_opens" in customers.columns else None
    redeem_col = "coupon_redeem_count" if "coupon_redeem_count" in customers.columns else "coupon_redeemed"

    exposed_customers = int((pd.to_numeric(customers.get(exposure_col), errors="coerce").fillna(0) > 0).sum()) if exposure_col in customers.columns else 0
    high_exposure_customers = int((pd.to_numeric(customers.get(exposure_col), errors="coerce").fillna(0) >= 3).sum()) if exposure_col in customers.columns else 0
    total_exposures = _safe_num(pd.to_numeric(customers.get(exposure_col), errors="coerce").fillna(0).sum()) if exposure_col in customers.columns else 0.0
    total_opens = _safe_num(pd.to_numeric(customers.get(open_col), errors="coerce").fillna(0).sum()) if open_col else 0.0
    total_redeems = _safe_num(pd.to_numeric(customers.get(redeem_col), errors="coerce").fillna(0).sum()) if redeem_col in customers.columns else 0.0

    open_rate = total_opens / total_exposures if total_exposures > 0 else np.nan
    redeem_rate = total_redeems / total_exposures if total_exposures > 0 else np.nan

    segment_df = pd.DataFrame()
    if not customers.empty and "persona" in customers.columns and exposure_col in customers.columns:
        agg_dict: Dict[str, tuple[str, str]] = {
            "customers": ("customer_id", "count"),
            "avg_coupon_exposure": (exposure_col, "mean"),
        }
        if "churn_probability" in customers.columns:
            agg_dict["avg_churn_probability"] = ("churn_probability", "mean")
        if "expected_roi" in customers.columns:
            agg_dict["avg_expected_roi"] = ("expected_roi", "mean")
        if open_col and open_col in customers.columns:
            agg_dict["avg_coupon_opens"] = (open_col, "mean")
        if redeem_col in customers.columns:
            agg_dict["avg_coupon_redeems"] = (redeem_col, "mean")
        segment_df = customers.groupby("persona", dropna=False).agg(**agg_dict).reset_index()
        segment_df = segment_df.sort_values("avg_coupon_exposure", ascending=False)

    recommendation_mix = pd.DataFrame()
    if not recommendations.empty and "recommended_category" in recommendations.columns:
        recommendation_mix = recommendations["recommended_category"].fillna("unknown").value_counts().rename_axis("recommended_category").reset_index(name="count")

    intensity_mix = _complete_intensity_mix(selected)

    flags = [
        {
            "issue": "고강도 개입 사전효과(effect prior)",
            "status": "주의" if _safe_num(dose_summary.get("effect_priors", {}).get("high", 0.0)) < 0 else "양호",
            "detail": "high 강도 arm의 사전효과가 음수이면 혜택을 강하게 줄수록 성과가 악화될 수 있습니다.",
        },
        {
            "issue": "고노출 고객 비중",
            "status": "주의" if (high_exposure_customers / max(len(customers), 1)) >= 0.15 else "양호",
            "detail": f"3회 이상 노출 고객 {high_exposure_customers:,}명",
        },
        {
            "issue": "쿠폰 리딤률",
            "status": "주의" if np.isfinite(redeem_rate) and redeem_rate < 0.02 else "양호",
            "detail": f"전체 노출 대비 리딤률 {redeem_rate:.2%}" if np.isfinite(redeem_rate) else "계산 불가",
        },
    ]

    return {
        "metrics": {
            "exposed_customers": exposed_customers,
            "high_exposure_customers": high_exposure_customers,
            "total_exposures": total_exposures,
            "open_rate": open_rate,
            "redeem_rate": redeem_rate,
            "coupon_spend_total": _safe_num(exposures.get("coupon_cost", pd.Series(dtype=float)).sum()),
        },
        "segment_df": segment_df,
        "recommendation_mix": recommendation_mix,
        "intensity_mix": intensity_mix,
        "flags_df": pd.DataFrame(flags),
    }


def build_data_diagnostics(bundle: DashboardInsightBundle) -> Dict[str, Any]:
    customers = bundle.customer_summary.copy()
    base_customers = bundle.customers.copy()
    events = bundle.events.copy()
    orders = bundle.orders.copy()
    exposures = bundle.campaign_exposures.copy()
    snapshots = bundle.state_snapshots.copy()
    treatment = bundle.treatment_assignments.copy()

    customer_ids = _id_values(customers)

    def _orphan_count(df: pd.DataFrame, id_col: str = "customer_id") -> int:
        if df.empty or id_col not in df.columns:
            return 0
        values = _id_values(df, id_col)
        return int(len(values - customer_ids))

    checks = [
        {
            "check": "customer_summary 중복 고객 여부",
            "status": "양호" if customers.empty or "customer_id" not in customers.columns or customers["customer_id"].is_unique else "주의",
            "detail": f"중복 수: {int(customers['customer_id'].duplicated().sum()) if not customers.empty and 'customer_id' in customers.columns else 0}",
        },
        {
            "check": "events의 고아 customer_id",
            "status": "양호" if _orphan_count(events) == 0 else "주의",
            "detail": f"고아 고객 수: {_orphan_count(events):,}",
        },
        {
            "check": "orders의 고아 customer_id",
            "status": "양호" if _orphan_count(orders) == 0 else "주의",
            "detail": f"고아 고객 수: {_orphan_count(orders):,}",
        },
        {
            "check": "state snapshot 고객 커버리지",
            "status": "양호" if snapshots.empty or _safe_nunique(snapshots) == max(len(customer_ids), 1) else "주의",
            "detail": f"snapshot 고객 수: {_safe_nunique(snapshots):,} / 전체 고객 수: {len(customer_ids):,}",
        },
        {
            "check": "treatment assignment 커버리지",
            "status": "양호" if treatment.empty or _safe_nunique(treatment) == max(len(customer_ids), 1) else "주의",
            "detail": f"assignment 고객 수: {_safe_nunique(treatment):,} / 전체 고객 수: {len(customer_ids):,}",
        },
        {
            "check": "핵심 스코어 결측",
            "status": "양호" if customers.empty else ("양호" if customers[[c for c in ["churn_probability", "uplift_score", "clv"] if c in customers.columns]].isna().sum().sum() == 0 else "주의"),
            "detail": (
                "결측 없음" if customers.empty else f"결측 합계: {int(customers[[c for c in ['churn_probability', 'uplift_score', 'clv'] if c in customers.columns]].isna().sum().sum())}"
            ),
        },
    ]
    checks_df = pd.DataFrame(checks)

    volumes_df = pd.DataFrame([
        {"dataset": "customer_summary", "rows": int(len(customers)), "unique_customers": _safe_nunique(customers)},
        {"dataset": "customers", "rows": int(len(base_customers)), "unique_customers": _safe_nunique(base_customers)},
        {"dataset": "events", "rows": int(len(events)), "unique_customers": _safe_nunique(events)},
        {"dataset": "orders", "rows": int(len(orders)), "unique_customers": _safe_nunique(orders)},
        {"dataset": "campaign_exposures", "rows": int(len(exposures)), "unique_customers": _safe_nunique(exposures)},
        {"dataset": "state_snapshots", "rows": int(len(snapshots)), "unique_customers": _safe_nunique(snapshots)},
        {"dataset": "treatment_assignments", "rows": int(len(treatment)), "unique_customers": _safe_nunique(treatment)},
    ])

    event_mix_df = pd.DataFrame()
    if not events.empty and "event_type" in events.columns:
        event_mix_df = events["event_type"].value_counts().rename_axis("event_type").reset_index(name="count")
        event_mix_df["share"] = event_mix_df["count"] / max(int(event_mix_df["count"].sum()), 1)

    distribution_df = pd.DataFrame()
    if not customers.empty:
        pieces = []
        for column in ["persona", "region", "device_type", "acquisition_channel"]:
            if column in customers.columns:
                part = customers[column].fillna("unknown").value_counts().rename_axis("value").reset_index(name="count")
                part.insert(0, "dimension", column)
                part["share"] = part["count"] / max(int(part["count"].sum()), 1)
                pieces.append(part)
        if pieces:
            distribution_df = pd.concat(pieces, ignore_index=True)

    return {
        "checks_df": checks_df,
        "volumes_df": volumes_df,
        "event_mix_df": event_mix_df,
        "distribution_df": distribution_df,
    }


def _build_reason_summary(row: pd.Series) -> str:
    reasons: List[str] = []
    churn_probability = _safe_num(row.get("churn_probability", row.get("realtime_churn_score", 0.0)))
    uplift_score = _safe_num(row.get("uplift_score", row.get("predicted_uplift", 0.0)))
    clv = _safe_num(row.get("clv", 0.0))
    expected_roi = _safe_num(row.get("expected_roi", row.get("queued_expected_roi", 0.0)))
    window_days = _safe_num(row.get("intervention_window_days", row.get("predicted_median_time_to_churn_days", np.nan)), default=np.nan)
    coupon_affinity = _safe_num(row.get("coupon_affinity", 0.0))
    price_sensitivity = _safe_num(row.get("price_sensitivity", 0.0))
    exposures = _safe_num(row.get("coupon_exposure_count", row.get("coupon_exposures", 0.0)))

    if churn_probability >= 0.6:
        reasons.append("이탈 위험이 높음")
    if uplift_score >= 0.02:
        reasons.append("개입 반응 가능성이 큼")
    if clv >= 1_000_000:
        reasons.append("고객 가치가 높음")
    if np.isfinite(window_days) and window_days <= 7:
        reasons.append("개입 시한이 촉박함")
    if expected_roi >= 0.3:
        reasons.append("예상 ROI가 양호함")
    if coupon_affinity >= 0.7 and price_sensitivity >= 0.7:
        reasons.append("할인 반응성이 높은 편")
    if exposures >= 3:
        reasons.append("최근 쿠폰 노출이 누적됨")
    return ", ".join(reasons[:4]) if reasons else "핵심 신호를 종합해 관찰 대상에 포함"


def _build_watchout_summary(row: pd.Series) -> str:
    watchouts: List[str] = []
    exposures = _safe_num(row.get("coupon_exposure_count", row.get("coupon_exposures", 0.0)))
    price_sensitivity = _safe_num(row.get("price_sensitivity", 0.0))
    support_propensity = _safe_num(row.get("support_contact_propensity", 0.0))
    risk_group = str(row.get("risk_group", "")).strip()

    if exposures >= 3:
        watchouts.append("과도한 쿠폰 학습 가능성")
    if price_sensitivity >= 0.8:
        watchouts.append("할인 없이는 반응 저하 가능")
    if support_propensity >= 0.5:
        watchouts.append("서비스 이슈형 대응 검토")
    if risk_group.lower().startswith("high"):
        watchouts.append("단기 이탈 가속 주의")
    return ", ".join(watchouts[:3]) if watchouts else "가격·서비스·타이밍 리스크를 함께 점검"


def build_customer_explanations(
    customers: pd.DataFrame,
    selected_customers: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    survival_predictions: pd.DataFrame,
    realtime_scores: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    if customers.empty:
        return pd.DataFrame()

    base = customers.copy()
    keep_columns = [
        col for col in [
            "customer_id",
            "persona",
            "churn_probability",
            "uplift_score",
            "clv",
            "expected_roi",
            "price_sensitivity",
            "coupon_affinity",
            "support_contact_propensity",
            "coupon_exposure_count",
            "coupon_exposures",
            "inactivity_days",
        ] if col in base.columns
    ]
    base = base[keep_columns].drop_duplicates(subset=["customer_id"]) if keep_columns else pd.DataFrame()

    merged = base
    for extra_df, columns in [
        (selected_customers, [
            "customer_id",
            "recommended_action",
            "intervention_intensity",
            "intervention_window_days",
            "expected_incremental_profit",
            "selection_score",
            "timing_priority_bucket",
        ]),
        (recommendation_df, [
            "customer_id",
            "recommended_category",
            "recommendation_score",
            "reason_tags",
        ]),
        (survival_predictions, [
            "customer_id",
            "predicted_median_time_to_churn_days",
            "risk_group",
            "survival_prob_30d",
        ]),
        (realtime_scores, [
            "customer_id",
            "realtime_churn_score",
            "latest_trigger_reason",
            "queued_recommended_action",
        ]),
    ]:
        available = [col for col in columns if not extra_df.empty and col in extra_df.columns]
        if len(available) >= 2 and "customer_id" in available:
            deduped = extra_df[available].drop_duplicates(subset=["customer_id"])
            merged = merged.merge(deduped, on="customer_id", how="left")

    sort_columns = [col for col in ["selection_score", "realtime_churn_score", "churn_probability", "uplift_score", "clv"] if col in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    merged = merged.head(max(int(top_n), 1)).copy()
    if merged.empty:
        return merged

    merged["selection_reason"] = merged.apply(_build_reason_summary, axis=1)
    merged["watchout"] = merged.apply(_build_watchout_summary, axis=1)
    if "reason_tags" in merged.columns:
        merged["reason_tags"] = merged["reason_tags"].fillna("").astype(str).str.replace(",", ", ", regex=False)

    preferred_cols = [
        col for col in [
            "customer_id",
            "persona",
            "selection_reason",
            "watchout",
            "recommended_action",
            "queued_recommended_action",
            "recommended_category",
            "intervention_intensity",
            "timing_priority_bucket",
            "churn_probability",
            "realtime_churn_score",
            "uplift_score",
            "clv",
            "expected_roi",
            "expected_incremental_profit",
            "predicted_median_time_to_churn_days",
            "risk_group",
            "latest_trigger_reason",
            "reason_tags",
        ] if col in merged.columns
    ]
    return merged[preferred_cols].copy()
