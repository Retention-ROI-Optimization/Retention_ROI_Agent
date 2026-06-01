from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.optimization.policy import normalize, safe_numeric
from src.optimization.timing import apply_survival_timing


@dataclass(frozen=True)
class CounterfactualAction:
    action_id: str
    label: str
    cost: float
    uplift_multiplier: float
    channel: str
    description: str


ACTION_CATALOG: tuple[CounterfactualAction, ...] = (
    CounterfactualAction(
        action_id="no_action",
        label="무개입",
        cost=0.0,
        uplift_multiplier=0.0,
        channel="none",
        description="아무 개입도 하지 않았을 때의 기대 순이익",
    ),
    CounterfactualAction(
        action_id="coupon_5000",
        label="5,000원 혜택",
        cost=5000.0,
        uplift_multiplier=1.00,
        channel="benefit",
        description="정해진 비용 안에서 고객에게 금융 혜택 또는 재방문 혜택을 제공하는 전략",
    ),
    CounterfactualAction(
        action_id="consult_call",
        label="상담 전화",
        cost=12000.0,
        uplift_multiplier=1.18,
        channel="call_center",
        description="가치가 높고 이탈 위험도 큰 고객에게 상담원이 직접 연락하는 전략",
    ),
    CounterfactualAction(
        action_id="push_email",
        label="푸시/이메일",
        cost=800.0,
        uplift_multiplier=0.42,
        channel="owned_message",
        description="앱 푸시나 이메일로 낮은 비용의 안내 메시지를 보내는 전략",
    ),
    CounterfactualAction(
        action_id="wait_7d",
        label="7일 대기",
        cost=0.0,
        uplift_multiplier=0.0,
        channel="defer",
        description="지금 바로 비용을 쓰지 않고 7일 동안 추가 행동을 지켜본 뒤 결정하는 전략",
    ),
)

_ACTION_LABEL_BY_ID = {action.action_id: action.label for action in ACTION_CATALOG}


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float(default)] * len(df), index=df.index, dtype=float)
    return safe_numeric(df[column], default=default)


def _first_numeric_column(df: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> pd.Series:
    result = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    for column in columns:
        if column in df.columns:
            candidate = pd.to_numeric(df[column], errors="coerce")
            result = result.where(result.notna(), candidate)
    return result.fillna(float(default)).astype(float)


def _has_non_null_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def _dedupe_customer_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "customer_id" not in df.columns:
        return df.head(0).copy()
    out = df.copy()
    out["_customer_id_text"] = out["customer_id"].astype(str)
    sort_cols = [col for col in ["selection_score", "priority_score", "expected_incremental_profit", "churn_probability"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    out = out.drop_duplicates("_customer_id_text", keep="first").drop(columns=["_customer_id_text"])
    return out.reset_index(drop=True)


def _prepare_base_frame(
    customers: pd.DataFrame,
    selected_customers: Optional[pd.DataFrame],
    survival_predictions: Optional[pd.DataFrame],
    *,
    threshold: float,
    top_n: Optional[int],
) -> pd.DataFrame:
    if customers is None or customers.empty or "customer_id" not in customers.columns:
        return pd.DataFrame()

    base = customers.copy()
    base["churn_probability"] = _first_numeric_column(base, ["churn_probability", "realtime_churn_score", "churn_score"], 0.0).clip(0.0, 1.0)
    base["uplift_score"] = _first_numeric_column(base, ["uplift_score", "expected_uplift", "incremental_retention_probability"], 0.0).clip(-0.30, 0.60)
    base["clv"] = _first_numeric_column(base, ["predicted_clv_12m", "clv", "customer_lifetime_value", "monetary"], 0.0).clip(lower=0.0)

    selected_ids: set[str] = set()
    if selected_customers is not None and not selected_customers.empty and "customer_id" in selected_customers.columns:
        selected_ids = set(selected_customers["customer_id"].astype(str))

    at_risk = base[base["churn_probability"] >= float(threshold)].copy()
    if selected_ids:
        selected_base = base[base["customer_id"].astype(str).isin(selected_ids)].copy()
        pool = pd.concat([selected_base, at_risk], ignore_index=True)
    else:
        pool = at_risk.copy()

    if pool.empty:
        pool = base.copy()

    pool = _dedupe_customer_rows(pool)

    # Enrich with the already optimized action/cost columns when available.
    if selected_customers is not None and not selected_customers.empty and "customer_id" in selected_customers.columns:
        selected = _dedupe_customer_rows(selected_customers)
        keep = [
            col
            for col in [
                "customer_id",
                "recommended_action",
                "intervention_intensity",
                "recommended_intervention_window",
                "coupon_cost",
                "expected_incremental_profit",
                "expected_roi",
                "priority_score",
                "selection_score",
            ]
            if col in selected.columns
        ]
        if len(keep) > 1:
            pool = pool.copy()
            selected = selected[keep].copy()
            pool["_customer_id_key"] = pool["customer_id"].astype(str)
            selected["_customer_id_key"] = selected["customer_id"].astype(str)
            selected = selected.drop(columns=["customer_id"])
            pool = pool.merge(selected, on="_customer_id_key", how="left", suffixes=("", "_selected"))
            pool = pool.drop(columns=["_customer_id_key"], errors="ignore")
            for col in [c for c in keep if c != "customer_id"]:
                selected_col = f"{col}_selected"
                if selected_col in pool.columns:
                    if col in pool.columns:
                        pool[col] = pool[col].where(pool[col].notna(), pool[selected_col])
                    else:
                        pool[col] = pool[selected_col]
                    pool = pool.drop(columns=[selected_col])

    # Survival enrichment is optional.  If customer_id cannot be parsed as numeric,
    # keep the base frame instead of silently dropping all string IDs.
    can_apply_survival = False
    if survival_predictions is not None and not survival_predictions.empty and "customer_id" in survival_predictions.columns:
        can_apply_survival = pd.to_numeric(pool["customer_id"], errors="coerce").notna().any()
    if can_apply_survival:
        enriched = apply_survival_timing(pool, survival_predictions=survival_predictions, customer_id_col="customer_id")
        if not enriched.empty:
            pool = enriched

    if "intervention_window_days" not in pool.columns:
        pool["intervention_window_days"] = 90
    if "recommended_intervention_window" not in pool.columns:
        pool["recommended_intervention_window"] = "60일 이후 관찰"
    if "timing_urgency_score" not in pool.columns:
        pool["timing_urgency_score"] = 0.0
    if "churn_timing_weight" not in pool.columns:
        pool["churn_timing_weight"] = 1.0

    pool["churn_probability"] = _first_numeric_column(pool, ["churn_probability", "realtime_churn_score", "churn_score"], 0.0).clip(0.0, 1.0)
    pool["uplift_score"] = _first_numeric_column(pool, ["uplift_score", "expected_uplift", "incremental_retention_probability"], 0.0).clip(-0.30, 0.60)
    pool["clv"] = _first_numeric_column(pool, ["predicted_clv_12m", "clv", "customer_lifetime_value", "monetary"], 0.0).clip(lower=0.0)
    pool["timing_urgency_score"] = _series_or_default(pool, "timing_urgency_score", 0.0).clip(0.0, 1.0)
    pool["churn_timing_weight"] = _series_or_default(pool, "churn_timing_weight", 1.0).clip(0.60, 1.60)
    pool["intervention_window_days"] = _series_or_default(pool, "intervention_window_days", 90.0).clip(lower=1.0)

    pool["_rank_score"] = (
        0.34 * pool["churn_probability"]
        + 0.26 * pool["uplift_score"].clip(lower=0.0)
        + 0.20 * normalize(pool["clv"])
        + 0.20 * pool["timing_urgency_score"]
    )
    pool = pool.sort_values(["_rank_score", "churn_probability", "clv", "customer_id"], ascending=[False, False, False, True])
    if top_n is not None and int(top_n) > 0:
        pool = pool.head(int(top_n))
    return pool.drop(columns=["_rank_score"], errors="ignore").reset_index(drop=True)


def _action_fit(frame: pd.DataFrame, action_id: str) -> pd.Series:
    churn = _series_or_default(frame, "churn_probability", 0.0).clip(0.0, 1.0)
    uplift = _series_or_default(frame, "uplift_score", 0.0).clip(-0.30, 0.60)
    urgency = _series_or_default(frame, "timing_urgency_score", 0.0).clip(0.0, 1.0)
    clv_rank = normalize(_series_or_default(frame, "clv", 0.0)).clip(0.0, 1.0)
    coupon_affinity = normalize(_series_or_default(frame, "coupon_affinity", 0.0)).clip(0.0, 1.0)
    price_sensitivity = normalize(_series_or_default(frame, "price_sensitivity", 0.0)).clip(0.0, 1.0)
    fatigue = normalize(_series_or_default(frame, "discount_pressure_score", 0.0)).clip(0.0, 1.0)
    brand_sensitivity = normalize(_series_or_default(frame, "brand_sensitivity", 0.0)).clip(0.0, 1.0)

    if action_id == "coupon_5000":
        fit = 0.82 + 0.22 * coupon_affinity + 0.22 * price_sensitivity + 0.12 * urgency - 0.18 * fatigue
    elif action_id == "consult_call":
        fit = 0.78 + 0.25 * clv_rank + 0.18 * churn + 0.14 * urgency + 0.08 * brand_sensitivity - 0.08 * coupon_affinity
    elif action_id == "push_email":
        fit = 0.68 + 0.18 * uplift.clip(lower=0.0) + 0.12 * (1.0 - fatigue) + 0.10 * (1.0 - urgency)
    else:
        fit = pd.Series([1.0] * len(frame), index=frame.index, dtype=float)
    return fit.clip(lower=0.25, upper=1.35)


def _confidence_label(score: float) -> str:
    if score >= 0.74:
        return "높음"
    if score >= 0.50:
        return "중간"
    return "낮음"


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(float(value)):,.0f}원"


def _build_reason(row: pd.Series) -> str:
    best_action = str(row.get("final_recommendation", ""))
    delta = float(row.get("incremental_vs_no_action", 0.0) or 0.0)
    confidence = str(row.get("confidence", ""))
    churn = float(row.get("churn_probability", 0.0) or 0.0)
    uplift = float(row.get("uplift_score", 0.0) or 0.0)
    if best_action == "무개입":
        return "개입 비용이 효과보다 크거나 고객 반응 가능성이 낮아, 지금은 아무 조치를 하지 않는 편이 예상 순이익이 가장 높습니다."
    if best_action == "7일 대기":
        return f"지금 바로 개입하기보다 7일 동안 고객 행동을 더 지켜보는 편이 예상 순이익이 높습니다. 현재 이탈 가능성은 {churn:.1%}, 개입 반응 가능성은 {uplift:.3f} 수준입니다."
    if confidence == "낮음":
        return f"{best_action}을 선택하면 아무것도 하지 않을 때보다 {_format_delta(delta)} 정도 나아질 것으로 보입니다. 다만 신뢰도가 낮으므로 바로 전체 고객에게 적용하지 말고 A/B 검증이나 검증용 미개입군을 함께 두는 것이 좋습니다."
    if confidence == "중간":
        return f"{best_action}을 선택하면 아무것도 하지 않을 때보다 {_format_delta(delta)} 정도 나아질 것으로 보입니다. 신뢰도는 중간 수준이므로 실험군과 대조군을 함께 운영해 실제 효과를 확인하는 것이 안전합니다."
    return f"{best_action}을 선택하면 아무것도 하지 않을 때보다 {_format_delta(delta)} 정도 나아질 것으로 보입니다. 입력 신호가 비교적 충분해 우선 실행 후보로 볼 수 있습니다."


def build_counterfactual_retention_lab(
    customers: pd.DataFrame,
    selected_customers: Optional[pd.DataFrame] = None,
    survival_predictions: Optional[pd.DataFrame] = None,
    *,
    top_n: Optional[int] = 100,
    threshold: float = 0.50,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Build customer-level counterfactual retention decision scenarios.

    The output intentionally separates two shapes:
    - lab_df: one row per customer, suitable for the dashboard table.
    - scenario_df: one row per customer × action, suitable for comparison charts.

    Values are expected net profit estimates, not realized causal effects.  They
    should be validated with holdout/A-B tests before operational rollout.
    """

    base = _prepare_base_frame(
        customers=customers,
        selected_customers=selected_customers,
        survival_predictions=survival_predictions,
        threshold=threshold,
        top_n=top_n,
    )
    if base.empty:
        empty_summary = {
            "customer_count": 0,
            "scenario_count": 0,
            "avg_incremental_vs_no_action": 0.0,
            "positive_recommendation_count": 0,
            "ab_test_recommended_count": 0,
            "best_action_counts": {},
            "model_note": "No eligible customers found.",
        }
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    base_churn = _series_or_default(base, "churn_probability", 0.0).clip(0.0, 1.0)
    clv = _series_or_default(base, "clv", 0.0).clip(lower=0.0)
    uplift = _series_or_default(base, "uplift_score", 0.0).clip(-0.30, 0.60)
    timing_weight = _series_or_default(base, "churn_timing_weight", 1.0).clip(0.60, 1.60)
    urgency = _series_or_default(base, "timing_urgency_score", 0.0).clip(0.0, 1.0)
    window_days = _series_or_default(base, "intervention_window_days", 90.0).clip(lower=1.0)

    no_action_net = clv * (1.0 - base_churn)
    scenario_frames: list[pd.DataFrame] = []
    action_net_by_id: dict[str, pd.Series] = {}

    for action in ACTION_CATALOG:
        scenario = base[[col for col in ["customer_id", "persona", "uplift_segment", "recommended_action"] if col in base.columns]].copy()
        scenario["action_id"] = action.action_id
        scenario["action_label"] = action.label
        scenario["channel"] = action.channel
        scenario["action_cost"] = float(action.cost)
        scenario["churn_probability"] = base_churn
        scenario["clv"] = clv
        scenario["uplift_score"] = uplift

        if action.action_id == "no_action":
            effect = pd.Series([0.0] * len(base), index=base.index, dtype=float)
            action_net = no_action_net.copy()
            treated_churn = base_churn.copy()
            option_value = pd.Series([0.0] * len(base), index=base.index, dtype=float)
        elif action.action_id == "wait_7d":
            # Approximate the option value of waiting: saving the immediate cost and
            # collecting more behavior signals, offset by short-term hazard exposure.
            seven_day_hazard = (7.0 / window_days).clip(0.0, 1.0) * (0.05 + 0.22 * urgency + 0.20 * base_churn)
            treated_churn = (base_churn + seven_day_hazard).clip(0.0, 1.0)
            information_value = (0.018 * clv * (1.0 - urgency) * uplift.clip(lower=0.0)).clip(lower=0.0)
            action_net = (clv * (1.0 - treated_churn)) + information_value
            effect = (base_churn - treated_churn).clip(-1.0, 1.0)
            option_value = information_value
        else:
            fit = _action_fit(base, action.action_id)
            effect = (uplift * float(action.uplift_multiplier) * timing_weight * fit).clip(-0.25, 0.45)
            # High-risk customers cannot be improved by more than their churn risk.
            effect = effect.clip(lower=-0.25, upper=base_churn)
            treated_churn = (base_churn - effect).clip(0.0, 1.0)
            option_value = pd.Series([0.0] * len(base), index=base.index, dtype=float)
            action_net = (clv * (1.0 - treated_churn)) - float(action.cost)

        scenario["treated_churn_probability"] = treated_churn
        scenario["estimated_churn_delta"] = (treated_churn - base_churn)
        scenario["estimated_retention_lift"] = effect
        scenario["option_value"] = option_value
        scenario["expected_net_profit"] = action_net
        scenario["incremental_vs_no_action"] = action_net - no_action_net
        scenario["expected_roi"] = (scenario["incremental_vs_no_action"] / float(action.cost)) if action.cost > 0 else np.nan
        scenario["description"] = action.description
        scenario_frames.append(scenario)
        action_net_by_id[action.action_id] = action_net

    scenario_df = pd.concat(scenario_frames, ignore_index=True)
    net_matrix = pd.DataFrame(action_net_by_id)
    best_action_id = net_matrix.idxmax(axis=1)
    best_net = net_matrix.max(axis=1)

    lab_df = base.copy()
    lab_df["expected_no_action_net_profit"] = no_action_net
    for action in ACTION_CATALOG:
        lab_df[f"expected_net_profit_{action.action_id}"] = action_net_by_id[action.action_id]
    lab_df["best_action_id"] = best_action_id.values
    lab_df["final_recommendation"] = lab_df["best_action_id"].map(_ACTION_LABEL_BY_ID).fillna(lab_df["best_action_id"].astype(str))
    lab_df["best_expected_net_profit"] = best_net.values
    lab_df["incremental_vs_no_action"] = lab_df["best_expected_net_profit"] - lab_df["expected_no_action_net_profit"]

    second_best = net_matrix.apply(lambda row: row.nlargest(2).iloc[-1] if len(row) >= 2 else row.max(), axis=1)
    decision_margin = (best_net - second_best).clip(lower=0.0)
    available_signals = (
        0.25
        + 0.18 * float(_has_non_null_column(base, "churn_probability"))
        + 0.18 * float(_has_non_null_column(base, "uplift_score"))
        + 0.16 * float(_has_non_null_column(base, "clv"))
        + 0.13 * float(_has_non_null_column(base, "intervention_window_days"))
        + 0.10 * float(_has_non_null_column(base, "timing_urgency_score"))
    )
    normalized_margin = (decision_margin / clv.where(clv > 0, 1.0)).clip(0.0, 0.35) / 0.35
    uplift_quality = uplift.abs().clip(0.0, 0.20) / 0.20
    lab_df["confidence_score"] = (0.55 * available_signals + 0.25 * normalized_margin + 0.20 * uplift_quality).clip(0.0, 1.0)
    lab_df["confidence"] = lab_df["confidence_score"].map(lambda x: _confidence_label(float(x)))
    lab_df["ab_test_recommended"] = (lab_df["confidence"] != "높음") | (lab_df["incremental_vs_no_action"].abs() < clv.where(clv > 0, 1.0) * 0.02)

    if "predicted_median_time_to_churn_days" in lab_df.columns:
        lab_df["expected_churn_period"] = _series_or_default(lab_df, "predicted_median_time_to_churn_days", 90.0)
    else:
        lab_df["expected_churn_period"] = window_days

    if "recommended_action" not in lab_df.columns:
        lab_df["recommended_action"] = lab_df["final_recommendation"]
    else:
        lab_df["recommended_action"] = lab_df["recommended_action"].fillna(lab_df["final_recommendation"])

    lab_df["recommendation_reason"] = lab_df.apply(_build_reason, axis=1)
    lab_df["model_caveat"] = "예측으로 계산한 비교 결과입니다. 실제 추가 이익은 A/B 검증이나 검증용 미개입군으로 확인해야 합니다."

    summary = {
        "customer_count": int(lab_df["customer_id"].nunique()) if "customer_id" in lab_df.columns else int(len(lab_df)),
        "scenario_count": int(len(scenario_df)),
        "avg_incremental_vs_no_action": round(float(lab_df["incremental_vs_no_action"].mean()), 2) if not lab_df.empty else 0.0,
        "total_incremental_vs_no_action": round(float(lab_df["incremental_vs_no_action"].sum()), 2) if not lab_df.empty else 0.0,
        "positive_recommendation_count": int((lab_df["incremental_vs_no_action"] > 0).sum()),
        "ab_test_recommended_count": int(lab_df["ab_test_recommended"].sum()),
        "best_action_counts": lab_df["final_recommendation"].value_counts().astype(int).to_dict(),
        "avg_confidence_score": round(float(lab_df["confidence_score"].mean()), 4) if not lab_df.empty else 0.0,
        "model_note": "예상 순이익 = 유지될 것으로 기대되는 고객가치 - 개입 비용입니다. 이 값은 실제 집행 결과가 아니라 예측 기반 비교값입니다.",
    }

    scenario_df = scenario_df.sort_values(["customer_id", "expected_net_profit"], ascending=[True, False]).reset_index(drop=True)
    lab_df = lab_df.sort_values(["incremental_vs_no_action", "best_expected_net_profit", "churn_probability"], ascending=[False, False, False]).reset_index(drop=True)
    return summary, lab_df, scenario_df


__all__ = ["ACTION_CATALOG", "build_counterfactual_retention_lab"]
