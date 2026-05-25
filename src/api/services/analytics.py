from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from pathlib import Path

import pandas as pd

from src.optimization.policy import build_intensity_action_candidates, normalize, safe_numeric
from src.optimization.timing import load_survival_predictions

DEFAULT_CUSTOMER_COLUMNS = [
    'customer_id', 'persona', 'uplift_segment_true', 'acquisition_month', 'recency_days', 'frequency', 'monetary',
    'churn_probability', 'uplift_score', 'clv', 'coupon_cost', 'expected_incremental_profit',
    'expected_roi', 'uplift_segment', 'treatment_group', 'inactivity_days',
]

DEFAULT_SEGMENT_ORDER = [
    'Persuadables',
    'Sure Things',
    'Sleeping Dogs',
    'Lost Causes',
]

INTENSITY_ORDER = ['low', 'mid', 'high']


def _budget_sensitivity_grid(base_budget: int, step: int = 1_000_000) -> list[int]:
    """Return a compact budget grid centered on the operator's current budget.

    The dashboard needs a table-style sensitivity map, not an expensive chart.
    We therefore keep 1,000,000 KRW intervals around the current setting and
    include the exact current budget even when it is not aligned to the step.
    """
    base = max(int(base_budget or 0), 0)
    step = max(int(step or 1_000_000), 1)
    if base <= 30 * step:
        start = 0
        end = max(10 * step, base + 5 * step)
    else:
        start = max(0, base - 10 * step)
        end = base + 10 * step

    budgets = set(range(start, end + step, step))
    budgets.update({0, base, base + step})
    return sorted(int(v) for v in budgets if v >= 0)


def _select_budget_candidates(
    candidate: pd.DataFrame,
    budget: int,
    max_customers: Optional[int] = None,
) -> pd.DataFrame:
    """Apply the same one-action-per-customer greedy policy to a prepared pool."""
    if candidate.empty or int(budget or 0) <= 0:
        return candidate.head(0).copy()

    selected_rows: list[dict] = []
    used_customers: set[int] = set()
    spent = 0.0
    selection_cap = int(max_customers) if max_customers is not None and int(max_customers) > 0 else None
    high_intensity_cap = max(1, int((selection_cap or max(len(candidate), 1)) * 0.35))
    high_intensity_used = 0

    for row in candidate.itertuples(index=False):
        if selection_cap is not None and len(selected_rows) >= selection_cap:
            break
        customer_id = int(getattr(row, "customer_id"))
        cost = float(getattr(row, "coupon_cost", 0.0))
        if customer_id in used_customers:
            continue
        if cost <= 0:
            continue
        if spent + cost > float(budget):
            continue
        intensity_value = str(getattr(row, "intervention_intensity", "")).lower()
        if intensity_value == "high" and high_intensity_used >= high_intensity_cap:
            continue
        selected_rows.append(row._asdict())
        used_customers.add(customer_id)
        spent += cost
        if intensity_value == "high":
            high_intensity_used += 1

    if not selected_rows:
        return candidate.head(0).copy()
    return pd.DataFrame(selected_rows)


def build_budget_sensitivity_map(
    customers: pd.DataFrame,
    base_budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
    survival_predictions: Optional[pd.DataFrame] = None,
    budget_step: int = 1_000_000,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Evaluate how profit, ROI, target count, and marginal gain change by budget.

    This function intentionally reuses one candidate pool across all budget
    levels.  It avoids rebuilding dose/timing candidates for every row, so the
    dashboard can show a sensitivity table without making the budget view slow.
    """
    base_budget = max(int(base_budget or 0), 0)
    budget_step = max(int(budget_step or 1_000_000), 1)

    empty_summary: Dict[str, object] = {
        "current_budget": base_budget,
        "current_spent": 0.0,
        "current_target_count": 0,
        "current_expected_profit": 0.0,
        "current_average_roi": 0.0,
        "next_1m_expected_profit_gain": 0.0,
        "current_marginal_roi": 0.0,
        "saturation_budget": None,
        "saturation_label": "분석 가능한 후보 고객이 없습니다.",
        "low_efficiency_budget": None,
        "low_efficiency_label": "분석 가능한 후보 고객이 없습니다.",
        "candidate_customers": 0,
        "budget_step": budget_step,
    }

    if customers.empty:
        return empty_summary, pd.DataFrame()

    candidate = _build_candidate_pool(customers, threshold=threshold, survival_predictions=survival_predictions)
    if candidate.empty:
        return empty_summary, pd.DataFrame()

    budgets = _budget_sensitivity_grid(base_budget, step=budget_step)
    rows: list[dict] = []
    previous: dict | None = None

    for budget_value in budgets:
        selected = _select_budget_candidates(candidate, budget=budget_value, max_customers=max_customers)
        spent = float(selected["coupon_cost"].sum()) if not selected.empty and "coupon_cost" in selected.columns else 0.0
        profit = float(selected["expected_incremental_profit"].sum()) if not selected.empty and "expected_incremental_profit" in selected.columns else 0.0
        target_count = int(selected["customer_id"].nunique()) if not selected.empty and "customer_id" in selected.columns else 0
        avg_roi = float(profit / spent) if spent > 0 else 0.0

        if previous is None:
            added_budget = 0.0
            added_spend = 0.0
            added_profit = 0.0
            added_targets = 0
            marginal_profit_per_1m = 0.0
            marginal_roi = 0.0
            status = "기준 구간"
            interpretation = "비교를 시작하기 위한 기준 예산 구간입니다."
        else:
            added_budget = float(budget_value - previous["budget"])
            added_spend = float(spent - previous["spent"])
            added_profit = float(profit - previous["expected_incremental_profit"])
            added_targets = int(target_count - previous["target_count"])
            marginal_profit_per_1m = float(added_profit / added_budget * 1_000_000) if added_budget > 0 else 0.0
            marginal_roi = float(added_profit / added_spend) if added_spend > 0 else 0.0

            if added_profit <= 0 and added_targets <= 0:
                status = "포화 또는 낭비 주의"
                interpretation = "예산을 더 늘려도 추가로 선택되는 고객이나 기대 순이익이 거의 없습니다."
            elif marginal_roi < 0.2:
                status = "효율 낮음"
                interpretation = "추가 예산 대비 기대 순이익이 낮아 예산 확대를 신중히 검토해야 합니다."
            elif avg_roi < float(previous.get("average_roi", 0.0)):
                status = "ROI 하락 시작"
                interpretation = "타깃 고객은 늘지만 평균 ROI가 낮아지기 시작하는 구간입니다."
            elif marginal_roi >= avg_roi:
                status = "확대 검토 가능"
                interpretation = "추가 예산의 효율이 현재 평균과 비슷하거나 더 높아 확대를 검토할 수 있습니다."
            else:
                status = "점진 확대 가능"
                interpretation = "추가 예산의 효율은 양호하지만 평균 효율보다 낮아 단계적 확대가 적절합니다."

        row = {
            "budget": int(budget_value),
            "spent": round(spent, 2),
            "remaining": round(max(float(budget_value) - spent, 0.0), 2),
            "target_count": int(target_count),
            "expected_incremental_profit": round(profit, 2),
            "average_roi": round(avg_roi, 6),
            "added_budget": round(added_budget, 2),
            "added_spend": round(added_spend, 2),
            "added_target_count": int(added_targets),
            "added_profit": round(added_profit, 2),
            "marginal_profit_per_1m": round(marginal_profit_per_1m, 2),
            "marginal_roi": round(marginal_roi, 6),
            "budget_status": status,
            "operator_message": interpretation,
        }
        rows.append(row)
        previous = {
            "budget": int(budget_value),
            "spent": spent,
            "target_count": target_count,
            "expected_incremental_profit": profit,
            "average_roi": avg_roi,
        }

    table = pd.DataFrame(rows)
    if table.empty:
        return empty_summary, table

    current_idx = (table["budget"] - base_budget).abs().idxmin()
    current_row = table.loc[current_idx]
    next_budget = base_budget + budget_step
    next_candidates = table[table["budget"] >= next_budget]
    next_row = next_candidates.iloc[0] if not next_candidates.empty else current_row

    positive_marginal = pd.to_numeric(table.get("marginal_profit_per_1m", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    best_marginal = float(positive_marginal.max()) if not positive_marginal.empty else 0.0
    saturation_threshold = max(best_marginal * 0.15, 0.0)
    saturation_df = table[(table["budget"] > 0) & (table["marginal_profit_per_1m"] <= saturation_threshold) & (table["added_target_count"] <= 0)]
    if saturation_df.empty and best_marginal > 0:
        saturation_df = table[(table["budget"] > 0) & (table["marginal_profit_per_1m"] <= saturation_threshold)]
    if saturation_df.empty:
        saturation_budget = None
        saturation_label = "현재 분석 범위에서는 뚜렷한 포화점이 보이지 않습니다."
    else:
        saturation_budget = int(saturation_df.iloc[0]["budget"])
        saturation_label = f"약 {saturation_budget:,}원 이후부터 추가 예산 효율이 크게 낮아질 수 있습니다."

    low_eff_df = table[(table["budget"] > 0) & ((table["marginal_roi"] < 0.2) | (table["marginal_profit_per_1m"] <= 0))]
    if low_eff_df.empty:
        low_efficiency_budget = None
        low_efficiency_label = "현재 분석 범위에서는 명확한 저효율 예산 구간이 확인되지 않습니다."
    else:
        low_efficiency_budget = int(low_eff_df.iloc[0]["budget"])
        low_efficiency_label = f"약 {low_efficiency_budget:,}원 구간부터 추가 집행 효율을 점검하는 것이 좋습니다."

    summary: Dict[str, object] = {
        "current_budget": int(base_budget),
        "current_spent": float(current_row.get("spent", 0.0)),
        "current_target_count": int(current_row.get("target_count", 0)),
        "current_expected_profit": float(current_row.get("expected_incremental_profit", 0.0)),
        "current_average_roi": float(current_row.get("average_roi", 0.0)),
        "next_1m_expected_profit_gain": float(next_row.get("added_profit", 0.0)) if int(next_row.get("budget", base_budget)) >= next_budget else 0.0,
        "current_marginal_roi": float(current_row.get("marginal_roi", 0.0)),
        "saturation_budget": saturation_budget,
        "saturation_label": saturation_label,
        "low_efficiency_budget": low_efficiency_budget,
        "low_efficiency_label": low_efficiency_label,
        "candidate_customers": int(candidate["customer_id"].nunique()) if "customer_id" in candidate.columns else int(len(candidate)),
        "budget_step": int(budget_step),
    }
    return summary, table


def get_churn_status(customers: pd.DataFrame, threshold: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    df = customers.copy()
    if df.empty:
        summary = {
            'total_customers': 0,
            'at_risk_customers': 0,
            'risk_rate': 0.0,
            'avg_churn_prob': 0.0,
        }
        return summary, df

    df['churn_probability'] = pd.to_numeric(df['churn_probability'], errors='coerce').fillna(0.0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)
    df['is_churn_risk'] = df['churn_probability'] >= float(threshold)

    summary = {
        'total_customers': int(len(df)),
        'at_risk_customers': int(df['is_churn_risk'].sum()),
        'risk_rate': float(df['is_churn_risk'].mean()) if len(df) else 0.0,
        'avg_churn_prob': float(df['churn_probability'].mean()) if len(df) else 0.0,
    }
    risk_customers = df[df['is_churn_risk']].sort_values(['churn_probability', 'clv'], ascending=[False, False])
    return summary, risk_customers


def get_top_high_value_customers(customers: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    df = customers.copy()
    if df.empty:
        return df.head(0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)
    df['uplift_score'] = pd.to_numeric(df.get('uplift_score', 0.0), errors='coerce').fillna(0.0)
    df['value_score'] = df['clv'] * df['uplift_score']
    ranked = df.sort_values(['value_score', 'clv', 'customer_id'], ascending=[False, False, True])
    if top_n is None or int(top_n) <= 0:
        return ranked
    return ranked.head(int(top_n))


def get_retention_targets(customers: pd.DataFrame, threshold: float, top_n: int | None = None) -> pd.DataFrame:
    df = customers.copy()
    if df.empty:
        return df.head(0)

    df['churn_probability'] = pd.to_numeric(df.get('churn_probability', 0.0), errors='coerce').fillna(0.0)
    df['uplift_score'] = pd.to_numeric(df.get('uplift_score', 0.0), errors='coerce').fillna(0.0)
    df['clv'] = pd.to_numeric(df.get('clv', 0.0), errors='coerce').fillna(0.0)

    condition = (
        (df['churn_probability'] >= float(threshold))
        & (df['uplift_score'] > 0.08)
        & (df['clv'] > df['clv'].median())
        & (df['uplift_segment'] != 'Sleeping Dogs')
    )
    target = df[condition].copy()
    if target.empty:
        return target

    max_clv = max(float(target['clv'].max()), 1.0)
    target['priority_score'] = (
        0.45 * target['churn_probability']
        + 0.25 * target['uplift_score']
        + 0.30 * (target['clv'] / max_clv)
    )
    ranked = target.sort_values(['priority_score', 'expected_roi', 'customer_id'], ascending=[False, False, True])
    if top_n is None or int(top_n) <= 0:
        return ranked
    return ranked.head(int(top_n))


def _segment_order(customers: pd.DataFrame) -> List[str]:
    present = [str(x) for x in customers.get('uplift_segment', pd.Series(dtype=object)).dropna().unique()]
    ordered = [seg for seg in DEFAULT_SEGMENT_ORDER if seg in present]
    remaining = sorted(seg for seg in present if seg not in ordered)
    if not ordered and not remaining:
        return DEFAULT_SEGMENT_ORDER.copy()
    return ordered + remaining




def _first_numeric_series(df: pd.DataFrame, columns: list[str], default: float | None = None) -> pd.Series:
    """Return the first usable numeric value across aliases, row by row."""
    out = pd.Series(float("nan"), index=df.index, dtype="float64")
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), values)
    if default is not None:
        out = out.fillna(float(default))
    return out


def _first_positive_numeric_series(df: pd.DataFrame, columns: list[str], default: float | None = None) -> pd.Series:
    """Return the first positive numeric value across aliases, row by row.

    Live DB rows often contain zero-filled placeholder columns.  For budget
    optimization, a placeholder 0 for CLV/uplift/profit/cost must be treated as
    missing; otherwise every customer becomes either free, unprofitable, or both.
    """
    out = pd.Series(float("nan"), index=df.index, dtype="float64")
    for col in columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.where(values > 0)
        out = out.where(out.notna(), values)
    if default is not None:
        out = out.fillna(float(default))
    return out


def _ensure_budgetable_customer_columns(customers: pd.DataFrame) -> pd.DataFrame:
    """Fill only missing/zero economics needed by the budget optimizer.

    Offline artifacts may already have real coupon_cost, CLV, uplift and profit;
    those positive values are preserved.  PostgreSQL user-live score rows can be
    sparse or zero-filled, so missing economics are derived only as a fallback.
    This keeps the existing offline pipeline intact while making the live budget
    view respond to the current sidebar budget.
    """
    if customers.empty:
        return customers.copy()

    df = customers.copy()

    if "churn_probability" not in df.columns and "churn_score" in df.columns:
        df["churn_probability"] = pd.to_numeric(df["churn_score"], errors="coerce")
    df["churn_probability"] = _first_numeric_series(df, ["churn_probability", "churn_score", "risk_score"], 0.0).clip(0.0, 1.0)

    clv_aliases = ["clv", "predicted_clv_12m", "predicted_clv", "customer_lifetime_value", "ltv"]
    clv = _first_positive_numeric_series(df, clv_aliases, None)
    if "monetary" in df.columns:
        monetary_based = pd.to_numeric(df["monetary"], errors="coerce") * 6.0
        clv = clv.where((clv.notna()) & (clv > 0), monetary_based.where(monetary_based > 0))
    # Zero-filled live CLV is a placeholder, not a real zero-value customer.
    df["clv"] = clv.fillna(50_000.0).clip(lower=1.0)

    uplift = _first_positive_numeric_series(df, ["uplift_score", "uplift", "incremental_response", "treatment_effect"], None)
    profit_alias = _first_positive_numeric_series(
        df,
        ["expected_incremental_profit", "expected_profit", "incremental_profit", "expected_net_profit"],
        None,
    )
    # If only profit and CLV are available, infer uplift. Otherwise use a small
    # conservative default so live score rows can still be ranked by budget.
    implied_uplift = profit_alias / df["clv"].where(df["clv"] > 0, pd.NA)
    uplift = uplift.where((uplift.notna()) & (uplift > 0), implied_uplift)
    df["uplift_score"] = uplift.fillna(0.03).clip(lower=0.001)

    roi = _first_positive_numeric_series(df, ["expected_roi", "roi"], None)
    strategy_cost = _first_positive_numeric_series(df, ["strategy_cost", "coupon_cost", "action_cost", "intervention_cost", "cost"], None)
    implied_cost_from_roi = profit_alias / roi.where(roi > 0, pd.NA)
    # The previous fallback often produced 1,000~1,500 KRW costs, so the default
    # 5M budget could already saturate the target cap and the metric cards looked
    # fixed.  Use a still-conservative but budget-visible live fallback cost.
    value_based_cost = (df["clv"] * 0.12).clip(lower=10_000.0, upper=80_000.0)
    strategy_cost = strategy_cost.where((strategy_cost.notna()) & (strategy_cost > 0), implied_cost_from_roi)
    strategy_cost = strategy_cost.where((strategy_cost.notna()) & (strategy_cost > 0), value_based_cost)
    df["strategy_cost"] = strategy_cost.fillna(10_000.0).clip(lower=1.0)

    # Keep coupon_cost available for downstream summaries, but do not overwrite a
    # meaningful non-zero value from existing offline artifacts.
    coupon_cost = _first_positive_numeric_series(df, ["coupon_cost"], None)
    df["coupon_cost"] = coupon_cost.where((coupon_cost.notna()) & (coupon_cost > 0), df["strategy_cost"]).fillna(df["strategy_cost"])

    profit = profit_alias
    profit_from_roi = roi.where(roi > 0, pd.NA) * df["strategy_cost"]
    profit_from_scores = df["clv"] * df["uplift_score"] * df["churn_probability"].clip(lower=0.50)
    profit = profit.where((profit.notna()) & (profit > 0), profit_from_roi)
    profit = profit.where((profit.notna()) & (profit > 0), profit_from_scores)
    df["expected_incremental_profit"] = profit.fillna(0.0).clip(lower=0.0)

    df["expected_roi"] = roi.where(
        (roi.notna()) & (roi > 0),
        df["expected_incremental_profit"] / df["strategy_cost"].where(df["strategy_cost"] > 0, pd.NA),
    ).fillna(0.0)

    if "uplift_segment" not in df.columns:
        df["uplift_segment"] = "Persuadables"
    else:
        segment = df["uplift_segment"].fillna("").astype(str).str.strip()
        missing_segment = segment.eq("") | segment.str.lower().isin({"nan", "none", "unknown", "live", "unknown_segment"})
        inferred = pd.Series("Persuadables", index=df.index, dtype=object)
        inferred = inferred.where(df["uplift_score"] >= 0.02, "Sure Things")
        inferred = inferred.where(df["expected_incremental_profit"] > 0, "Lost Causes")
        df["uplift_segment"] = segment.where(~missing_segment, inferred)

    return df


def _build_candidate_pool(
    customers: pd.DataFrame,
    threshold: float,
    survival_predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if customers.empty:
        return customers.head(0).copy()

    df = _ensure_budgetable_customer_columns(customers)
    df["churn_probability"] = safe_numeric(df.get("churn_probability"), default=0.0)
    df["uplift_score"] = safe_numeric(df.get("uplift_score"), default=0.0)
    df["clv"] = safe_numeric(df.get("clv"), default=0.0)
    df["coupon_cost"] = safe_numeric(df.get("coupon_cost"), default=0.0)
    df["strategy_cost"] = safe_numeric(df.get("strategy_cost"), default=0.0)
    df["expected_incremental_profit"] = safe_numeric(df.get("expected_incremental_profit"), default=0.0)
    df["expected_roi"] = safe_numeric(df.get("expected_roi"), default=0.0)

    candidate = build_intensity_action_candidates(df, survival_predictions=survival_predictions)
    candidate["optimization_score"] = candidate["expected_incremental_profit"] / candidate["coupon_cost"].where(
        candidate["coupon_cost"] > 0,
        1.0,
    )
    candidate = candidate[
        (candidate["churn_probability"] >= float(threshold))
        & (candidate["uplift_score"] > 0.0)
        & (candidate["expected_incremental_profit"] > 0.0)
        & (candidate["coupon_cost"] > 0.0)
        & (~candidate.get("uplift_segment", pd.Series("", index=candidate.index)).astype(str).isin(["Sleeping Dogs"]))
    ].copy()

    if candidate.empty:
        return candidate

    candidate["roi_rank_score"] = normalize(candidate["expected_roi"])
    candidate["profit_rank_score"] = normalize(candidate["expected_incremental_profit"])
    candidate["clv_rank_score"] = normalize(safe_numeric(candidate.get("clv"), default=0.0))
    candidate["timing_rank_score"] = normalize(candidate["timing_urgency_score"])
    candidate["window_rank_score"] = 1.0 - normalize(candidate["intervention_window_days"])
    candidate["intensity_fit_rank_score"] = normalize(candidate["intensity_effect_multiplier"])
    candidate["optimization_rank_score"] = normalize(candidate["optimization_score"])

    candidate["priority_score"] = (
        0.18 * candidate["roi_rank_score"]
        + 0.18 * candidate["profit_rank_score"]
        + 0.14 * candidate["churn_probability"]
        + 0.10 * candidate["uplift_score"]
        + 0.10 * candidate["clv_rank_score"]
        + 0.12 * candidate["timing_rank_score"]
        + 0.08 * candidate["window_rank_score"]
        + 0.10 * candidate["intensity_fit_rank_score"]
    )

    candidate["selection_score"] = 0.55 * candidate["priority_score"] + 0.45 * candidate["optimization_rank_score"]

    candidate = candidate.sort_values(
        [
            "selection_score",
            "priority_score",
            "optimization_score",
            "timing_urgency_score",
            "expected_roi",
            "expected_incremental_profit",
            "intervention_window_days",
            "coupon_cost",
            "customer_id",
        ],
        ascending=[False, False, False, False, False, False, True, True, True],
    ).reset_index(drop=True)
    return candidate


def budget_allocation_by_segment(
    selected_customers: pd.DataFrame,
    all_segments: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    all_segments = list(all_segments or DEFAULT_SEGMENT_ORDER)

    if selected_customers.empty:
        return pd.DataFrame(
            [
                {
                    "uplift_segment": segment,
                    "intervention_intensity": intensity,
                    "customer_count": 0,
                    "allocated_budget": 0.0,
                    "expected_profit": 0.0,
                }
                for segment in all_segments
                for intensity in INTENSITY_ORDER
            ]
        )

    grouped = (
        selected_customers.groupby(["uplift_segment", "intervention_intensity"], as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            allocated_budget=("coupon_cost", "sum"),
            expected_profit=("expected_incremental_profit", "sum"),
        )
        .reset_index(drop=True)
    )
    full_index = pd.MultiIndex.from_product([all_segments, INTENSITY_ORDER], names=["uplift_segment", "intervention_intensity"])
    grouped = grouped.set_index(["uplift_segment", "intervention_intensity"]).reindex(full_index, fill_value=0).reset_index()
    grouped["customer_count"] = grouped["customer_count"].astype(int)
    grouped["allocated_budget"] = pd.to_numeric(grouped["allocated_budget"], errors="coerce").fillna(0.0)
    grouped["expected_profit"] = pd.to_numeric(grouped["expected_profit"], errors="coerce").fillna(0.0)
    grouped["intensity_order"] = grouped["intervention_intensity"].map({key: idx for idx, key in enumerate(INTENSITY_ORDER)})
    grouped = grouped.sort_values(["uplift_segment", "intensity_order"]).drop(columns=["intensity_order"]).reset_index(drop=True)
    return grouped


def allocate_budget(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    selected, summary, _ = get_budget_result(
        customers=customers,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
    )
    return selected, summary


def get_budget_result(
    customers: pd.DataFrame,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
    survival_predictions: Optional[pd.DataFrame] = None,
    result_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    if customers.empty or budget <= 0:
        empty = customers.head(0).copy()
        summary = {
            'budget': int(budget),
            'spent': 0,
            'remaining': int(max(budget, 0)),
            'num_targeted': 0,
            'candidate_customers': 0,
            'expected_incremental_profit': 0.0,
            'overall_roi': 0.0,
            'threshold': float(threshold),
            'max_customers_cap': int(max_customers or 0),
            'candidate_segment_counts': {seg: 0 for seg in _segment_order(customers)},
            'survival_enriched': False,
            'dose_response_enriched': False,
            'dose_response_model_version': None,
        }
        return empty, summary, budget_allocation_by_segment(empty, _segment_order(customers))

    all_segments = _segment_order(customers)
    resolved_survival = survival_predictions
    if resolved_survival is None and result_dir is not None:
        resolved_survival = load_survival_predictions(result_dir)
    candidate = _build_candidate_pool(customers, threshold=threshold, survival_predictions=resolved_survival)

    if candidate.empty:
        summary = {
            'budget': int(budget),
            'spent': 0,
            'remaining': int(budget),
            'num_targeted': 0,
            'candidate_customers': 0,
            'expected_incremental_profit': 0.0,
            'overall_roi': 0.0,
            'threshold': float(threshold),
            'max_customers_cap': int(max_customers or 0),
            'candidate_segment_counts': {seg: 0 for seg in all_segments},
            'survival_enriched': False,
            'dose_response_enriched': False,
            'dose_response_model_version': None,
        }
        return candidate, summary, budget_allocation_by_segment(candidate, all_segments)

    candidate_segment_counts = (
        candidate.groupby('uplift_segment')['customer_id'].nunique().reindex(all_segments, fill_value=0).astype(int).to_dict()
    )

    selected_rows: list[dict] = []
    used_customers: set[int] = set()
    spent = 0.0
    selection_cap = int(max_customers) if max_customers is not None and int(max_customers) > 0 else None
    high_intensity_cap = max(1, int((selection_cap or max(len(candidate), 1)) * 0.35))
    high_intensity_used = 0

    # Do not pre-seed one action per intensity.  A forced high/mid/low seed can
    # make small budget changes appear ineffective and may choose a high-cost
    # action before a better ROI action.  The single greedy pass below treats
    # all customer × intensity candidates uniformly while enforcing one action
    # per customer, budget cap, max-customer cap, and high-intensity cap.
    for row in candidate.itertuples(index=False):
        if selection_cap is not None and len(selected_rows) >= selection_cap:
            break
        customer_id = int(getattr(row, "customer_id"))
        cost = float(getattr(row, "coupon_cost", 0.0))
        if customer_id in used_customers:
            continue
        if cost <= 0:
            continue
        if spent + cost > float(budget):
            continue
        intensity_value = str(getattr(row, "intervention_intensity", "")).lower()
        if intensity_value == "high" and high_intensity_used >= high_intensity_cap:
            continue
        selected_rows.append(row._asdict())
        used_customers.add(customer_id)
        spent += cost
        if intensity_value == "high":
            high_intensity_used += 1

    if selected_rows:
        selected = pd.DataFrame(selected_rows)
    else:
        selected = candidate.head(0).copy()

    spent = float(selected['coupon_cost'].sum()) if not selected.empty else 0.0
    expected_profit = float(selected['expected_incremental_profit'].sum()) if not selected.empty else 0.0
    overall_roi = float(expected_profit / spent) if spent > 0 else 0.0

    summary = {
        'budget': int(budget),
        'spent': int(round(spent)),
        'remaining': int(round(budget - spent)),
        'num_targeted': int(selected['customer_id'].nunique()) if not selected.empty and 'customer_id' in selected.columns else 0,
        'candidate_customers': int(candidate['customer_id'].nunique()),
        'candidate_actions': int(len(candidate)),
        'selected_actions': int(len(selected)),
        'budget_binding': bool(spent >= float(budget) * 0.98) if budget > 0 else False,
        'expected_incremental_profit': round(expected_profit, 2),
        'overall_roi': round(overall_roi, 6),
        'threshold': float(threshold),
        'max_customers_cap': int(max_customers or len(candidate)),
        'candidate_segment_counts': candidate_segment_counts,
        'survival_enriched': bool(resolved_survival is not None and not resolved_survival.empty),
        'dose_response_enriched': bool(candidate.get('dose_response_enabled', pd.Series(dtype=bool)).fillna(False).any()) if not candidate.empty else False,
        'dose_response_model_version': str(candidate['dose_response_model_version'].iloc[0]) if not candidate.empty and 'dose_response_model_version' in candidate.columns else None,
        'selected_intensity_counts': selected['intervention_intensity'].value_counts().to_dict() if not selected.empty else {},
        'avg_timing_urgency_score': round(float(candidate['timing_urgency_score'].mean()), 6),
        'avg_intervention_window_days': round(float(candidate['intervention_window_days'].mean()), 2),
        'optimization_method': 'constrained multiple-choice greedy: one action per customer, positive uplift/profit only, Sleeping Dogs excluded, high-intensity cap applied',
        'high_intensity_cap': int(high_intensity_cap),
    }
    segment_allocation = budget_allocation_by_segment(selected, all_segments=all_segments)
    return selected, summary, segment_allocation


def distribution_table(df: pd.DataFrame, column: str, limit: int | None = None) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=['name', 'count', 'share'])
    counts = df[column].fillna('Unknown').astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    total = max(int(len(df)), 1)
    return pd.DataFrame(
        {
            'name': counts.index.tolist(),
            'count': counts.astype(int).tolist(),
            'share': (counts / total).astype(float).tolist(),
        }
    )
