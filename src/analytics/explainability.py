from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ExplainabilityArtifacts:
    explanations_path: str
    summary_path: str
    markdown_path: str


RISK_RULES = [
    ("inactivity_days", "high", 0.80, "extended_inactivity", "장기간 비활성 상태가 이어졌습니다."),
    ("recency_days", "high", 0.80, "purchase_recency_drop", "최근 구매 공백이 길어졌습니다."),
    ("visit_change_rate_14d", "low", 0.20, "visit_decline", "최근 방문 빈도가 눈에 띄게 감소했습니다."),
    ("purchase_change_rate_14d", "low", 0.20, "purchase_decline", "최근 구매 빈도가 감소했습니다."),
    ("support_contact_30d", "high", 0.80, "support_contact_spike", "문의/불만성 접촉이 증가했습니다."),
    ("cart_conversion_change_rate", "low", 0.20, "cart_conversion_drop", "장바구니에서 구매로 이어지는 전환이 약해졌습니다."),
    ("purchase_cycle_anomaly", "high", 0.80, "purchase_cycle_break", "평소 구매 주기 대비 이탈 징후가 커졌습니다."),
]

LEVER_RULES = [
    ("uplift_score", "high", 0.70, "high_uplift", "개입 시 행동을 바꿀 가능성이 상대적으로 높습니다."),
    ("coupon_affinity", "high", 0.70, "coupon_affinity", "쿠폰/프로모션 반응 성향이 높습니다."),
    ("clv", "high", 0.75, "high_clv", "고객 생애가치가 높아 방어 우선순위가 큽니다."),
    ("timing_urgency_score", "high", 0.70, "timing_urgent", "개입 시점을 늦추면 효과가 빠르게 떨어질 수 있습니다."),
]

GUARDRAIL_RULES = [
    ("discount_pressure_score", "high", 0.75, "discount_pressure", "프로모션 과다 노출 위험이 커 강한 쿠폰 개입은 비효율적일 수 있습니다."),
    ("coupon_fatigue_score", "high", 0.75, "coupon_fatigue", "최근 혜택 노출이 누적돼 반응 피로가 관찰됩니다."),
    ("brand_sensitivity", "high", 0.70, "brand_sensitivity", "브랜드/정가 훼손 민감도가 높아 공격적 할인은 주의가 필요합니다."),
]


def _safe_numeric(series: pd.Series | Iterable[float] | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(float(default))


def _load_feature_labels(feature_store_dir: Path) -> dict[str, str]:
    metadata_path = feature_store_dir / "customer_features_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    labels = metadata.get("feature_dictionary") or metadata.get("feature_labels") or {}
    if not isinstance(labels, dict):
        return {}
    return {str(k): str(v) for k, v in labels.items()}


def _percentile_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame(index=df.index)
    numeric = numeric.loc[:, ~numeric.columns.duplicated()].copy()
    ranked = {col: _safe_numeric(numeric[col]).rank(pct=True, method="average") for col in numeric.columns}
    return pd.DataFrame(ranked, index=df.index)


def _collect_rules(row: pd.Series, pct_row: pd.Series, rules: list[tuple[str, str, float, str, str]]) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    for feature, direction, threshold, code, text in rules:
        if feature not in pct_row.index:
            continue
        value = float(pct_row.get(feature, 0.5))
        matched = value >= threshold if direction == "high" else value <= threshold
        if matched:
            findings.append((code, feature, text))
    return findings


def run_operational_explainability(
    data_dir: Path,
    result_dir: Path,
    feature_store_dir: Path,
    max_rows: int = 1500,
) -> ExplainabilityArtifacts:
    customer_summary = pd.read_csv(data_dir / "customer_summary.csv")
    feature_path = feature_store_dir / "customer_features.csv"
    features = pd.read_csv(feature_path) if feature_path.exists() else customer_summary.copy()
    labels = _load_feature_labels(feature_store_dir)

    targets_path = result_dir / "optimization_selected_customers.csv"
    if targets_path.exists():
        targets = pd.read_csv(targets_path)
    else:
        targets = customer_summary.sort_values(["churn_probability", "clv"], ascending=[False, False]).head(max_rows).copy()

    base = targets.merge(customer_summary, on="customer_id", how="left", suffixes=("", "_summary"))
    base = base.merge(features, on="customer_id", how="left", suffixes=("", "_feature"))
    if len(base) > max_rows:
        base = base.head(max_rows).copy()

    pct = _percentile_frame(base)
    rows = []
    for idx, row in base.iterrows():
        pct_row = pct.loc[idx] if idx in pct.index else pd.Series(dtype=float)
        risk_hits = _collect_rules(row, pct_row, RISK_RULES)
        lever_hits = _collect_rules(row, pct_row, LEVER_RULES)
        guardrail_hits = _collect_rules(row, pct_row, GUARDRAIL_RULES)

        if not risk_hits and float(row.get("churn_probability", 0.0)) >= 0.5:
            risk_hits.append(("high_churn_score", "churn_probability", "예측 이탈 점수가 높습니다."))
        if not lever_hits and float(row.get("expected_roi", 0.0)) > 0:
            lever_hits.append(("positive_expected_roi", "expected_roi", "예상 ROI가 양수여서 개입 타당성이 있습니다."))

        risk_text = " / ".join(text for _, _, text in risk_hits[:3])
        lever_text = " / ".join(text for _, _, text in lever_hits[:2])
        guardrail_text = " / ".join(text for _, _, text in guardrail_hits[:2]) or "특별한 과다혜택 경고 신호는 크지 않습니다."

        dominant_features = [labels.get(feature, feature) for _, feature, _ in (risk_hits[:2] + lever_hits[:2])]
        note = str(row.get("recommended_action", "")) if pd.notna(row.get("recommended_action", np.nan)) else ""
        if guardrail_hits and str(row.get("intervention_intensity", "")) == "high":
            note = (note + " | 고강도 개입 전 빈도 캡 필요").strip(" |")

        rows.append(
            {
                "customer_id": int(row.get("customer_id")),
                "persona": row.get("persona", row.get("persona_summary")),
                "uplift_segment": row.get("uplift_segment", row.get("uplift_segment_summary")),
                "risk_reason_codes": ", ".join(code for code, _, _ in risk_hits[:3]),
                "risk_reason_text": risk_text,
                "action_reason_codes": ", ".join(code for code, _, _ in lever_hits[:2]),
                "action_reason_text": lever_text,
                "guardrail_codes": ", ".join(code for code, _, _ in guardrail_hits[:2]),
                "guardrail_text": guardrail_text,
                "dominant_features": ", ".join(dominant_features),
                "recommended_action_note": note,
                "expected_roi": float(row.get("expected_roi", 0.0) or 0.0),
                "expected_incremental_profit": float(row.get("expected_incremental_profit", 0.0) or 0.0),
            }
        )

    explanation_df = pd.DataFrame(rows).sort_values(["expected_incremental_profit", "expected_roi", "customer_id"], ascending=[False, False, True])
    summary = {
        "rows": int(len(explanation_df)),
        "top_risk_reasons": explanation_df["risk_reason_codes"].fillna("").str.split(", ").explode().replace("", np.nan).dropna().value_counts().head(10).to_dict(),
        "top_guardrails": explanation_df["guardrail_codes"].fillna("").str.split(", ").explode().replace("", np.nan).dropna().value_counts().head(10).to_dict(),
        "generated_for": "optimization targets" if targets_path.exists() else "top-risk customers",
    }

    markdown_lines = [
        "# Operational explainability report",
        "",
        f"- Generated rows: **{len(explanation_df):,}**",
        f"- Scope: **{summary['generated_for']}**",
        "",
        "## Frequent risk reasons",
    ]
    for key, value in summary["top_risk_reasons"].items():
        markdown_lines.append(f"- {key}: {value}")
    markdown_lines.append("")
    markdown_lines.append("## Frequent guardrails")
    for key, value in summary["top_guardrails"].items():
        markdown_lines.append(f"- {key}: {value}")

    explanations_path = result_dir / "customer_operational_explanations.csv"
    summary_path = result_dir / "customer_operational_explanations_summary.json"
    markdown_path = result_dir / "customer_operational_explanations.md"
    explanation_df.to_csv(explanations_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return ExplainabilityArtifacts(str(explanations_path), str(summary_path), str(markdown_path))
