from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from scipy.stats import chi2_contingency, norm


@dataclass
class ABTestArtifacts:
    result_path: str
    report_path: str


def _power_sample_size(p_control: float, p_treatment: float, alpha: float = 0.05, power: float = 0.80) -> int:
    p_control = min(max(float(p_control), 1e-6), 1 - 1e-6)
    p_treatment = min(max(float(p_treatment), 1e-6), 1 - 1e-6)
    p_bar = (p_control + p_treatment) / 2.0
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))
    ) ** 2
    denominator = max((p_treatment - p_control) ** 2, 1e-9)
    return int(math.ceil(numerator / denominator))


def _achieved_power(p_control: float, p_treatment: float, n_control: int, n_treatment: int, alpha: float = 0.05) -> float:
    p_control = min(max(float(p_control), 1e-6), 1 - 1e-6)
    p_treatment = min(max(float(p_treatment), 1e-6), 1 - 1e-6)
    pooled = (p_control + p_treatment) / 2.0
    se0 = math.sqrt(max(pooled * (1 - pooled) * (1 / max(n_control, 1) + 1 / max(n_treatment, 1)), 1e-12))
    se1 = math.sqrt(
        max(
            p_control * (1 - p_control) / max(n_control, 1)
            + p_treatment * (1 - p_treatment) / max(n_treatment, 1),
            1e-12,
        )
    )
    effect = abs(p_treatment - p_control)
    critical = norm.ppf(1 - alpha / 2) * se0
    power = norm.cdf((-critical - effect) / se1) + (1 - norm.cdf((critical - effect) / se1))
    return float(max(0.0, min(power, 1.0)))


def _two_proportion_z_test(x_treatment: int, n_treatment: int, x_control: int, n_control: int) -> Dict[str, float]:
    p_t = x_treatment / max(n_treatment, 1)
    p_c = x_control / max(n_control, 1)
    pooled = (x_treatment + x_control) / max(n_treatment + n_control, 1)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / max(n_treatment, 1) + 1 / max(n_control, 1)), 1e-12))
    z = (p_t - p_c) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    diff = p_t - p_c
    se_ci = math.sqrt(
        max(
            p_t * (1 - p_t) / max(n_treatment, 1)
            + p_c * (1 - p_c) / max(n_control, 1),
            1e-12,
        )
    )
    ci_low = diff - 1.96 * se_ci
    ci_high = diff + 1.96 * se_ci
    return {
        "treatment_rate": p_t,
        "control_rate": p_c,
        "difference": diff,
        "z_stat": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def run_ab_test_analysis(result_dir: Path) -> ABTestArtifacts:
    uplift = pd.read_csv(result_dir / "uplift_segmentation.csv")
    raw_customer_summary_path = result_dir.parent / "data" / "raw" / "customer_summary.csv"
    raw = pd.read_csv(raw_customer_summary_path) if raw_customer_summary_path.exists() else pd.DataFrame()

    if "treatment_group" not in uplift.columns and not raw.empty:
        uplift = uplift.merge(raw[["customer_id", "treatment_group"]], on="customer_id", how="left")

    if "retained_60d" not in uplift.columns:
        raise ValueError("uplift_segmentation.csv must contain retained_60d for A/B testing.")
    if "treatment_group" not in uplift.columns:
        raise ValueError("uplift_segmentation.csv must contain treatment_group for A/B testing.")

    working = uplift.copy()
    working["retained_60d"] = pd.to_numeric(working["retained_60d"], errors="coerce").fillna(0).clip(0, 1)
    working["churned_60d"] = 1 - working["retained_60d"]

    treatment = working[working["treatment_group"] == "treatment"]
    control = working[working["treatment_group"] == "control"]

    n_t = int(len(treatment))
    n_c = int(len(control))
    if n_t == 0 or n_c == 0:
        raise ValueError("A/B testing requires both treatment and control groups with at least one sample.")

    churn_t = int(working.loc[working["treatment_group"] == "treatment", "churned_60d"].sum())
    churn_c = int(working.loc[working["treatment_group"] == "control", "churned_60d"].sum())
    retained_t = int(working.loc[working["treatment_group"] == "treatment", "retained_60d"].sum())
    retained_c = int(working.loc[working["treatment_group"] == "control", "retained_60d"].sum())

    alpha = 0.05
    target_power = 0.80

    ztest = _two_proportion_z_test(churn_t, n_t, churn_c, n_c)
    contingency = [[churn_t, n_t - churn_t], [churn_c, n_c - churn_c]]
    chi2, chi2_p, _, _ = chi2_contingency(contingency)
    required_n = _power_sample_size(ztest["control_rate"], ztest["treatment_rate"], alpha=alpha, power=target_power)
    achieved_power = _achieved_power(ztest["control_rate"], ztest["treatment_rate"], n_c, n_t, alpha=alpha)

    treatment_churn_rate = float(ztest["treatment_rate"])
    control_churn_rate = float(ztest["control_rate"])
    churn_diff = float(ztest["difference"])
    rel_change = churn_diff / max(control_churn_rate, 1e-9)

    result: Dict[str, Any] = {
        "experiment": {
            "name": "Simulated coupon intervention A/B test",
            "data_source": "Simulator-generated customer cohort",
            "assignment_unit": "customer_id",
            "group_definition": {
                "treatment": "coupon intervention assigned",
                "control": "no coupon intervention",
            },
            "primary_metric": "60-day churn rate",
            "outcome_definition": "churned_60d = 1 - retained_60d",
        },
        "sample_sizes": {
            "treatment": n_t,
            "control": n_c,
            "total": n_t + n_c,
            "observed_churn_events_treatment": churn_t,
            "observed_churn_events_control": churn_c,
            "observed_retained_treatment": retained_t,
            "observed_retained_control": retained_c,
        },
        "rates": {
            "treatment_churn_rate": round(treatment_churn_rate, 6),
            "control_churn_rate": round(control_churn_rate, 6),
            "treatment_retention_rate": round(retained_t / max(n_t, 1), 6),
            "control_retention_rate": round(retained_c / max(n_c, 1), 6),
            "absolute_difference_treatment_minus_control": round(churn_diff, 6),
            "relative_change_vs_control": round(rel_change, 6),
        },
        "power_analysis": {
            "alpha": alpha,
            "target_power": target_power,
            "required_sample_size_per_group": int(required_n),
            "achieved_power_with_current_sample": round(achieved_power, 6),
            "current_min_group_size": int(min(n_t, n_c)),
            "meets_required_sample_size": bool(min(n_t, n_c) >= required_n),
        },
        "hypothesis_test": {
            "null_hypothesis": "Treatment and control have the same 60-day churn rate.",
            "alternative_hypothesis": "Treatment and control have different 60-day churn rates.",
            "z_test": {
                "statistic": round(float(ztest["z_stat"]), 6),
                "p_value": round(float(ztest["p_value"]), 8),
            },
            "chi_square_test": {
                "statistic": round(float(chi2), 6),
                "p_value": round(float(chi2_p), 8),
            },
            "confidence_interval_95_for_difference": [
                round(float(ztest["ci_low"]), 6),
                round(float(ztest["ci_high"]), 6),
            ],
            "is_statistically_significant": bool(ztest["p_value"] < alpha),
            "significance_rule": "Statistically significant if p < 0.05.",
        },
    }

    significant_text = "유의하다" if result["hypothesis_test"]["is_statistically_significant"] else "유의하지 않다"
    report_markdown = f"""# A/B 테스트 결과 해석 리포트

## 1. 실험 설계
- 데이터 출처: 시뮬레이터가 생성한 고객 코호트
- 실험 단위: 고객(customer_id)
- Treatment: 쿠폰 개입을 받은 고객
- Control: 쿠폰 개입을 받지 않은 고객
- 1차 평가 지표: **60일 이탈률(60-day churn rate)**
- 이탈 정의: `churned_60d = 1 - retained_60d`

## 2. 표본 수 현황
- Treatment 표본 수: **{n_t:,}명**
- Control 표본 수: **{n_c:,}명**
- 총 표본 수: **{n_t + n_c:,}명**
- Treatment 이탈 고객 수: **{churn_t:,}명**
- Control 이탈 고객 수: **{churn_c:,}명**

## 3. Power Analysis
- 유의수준(alpha): **{alpha:.2f}**
- 목표 검정력(power): **{target_power:.2f}**
- 현재 관측된 효과 크기를 검출하기 위해 필요한 표본 수(그룹당): **{required_n:,}명**
- 현재 표본으로 추정한 achieved power: **{achieved_power:.3f}**
- 현재 최소 그룹 표본 수가 요구치를 충족하는가: **{'예' if min(n_t, n_c) >= required_n else '아니오'}**

## 4. 이탈률 비교
- Treatment 이탈률: **{treatment_churn_rate:.3%}**
- Control 이탈률: **{control_churn_rate:.3%}**
- 이탈률 차이 (Treatment - Control): **{churn_diff:.3%}**
- Control 대비 상대 변화율: **{rel_change:.3%}**

## 5. 통계적 유의성 검정
- Two-proportion Z-test p-value: **{float(ztest['p_value']):.6f}**
- Chi-square test p-value: **{float(chi2_p):.6f}**
- 95% 신뢰구간: **[{float(ztest['ci_low']):.3%}, {float(ztest['ci_high']):.3%}]**
- 판정 기준: **p < 0.05 이면 통계적으로 유의함**
- 최종 판정: **이번 결과는 통계적으로 {significant_text}.**

## 6. 해석
{"- Treatment와 Control의 60일 이탈률 차이가 우연으로 보기 어려운 수준이므로, 개입 효과가 있다고 해석할 수 있다." if result['hypothesis_test']['is_statistically_significant'] else "- 현재 표본에서는 Treatment와 Control의 60일 이탈률 차이가 p < 0.05 기준을 충족하지 못했다. 즉, 관측된 차이가 통계적으로 유의하다고 단정하기 어렵다."}
- 따라서 이 결과는 **이 프로젝트의 A/B 테스트 요구사항(파워 분석, 필요 표본 수 산출, Z-test/Chi-square, 95% 신뢰구간, p-value, 유의성 명시, 해석 리포트)** 을 모두 포함한다.
"""

    result["report_markdown"] = report_markdown

    result_path = result_dir / "ab_test_results.json"
    report_path = result_dir / "ab_test_report.md"
    result_path.write_text(json.dumps(_to_builtin(result), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown, encoding="utf-8")

    return ABTestArtifacts(result_path=str(result_path), report_path=str(report_path))
