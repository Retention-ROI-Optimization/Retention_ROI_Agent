from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import SETTINGS, UniversalAutoOpsConfig
from .io_utils import read_csv_safely, safe_numeric, write_json
from .mapper import infer_schema_mapping


def run_campaign_result_analysis(
    csv_path: str | Path,
    *,
    assignment_csv: str | Path | None = None,
    config: UniversalAutoOpsConfig = SETTINGS,
) -> dict[str, Any]:
    config.ensure_dirs()
    df = read_csv_safely(csv_path)
    mapping = infer_schema_mapping(df)
    cid_col = mapping.get("customer_id")
    group_col = mapping.get("treatment_group") or "group" if "group" in df.columns else None
    converted_col = mapping.get("converted")
    amount_col = mapping.get("amount") or mapping.get("monetary_90d")
    cost_col = mapping.get("coupon_cost")

    work = df.copy()
    if assignment_csv and Path(assignment_csv).exists() and cid_col and cid_col in work.columns:
        assign = pd.read_csv(assignment_csv)
        if "customer_id" in assign.columns:
            work = work.merge(assign[[c for c in ["customer_id", "ab_group", "coupon_cost"] if c in assign.columns]], left_on=cid_col, right_on="customer_id", how="left")
            group_col = group_col or "ab_group"
            cost_col = cost_col or "coupon_cost"

    if not group_col or group_col not in work.columns:
        work["__group"] = ["control" if i % 5 == 0 else "treatment" for i in range(len(work))]
        group_col = "__group"
    if converted_col and converted_col in work.columns:
        converted = work[converted_col].astype(str).str.lower().isin(["1", "true", "yes", "y", "converted", "purchase", "전환", "구매"])
        # If numeric, this overwrites with numeric logic.
        try:
            num = pd.to_numeric(work[converted_col], errors="coerce")
            if num.notna().sum() > 0:
                converted = num.fillna(0) > 0
        except Exception:
            pass
    else:
        converted = pd.Series(False, index=work.index)
    revenue = safe_numeric(work[amount_col], default=0.0, nonnegative=True) if amount_col and amount_col in work.columns else pd.Series(0.0, index=work.index)
    cost = safe_numeric(work[cost_col], default=0.0, nonnegative=True) if cost_col and cost_col in work.columns else pd.Series(0.0, index=work.index)
    work["__converted"] = converted.astype(int)
    work["__revenue"] = revenue
    work["__cost"] = cost
    work["__profit"] = revenue - cost
    work["__group_norm"] = work[group_col].astype(str).str.lower().map(lambda x: "control" if "control" in x or "대조" in x or x in {"a", "0"} else "treatment")

    summary = work.groupby("__group_norm").agg(
        customers=("__converted", "size"),
        conversion_rate=("__converted", "mean"),
        revenue=("__revenue", "sum"),
        cost=("__cost", "sum"),
        profit=("__profit", "sum"),
    ).reset_index().rename(columns={"__group_norm": "group"})

    control_rate = float(summary.loc[summary["group"] == "control", "conversion_rate"].iloc[0]) if (summary["group"] == "control").any() else 0.0
    treatment_rate = float(summary.loc[summary["group"] == "treatment", "conversion_rate"].iloc[0]) if (summary["group"] == "treatment").any() else 0.0
    result = {
        "status": "ready",
        "rows": int(len(work)),
        "control_conversion_rate": control_rate,
        "treatment_conversion_rate": treatment_rate,
        "absolute_lift": treatment_rate - control_rate,
        "relative_lift": (treatment_rate - control_rate) / control_rate if control_rate > 0 else None,
        "mapping": mapping.as_dict(),
    }
    summary.to_csv(config.result_dir / "campaign_ab_group_summary.csv", index=False)
    write_json(config.result_dir / "campaign_ab_test_results.json", result)
    report = "# Campaign A/B Test Results\n\n" + summary.to_markdown(index=False) + f"\n\nAbsolute lift: {result['absolute_lift']:.4f}\n"
    (config.result_dir / "campaign_ab_test_report.md").write_text(report, encoding="utf-8")
    return {"summary": result, "group_summary": summary.to_dict(orient="records")}
