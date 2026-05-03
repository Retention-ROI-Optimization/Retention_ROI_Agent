from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class SegmentationArtifacts:
    customer_segments: pd.DataFrame
    customer_segments_path: str
    summary_path: str
    visualization_path: str


def _assign_segment(row: pd.Series) -> str:
    high_value = bool(row["is_high_value_top20pct"])
    uplift_segment = str(row["uplift_segment"])
    is_new = bool(row.get("is_new_customer", False))
    if is_new:
        return "New Customers"
    if high_value and uplift_segment == "Persuadables":
        return "High Value-Persuadables"
    if high_value and uplift_segment == "Sure Things":
        return "High Value-Sure Things"
    if high_value and uplift_segment in {"Lost Causes", "Sleeping Dogs"}:
        return "High Value-Lost Causes"
    if (not high_value) and uplift_segment == "Persuadables":
        return "Low Value-Persuadables"
    if (not high_value) and uplift_segment in {"Lost Causes", "Sleeping Dogs"}:
        return "Low Value-Lost Causes"
    return "Low Value-Sure Things"


def _plot_segments(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(summary["segment_name"], summary["customer_count"])
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Segment")
    plt.ylabel("Customer Count")
    plt.title("Customer Segmentation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_segmentation_pipeline(result_dir: Path, data_dir: Path) -> SegmentationArtifacts:
    uplift = pd.read_csv(result_dir / "uplift_segmentation.csv")
    clv = pd.read_csv(result_dir / "clv_predictions.csv")
    customer_summary = pd.read_csv(data_dir / "customer_summary.csv")

    df = uplift.merge(
        clv[["customer_id", "predicted_clv_12m", "is_high_value_top20pct", "tenure_days"]],
        on="customer_id",
        how="left",
    ).merge(
        customer_summary[["customer_id", "persona", "churn_probability"]],
        on="customer_id",
        how="left",
        suffixes=("", "_summary"),
    )

    df["is_new_customer"] = (pd.to_numeric(df["tenure_days"], errors="coerce").fillna(0.0) < 90) | (df["persona"] == "new_signup")
    df["retention_priority_score"] = pd.to_numeric(df["predicted_uplift"], errors="coerce").fillna(0.0).clip(lower=0.0) * pd.to_numeric(df["predicted_clv_12m"], errors="coerce").fillna(0.0)
    df["customer_segment"] = df.apply(_assign_segment, axis=1)

    segment_summary = (
        df.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_clv=("predicted_clv_12m", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_uplift=("predicted_uplift", "mean"),
            avg_priority_score=("retention_priority_score", "mean"),
        )
        .rename(columns={"customer_segment": "segment_name"})
    )
    segment_summary["customer_ratio"] = segment_summary["customer_count"] / max(len(df), 1)
    segment_summary = segment_summary.sort_values(["avg_priority_score", "customer_count"], ascending=[False, False]).reset_index(drop=True)

    customer_segments_path = result_dir / "customer_segments.csv"
    summary_path = result_dir / "customer_segment_summary.json"
    visualization_path = result_dir / "customer_segments.png"

    df.sort_values(["retention_priority_score", "predicted_clv_12m"], ascending=[False, False]).to_csv(customer_segments_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "segment_count": int(segment_summary["segment_name"].nunique()),
                "segments": segment_summary.round(6).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_segments(segment_summary, visualization_path)

    return SegmentationArtifacts(
        customer_segments=df,
        customer_segments_path=str(customer_segments_path),
        summary_path=str(summary_path),
        visualization_path=str(visualization_path),
    )
