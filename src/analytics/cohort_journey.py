from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulator.cohort_analysis import build_cohort_retention
from src.simulator.config import DEFAULT_CONFIG


CANONICAL_ACTIVITY_DEFINITION = "core_engagement"
CANONICAL_RETENTION_MODE = "rolling"
RETENTION_MILESTONES = (1, 3, 6, 12)
MAJOR_PRE_CHURN_EVENTS = [
    "remove_from_cart",
    "support_contact",
    "add_to_cart",
    "search",
    "purchase",
    "coupon_open",
    "coupon_redeem",
    "visit",
    "page_view",
]


PLOT_STAGE_LABELS = {
    "가입": "Signup",
    "첫구매": "First Purchase",
    "재구매": "Repeat Purchase",
    "충성": "Loyal",
    "이탈": "Churn",
}

MEANINGFUL_EVENT_TYPES = [
    "visit",
    "page_view",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "support_contact",
]


@dataclass
class CohortJourneyArtifacts:
    summary_path: str
    retention_curve_path: str
    churn_heatmap_path: str
    retention_milestone_csv_path: str
    sequence_csv_path: str
    sequence_plot_path: str
    pre_churn_event_csv_path: str
    pre_churn_event_plot_path: str
    funnel_csv_path: str
    funnel_plot_path: str
    churn_timing_csv_path: str
    churn_timing_plot_path: str
    report_path: str


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"]),
        "events": pd.read_csv(data_dir / "events.csv", parse_dates=["timestamp"]),
        "orders": pd.read_csv(data_dir / "orders.csv", parse_dates=["order_time"]),
        "snapshots": pd.read_csv(data_dir / "state_snapshots.csv", parse_dates=["snapshot_date"]),
    }


def _load_or_build_canonical_cohort_curve(data_dir: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cohort_path = data_dir / "cohort_retention.csv"
    if cohort_path.exists():
        current = pd.read_csv(cohort_path)
        current["period"] = pd.to_numeric(current["period"], errors="coerce")
        has_required_schema = {"activity_definition", "retention_mode"}.issubset(current.columns)
        if has_required_schema:
            canonical = current[
                (current["activity_definition"].astype(str) == CANONICAL_ACTIVITY_DEFINITION)
                & (current["retention_mode"].astype(str) == CANONICAL_RETENTION_MODE)
            ].copy()
            if not canonical.empty and int(canonical["period"].max()) >= 12:
                return _canonical_cohort_view(canonical)

    canonical = build_cohort_retention(
        customers=tables["customers"],
        events=tables["events"],
        periods=13,
        end_date=DEFAULT_CONFIG.end_date,
        activity_definition=CANONICAL_ACTIVITY_DEFINITION,
        retention_mode=CANONICAL_RETENTION_MODE,
    )
    return _canonical_cohort_view(canonical)


def _canonical_cohort_view(cohort: pd.DataFrame) -> pd.DataFrame:
    df = cohort.copy()
    if "activity_definition" in df.columns:
        df = df[df["activity_definition"].astype(str) == CANONICAL_ACTIVITY_DEFINITION].copy()
    if "retention_mode" in df.columns:
        df = df[df["retention_mode"].astype(str) == CANONICAL_RETENTION_MODE].copy()
    df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
    df["retention_rate"] = pd.to_numeric(df["retention_rate"], errors="coerce")
    df["cohort_size"] = pd.to_numeric(df["cohort_size"], errors="coerce")
    return df.sort_values(["cohort_month", "period"]).reset_index(drop=True)


def _build_retention_milestone_table(cohort_curve: pd.DataFrame) -> pd.DataFrame:
    milestone_df = cohort_curve[cohort_curve["period"].isin(RETENTION_MILESTONES)].copy()
    milestone_df["churn_rate"] = 1.0 - milestone_df["retention_rate"]
    return milestone_df[["cohort_month", "period", "cohort_size", "retention_rate", "churn_rate", "observed"]].copy()


def _plot_retention_curve(cohort_curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plot_df = cohort_curve.dropna(subset=["retention_rate"]).copy()
    for cohort_month, grp in plot_df.groupby("cohort_month"):
        ax.plot(grp["period"], grp["retention_rate"], marker="o", linewidth=1.2, alpha=0.65, label=str(cohort_month))

    avg_curve = (
        plot_df.groupby("period", as_index=False)["retention_rate"]
        .mean()
        .sort_values("period")
    )
    if not avg_curve.empty:
        ax.plot(avg_curve["period"], avg_curve["retention_rate"], marker="o", linewidth=3.0, label="Average")

    for milestone in RETENTION_MILESTONES:
        ax.axvline(milestone, linestyle="--", linewidth=0.8)

    ax.set_title("Cohort Retention Curve (M1 / M3 / M6 / M12)")
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Retention rate")
    ax.set_xticks(range(0, 13))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_churn_heatmap(milestone_df: pd.DataFrame, output_path: Path) -> None:
    pivot = milestone_df.pivot(index="cohort_month", columns="period", values="churn_rate").reindex(columns=list(RETENTION_MILESTONES))
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    matrix = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto")
    ax.set_title("Cohort Churn-Rate Difference Heatmap")
    ax.set_xlabel("Milestone month")
    ax.set_ylabel("Acquisition cohort")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[f"M{int(col)}" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=list(pivot.index))

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            value = matrix[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Churn rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _final_churn_run_start(snapshots: pd.DataFrame) -> pd.DataFrame:
    snap = snapshots.sort_values(["customer_id", "snapshot_date"]).copy()
    rows: list[dict] = []
    for customer_id, grp in snap.groupby("customer_id", sort=False):
        statuses = grp["current_status"].astype(str).tolist()
        dates = grp["snapshot_date"].tolist()
        if not statuses or statuses[-1] != "churn_risk":
            continue
        idx = len(statuses) - 1
        while idx - 1 >= 0 and statuses[idx - 1] == "churn_risk":
            idx -= 1
        rows.append(
            {
                "customer_id": int(customer_id),
                "final_churn_start_date": pd.Timestamp(dates[idx]),
                "final_snapshot_date": pd.Timestamp(dates[-1]),
            }
        )
    return pd.DataFrame(rows)


def _resolve_behavior_anchor(events: pd.DataFrame, churners: pd.DataFrame) -> pd.DataFrame:
    if churners.empty:
        return churners.assign(behavior_anchor_time=pd.NaT, anchor_source="no_customer")

    relevant = events[events["customer_id"].isin(churners["customer_id"])].copy()
    meaningful = relevant[relevant["event_type"].isin(MEANINGFUL_EVENT_TYPES)].copy()
    last_meaningful = meaningful.groupby("customer_id", as_index=False)["timestamp"].max().rename(columns={"timestamp": "last_meaningful_time"})
    last_any = relevant.groupby("customer_id", as_index=False)["timestamp"].max().rename(columns={"timestamp": "last_any_time"})

    out = churners.merge(last_meaningful, on="customer_id", how="left").merge(last_any, on="customer_id", how="left")
    out["behavior_anchor_time"] = out["last_meaningful_time"].combine_first(out["last_any_time"])
    out["behavior_anchor_time"] = out["behavior_anchor_time"].fillna(out["final_churn_start_date"])
    out["anchor_source"] = np.select(
        [
            out["last_meaningful_time"].notna(),
            out["last_any_time"].notna(),
        ],
        ["last_meaningful_event", "last_any_event"],
        default="final_churn_start_date",
    )
    return out[["customer_id", "final_churn_start_date", "final_snapshot_date", "behavior_anchor_time", "anchor_source"]]


def _compress_sequence(event_types: Iterable[str]) -> list[str]:
    compact: list[str] = []
    prev = None
    for event_type in event_types:
        if event_type != prev:
            compact.append(str(event_type))
            prev = event_type
    return compact


def _extract_top5_patterns(events: pd.DataFrame, churners: pd.DataFrame) -> pd.DataFrame:
    if churners.empty:
        return pd.DataFrame(columns=["rank", "pattern", "customer_count", "share_of_churn_customers"])

    merged = events.merge(churners[["customer_id", "behavior_anchor_time"]], on="customer_id", how="inner")
    window = merged.loc[
        (merged["timestamp"] <= merged["behavior_anchor_time"])
        & (merged["timestamp"] >= merged["behavior_anchor_time"] - pd.Timedelta(days=30)),
        ["customer_id", "timestamp", "event_type"],
    ].sort_values(["customer_id", "timestamp"])

    sequence_rows: list[tuple[int, str]] = []
    seen_customers: set[int] = set()
    for customer_id, grp in window.groupby("customer_id", sort=False):
        compact = _compress_sequence(grp["event_type"].tolist())
        pattern = " > ".join(compact[-5:]) if compact else "no_history"
        sequence_rows.append((int(customer_id), pattern))
        seen_customers.add(int(customer_id))

    missing_customers = sorted(set(churners["customer_id"].astype(int)) - seen_customers)
    sequence_rows.extend((customer_id, "no_history") for customer_id in missing_customers)

    sequence_df = pd.DataFrame(sequence_rows, columns=["customer_id", "pattern"])
    top5 = (
        sequence_df.groupby("pattern", as_index=False)
        .agg(customer_count=("customer_id", "nunique"))
        .sort_values(["customer_count", "pattern"], ascending=[False, True])
        .head(5)
        .reset_index(drop=True)
    )
    total = max(int(churners["customer_id"].nunique()), 1)
    top5["share_of_churn_customers"] = top5["customer_count"] / total
    top5.insert(0, "rank", np.arange(1, len(top5) + 1))
    return top5


def _plot_top5_patterns(pattern_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if pattern_df.empty:
        ax.text(0.5, 0.5, "No churn sequence patterns available", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [f"#{row.rank} {row.pattern}" for row in pattern_df.itertuples(index=False)]
        ax.barh(labels, pattern_df["customer_count"])
        ax.invert_yaxis()
        ax.set_title("Top 5 Common Patterns in the Last 30 Days of Churn Customers")
        ax.set_xlabel("Customers")
        for y, value in enumerate(pattern_df["customer_count"]):
            ax.text(value, y, f" {int(value):,}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_pre_churn_event_frequency(events: pd.DataFrame, churners: pd.DataFrame) -> pd.DataFrame:
    if churners.empty:
        return pd.DataFrame(columns=["event_type", "event_count", "customer_count", "customer_share"])

    merged = events.merge(churners[["customer_id", "behavior_anchor_time"]], on="customer_id", how="inner")
    window = merged.loc[
        (merged["timestamp"] <= merged["behavior_anchor_time"])
        & (merged["timestamp"] >= merged["behavior_anchor_time"] - pd.Timedelta(days=30)),
        ["customer_id", "event_type"],
    ].copy()

    total_customers = max(int(churners["customer_id"].nunique()), 1)
    rows: list[dict] = []
    for event_type in MAJOR_PRE_CHURN_EVENTS:
        subset = window[window["event_type"] == event_type]
        rows.append(
            {
                "event_type": event_type,
                "event_count": int(len(subset)),
                "customer_count": int(subset["customer_id"].nunique()),
                "customer_share": float(subset["customer_id"].nunique() / total_customers),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_count", "customer_count", "event_type"], ascending=[False, False, True]).reset_index(drop=True)


def _plot_pre_churn_event_frequency(event_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if event_df.empty:
        ax.text(0.5, 0.5, "No pre-churn event frequency available", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(event_df["event_type"], event_df["event_count"])
        ax.set_title("Pre-Churn Major Event Frequency (Last 30 Days)")
        ax.set_ylabel("Event count")
        ax.tick_params(axis="x", rotation=35)
        for x, value in enumerate(event_df["event_count"]):
            ax.text(x, value, f"{int(value):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_funnel(customers: pd.DataFrame, orders: pd.DataFrame, churners: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_base = customers[["customer_id", "signup_date"]].copy()
    order_hist = orders.sort_values(["customer_id", "order_time"]).copy()
    order_hist["purchase_number"] = order_hist.groupby("customer_id").cumcount() + 1
    purchase_counts = order_hist.groupby("customer_id").size().rename("purchase_count")

    stage_df = customer_base.copy()
    stage_df["purchase_count"] = stage_df["customer_id"].map(purchase_counts).fillna(0).astype(int)
    stage_df["reached_first_purchase"] = stage_df["purchase_count"] >= 1
    stage_df["reached_repeat_purchase"] = stage_df["purchase_count"] >= 2
    stage_df["reached_loyal"] = stage_df["purchase_count"] >= 3
    stage_df["is_churned"] = stage_df["customer_id"].isin(set(churners["customer_id"]))

    total_signup = int(len(stage_df))
    first_purchase_count = int(stage_df["reached_first_purchase"].sum())
    repeat_purchase_count = int(stage_df["reached_repeat_purchase"].sum())
    loyal_count = int(stage_df["reached_loyal"].sum())
    churn_count = int(stage_df["is_churned"].sum())

    funnel = pd.DataFrame(
        [
            {
                "stage": "가입",
                "customer_count": total_signup,
                "stage_rate_from_signup": 1.0,
                "transition_rate_from_prev": 1.0,
            },
            {
                "stage": "첫구매",
                "customer_count": first_purchase_count,
                "stage_rate_from_signup": first_purchase_count / max(total_signup, 1),
                "transition_rate_from_prev": first_purchase_count / max(total_signup, 1),
            },
            {
                "stage": "재구매",
                "customer_count": repeat_purchase_count,
                "stage_rate_from_signup": repeat_purchase_count / max(total_signup, 1),
                "transition_rate_from_prev": repeat_purchase_count / max(first_purchase_count, 1),
            },
            {
                "stage": "충성",
                "customer_count": loyal_count,
                "stage_rate_from_signup": loyal_count / max(total_signup, 1),
                "transition_rate_from_prev": loyal_count / max(repeat_purchase_count, 1),
            },
            {
                "stage": "이탈",
                "customer_count": churn_count,
                "stage_rate_from_signup": churn_count / max(total_signup, 1),
                "transition_rate_from_prev": churn_count / max(loyal_count, 1),
            },
        ]
    )

    churn_timing = churners[["customer_id", "final_churn_start_date"]].merge(customer_base, on="customer_id", how="left")
    orders_before_churn = order_hist.merge(churn_timing[["customer_id", "final_churn_start_date"]], on="customer_id", how="inner")
    orders_before_churn = orders_before_churn[orders_before_churn["order_time"] <= orders_before_churn["final_churn_start_date"]].copy()
    purchase_before_churn = orders_before_churn.groupby("customer_id").size().rename("purchase_count_before_churn")

    churn_timing["purchase_count_before_churn"] = churn_timing["customer_id"].map(purchase_before_churn).fillna(0).astype(int)
    churn_timing["churn_stage"] = np.select(
        [
            churn_timing["purchase_count_before_churn"] >= 3,
            churn_timing["purchase_count_before_churn"] == 2,
            churn_timing["purchase_count_before_churn"] == 1,
        ],
        ["충성", "재구매", "첫구매"],
        default="가입",
    )
    churn_timing["days_to_churn"] = (
        churn_timing["final_churn_start_date"] - churn_timing["signup_date"]
    ).dt.days.clip(lower=0)

    churn_stage_summary = (
        churn_timing.groupby("churn_stage", as_index=False)
        .agg(
            churned_customers=("customer_id", "nunique"),
            avg_days_to_churn=("days_to_churn", "mean"),
            median_days_to_churn=("days_to_churn", "median"),
        )
        .sort_values(["churned_customers", "churn_stage"], ascending=[False, True])
        .reset_index(drop=True)
    )
    churn_stage_summary["share_of_churn_customers"] = churn_stage_summary["churned_customers"] / max(int(len(churn_timing)), 1)
    return funnel, churn_stage_summary


def _plot_funnel(funnel_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    labels = [PLOT_STAGE_LABELS.get(str(x), str(x)) for x in funnel_df["stage"]]
    ax.bar(labels, funnel_df["customer_count"])
    ax.set_title("Customer Journey Funnel Conversion")
    ax.set_ylabel("Customers")
    for x, row in enumerate(funnel_df.itertuples(index=False)):
        ax.text(x, row.customer_count, f"{row.stage_rate_from_signup:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_churn_timing(churn_timing_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if churn_timing_df.empty:
        ax.text(0.5, 0.5, "No churn timing information available", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [PLOT_STAGE_LABELS.get(str(x), str(x)) for x in churn_timing_df["churn_stage"]]
        ax.bar(labels, churn_timing_df["avg_days_to_churn"])
        ax.set_title("Average Churn Timing by Funnel Stage")
        ax.set_ylabel("Average days from signup to churn")
        for x, row in enumerate(churn_timing_df.itertuples(index=False)):
            ax.text(x, row.avg_days_to_churn, f"{row.avg_days_to_churn:.1f}d", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    report_path: Path,
    milestone_df: pd.DataFrame,
    top5_patterns: pd.DataFrame,
    event_frequency: pd.DataFrame,
    funnel_df: pd.DataFrame,
    churn_timing_df: pd.DataFrame,
) -> None:
    observed = milestone_df[milestone_df["observed"] == True].copy()
    lines = ["# Cohort and Journey Analysis", ""]
    if not observed.empty:
        milestone_summary = (
            observed.groupby("period", as_index=False)
            .agg(avg_retention=("retention_rate", "mean"), avg_churn=("churn_rate", "mean"))
            .sort_values("period")
        )
        lines.append("## Cohort retention / churn milestones")
        for row in milestone_summary.itertuples(index=False):
            lines.append(f"- M{int(row.period)} 평균 리텐션 {row.avg_retention:.2%}, 평균 이탈률 {row.avg_churn:.2%}")
        lines.append("")

    lines.append("## Last-30-day common patterns of churn customers")
    if top5_patterns.empty:
        lines.append("- 추출 가능한 패턴이 없습니다.")
    else:
        for row in top5_patterns.itertuples(index=False):
            lines.append(f"- #{int(row.rank)} {row.pattern}: {int(row.customer_count):,}명 ({row.share_of_churn_customers:.2%})")
    lines.append("")

    lines.append("## Pre-churn major event frequency")
    if event_frequency.empty:
        lines.append("- 분석 가능한 이벤트가 없습니다.")
    else:
        for row in event_frequency.head(5).itertuples(index=False):
            lines.append(f"- {row.event_type}: {int(row.event_count):,}회, {int(row.customer_count):,}명 ({row.customer_share:.2%})")
    lines.append("")

    lines.append("## Journey funnel")
    for row in funnel_df.itertuples(index=False):
        lines.append(
            f"- {row.stage}: {int(row.customer_count):,}명, 가입 기준 {row.stage_rate_from_signup:.2%}, 직전 단계 대비 {row.transition_rate_from_prev:.2%}"
        )
    lines.append("")

    lines.append("## Churn timing by stage")
    if churn_timing_df.empty:
        lines.append("- 분석 가능한 이탈 시점 정보가 없습니다.")
    else:
        for row in churn_timing_df.itertuples(index=False):
            lines.append(
                f"- {row.churn_stage}: {int(row.churned_customers):,}명, 평균 {row.avg_days_to_churn:.1f}일, 중앙값 {row.median_days_to_churn:.1f}일"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_cohort_and_journey_analysis(
    data_dir: str | Path = "data/raw",
    result_dir: str | Path = "results",
) -> CohortJourneyArtifacts:
    data_path = Path(data_dir)
    result_path = _ensure_directory(Path(result_dir))

    tables = _load_csvs(data_path)
    cohort_curve = _load_or_build_canonical_cohort_curve(data_path, tables)
    milestone_df = _build_retention_milestone_table(cohort_curve)

    churners = _final_churn_run_start(tables["snapshots"])
    churners = _resolve_behavior_anchor(tables["events"], churners)
    top5_patterns = _extract_top5_patterns(tables["events"], churners)
    event_frequency = _build_pre_churn_event_frequency(tables["events"], churners)
    funnel_df, churn_timing_df = _build_funnel(tables["customers"], tables["orders"], churners)

    retention_curve_path = result_path / "cohort_retention_curve.png"
    churn_heatmap_path = result_path / "cohort_churn_rate_heatmap.png"
    retention_milestone_csv_path = result_path / "cohort_retention_milestones.csv"
    sequence_csv_path = result_path / "churn_last30_top5_patterns.csv"
    sequence_plot_path = result_path / "churn_last30_top5_patterns.png"
    pre_churn_event_csv_path = result_path / "pre_churn_event_frequency.csv"
    pre_churn_event_plot_path = result_path / "pre_churn_event_frequency.png"
    funnel_csv_path = result_path / "journey_funnel_conversion.csv"
    funnel_plot_path = result_path / "journey_funnel_conversion.png"
    churn_timing_csv_path = result_path / "journey_churn_timing_by_stage.csv"
    churn_timing_plot_path = result_path / "journey_churn_timing_by_stage.png"
    summary_path = result_path / "cohort_journey_summary.json"
    report_path = result_path / "cohort_journey_report.md"

    milestone_df.to_csv(retention_milestone_csv_path, index=False)
    top5_patterns.to_csv(sequence_csv_path, index=False)
    event_frequency.to_csv(pre_churn_event_csv_path, index=False)
    funnel_df.to_csv(funnel_csv_path, index=False)
    churn_timing_df.to_csv(churn_timing_csv_path, index=False)

    _plot_retention_curve(cohort_curve, retention_curve_path)
    _plot_churn_heatmap(milestone_df, churn_heatmap_path)
    _plot_top5_patterns(top5_patterns, sequence_plot_path)
    _plot_pre_churn_event_frequency(event_frequency, pre_churn_event_plot_path)
    _plot_funnel(funnel_df, funnel_plot_path)
    _plot_churn_timing(churn_timing_df, churn_timing_plot_path)
    _write_report(report_path, milestone_df, top5_patterns, event_frequency, funnel_df, churn_timing_df)

    summary = {
        "analysis_basis": {
            "cohort_activity_definition": CANONICAL_ACTIVITY_DEFINITION,
            "cohort_retention_mode": CANONICAL_RETENTION_MODE,
            "retention_milestones": list(RETENTION_MILESTONES),
            "final_churn_customer_count": int(churners["customer_id"].nunique()),
            "behavior_anchor_source_distribution": churners["anchor_source"].value_counts().to_dict(),
        },
        "cohort_milestones": milestone_df.to_dict(orient="records"),
        "last30_top5_patterns": top5_patterns.to_dict(orient="records"),
        "pre_churn_event_frequency": event_frequency.to_dict(orient="records"),
        "journey_funnel": funnel_df.to_dict(orient="records"),
        "journey_churn_timing": churn_timing_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return CohortJourneyArtifacts(
        summary_path=str(summary_path),
        retention_curve_path=str(retention_curve_path),
        churn_heatmap_path=str(churn_heatmap_path),
        retention_milestone_csv_path=str(retention_milestone_csv_path),
        sequence_csv_path=str(sequence_csv_path),
        sequence_plot_path=str(sequence_plot_path),
        pre_churn_event_csv_path=str(pre_churn_event_csv_path),
        pre_churn_event_plot_path=str(pre_churn_event_plot_path),
        funnel_csv_path=str(funnel_csv_path),
        funnel_plot_path=str(funnel_plot_path),
        churn_timing_csv_path=str(churn_timing_csv_path),
        churn_timing_plot_path=str(churn_timing_plot_path),
        report_path=str(report_path),
    )
