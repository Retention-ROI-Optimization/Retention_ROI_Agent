from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_writer import write_json


def _as_float(series: pd.Series | float | int, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)
    return pd.Series(float(series))


def refresh_realtime_artifacts(
    predictions: pd.DataFrame,
    result_dir: str | Path,
    *,
    top_n: int = 200,
    processed_events: int = 0,
    source: str = "uploaded_customer_dataset",
) -> dict[str, Any]:
    """Refresh dashboard-compatible realtime snapshot files from the latest uploaded dataset.

    The existing realtime API already falls back to `results/realtime_scores_snapshot.csv` when Redis is not available.
    Writing these files makes the realtime dashboard reflect the newly onboarded customers without editing existing code.
    """
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    df = predictions.copy()
    if df.empty:
        snapshot = pd.DataFrame()
    else:
        churn = _as_float(df.get("churn_probability", df.get("proxy_churn_risk", 0.5)), 0.5)
        uplift = _as_float(df.get("uplift_score", 0.0), 0.0)
        recency = _as_float(df.get("recency_days", 0.0), 0.0)
        recency_boost = np.clip(recency / max(float(recency.quantile(0.95)) if len(recency) else 1.0, 1.0), 0.0, 1.0)
        realtime_score = np.clip(0.78 * churn + 0.14 * recency_boost + 0.08 * np.maximum(uplift, 0.0), 0.0, 1.0)
        base_score = churn
        score_delta = realtime_score - base_score
        action_status = np.where(realtime_score >= 0.70, "queued", np.where(realtime_score >= 0.50, "watch", "none"))
        trigger = np.where(realtime_score >= 0.70, "uploaded_data_high_risk", np.where(score_delta >= 0.08, "risk_delta", "baseline"))

        snapshot = pd.DataFrame(
            {
                "customer_id": df["customer_id"].astype(str),
                "persona": df.get("persona", "general"),
                "uplift_segment": df.get("uplift_segment", "Unknown"),
                "base_churn_probability": base_score,
                "realtime_churn_score": realtime_score,
                "score_delta": score_delta,
                "risk_percentile": pd.Series(realtime_score).rank(pct=True).to_numpy(),
                "last_event_type": "dataset_refresh",
                "latest_trigger_reason": trigger,
                "action_queue_status": action_status,
                "queued_recommended_action": np.where(action_status == "queued", df.get("recommended_action", "personalized_coupon"), ""),
                "queued_intervention_intensity": np.where(action_status == "queued", df.get("intervention_intensity", "medium"), ""),
                "queued_coupon_cost": np.where(action_status == "queued", _as_float(df.get("coupon_cost", 0.0), 0.0), 0.0),
                "queued_expected_profit": np.where(action_status == "queued", _as_float(df.get("expected_incremental_profit", 0.0), 0.0), 0.0),
                "queued_expected_roi": np.where(action_status == "queued", _as_float(df.get("expected_roi", 0.0), 0.0), 0.0),
                "reoptimization_count": np.where(action_status == "queued", 1, 0),
                "last_reoptimized_at": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
            }
        ).sort_values("realtime_churn_score", ascending=False)

    snapshot_path = result_dir / "realtime_scores_snapshot.csv"
    snapshot.head(int(top_n)).to_csv(snapshot_path, index=False)

    queued = snapshot[snapshot.get("action_queue_status", pd.Series(dtype=str)).astype(str) == "queued"].copy() if not snapshot.empty else pd.DataFrame()
    queue_path = result_dir / "realtime_action_queue_snapshot.csv"
    queued.to_csv(queue_path, index=False)

    summary = {
        "source": "snapshot",
        "dataset_source": source,
        "tracked_customers": int(len(snapshot)),
        "high_risk_customers": int((snapshot.get("realtime_churn_score", pd.Series(dtype=float)) >= 0.50).sum()) if not snapshot.empty else 0,
        "critical_risk_customers": int((snapshot.get("realtime_churn_score", pd.Series(dtype=float)) >= 0.80).sum()) if not snapshot.empty else 0,
        "snapshot_rows": int(min(len(snapshot), top_n)),
        "processed_events": int(processed_events),
        "triggered_reoptimizations": int(len(queued)),
        "action_queue_size": int(len(queued)),
        "high_priority_queue_size": int((queued.get("realtime_churn_score", pd.Series(dtype=float)) >= 0.80).sum()) if not queued.empty else 0,
        "closed_loop_budget_spent": int(round(pd.to_numeric(queued.get("queued_coupon_cost", 0), errors="coerce").fillna(0).sum())) if not queued.empty else 0,
        "daily_channel_capacity": 500,
        "daily_channel_allocated": int(min(len(queued), 500)),
        "generated_at": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
    }
    write_json(result_dir / "realtime_scores_summary.json", summary)
    write_json(result_dir / "realtime_action_queue_summary.json", {"queue_size": int(len(queued)), "high_priority_queue_size": summary["high_priority_queue_size"]})
    return {"summary": summary, "snapshot_path": str(snapshot_path), "queue_path": str(queue_path)}


def refresh_realtime_tick(result_dir: str | Path, *, batch_size: int = 250, top_n: int = 200) -> dict[str, Any]:
    result_dir = Path(result_dir)
    latest = result_dir / "customer_summary_latest.csv"
    if not latest.exists():
        latest = result_dir / "customer_segments.csv"
    if not latest.exists():
        latest = result_dir / "personalized_recommendations.csv"
    df = pd.read_csv(latest) if latest.exists() else pd.DataFrame()
    return refresh_realtime_artifacts(df, result_dir, top_n=top_n, processed_events=batch_size, source="autoops_tick")
