from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.workflows.pipeline_runner import (
    ensure_simulation_outputs,
    run_ab_test_pipeline,
    run_churn_training_pipeline,
    run_cohort_journey_pipeline,
    run_feature_engineering_pipeline,
    run_optimize_pipeline,
    run_recommendation_pipeline,
    run_realtime_bootstrap_pipeline,
    run_realtime_replay_pipeline,
    run_survival_pipeline,
    run_uplift_pipeline,
)


def bootstrap_data(data_dir: Path) -> None:
    """Make sure the API has the minimum raw simulator outputs it needs on startup."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_simulation_outputs(data_dir)


def run_mode(
    mode: str,
    data_dir: Path,
    model_dir: Path,
    result_dir: Path,
    *,
    feature_store_dir: Optional[Path] = None,
    budget: Optional[int] = None,
    threshold: float = 0.50,
    max_customers: Optional[int] = 1000,
    per_customer: int = 3,
    force_simulation: bool = False,
) -> Dict[str, Any]:
    resolved_mode = str(mode).strip().lower()

    common_kwargs = {
        "force_simulation": bool(force_simulation),
    }

    if resolved_mode == "features":
        return run_feature_engineering_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            feature_store_dir=feature_store_dir,
            **common_kwargs,
        )

    if resolved_mode == "train":
        return run_churn_training_pipeline(
            data_dir=data_dir,
            model_dir=model_dir,
            result_dir=result_dir,
            feature_store_dir=feature_store_dir,
            **common_kwargs,
        )

    if resolved_mode == "uplift":
        return run_uplift_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            **common_kwargs,
        )

    if resolved_mode == "optimize":
        return run_optimize_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            budget=int(budget or 5_000_000),
            **common_kwargs,
        )

    if resolved_mode == "abtest":
        return run_ab_test_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            **common_kwargs,
        )

    if resolved_mode == "cohort":
        return run_cohort_journey_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            **common_kwargs,
        )

    if resolved_mode == "recommend":
        return run_recommendation_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            budget=int(budget or 5_000_000),
            threshold=float(threshold),
            max_customers=max_customers,
            per_customer=int(per_customer),
            **common_kwargs,
        )

    if resolved_mode == "survival":
        return run_survival_pipeline(
            data_dir=data_dir,
            model_dir=model_dir,
            result_dir=result_dir,
            feature_store_dir=feature_store_dir,
            **common_kwargs,
        )

    if resolved_mode == "realtime-bootstrap":
        return run_realtime_bootstrap_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            **common_kwargs,
        )

    if resolved_mode == "realtime-replay":
        return run_realtime_replay_pipeline(
            data_dir=data_dir,
            result_dir=result_dir,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported pipeline mode: {mode}")
