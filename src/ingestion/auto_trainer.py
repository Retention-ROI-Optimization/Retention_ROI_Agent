"""
auto_trainer.py — End-to-end auto-training orchestrator.

After the preprocessor generates internal-schema tables, this module
runs the complete ML pipeline:
  1. Feature engineering
  2. Churn model training (XGBoost/LightGBM)
  3. Uplift modeling (T-Learner / S-Learner)
  4. CLV prediction
  5. Survival analysis (Cox PH)
  6. Segmentation & prioritization
  7. Budget optimization (dose-response + timing)
  8. Personalized recommendations
  9. A/B test analysis
  10. Explainability
  11. Cohort/journey analysis

All results are saved to the standard results/ directory so the
existing dashboard renders them without modification.
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AutoTrainResult:
    """Result of the auto-training pipeline."""
    success: bool
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _run_stage(name: str, fn, result: AutoTrainResult, **kwargs) -> Optional[Any]:
    """Run a pipeline stage with error handling."""
    try:
        output = fn(**kwargs)
        result.stages_completed.append(name)
        return output
    except Exception as exc:
        result.stages_failed[name] = f"{type(exc).__name__}: {exc}"
        print(f"[AutoTrainer] Stage '{name}' failed: {exc}")
        traceback.print_exc()
        return None


def run_auto_training_pipeline(
    data_dir: str | Path,
    model_dir: str | Path = "models",
    result_dir: str | Path = "results",
    feature_store_dir: str | Path = "data/feature_store",
    *,
    budget: int = 5_000_000,
    threshold: float = 0.50,
    max_customers: int = 1500,
    skip_realtime: bool = True,
) -> AutoTrainResult:
    """
    Run the complete ML training pipeline on preprocessed data.

    This is the main entry point after data ingestion/preprocessing.
    Each stage is independent and failure-tolerant — if one stage fails,
    the pipeline continues with the remaining stages.
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    feature_store_dir = Path(feature_store_dir)

    for d in [data_dir, model_dir, result_dir, feature_store_dir]:
        d.mkdir(parents=True, exist_ok=True)

    result = AutoTrainResult(success=False)

    # Verify required files exist
    required = ["customer_summary.csv", "events.csv", "orders.csv", "treatment_assignments.csv", "customers.csv"]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        result.stages_failed["validation"] = f"필수 파일이 없습니다: {', '.join(missing)}"
        return result

    result.stages_completed.append("validation")

    # ── Stage 1: Feature Engineering ──
    from src.workflows.pipeline_runner import run_feature_engineering_pipeline
    fe_result = _run_stage(
        "feature_engineering",
        run_feature_engineering_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
        feature_store_dir=feature_store_dir,
    )
    if fe_result:
        result.artifacts["feature_engineering"] = fe_result

    # ── Stage 2: Churn Model Training ──
    from src.workflows.pipeline_runner import run_churn_training_pipeline
    train_result = _run_stage(
        "churn_training",
        run_churn_training_pipeline,
        result,
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        feature_store_dir=feature_store_dir,
    )
    if train_result:
        result.artifacts["churn_training"] = train_result

    # ── Stage 3: Uplift Modeling ──
    from src.workflows.pipeline_runner import run_uplift_pipeline
    uplift_result = _run_stage(
        "uplift_modeling",
        run_uplift_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if uplift_result:
        result.artifacts["uplift_modeling"] = uplift_result

    # ── Stage 4: CLV Prediction ──
    from src.workflows.pipeline_runner import run_clv_prediction_pipeline
    clv_result = _run_stage(
        "clv_prediction",
        run_clv_prediction_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if clv_result:
        result.artifacts["clv_prediction"] = clv_result

    # ── Stage 5: Survival Analysis ──
    from src.workflows.pipeline_runner import run_survival_pipeline
    survival_result = _run_stage(
        "survival_analysis",
        run_survival_pipeline,
        result,
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
        feature_store_dir=feature_store_dir,
    )
    if survival_result:
        result.artifacts["survival_analysis"] = survival_result

    # ── Stage 6: Segmentation ──
    from src.workflows.pipeline_runner import run_segmentation_priority_pipeline
    seg_result = _run_stage(
        "segmentation",
        run_segmentation_priority_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if seg_result:
        result.artifacts["segmentation"] = seg_result

    # ── Stage 7: A/B Test Analysis ──
    from src.workflows.pipeline_runner import run_ab_test_pipeline
    ab_result = _run_stage(
        "ab_test",
        run_ab_test_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if ab_result:
        result.artifacts["ab_test"] = ab_result

    # ── Stage 8: Budget Optimization ──
    from src.workflows.pipeline_runner import run_optimize_pipeline
    opt_result = _run_stage(
        "budget_optimization",
        run_optimize_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
        budget=budget,
        model_dir=model_dir,
        feature_store_dir=feature_store_dir,
    )
    if opt_result:
        result.artifacts["budget_optimization"] = opt_result

    # ── Stage 9: Personalized Recommendations ──
    from src.workflows.pipeline_runner import run_recommendation_pipeline
    rec_result = _run_stage(
        "recommendations",
        run_recommendation_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
        budget=budget,
        threshold=threshold,
        max_customers=max_customers,
        model_dir=model_dir,
        feature_store_dir=feature_store_dir,
    )
    if rec_result:
        result.artifacts["recommendations"] = rec_result

    # ── Stage 10: Cohort & Journey Analysis ──
    from src.workflows.pipeline_runner import run_cohort_journey_pipeline
    cohort_result = _run_stage(
        "cohort_journey",
        run_cohort_journey_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if cohort_result:
        result.artifacts["cohort_journey"] = cohort_result

    # ── Stage 11: Explainability ──
    from src.workflows.pipeline_runner import run_explainability_pipeline
    explain_result = _run_stage(
        "explainability",
        run_explainability_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
        feature_store_dir=feature_store_dir,
    )
    if explain_result:
        result.artifacts["explainability"] = explain_result

    # ── Stage 12: Simulation Fidelity ──
    from src.workflows.pipeline_runner import run_simulation_fidelity_pipeline
    fidelity_result = _run_stage(
        "fidelity_audit",
        run_simulation_fidelity_pipeline,
        result,
        data_dir=data_dir,
        result_dir=result_dir,
    )
    if fidelity_result:
        result.artifacts["fidelity_audit"] = fidelity_result

    # ── Finalize ──
    total_stages = len(result.stages_completed) + len(result.stages_failed)
    result.success = len(result.stages_completed) >= 3  # at least validation + 2 ML stages
    result.metadata = {
        "total_stages": total_stages,
        "completed": len(result.stages_completed),
        "failed": len(result.stages_failed),
        "stages_completed": result.stages_completed,
        "stages_failed": result.stages_failed,
        "budget": budget,
        "threshold": threshold,
        "max_customers": max_customers,
    }

    # Save pipeline summary
    summary_path = result_dir / "auto_training_summary.json"
    summary_path.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result
