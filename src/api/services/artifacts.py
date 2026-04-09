from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.api.settings import ApiSettings
from src.api.services.analytics import get_budget_result
from src.optimization.timing import load_survival_predictions
from src.workflows.pipeline_runner import (
    run_churn_training_pipeline,
    run_optimize_pipeline,
    run_uplift_pipeline,
)


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _safe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _latest_mtime(paths: list[Path]) -> float:
    existing = [path.stat().st_mtime for path in paths if path.exists()]
    return max(existing) if existing else -1.0


def _is_stale(outputs: list[Path], dependencies: list[Path]) -> bool:
    if not outputs or any(not path.exists() for path in outputs):
        return True
    return _latest_mtime(outputs) < _latest_mtime(dependencies)


def _find_model_path(model_dir: Path) -> Optional[str]:
    candidates = list(model_dir.glob("churn_model_*.joblib"))
    if not candidates:
        candidates = list(model_dir.glob("churn_model*.joblib"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def resolve_training_request(
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    shap_sample_size: int = 300,
    models: str = "xgboost,lightgbm",
    threshold_tp_value: float = 120000.0,
    threshold_fp_cost: float = 18000.0,
    threshold_fn_cost: float = 60000.0,
) -> Dict[str, Any]:
    requested_models = [item.strip().lower() for item in str(models).split(",") if item.strip()]
    if not requested_models:
        requested_models = ["xgboost", "lightgbm"]

    return {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "shap_sample_size": int(shap_sample_size),
        "candidate_models": requested_models,
        "threshold_tp_value": float(threshold_tp_value),
        "threshold_fp_cost": float(threshold_fp_cost),
        "threshold_fn_cost": float(threshold_fn_cost),
    }


def training_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    feature_store_dir = settings.resolved_feature_store_dir
    required = [
        result_dir / "churn_metrics.json",
        result_dir / "churn_threshold_analysis.json",
        feature_store_dir / "customer_features.csv",
        feature_store_dir / "customer_features_metadata.json",
    ]
    if any(not path.exists() for path in required):
        return True
    if _find_model_path(settings.resolved_model_dir) is None:
        return True

    dependencies = [
        settings.resolved_data_dir / "customers.csv",
        settings.resolved_data_dir / "events.csv",
        settings.resolved_data_dir / "orders.csv",
        settings.resolved_data_dir / "state_snapshots.csv",
        settings.resolved_data_dir / "campaign_exposures.csv",
        settings.resolved_data_dir / "treatment_assignments.csv",
        settings.resolved_data_dir / "customer_summary.csv",
    ]
    outputs = required + [Path(_find_model_path(settings.resolved_model_dir) or "")]
    return _is_stale(outputs, dependencies)


def training_parameter_mismatch(settings: ApiSettings, requested: Dict[str, Any]) -> bool:
    metrics_path = settings.resolved_result_dir / "churn_metrics.json"
    if not metrics_path.exists():
        return True

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    saved = metrics.get("training_parameters", {})
    if not isinstance(saved, dict):
        return True

    normalized_saved = {
        "test_size": round(float(saved.get("test_size", -1.0)), 6),
        "random_state": int(saved.get("random_state", -1)),
        "shap_sample_size": int(saved.get("shap_sample_size", -1)),
        "candidate_models": [str(x).strip().lower() for x in saved.get("requested_models", [])],
        "threshold_tp_value": float(saved.get("threshold_tp_value", -1.0)),
        "threshold_fp_cost": float(saved.get("threshold_fp_cost", -1.0)),
        "threshold_fn_cost": float(saved.get("threshold_fn_cost", -1.0)),
    }
    normalized_requested = {
        "test_size": round(float(requested["test_size"]), 6),
        "random_state": int(requested["random_state"]),
        "shap_sample_size": int(requested["shap_sample_size"]),
        "candidate_models": [str(x).strip().lower() for x in requested["candidate_models"]],
        "threshold_tp_value": float(requested["threshold_tp_value"]),
        "threshold_fp_cost": float(requested["threshold_fp_cost"]),
        "threshold_fn_cost": float(requested["threshold_fn_cost"]),
    }
    return normalized_saved != normalized_requested


def uplift_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    outputs = [
        result_dir / "uplift_summary.json",
        result_dir / "uplift_segmentation.csv",
    ]
    dependencies = [
        settings.resolved_data_dir / "customer_summary.csv",
        settings.resolved_data_dir / "treatment_assignments.csv",
        settings.resolved_data_dir / "orders.csv",
    ]
    return _is_stale(outputs, dependencies)


def optimization_artifacts_missing(settings: ApiSettings) -> bool:
    result_dir = settings.resolved_result_dir
    outputs = [
        result_dir / "optimization_summary.json",
        result_dir / "optimization_segment_budget.csv",
        result_dir / "optimization_selected_customers.csv",
    ]
    dependencies = [
        result_dir / "customer_segments.csv",
        result_dir / "survival_predictions.csv",
    ]
    return _is_stale(outputs, dependencies)


def optimization_budget_mismatch(settings: ApiSettings, budget: int) -> bool:
    summary_path = settings.resolved_result_dir / "optimization_summary.json"
    if not summary_path.exists():
        return True
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return int(summary.get("budget", -1)) != int(budget)


def ensure_training_artifacts(
    settings: ApiSettings,
    *,
    rebuild: bool = False,
    training_request: Dict[str, Any] | None = None,
) -> None:
    training_request = training_request or resolve_training_request()
    if rebuild or training_artifacts_missing(settings) or training_parameter_mismatch(settings, training_request):
        run_churn_training_pipeline(
            data_dir=settings.resolved_data_dir,
            model_dir=settings.resolved_model_dir,
            result_dir=settings.resolved_result_dir,
            feature_store_dir=settings.resolved_feature_store_dir,
            test_size=float(training_request["test_size"]),
            random_state=int(training_request["random_state"]),
            shap_sample_size=int(training_request["shap_sample_size"]),
            candidate_models=list(training_request["candidate_models"]),
            threshold_tp_value=float(training_request["threshold_tp_value"]),
            threshold_fp_cost=float(training_request["threshold_fp_cost"]),
            threshold_fn_cost=float(training_request["threshold_fn_cost"]),
        )


def ensure_saved_results_artifacts(settings: ApiSettings, budget: int, rebuild: bool = False) -> None:
    if rebuild or uplift_artifacts_missing(settings):
        run_uplift_pipeline(
            data_dir=settings.resolved_data_dir,
            result_dir=settings.resolved_result_dir,
        )
    if rebuild or optimization_artifacts_missing(settings) or optimization_budget_mismatch(settings, budget):
        run_optimize_pipeline(
            data_dir=settings.resolved_data_dir,
            result_dir=settings.resolved_result_dir,
            budget=int(budget),
            model_dir=settings.resolved_model_dir,
            feature_store_dir=settings.resolved_feature_store_dir,
        )


def load_training_artifacts_payload(
    settings: ApiSettings,
    training_request: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result_dir = settings.resolved_result_dir
    model_dir = settings.resolved_model_dir
    feature_store_dir = settings.resolved_feature_store_dir
    churn_metrics = _safe_json(result_dir / "churn_metrics.json")

    return {
        "directories": {
            "result_dir": str(result_dir),
            "model_dir": str(model_dir),
            "feature_store_dir": str(feature_store_dir),
        },
        "feature_summary": _safe_json(result_dir / "feature_engineering_summary.json"),
        "customer_features": _safe_csv(feature_store_dir / "customer_features.csv").head(200).to_dict(orient="records"),
        "customer_features_metadata": _safe_json(feature_store_dir / "customer_features_metadata.json"),
        "churn_metrics": churn_metrics,
        "threshold_analysis": _safe_json(result_dir / "churn_threshold_analysis.json"),
        "top_feature_importance": _safe_json_df(result_dir / "churn_top10_feature_importance.json").to_dict(orient="records"),
        "image_paths": {
            "churn_auc_roc": _safe_path(result_dir / "churn_auc_roc.png"),
            "churn_precision_recall_tradeoff": _safe_path(result_dir / "churn_precision_recall_tradeoff.png"),
            "churn_shap_summary": _safe_path(result_dir / "churn_shap_summary.png"),
            "churn_shap_local": _safe_path(result_dir / "churn_shap_local.png"),
        },
        "model_paths": {
            "churn_model": _find_model_path(model_dir),
        },
        "training_parameters": training_request or churn_metrics.get("training_parameters", {}),
    }


def load_saved_results_payload(
    settings: ApiSettings,
    budget: int,
    threshold: float = 0.50,
    max_customers: Optional[int] = None,
) -> Dict[str, Any]:
    result_dir = settings.resolved_result_dir
    data_dir = settings.resolved_data_dir

    uplift_summary = _safe_json(result_dir / "uplift_summary.json")
    uplift_segmentation = _safe_csv(result_dir / "uplift_segmentation.csv")

    customers = _safe_csv(data_dir / "customer_summary.csv")
    selected_customers, optimization_summary, optimization_segment_budget = get_budget_result(
        customers=customers,
        budget=int(budget),
        threshold=float(threshold),
        max_customers=max_customers,
        survival_predictions=load_survival_predictions(result_dir),
    )
    optimization_summary["live_generated"] = True

    return {
        "result_dir": str(result_dir),
        "uplift_summary": uplift_summary,
        "uplift_segmentation": uplift_segmentation.head(200).to_dict(orient="records"),
        "optimization_summary": optimization_summary,
        "optimization_segment_budget": optimization_segment_budget.to_dict(orient="records"),
        "optimization_selected_customers": selected_customers.head(200).to_dict(orient="records"),
        "parameters": {
            "budget": int(budget),
            "threshold": float(threshold),
            "max_customers": int(max_customers) if max_customers is not None else None,
        },
    }
